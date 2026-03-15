"""Pipeline package — pure-function rewrite of create_typical_periods."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsam.options import options
from tsam.pipeline.accuracy import reconstruct
from tsam.pipeline.clustering import (
    cluster_periods,
    cluster_sorted_periods,
    use_predefined_assignments,
)
from tsam.pipeline.extremes import add_extreme_periods
from tsam.pipeline.normalize import denormalize, normalize
from tsam.pipeline.periods import add_period_sum_features, unstack_to_periods
from tsam.pipeline.rescale import rescale_representatives
from tsam.pipeline.segment import segment_typical_periods
from tsam.pipeline.types import (
    ClusteringOutput,
    FormattedOutput,
    PipelineConfig,
    PipelineResult,
    PredefParams,  # noqa: F401 (re-exported)
    PreparedData,
)

if TYPE_CHECKING:
    from tsam.config import (
        Distribution,
        MinMaxMean,
    )


def _count_occurrences(cluster_order: np.ndarray) -> dict[int, float]:
    """Count how many original periods each cluster represents.

    Returns float values because partial-period adjustment (step 9)
    can produce fractional counts downstream.
    """
    nums, counts = np.unique(cluster_order, return_counts=True)
    return {int(num): float(counts[ii]) for ii, num in enumerate(nums)}


def _representatives_to_dataframe(
    cluster_periods_list: list[np.ndarray],
    column_index: pd.MultiIndex,
) -> pd.DataFrame:
    """Reshape flat cluster period vectors into a MultiIndex DataFrame.

    Converts a list of 1-D arrays (one per cluster) into a DataFrame
    indexed by (PeriodNum, TimeStep) with the original column names.
    """
    df = (
        pd.concat(
            [pd.Series(s, index=column_index) for s in cluster_periods_list],
            axis=1,
        )
        .unstack("TimeStep")
        .T
    )
    assert isinstance(df, pd.DataFrame)
    return df


def _warn_if_out_of_bounds(
    typical_periods: pd.DataFrame,
    original_data: pd.DataFrame,
    tolerance: float,
) -> None:
    """Warn if aggregated values exceed original data bounds."""
    exceeds_max = typical_periods.max(axis=0) > original_data.max(axis=0)
    if exceeds_max.any():
        diff = typical_periods.max(axis=0) - original_data.max(axis=0)
        exceeding_diff = diff[exceeds_max]
        if exceeding_diff.max() > tolerance:
            warnings.warn(
                "At least one maximal value of the "
                + "aggregated time series exceeds the maximal value "
                + "the input time series for: "
                + f"{exceeding_diff.to_dict()}"
                + ". To silence the warning set the 'numerical_tolerance' to a higher value."
            )
    below_min = typical_periods.min(axis=0) < original_data.min(axis=0)
    if below_min.any():
        diff = original_data.min(axis=0) - typical_periods.min(axis=0)
        exceeding_diff = diff[below_min]
        if exceeding_diff.max() > tolerance:
            warnings.warn(
                "Something went wrong... At least one minimal value of the "
                + "aggregated time series exceeds the minimal value "
                + "the input time series for: "
                + f"{exceeding_diff.to_dict()}"
                + ". To silence the warning set the 'numerical_tolerance' to a higher value."
            )


def _weight_candidates(
    candidates: np.ndarray,
    weight_values: list[float],
    n_timesteps: int,
) -> np.ndarray:
    """Apply per-column weights to the flat candidates array.

    Each row of candidates has shape (n_columns * n_timesteps,) with columns
    interleaved in the MultiIndex order (col0_t0, col0_t1, ..., col1_t0, ...).

    ``weight_values`` must have exactly one entry per column, in the same
    order as the columns in the candidates array.
    """
    weighted: np.ndarray = candidates.copy()
    for ci, w in enumerate(weight_values):
        start = ci * n_timesteps
        end = start + n_timesteps
        weighted[:, start:end] *= w
    return weighted


def _build_weight_vector(
    columns: pd.Index,
    weights: dict[str, float] | None,
) -> list[float] | None:
    """Build a weight list aligned to *columns*, defaulting unlisted columns to 1.0.

    Returns ``None`` if all weights are 1.0 (no weighting needed).
    """
    if not weights:
        return None
    result: list[float] = []
    any_non_unit = False
    for col in columns:
        w = weights.get(col, 1.0)
        if w < options.min_weight:
            warnings.warn(
                f'weight of "{col}" set to the minimal tolerable weighting',
                stacklevel=2,
            )
            w = options.min_weight
        if w != 1.0:
            any_non_unit = True
        result.append(w)
    return result if any_non_unit else None


def _build_representation_dict(
    columns: pd.Index,
    cluster_representation: str | Distribution | MinMaxMean | None,
) -> dict[str, str]:
    """Build the representation dict (mean/min/max per column) from config."""
    from tsam.config import MinMaxMean

    representation_dict: dict[str, str] = dict.fromkeys(columns, "mean")
    if isinstance(cluster_representation, MinMaxMean):
        for col in cluster_representation.max_columns:
            if col in representation_dict:
                representation_dict[col] = "max"
        for col in cluster_representation.min_columns:
            if col in representation_dict:
                representation_dict[col] = "min"
    return representation_dict


def _prepare_data(
    data: pd.DataFrame,
    cfg: PipelineConfig,
) -> PreparedData:
    """Phase 1: Build representation dict, normalize, unstack, weight (steps 1-3)."""
    cluster = cfg.cluster
    cluster_representation = cluster.get_representation()
    representation_dict = _build_representation_dict(
        data.columns, cluster_representation
    )
    original_column_order = list(data.columns)
    original_data = data.copy()

    # Step 1: Normalize
    norm_data = normalize(data, cluster.scale_by_column_means)

    # Step 2: Unstack to periods
    period_profiles = unstack_to_periods(norm_data.values, cfg.n_timesteps_per_period)
    candidates = period_profiles.profiles_dataframe.values

    # Step 2b: Create weighted candidates for clustering distance
    validated_weights = _build_weight_vector(norm_data.values.columns, cluster.weights)
    weighted_candidates: np.ndarray | None = None
    if validated_weights is not None:
        weighted_candidates = _weight_candidates(
            candidates,
            validated_weights,
            period_profiles.n_timesteps_per_period,
        )

    # Step 3: Add period sum features if requested
    # Period sums are extra columns appended for clustering distance only;
    # they must NOT reach representations() which expects original columns.
    n_feature_cols = candidates.shape[1]
    has_period_sums = False
    if cluster.include_period_sums:
        has_period_sums = True
        if weighted_candidates is not None:
            weighted_candidates, _n_extra = add_period_sum_features(
                period_profiles.profiles_dataframe, weighted_candidates
            )
        else:
            # Augmented matrix is used for clustering distance only;
            # candidates stays unaugmented for representation.
            augmented, _n_extra = add_period_sum_features(
                period_profiles.profiles_dataframe, candidates
            )
            weighted_candidates = augmented

    return PreparedData(
        norm_data=norm_data,
        period_profiles=period_profiles,
        candidates=candidates,
        weighted_candidates=weighted_candidates,
        representation_dict=representation_dict,
        n_feature_cols=n_feature_cols,
        original_column_order=original_column_order,
        original_data=original_data,
        has_period_sums=has_period_sums,
    )


def _cluster_and_postprocess(
    prepared: PreparedData,
    cfg: PipelineConfig,
    data_length: int,
) -> ClusteringOutput:
    """Phase 2: Cluster, trim, extremes, counts, rescale, partial (steps 4-9)."""
    cluster = cfg.cluster
    cluster_representation = cluster.get_representation()
    candidates = prepared.candidates
    weighted_candidates = prepared.weighted_candidates
    period_profiles = prepared.period_profiles

    # Step 4: Cluster
    clustering_duration = 0.0
    cluster_center_indices: list[int] | None = None

    if cfg.predef is not None:
        cluster_centers, cluster_center_indices, cluster_order = (
            use_predefined_assignments(
                candidates,
                cfg.predef,
                cluster_representation,
                prepared.representation_dict,
                cfg.n_timesteps_per_period,
            )
        )
    else:
        t_start = time.time()
        if not cluster.use_duration_curves:
            cluster_centers, cluster_center_indices, cluster_order = cluster_periods(
                candidates,
                cfg.n_clusters,
                cluster,
                prepared.representation_dict,
                cfg.n_timesteps_per_period,
                weighted_candidates=weighted_candidates,
            )
        else:
            # Duration-curve clustering sorts profiles — period-sum features
            # are not sortable columns, so don't pass them as weighted candidates.
            dc_weighted = (
                weighted_candidates
                if weighted_candidates is not None and not prepared.has_period_sums
                else None
            )
            if (
                weighted_candidates is not None
                and prepared.has_period_sums
                and dc_weighted is None
            ):
                warnings.warn(
                    "Column weights are ignored for duration-curve clustering when "
                    "include_period_sums=True, because period-sum features cannot "
                    "be sorted. Either disable include_period_sums or "
                    "use_duration_curves to apply weights.",
                    UserWarning,
                    stacklevel=2,
                )
            cluster_centers, cluster_center_indices, cluster_order = (
                cluster_sorted_periods(
                    candidates,
                    period_profiles,
                    cfg.n_clusters,
                    cluster,
                    prepared.representation_dict,
                    cfg.n_timesteps_per_period,
                    weighted_candidates=dc_weighted,
                )
            )
        clustering_duration = time.time() - t_start

    # Ensure cluster_order is always np.ndarray
    cluster_order = np.asarray(cluster_order)

    # Step 5: Trim eval features from representatives
    cluster_periods_list: list[np.ndarray] = [
        center[: prepared.n_feature_cols] for center in cluster_centers
    ]

    # Step 6: Add extreme periods if configured
    extreme_periods_info: dict[str, dict] = {}
    extreme_cluster_idx: list[int] = []

    if cfg.extremes is not None:
        (
            cluster_periods_list,
            cluster_order,
            extreme_cluster_idx,
            extreme_periods_info,
        ) = add_extreme_periods(
            period_profiles.profiles_dataframe,
            cluster_periods_list,
            cluster_order,
            cfg.extremes,
        )
        cluster_order = np.asarray(cluster_order)
    else:
        if cfg.predef is not None and cfg.predef.extreme_cluster_idx is not None:
            extreme_cluster_idx = list(cfg.predef.extreme_cluster_idx)

    # Step 7: Compute cluster counts
    cluster_counts = _count_occurrences(cluster_order)

    # Step 8: Rescale if requested
    rescale_deviations: dict[str, dict] = {}
    rescale_exclude = cfg.rescale_exclude_columns or []
    if cfg.rescale_cluster_periods:
        cluster_periods_list, rescale_deviations = rescale_representatives(  # type: ignore[assignment]
            cluster_periods_list,
            cluster_counts,
            extreme_cluster_idx,
            period_profiles.profiles_dataframe,
            prepared.original_data,
            prepared.norm_data.scale_by_column_means,
            cfg.n_timesteps_per_period,
            rescale_exclude,
        )
        cluster_periods_list = list(cluster_periods_list)

    # Step 9: Adjust for partial periods
    if data_length % cfg.n_timesteps_per_period != 0:
        last_cluster = int(cluster_order[-1])
        cluster_counts[last_cluster] -= (
            1
            - float(data_length % cfg.n_timesteps_per_period)
            / cfg.n_timesteps_per_period
        )

    return ClusteringOutput(
        cluster_periods_list=cluster_periods_list,
        cluster_order=cluster_order,
        cluster_counts=cluster_counts,
        cluster_center_indices=cluster_center_indices,
        extreme_cluster_idx=extreme_cluster_idx,
        extreme_periods_info=extreme_periods_info,
        clustering_duration=clustering_duration,
        rescale_deviations=rescale_deviations,
    )


def _format_and_reconstruct(
    prepared: PreparedData,
    clustered: ClusteringOutput,
    cfg: PipelineConfig,
) -> FormattedOutput:
    """Phase 3: Format, segment, denorm, bounds, reconstruct + accuracy (steps 10-14)."""
    norm_data = prepared.norm_data
    period_profiles = prepared.period_profiles

    # Step 10: Format representatives to MultiIndex DataFrame
    normalized_typical_periods = _representatives_to_dataframe(
        clustered.cluster_periods_list, period_profiles.column_index
    )

    # Step 11: Segmentation if configured
    segmented_df = None
    predicted_segmented_df = None
    segment_center_indices = None

    if cfg.segments is not None:
        segmented_df, predicted_segmented_df, segment_center_indices = (
            segment_typical_periods(
                normalized_typical_periods,
                cfg.n_timesteps_per_period,
                cfg.segments,
                prepared.representation_dict,
                cfg.predef,
            )
        )
        segmented_normalized = segmented_df.reset_index(level=3, drop=True)
        denorm_source = segmented_normalized
        reconstruct_source = predicted_segmented_df
    else:
        denorm_source = normalized_typical_periods
        reconstruct_source = normalized_typical_periods

    # Step 12: Denormalize -> typical_periods
    typical_periods = denormalize(denorm_source, norm_data)
    if cfg.round_decimals is not None:
        typical_periods = typical_periods.round(decimals=cfg.round_decimals)

    # Step 13: Bounds check + warnings
    _warn_if_out_of_bounds(
        typical_periods, prepared.original_data, cfg.numerical_tolerance
    )

    # Step 14: Reconstruct + compute accuracy
    reconstructed_data, normalized_predicted = reconstruct(
        reconstruct_source,
        clustered.cluster_order,
        period_profiles,
        norm_data,
        prepared.original_data,
    )
    if cfg.round_decimals is not None:
        reconstructed_data = reconstructed_data.round(decimals=cfg.round_decimals)

    # Restore original column order
    typical_periods = typical_periods[prepared.original_column_order]
    reconstructed_data = reconstructed_data[prepared.original_column_order]

    return FormattedOutput(
        typical_periods=typical_periods,
        reconstructed_data=reconstructed_data,
        normalized_predicted=normalized_predicted,
        segmented_df=segmented_df,
        segment_center_indices=segment_center_indices,
    )


def _assemble_result(
    prepared: PreparedData,
    clustered: ClusteringOutput,
    formatted: FormattedOutput,
    cfg: PipelineConfig,
) -> PipelineResult:
    """Phase 4: Build ClusteringResult + PipelineResult (steps 15-16)."""
    from tsam.config import ClusteringResult as _ClusteringResult

    original_data_out = prepared.original_data[prepared.original_column_order]

    clustering_result = _ClusteringResult.from_pipeline(
        cluster_center_indices=clustered.cluster_center_indices,
        extreme_periods_info=clustered.extreme_periods_info,
        extremes_config=cfg.extremes,
        cluster_order=clustered.cluster_order,
        segmented_df=formatted.segmented_df,
        segment_center_indices=formatted.segment_center_indices,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        temporal_resolution=cfg.temporal_resolution,
        original_data=original_data_out,
        cluster_config=cfg.cluster,
        segment_config=cfg.segments,
        rescale_cluster_periods=cfg.rescale_cluster_periods,
        rescale_exclude_columns=cfg.rescale_exclude_columns or [],
        extreme_cluster_idx=clustered.extreme_cluster_idx,
    )

    return PipelineResult(
        typical_periods=formatted.typical_periods,
        cluster_counts=clustered.cluster_counts,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        time_index=prepared.period_profiles.time_index,
        original_data=original_data_out,
        clustering_duration=clustered.clustering_duration,
        rescale_deviations=clustered.rescale_deviations,
        segmented_df=formatted.segmented_df,
        reconstructed_data=formatted.reconstructed_data,
        _norm_values=prepared.norm_data.values,
        _normalized_predicted=formatted.normalized_predicted,
        clustering_result=clustering_result,
    )


def run_pipeline(
    data: pd.DataFrame,
    cfg: PipelineConfig,
) -> PipelineResult:
    """Run the full aggregation pipeline.

    This replaces create_typical_periods() + predict_original_data() + accuracy_indicators().
    """
    prepared = _prepare_data(data, cfg)

    clustered = _cluster_and_postprocess(
        prepared,
        cfg,
        data_length=len(data),
    )

    formatted = _format_and_reconstruct(
        prepared,
        clustered,
        cfg,
    )

    return _assemble_result(
        prepared,
        clustered,
        formatted,
        cfg,
    )
