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
    PipelineConfig,
    PipelineResult,
    PredefParams,  # noqa: F401 (re-exported)
)

if TYPE_CHECKING:
    from tsam.config import (
        Distribution,
        MinMaxMean,
    )


# ──────────────────────────────────────────────────────────────────────
# Transformation helpers used once by run_pipeline().
# Each one is a pure transform — no orchestration logic. They live here
# because they're only meaningful in pipeline context; inlining them
# would make run_pipeline() harder to read without removing real state.
# ──────────────────────────────────────────────────────────────────────


def _representatives_to_dataframe(
    cluster_periods_list: list[np.ndarray],
    column_index: pd.MultiIndex,
) -> pd.DataFrame:
    """Reshape flat cluster period vectors into a (PeriodNum, TimeStep) DataFrame."""
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


def _apply_weights_df(
    df: pd.DataFrame, weights: dict[str, float] | None
) -> pd.DataFrame:
    """Multiply DataFrame columns by weights for segmentation distance."""
    if not weights:
        return df
    out = df.copy()
    for col, w in weights.items():
        if col in out.columns:
            out[col] *= w
    return pd.DataFrame(out)


def _remove_weights_df(
    df: pd.DataFrame, weights: dict[str, float] | None
) -> pd.DataFrame:
    """Divide out weights after segmentation to restore unweighted values."""
    if not weights:
        return df
    out = df.copy()
    for col, w in weights.items():
        if col in out.columns:
            out[col] /= w
    return pd.DataFrame(out)


def _build_weight_vector(
    columns: pd.Index,
    weights: dict[str, float] | None,
) -> np.ndarray | None:
    """Build a weight array aligned to *columns*, defaulting unlisted columns to 1.0.

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
    return np.array(result) if any_non_unit else None


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


# ──────────────────────────────────────────────────────────────────────
# Orchestrator: the single source of truth for stage ordering,
# conditional branching, and how PipelineConfig → PipelineResult flows.
# ──────────────────────────────────────────────────────────────────────


def run_pipeline(
    data: pd.DataFrame,
    cfg: PipelineConfig,
) -> PipelineResult:
    """Run the full aggregation pipeline end to end.

    All stage ordering, conditional branching, and config consumption lives
    here. The eight stage modules (``normalize``, ``periods``, ``clustering``,
    ``extremes``, ``rescale``, ``segment``, ``accuracy``) are pure transforms
    with no pipeline awareness; this function wires them together.

    Replaces create_typical_periods() + predict_original_data() +
    accuracy_indicators() from the v3 class-based API.
    """
    from tsam.config import ClusteringResult as _ClusteringResult

    cluster = cfg.cluster
    cluster_representation = cluster.get_representation()
    representation_dict = _build_representation_dict(
        data.columns, cluster_representation
    )
    original_column_order = list(data.columns)
    original_data = data.copy()

    # ── Step 1 — normalize ──────────────────────────────────────────────
    norm_data = normalize(data, cluster.scale_by_column_means)

    # ── Step 2 — unstack to periods ─────────────────────────────────────
    period_profiles = unstack_to_periods(norm_data.values, cfg.n_timesteps_per_period)
    candidates = period_profiles.profiles_dataframe.values

    # ── Step 2a — apply weights (optional) ──────────────────────────────
    # Weights are baked into the candidates array so clustering distance
    # respects them. A second weighted DataFrame is kept around with column
    # labels for extremes/segmentation.
    weight_vector = _build_weight_vector(norm_data.values.columns, cluster.weights)
    weighted_profiles_df: pd.DataFrame | None = None
    if weight_vector is not None:
        weight_tile = np.repeat(weight_vector, period_profiles.n_timesteps_per_period)
        candidates = candidates * weight_tile
        wpdf = period_profiles.profiles_dataframe.copy()
        for col_name, w in zip(
            wpdf.columns.get_level_values(0).unique(),
            weight_vector,
        ):
            wpdf[col_name] *= w
        weighted_profiles_df = wpdf

    # ── Step 2b — period-sum features (optional) ────────────────────────
    # Period sums are extra columns appended for clustering distance only;
    # they must NOT reach representations() which expects original columns.
    # We remember the original width so we can trim them off after clustering.
    n_feature_cols = candidates.shape[1]
    if cluster.include_period_sums:
        candidates, _n_extra = add_period_sum_features(
            period_profiles.profiles_dataframe, candidates
        )

    # ── Step 3 — cluster ────────────────────────────────────────────────
    clustering_duration = 0.0
    cluster_center_indices: list[int] | None = None

    if cfg.predef is not None:
        cluster_centers, cluster_center_indices, cluster_order = (
            use_predefined_assignments(
                candidates,
                cfg.predef,
                cluster_representation,
                representation_dict,
                cfg.n_timesteps_per_period,
            )
        )
    else:
        t_start = time.time()
        # When period-sum features are appended, representations must run on
        # the non-augmented prefix so period-sum columns don't leak in.
        rep_candidates: np.ndarray | None = None
        if candidates.shape[1] != n_feature_cols:
            rep_candidates = candidates[:, :n_feature_cols]

        if not cluster.use_duration_curves:
            cluster_centers, cluster_center_indices, cluster_order = cluster_periods(
                candidates,
                cfg.n_clusters,
                cluster,
                representation_dict,
                cfg.n_timesteps_per_period,
                representation_candidates=rep_candidates,
            )
        else:
            cluster_centers, cluster_center_indices, cluster_order = (
                cluster_sorted_periods(
                    candidates,
                    period_profiles.n_columns,
                    cfg.n_clusters,
                    cluster,
                    representation_dict,
                    cfg.n_timesteps_per_period,
                )
            )
        clustering_duration = time.time() - t_start

    cluster_order = np.asarray(cluster_order)

    # ── Step 4 — trim period-sum extras (still weighted) ────────────────
    cluster_periods_list: list[np.ndarray] = [
        center[:n_feature_cols] for center in cluster_centers
    ]

    # ── Step 5 — extreme periods (optional, still weighted) ─────────────
    # Extremes run in weighted space: weighted profiles determine which
    # period is extreme, and extracted profiles carry weights. Unweighting
    # below treats every center (regular + extreme) the same way.
    extreme_periods_info: dict[str, dict] = {}
    extreme_cluster_idx: list[int] = []

    if cfg.extremes is not None:
        profiles_for_extremes = (
            weighted_profiles_df
            if weighted_profiles_df is not None
            else period_profiles.profiles_dataframe
        )
        (
            cluster_periods_list,
            cluster_order,
            extreme_cluster_idx,
            extreme_periods_info,
        ) = add_extreme_periods(
            profiles_for_extremes,
            cluster_periods_list,
            cluster_order,
            cfg.extremes,
        )
        cluster_order = np.asarray(cluster_order)
    elif cfg.predef is not None and cfg.predef.extreme_cluster_idx is not None:
        extreme_cluster_idx = list(cfg.predef.extreme_cluster_idx)

    # ── Unweight representatives ────────────────────────────────────────
    # Downstream steps (rescale, denormalize, reconstruct) expect unweighted
    # data, so remove weights from every center now.
    if weight_vector is not None:
        inv_tile = np.repeat(1.0 / weight_vector, cfg.n_timesteps_per_period)
        cluster_periods_list = [center * inv_tile for center in cluster_periods_list]

    # ── Step 6 — count cluster occurrences ──────────────────────────────
    occurrence_nums, occurrence_counts = np.unique(cluster_order, return_counts=True)
    cluster_counts: dict[int, float] = {
        int(num): float(occurrence_counts[ii]) for ii, num in enumerate(occurrence_nums)
    }

    # ── Step 7 — rescale (optional) ─────────────────────────────────────
    rescale_deviations: dict[str, dict] = {}
    rescale_exclude = cfg.rescale_exclude_columns or []
    if cfg.rescale_cluster_periods:
        cluster_periods_list, rescale_deviations = rescale_representatives(  # type: ignore[assignment]
            cluster_periods_list,
            cluster_counts,
            extreme_cluster_idx,
            period_profiles.profiles_dataframe,
            original_data,
            norm_data.scale_by_column_means,
            cfg.n_timesteps_per_period,
            rescale_exclude,
        )
        cluster_periods_list = list(cluster_periods_list)

    # ── Step 8 — partial-period adjustment ──────────────────────────────
    # If the input doesn't divide evenly into periods the last one was
    # padded in step 2; shrink its weight proportionally so totals match.
    data_length = len(data)
    if data_length % cfg.n_timesteps_per_period != 0:
        last_cluster = int(cluster_order[-1])
        cluster_counts[last_cluster] -= (
            1
            - float(data_length % cfg.n_timesteps_per_period)
            / cfg.n_timesteps_per_period
        )

    # ── Step 9 — format representatives to a MultiIndex DataFrame ───────
    normalized_typical_periods = _representatives_to_dataframe(
        cluster_periods_list, period_profiles.column_index
    )

    # ── Step 10 — segment typical periods (optional) ────────────────────
    # Segmentation runs in weighted space so high-weight columns drive
    # segment boundaries. Weights are stripped from both outputs before
    # denormalization and reconstruction.
    segmented_df: pd.DataFrame | None = None
    predicted_segmented_df: pd.DataFrame | None = None
    segment_center_indices: list | None = None

    if cfg.segments is not None:
        weights = cluster.weights
        segmentation_input = _apply_weights_df(normalized_typical_periods, weights)
        segmented_df, predicted_segmented_df, segment_center_indices = (
            segment_typical_periods(
                segmentation_input,
                cfg.n_timesteps_per_period,
                cfg.segments,
                representation_dict,
                cfg.predef,
            )
        )
        segmented_df = _remove_weights_df(segmented_df, weights)
        predicted_segmented_df = _remove_weights_df(predicted_segmented_df, weights)
        segmented_normalized = segmented_df.reset_index(level=3, drop=True)
        denorm_source = segmented_normalized
        reconstruct_source = predicted_segmented_df
    else:
        denorm_source = normalized_typical_periods
        reconstruct_source = normalized_typical_periods

    # ── Step 11 — denormalize ───────────────────────────────────────────
    typical_periods = denormalize(denorm_source, norm_data)
    if cfg.round_decimals is not None:
        typical_periods = typical_periods.round(decimals=cfg.round_decimals)

    # ── Step 12 — bounds check ──────────────────────────────────────────
    _warn_if_out_of_bounds(typical_periods, original_data, cfg.numerical_tolerance)

    # ── Step 13 — reconstruct + accuracy ────────────────────────────────
    reconstructed_data, normalized_predicted = reconstruct(
        reconstruct_source,
        cluster_order,
        period_profiles,
        norm_data,
        original_data,
    )
    if cfg.round_decimals is not None:
        reconstructed_data = reconstructed_data.round(decimals=cfg.round_decimals)

    # Restore original column order
    typical_periods = typical_periods[original_column_order]
    reconstructed_data = reconstructed_data[original_column_order]

    # ── Step 14 — assemble result ───────────────────────────────────────
    original_data_out = original_data[original_column_order]
    input_time_index = (
        original_data_out.index
        if isinstance(original_data_out.index, pd.DatetimeIndex)
        else None
    )

    clustering_result = _ClusteringResult.from_pipeline(
        cluster_center_indices=cluster_center_indices,
        extreme_periods_info=extreme_periods_info,
        extremes_config=cfg.extremes,
        cluster_order=cluster_order,
        segmented_df=segmented_df,
        segment_center_indices=segment_center_indices,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        temporal_resolution=cfg.temporal_resolution,
        original_data=original_data_out,
        cluster_config=cfg.cluster,
        segment_config=cfg.segments,
        rescale_cluster_periods=cfg.rescale_cluster_periods,
        rescale_exclude_columns=cfg.rescale_exclude_columns or [],
        extreme_cluster_idx=extreme_cluster_idx,
        time_index=input_time_index,
    )

    return PipelineResult(
        typical_periods=typical_periods,
        cluster_counts=cluster_counts,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        time_index=period_profiles.time_index,
        original_data=original_data_out,
        clustering_duration=clustering_duration,
        rescale_deviations=rescale_deviations,
        segmented_df=segmented_df,
        reconstructed_data=reconstructed_data,
        _norm_values=norm_data.values,
        _normalized_predicted=normalized_predicted,
        clustering_result=clustering_result,
    )
