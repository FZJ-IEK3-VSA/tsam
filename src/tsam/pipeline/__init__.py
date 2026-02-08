"""Pipeline package — pure-function rewrite of create_typical_periods."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsam.pipeline.accuracy import compute_accuracy, reconstruct
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
from tsam.pipeline.types import PipelineResult, PredefParams

if TYPE_CHECKING:
    from tsam.config import (
        ClusterConfig,
        ClusteringResult,
        ExtremeConfig,
        SegmentConfig,
    )


def _infer_resolution(data: pd.DataFrame) -> float:
    """Infer temporal resolution from data index."""
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        return (data.index[1] - data.index[0]).total_seconds() / 3600
    return 1.0


def _count_occurrences(cluster_order: list | np.ndarray) -> dict[int, float]:
    """Count how many original periods each cluster represents."""
    nums, counts = np.unique(cluster_order, return_counts=True)
    return {int(num): int(counts[ii]) for ii, num in enumerate(nums)}


def _representatives_to_dataframe(
    cluster_periods_list: list,
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


def _build_representation_dict(
    columns: pd.Index,
    cluster_rep,
) -> dict[str, str]:
    """Build the representation dict (mean/min/max per column) from config."""
    from tsam.config import MinMaxMean

    representation_dict: dict[str, str] = dict.fromkeys(sorted(columns), "mean")
    if isinstance(cluster_rep, MinMaxMean):
        for col in cluster_rep.max_columns:
            if col in representation_dict:
                representation_dict[col] = "max"
        for col in cluster_rep.min_columns:
            if col in representation_dict:
                representation_dict[col] = "min"
    return representation_dict


def run_pipeline(
    data: pd.DataFrame,
    n_clusters: int,
    n_timesteps_per_period: int,
    *,
    cluster: ClusterConfig,
    extremes: ExtremeConfig | None = None,
    segments: SegmentConfig | None = None,
    rescale_cluster_periods: bool = True,
    rescale_exclude_columns: list[str] | None = None,
    round_decimals: int | None = None,
    numerical_tolerance: float = 1e-13,
    temporal_resolution: float | None = None,
    # Predefined parameters (for transfer/apply)
    predef: PredefParams | None = None,
) -> PipelineResult:
    """Run the full aggregation pipeline.

    This replaces create_typical_periods() + predict_original_data() + accuracy_indicators().
    """
    rescale_exclude_columns = rescale_exclude_columns or []

    cluster_rep = cluster.get_representation()
    representation_dict = _build_representation_dict(data.columns, cluster_rep)

    # Store original column order (before sort in normalize)
    original_column_order = list(data.columns)

    # Step 1: Normalize
    norm_data = normalize(data, cluster.weights, cluster.normalize_column_means)

    # Step 2: Unstack to periods
    period_profiles = unstack_to_periods(norm_data.values, n_timesteps_per_period)
    candidates = period_profiles.profiles_dataframe.values

    # Step 3: Add period sum features if requested
    n_feature_cols = candidates.shape[1]
    if cluster.include_period_sums:
        candidates, n_extra = add_period_sum_features(
            period_profiles.profiles_dataframe, candidates
        )
        n_feature_cols = candidates.shape[1] - n_extra

    # Step 4: Cluster (or predefined, or duration-curve variant)
    clustering_duration = 0.0
    cluster_center_indices: list | None = None

    if predef is not None:
        cluster_centers, cluster_center_indices, cluster_order = (
            use_predefined_assignments(
                candidates,
                predef,
                cluster_rep,
                representation_dict,
                n_timesteps_per_period,
            )
        )
    else:
        t_start = time.time()
        if not cluster.use_duration_curves:
            cluster_centers, cluster_center_indices, cluster_order = cluster_periods(
                candidates,
                n_clusters,
                cluster,
                representation_dict,
                n_timesteps_per_period,
            )
        else:
            cluster_centers, cluster_center_indices, cluster_order = (
                cluster_sorted_periods(
                    candidates,
                    period_profiles,
                    n_clusters,
                    cluster,
                    representation_dict,
                    n_timesteps_per_period,
                )
            )
        clustering_duration = time.time() - t_start

    # Step 5: Trim eval features from representatives
    cluster_periods_list = [center[:n_feature_cols] for center in cluster_centers]

    # Step 6: Add extreme periods if configured
    extreme_periods_info: dict = {}
    extreme_cluster_idx: list[int] = []

    if extremes is not None:
        (
            cluster_periods_list,
            cluster_order,
            extreme_cluster_idx,
            extreme_periods_info,
        ) = add_extreme_periods(
            period_profiles.profiles_dataframe,
            cluster_periods_list,
            cluster_order,
            extremes,
        )
    else:
        if predef is not None and predef.extreme_cluster_idx is not None:
            extreme_cluster_idx = list(predef.extreme_cluster_idx)

    # Step 7: Compute cluster weights
    cluster_period_no_occur = _count_occurrences(cluster_order)

    # Step 8: Rescale if requested
    rescale_deviations: dict = {}
    if rescale_cluster_periods:
        cluster_periods_list, rescale_deviations = rescale_representatives(  # type: ignore[assignment]
            cluster_periods_list,
            cluster_period_no_occur,
            extreme_cluster_idx,
            period_profiles.profiles_dataframe,
            norm_data,
            n_timesteps_per_period,
            rescale_exclude_columns,
        )
        cluster_periods_list = list(cluster_periods_list)

    # Step 9: Adjust for partial periods
    if len(data) % n_timesteps_per_period != 0:
        last_cluster = (
            cluster_order[-1]
            if isinstance(cluster_order, list)
            else int(cluster_order[-1])
        )
        cluster_period_no_occur[last_cluster] -= (
            1 - float(len(data) % n_timesteps_per_period) / n_timesteps_per_period
        )

    # Step 10: Format representatives to MultiIndex DataFrame
    normalized_typical_periods = _representatives_to_dataframe(
        cluster_periods_list, period_profiles.column_index
    )

    # Step 11: Segmentation if configured
    segmented_df = None
    predicted_segmented_df = None
    segment_center_indices = None

    if segments is not None:
        segmented_df, predicted_segmented_df, segment_center_indices = (
            segment_typical_periods(
                normalized_typical_periods,
                n_timesteps_per_period,
                segments,
                representation_dict,
                predef,
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
    if round_decimals is not None:
        typical_periods = typical_periods.round(decimals=round_decimals)

    # Step 13: Bounds check + warnings
    _warn_if_out_of_bounds(
        typical_periods, norm_data.original_data, numerical_tolerance
    )

    # Step 14: Reconstruct + compute accuracy
    reconstructed_data, normalized_predicted = reconstruct(
        reconstruct_source,
        cluster_order,
        period_profiles,
        norm_data,
    )
    if round_decimals is not None:
        reconstructed_data = reconstructed_data.round(decimals=round_decimals)

    accuracy_df = compute_accuracy(
        norm_data.values,
        normalized_predicted,
        norm_data,
    )

    # Restore original column order in output DataFrames
    # The pipeline sorts columns internally but outputs should match input order
    original_data_out = norm_data.original_data[original_column_order]
    reconstructed_data_out = reconstructed_data[original_column_order]
    typical_periods = typical_periods[original_column_order]

    # Step 15: Build ClusteringResult
    clustering_result = _build_clustering_result(
        cluster_center_indices=cluster_center_indices,
        extreme_periods_info=extreme_periods_info,
        extremes_config=extremes,
        cluster_order=cluster_order,
        segmented_df=segmented_df,
        segment_center_indices=segment_center_indices,
        n_timesteps_per_period=n_timesteps_per_period,
        temporal_resolution=temporal_resolution,
        original_data=original_data_out,
        cluster_config=cluster,
        segment_config=segments,
        rescale_cluster_periods=rescale_cluster_periods,
        rescale_exclude_columns=rescale_exclude_columns,
        extreme_cluster_idx=extreme_cluster_idx,
    )

    # Step 16: Return PipelineResult
    return PipelineResult(
        typical_periods=typical_periods,
        cluster_weights=cluster_period_no_occur,
        n_timesteps_per_period=n_timesteps_per_period,
        time_index=period_profiles.time_index,
        original_data=original_data_out,
        clustering_duration=clustering_duration,
        rescale_deviations=rescale_deviations,
        segmented_df=segmented_df,
        reconstructed_data=reconstructed_data_out,
        accuracy_indicators=accuracy_df,
        clustering_result=clustering_result,
    )


def _extract_segment_data(
    segmented_df: pd.DataFrame,
    segment_center_indices: list | None,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...] | None,
]:
    """Extract segment assignments, durations, and centers from a segmented DataFrame.

    Returns (assignments, durations, centers).
    """
    assignments_list = []
    durations_list = []

    for period_idx in segmented_df.index.get_level_values(0).unique():
        period_data = segmented_df.loc[period_idx]
        assignments = []
        durations = []
        for seg_step, seg_dur, _orig_start in period_data.index:
            assignments.extend([int(seg_step)] * int(seg_dur))
            durations.append(int(seg_dur))
        assignments_list.append(tuple(assignments))
        durations_list.append(tuple(durations))

    centers: tuple[tuple[int, ...], ...] | None = None
    if segment_center_indices is not None:
        if all(pc is not None for pc in segment_center_indices):
            centers = tuple(
                tuple(int(x) for x in period_centers)
                for period_centers in segment_center_indices
            )

    return tuple(assignments_list), tuple(durations_list), centers


def _build_clustering_result(
    *,
    cluster_center_indices: list | None,
    extreme_periods_info: dict,
    extremes_config: ExtremeConfig | None,
    cluster_order: list | np.ndarray,
    segmented_df: pd.DataFrame | None,
    segment_center_indices: list | None,
    n_timesteps_per_period: int,
    temporal_resolution: float | None,
    original_data: pd.DataFrame,
    cluster_config: ClusterConfig,
    segment_config: SegmentConfig | None,
    rescale_cluster_periods: bool,
    rescale_exclude_columns: list[str] | None,
    extreme_cluster_idx: list[int],
) -> ClusteringResult:
    """Build a ClusteringResult from pipeline intermediate data."""
    from tsam.config import ClusteringResult

    # Get cluster centers
    cluster_centers: tuple[int, ...] | None = None
    if cluster_center_indices is not None:
        center_indices = [int(x) for x in cluster_center_indices]

        if (
            extreme_periods_info
            and extremes_config is not None
            and extremes_config.method in ("new_cluster", "append")
        ):
            for period_type in extreme_periods_info:
                center_indices.append(int(extreme_periods_info[period_type]["step_no"]))

        cluster_centers = tuple(center_indices)

    # Compute segment data if segmentation was used
    segment_assignments: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None
    segment_centers: tuple[tuple[int, ...], ...] | None = None

    if segment_config is not None and segmented_df is not None:
        segment_assignments, segment_durations, segment_centers = _extract_segment_data(
            segmented_df, segment_center_indices
        )

    # Extract representation from configs
    representation = cluster_config.get_representation()
    segment_representation = segment_config.representation if segment_config else None

    # Extract extreme cluster indices
    extreme_cluster_indices_tuple: tuple[int, ...] | None = None
    if extreme_cluster_idx:
        extreme_cluster_indices_tuple = tuple(int(x) for x in extreme_cluster_idx)

    # Compute period_duration
    effective_resolution = (
        temporal_resolution
        if temporal_resolution is not None
        else _infer_resolution(original_data)
    )
    period_duration = n_timesteps_per_period * effective_resolution

    return ClusteringResult(
        period_duration=period_duration,
        cluster_assignments=tuple(int(x) for x in cluster_order),
        cluster_centers=cluster_centers,
        segment_assignments=segment_assignments,
        segment_durations=segment_durations,
        segment_centers=segment_centers,
        preserve_column_means=rescale_cluster_periods,
        rescale_exclude_columns=tuple(rescale_exclude_columns)
        if rescale_exclude_columns
        else None,
        representation=representation,
        segment_representation=segment_representation,
        temporal_resolution=temporal_resolution,
        n_timesteps_per_period=n_timesteps_per_period,
        extreme_cluster_indices=extreme_cluster_indices_tuple,
        cluster_config=cluster_config,
        segment_config=segment_config,
        extremes_config=extremes_config,
    )
