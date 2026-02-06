"""Pipeline package — pure-function rewrite of createTypicalPeriods."""

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
from tsam.pipeline.extremes import add_extreme_periods, count_extreme_periods
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

    This replaces createTypicalPeriods() + predictOriginalData() + accuracyIndicators().
    """
    rescale_exclude_columns = rescale_exclude_columns or []

    # Build representationDict default (monolith lines 504-512)
    representation_dict: dict[str, str] = dict.fromkeys(list(data.columns), "mean")
    representation_dict = dict(pd.Series(representation_dict).sort_index(axis=0))

    # Resolve segment representation: inherit from cluster config if not specified
    segment_representation = (
        segments.representation if segments else cluster.get_representation()
    )

    # Store original column order (before sort in normalize)
    original_column_order = list(data.columns)

    # Step 1: Normalize
    norm_data = normalize(data, cluster.weights, cluster.normalize_column_means)

    # Step 2: Unstack to periods
    period_profiles = unstack_to_periods(norm_data.values, n_timesteps_per_period)
    candidates = period_profiles.profiles_dataframe.values

    # Step 3: Add period sum features if requested
    if cluster.include_period_sums:
        candidates, n_extra = add_period_sum_features(
            period_profiles.profiles_dataframe, candidates
        )
        del_cluster_params = -n_extra
    else:
        del_cluster_params = None

    # Step 4: Warn if extremePreserveNumClusters is ignored due to predefined
    extreme_preserve = extremes is not None and extremes._effective_preserve_n_clusters
    if (
        predef is not None
        and extreme_preserve
        and extremes is not None
        and extremes.method not in (None, "replace")
    ):
        warnings.warn(
            "extremePreserveNumClusters=True is ignored when predefClusterOrder "
            "is set. Extreme periods will be appended via _addExtremePeriods "
            "without reserving clusters upfront. To avoid this warning, set "
            "extremePreserveNumClusters=False or remove predefClusterOrder.",
            UserWarning,
            stacklevel=2,
        )

    # Count extreme periods upfront if preserve_n_clusters is True
    n_extremes = 0
    if (
        extreme_preserve
        and extremes is not None
        and extremes.method not in (None, "replace")
        and predef is None
    ):
        n_extremes = count_extreme_periods(
            period_profiles.profiles_dataframe,
            extremes.max_value,
            extremes.min_value,
            extremes.max_period,
            extremes.min_period,
        )
        if n_clusters <= n_extremes:
            raise ValueError(
                f"n_clusters ({n_clusters}) must be greater than "
                f"the number of extreme periods ({n_extremes}) when "
                "preserve_n_clusters=True"
            )

    effective_n_clusters = n_clusters - n_extremes

    representation_method = cluster.get_representation()

    # Step 5: Cluster (or predefined, or duration-curve variant)
    clustering_duration = 0.0
    cluster_center_indices: list | None = None

    if predef is not None:
        cluster_centers, cluster_center_indices, cluster_order = (
            use_predefined_assignments(
                candidates,
                predef.cluster_order,
                predef.cluster_center_indices,
                representation_method,
                representation_dict,
                n_timesteps_per_period,
            )
        )
    else:
        t_start = time.time()
        if not cluster.use_duration_curves:
            cluster_centers, cluster_center_indices, cluster_order = cluster_periods(
                candidates,
                effective_n_clusters,
                cluster.method,
                cluster.solver,
                representation_method,
                representation_dict,
                n_timesteps_per_period,
            )
        else:
            cluster_centers, cluster_center_indices, cluster_order = (
                cluster_sorted_periods(
                    candidates,
                    period_profiles.profiles_dataframe.values,
                    period_profiles.n_columns,
                    effective_n_clusters,
                    cluster.method,
                    cluster.solver,
                    representation_method,
                    representation_dict,
                    n_timesteps_per_period,
                )
            )
        clustering_duration = time.time() - t_start

    # Step 6: Trim eval features from representatives
    cluster_periods_list = []
    for center in cluster_centers:
        cluster_periods_list.append(center[:del_cluster_params])

    # Step 7: Add extreme periods if configured
    extreme_periods_info: dict = {}
    extreme_cluster_idx: list[int] = []

    if extremes is not None:
        columns = list(norm_data.original_data.columns)
        (
            cluster_periods_list,
            cluster_order,
            extreme_cluster_idx,
            extreme_periods_info,
        ) = add_extreme_periods(
            period_profiles.profiles_dataframe,
            cluster_periods_list,
            cluster_order,
            extremes.method,
            extremes.max_value,
            extremes.min_value,
            extremes.max_period,
            extremes.min_period,
            columns,
        )
    else:
        if predef is not None and predef.extreme_cluster_idx is not None:
            extreme_cluster_idx = list(predef.extreme_cluster_idx)

    # Step 8: Compute cluster weights
    nums, counts = np.unique(cluster_order, return_counts=True)
    cluster_period_no_occur: dict[int, float] = {
        int(num): int(counts[ii]) for ii, num in enumerate(nums)
    }

    # Step 9: Rescale if requested
    rescale_deviations: dict = {}
    if rescale_cluster_periods:
        cluster_periods_list, rescale_deviations = rescale_representatives(  # type: ignore[assignment]
            cluster_periods_list,
            cluster_period_no_occur,
            extreme_cluster_idx,
            period_profiles.profiles_dataframe,
            norm_data.original_data,
            list(norm_data.original_data.columns),
            n_timesteps_per_period,
            cluster.normalize_column_means,
            cluster.weights,
            rescale_exclude_columns,
        )

    # Step 10: Adjust for partial periods
    if len(data) % n_timesteps_per_period != 0:
        last_cluster = (
            cluster_order[-1]
            if isinstance(cluster_order, list)
            else int(cluster_order[-1])
        )
        cluster_period_no_occur[last_cluster] -= (
            1 - float(len(data) % n_timesteps_per_period) / n_timesteps_per_period
        )

    # Step 11: Format representatives to MultiIndex DataFrame
    normalized_typical_periods = (
        pd.concat(
            [
                pd.Series(s, index=period_profiles.column_index)
                for s in (
                    cluster_periods_list
                    if isinstance(cluster_periods_list, list)
                    else [
                        cluster_periods_list[i]
                        for i in range(len(cluster_periods_list))
                    ]
                )
            ],
            axis=1,
        )
        .unstack("TimeStep")
        .T
    )
    assert isinstance(normalized_typical_periods, pd.DataFrame)

    # Step 12: Segmentation if configured
    segmented_df = None
    predicted_segmented_df = None
    segment_center_indices = None

    if segments is not None:
        segmented_df, predicted_segmented_df, segment_center_indices = (
            segment_typical_periods(
                normalized_typical_periods,
                segments.n_segments,
                n_timesteps_per_period,
                segment_representation,
                representation_dict,
                predef.segment_order if predef is not None else None,
                predef.segment_durations if predef is not None else None,
                predef.segment_centers if predef is not None else None,
            )
        )
        # Replace normalized_typical_periods with segmented version (drop Original Start Step level)
        normalized_typical_periods = segmented_df.reset_index(level=3, drop=True)

    # Step 13: Denormalize -> typical_periods
    typical_periods = denormalize(
        normalized_typical_periods,
        norm_data,
        cluster.weights,
        cluster.normalize_column_means,
        round_decimals,
    )

    # Step 14: Bounds check + warnings
    exceeds_max = typical_periods.max(axis=0) > norm_data.original_data.max(axis=0)
    if exceeds_max.any():
        diff = typical_periods.max(axis=0) - norm_data.original_data.max(axis=0)
        exceeding_diff = diff[exceeds_max]
        if exceeding_diff.max() > numerical_tolerance:
            warnings.warn(
                "At least one maximal value of the "
                + "aggregated time series exceeds the maximal value "
                + "the input time series for: "
                + f"{exceeding_diff.to_dict()}"
                + ". To silence the warning set the 'numericalTolerance' to a higher value."
            )
    below_min = typical_periods.min(axis=0) < norm_data.original_data.min(axis=0)
    if below_min.any():
        diff = norm_data.original_data.min(axis=0) - typical_periods.min(axis=0)
        exceeding_diff = diff[below_min]
        if exceeding_diff.max() > numerical_tolerance:
            warnings.warn(
                "Something went wrong... At least one minimal value of the "
                + "aggregated time series exceeds the minimal value "
                + "the input time series for: "
                + f"{exceeding_diff.to_dict()}"
                + ". To silence the warning set the 'numericalTolerance' to a higher value."
            )

    # Step 15: Reconstruct + compute accuracy
    reconstructed_data, normalized_predicted = reconstruct(
        normalized_typical_periods,
        cluster_order,
        period_profiles,
        norm_data,
        cluster.normalize_column_means,
        cluster.weights,
        round_decimals,
        segmentation_active=segments is not None,
        predicted_segmented_df=predicted_segmented_df,
    )

    accuracy_df = compute_accuracy(
        norm_data.values,
        normalized_predicted,
        cluster.weights,
    )

    # Restore original column order in output DataFrames
    # The pipeline sorts columns internally but outputs should match input order
    original_data_out = norm_data.original_data[original_column_order]
    reconstructed_data_out = reconstructed_data[original_column_order]
    typical_periods = typical_periods[original_column_order]

    # Step 16: Build ClusteringResult
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

    # Step 17: Return PipelineResult
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
                center_indices.append(int(extreme_periods_info[period_type]["stepNo"]))

        cluster_centers = tuple(center_indices)

    # Compute segment data if segmentation was used
    segment_assignments: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None
    segment_centers: tuple[tuple[int, ...], ...] | None = None

    if segment_config is not None and segmented_df is not None:
        segment_assignments_list = []
        segment_durations_list = []

        for period_idx in segmented_df.index.get_level_values(0).unique():
            period_data = segmented_df.loc[period_idx]
            assignments = []
            durations = []
            for seg_step, seg_dur, _orig_start in period_data.index:
                assignments.extend([int(seg_step)] * int(seg_dur))
                durations.append(int(seg_dur))
            segment_assignments_list.append(tuple(assignments))
            segment_durations_list.append(tuple(durations))

        segment_assignments = tuple(segment_assignments_list)
        segment_durations = tuple(segment_durations_list)

        if segment_center_indices is not None:
            if all(pc is not None for pc in segment_center_indices):
                segment_centers = tuple(
                    tuple(int(x) for x in period_centers)
                    for period_centers in segment_center_indices
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
