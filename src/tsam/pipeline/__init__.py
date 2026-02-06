"""Pipeline package — pure-function rewrite of createTypicalPeriods."""

from __future__ import annotations

import time
import warnings

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
from tsam.pipeline.types import PipelineResult


def run_pipeline(
    data: pd.DataFrame,
    n_clusters: int,
    n_timesteps_per_period: int,
    *,
    # Clustering parameters
    cluster_method: str,
    representation_method: str | None,
    representation_dict: dict | None,
    distribution_period_wise: bool = True,
    solver: str = "highs",
    use_duration_curves: bool = False,
    normalize_column_means: bool = False,
    include_period_sums: bool = False,
    weights: dict[str, float] | None = None,
    # Rescale parameters
    rescale_cluster_periods: bool = True,
    rescale_exclude_columns: list[str] | None = None,
    # Extreme parameters
    extreme_period_method: str = "None",
    extreme_preserve_n_clusters: bool = False,
    add_peak_max: list[str] | None = None,
    add_peak_min: list[str] | None = None,
    add_mean_max: list[str] | None = None,
    add_mean_min: list[str] | None = None,
    # Segmentation parameters
    segmentation: bool = False,
    n_segments: int = 10,
    segment_representation_method: str | None = None,
    # Output
    round_decimals: int | None = None,
    numerical_tolerance: float = 1e-13,
    # Predefined parameters (for transfer/apply)
    predef_cluster_order: list | np.ndarray | None = None,
    predef_cluster_center_indices: list | np.ndarray | None = None,
    predef_extreme_cluster_idx: list | None = None,
    predef_segment_order: list | None = None,
    predef_segment_durations: list | None = None,
    predef_segment_centers: list | None = None,
) -> PipelineResult:
    """Run the full aggregation pipeline.

    This replaces createTypicalPeriods() + predictOriginalData() + accuracyIndicators().
    """
    add_peak_max = add_peak_max or []
    add_peak_min = add_peak_min or []
    add_mean_max = add_mean_max or []
    add_mean_min = add_mean_min or []
    rescale_exclude_columns = rescale_exclude_columns or []

    # Build representationDict default (monolith lines 504-512)
    if representation_dict is None:
        representation_dict = dict.fromkeys(list(data.columns), "mean")
    # Sort representationDict alphabetically
    representation_dict = pd.Series(representation_dict).sort_index(axis=0).to_dict()

    # Inherit segment representation from main if not specified
    if segment_representation_method is None:
        segment_representation_method = representation_method

    # Store original column order (before sort in normalize)
    original_column_order = list(data.columns)

    # Step 1: Normalize
    norm_data = normalize(data, weights, normalize_column_means)

    # Step 2: Unstack to periods
    period_profiles = unstack_to_periods(norm_data.values, n_timesteps_per_period)

    # Step 3: Add period sum features if requested
    if include_period_sums:
        candidates, n_extra = add_period_sum_features(
            period_profiles.profiles_dataframe, period_profiles.profiles
        )
        del_cluster_params = -n_extra
    else:
        candidates = period_profiles.profiles
        del_cluster_params = None

    # Step 4: Warn if extremePreserveNumClusters is ignored due to predefined
    if (
        predef_cluster_order is not None
        and extreme_preserve_n_clusters
        and extreme_period_method not in ("None", "replace_cluster_center")
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
        extreme_preserve_n_clusters
        and extreme_period_method not in ("None", "replace_cluster_center")
        and predef_cluster_order is None
    ):
        n_extremes = count_extreme_periods(
            period_profiles.profiles_dataframe,
            add_peak_max,
            add_peak_min,
            add_mean_max,
            add_mean_min,
        )
        if n_clusters <= n_extremes:
            raise ValueError(
                f"n_clusters ({n_clusters}) must be greater than "
                f"the number of extreme periods ({n_extremes}) when "
                "preserve_n_clusters=True"
            )

    effective_n_clusters = n_clusters - n_extremes

    # Step 5: Cluster (or predefined, or duration-curve variant)
    clustering_duration = 0.0
    cluster_center_indices: list | None = None

    if predef_cluster_order is not None:
        cluster_centers, cluster_center_indices, cluster_order = (
            use_predefined_assignments(
                candidates,
                predef_cluster_order,
                predef_cluster_center_indices,
                representation_method,
                representation_dict,
                distribution_period_wise,
                n_timesteps_per_period,
            )
        )
    else:
        t_start = time.time()
        if not use_duration_curves:
            cluster_centers, cluster_center_indices, cluster_order = cluster_periods(
                candidates,
                effective_n_clusters,
                cluster_method,
                solver,
                representation_method,
                representation_dict,
                distribution_period_wise,
                n_timesteps_per_period,
            )
        else:
            cluster_centers, cluster_order, cluster_center_indices = (
                cluster_sorted_periods(
                    candidates,
                    period_profiles.profiles_dataframe.values,
                    period_profiles.n_columns,
                    effective_n_clusters,
                    cluster_method,
                    solver,
                    representation_method,
                    representation_dict,
                    distribution_period_wise,
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

    if extreme_period_method != "None":
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
            extreme_period_method,
            add_peak_max,
            add_peak_min,
            add_mean_max,
            add_mean_min,
            columns,
        )
    else:
        if predef_extreme_cluster_idx is not None:
            extreme_cluster_idx = list(predef_extreme_cluster_idx)

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
            normalize_column_means,
            weights,
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

    if segmentation:
        segmented_df, predicted_segmented_df, segment_center_indices = (
            segment_typical_periods(
                normalized_typical_periods,
                n_segments,
                n_timesteps_per_period,
                segment_representation_method,
                representation_dict,
                distribution_period_wise,
                predef_segment_order,
                predef_segment_durations,
                predef_segment_centers,
            )
        )
        # Replace normalized_typical_periods with segmented version (drop Original Start Step level)
        normalized_typical_periods = segmented_df.reset_index(level=3, drop=True)

    # Step 13: Denormalize → typical_periods
    typical_periods = denormalize(
        normalized_typical_periods,
        norm_data,
        weights,
        normalize_column_means,
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
        normalize_column_means,
        weights,
        round_decimals,
        segmentation_active=segmentation,
        predicted_segmented_df=predicted_segmented_df,
    )

    accuracy_df = compute_accuracy(
        norm_data.values,
        normalized_predicted,
        weights,
    )

    # Restore original column order in output DataFrames
    # The pipeline sorts columns internally but outputs should match input order
    original_data_out = norm_data.original_data[original_column_order]
    reconstructed_data_out = reconstructed_data[original_column_order]
    typical_periods = typical_periods[original_column_order]

    # Step 16: Return PipelineResult
    return PipelineResult(
        typical_periods=typical_periods,
        normalized_typical_periods=normalized_typical_periods,
        cluster_assignments=np.array(cluster_order),
        cluster_weights=cluster_period_no_occur,
        cluster_center_indices=cluster_center_indices,
        extreme_cluster_indices=extreme_cluster_idx,
        extreme_periods_info=extreme_periods_info,
        time_index=period_profiles.time_index,
        n_timesteps_per_period=n_timesteps_per_period,
        original_data=original_data_out,
        normalized_data=norm_data,
        period_profiles=period_profiles,
        clustering_duration=clustering_duration,
        rescale_deviations=rescale_deviations,
        segmented_df=segmented_df,
        predicted_segmented_df=predicted_segmented_df,
        segment_center_indices=segment_center_indices,
        normalize_column_means=normalize_column_means,
        weights=weights,
        round_decimals=round_decimals,
        reconstructed_data=reconstructed_data_out,
        accuracy_indicators=accuracy_df,
    )
