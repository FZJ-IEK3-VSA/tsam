"""Extreme period counting and integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def _append_col_with(column, append_with: str = " max."):
    """Append a string to the column name. For MultiIndexes, only last level is changed."""
    if isinstance(column, str):
        return column + append_with
    elif isinstance(column, tuple):
        col = list(column)
        col[-1] = col[-1] + append_with
        return tuple(col)


def count_extreme_periods(
    profiles_df: pd.DataFrame,
    add_peak_max: list[str],
    add_peak_min: list[str],
    add_mean_max: list[str],
    add_mean_min: list[str],
) -> int:
    """Count unique extreme periods without modifying any state.

    Replicates _countExtremePeriods (monolith lines 689-727).
    """
    extreme_period_indices: set = set()

    extreme_columns = (
        set(add_peak_max) | set(add_peak_min) | set(add_mean_max) | set(add_mean_min)
    )

    for column in extreme_columns:
        col_data = profiles_df[column]

        if column in add_peak_max:
            extreme_period_indices.add(col_data.max(axis=1).idxmax())  # type: ignore[arg-type]
        if column in add_peak_min:
            extreme_period_indices.add(col_data.min(axis=1).idxmin())  # type: ignore[arg-type]

        if column in add_mean_max or column in add_mean_min:
            mean_series = col_data.mean(axis=1)  # type: ignore[call-overload]
            if column in add_mean_max:
                extreme_period_indices.add(mean_series.idxmax())
            if column in add_mean_min:
                extreme_period_indices.add(mean_series.idxmin())

    return len(extreme_period_indices)


def add_extreme_periods(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extreme_method: str,
    add_peak_max: list[str],
    add_peak_min: list[str],
    add_mean_max: list[str],
    add_mean_min: list[str],
    columns: list[str],
) -> tuple[list, list | np.ndarray, list[int], dict]:
    """Add extreme periods to clustered data.

    Replicates _addExtremePeriods (monolith lines 729-918).

    Returns (new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods_info).
    """
    extreme_periods: dict = {}
    extreme_period_no: list = []

    cc_list = [center.tolist() for center in cluster_centers]

    # Check which extreme periods exist
    for column in columns:
        if column in add_peak_max:
            step_no = profiles_df[column].max(axis=1).idxmax()  # type: ignore[arg-type]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                max_col = _append_col_with(column, " max.")
                extreme_periods[max_col] = {
                    "stepNo": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

        if column in add_peak_min:
            step_no = profiles_df[column].min(axis=1).idxmin()  # type: ignore[arg-type]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                min_col = _append_col_with(column, " min.")
                extreme_periods[min_col] = {
                    "stepNo": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

        if column in add_mean_max:
            step_no = profiles_df[column].mean(axis=1).idxmax()  # type: ignore[call-overload]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                mean_max_col = _append_col_with(column, " daily max.")
                extreme_periods[mean_max_col] = {
                    "stepNo": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

        if column in add_mean_min:
            step_no = profiles_df[column].mean(axis=1).idxmin()  # type: ignore[call-overload]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                mean_min_col = _append_col_with(column, " daily min.")
                extreme_periods[mean_min_col] = {
                    "stepNo": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

    # Get current related clusters of extreme periods
    for period_type in extreme_periods:
        extreme_periods[period_type]["clusterNo"] = cluster_order[
            extreme_periods[period_type]["stepNo"]
        ]

    new_cluster_centers: list = []
    new_cluster_order = list(cluster_order)
    extreme_cluster_idx: list[int] = []

    if extreme_method == "append":
        for cluster_center in cluster_centers:
            new_cluster_centers.append(cluster_center)
        for i, period_type in enumerate(extreme_periods):
            extreme_cluster_idx.append(len(new_cluster_centers))
            new_cluster_centers.append(extreme_periods[period_type]["profile"])
            new_cluster_order[extreme_periods[period_type]["stepNo"]] = i + len(
                cluster_centers
            )

    elif extreme_method == "new_cluster_center":
        for cluster_center in cluster_centers:
            new_cluster_centers.append(cluster_center)
        for i, period_type in enumerate(extreme_periods):
            extreme_cluster_idx.append(len(new_cluster_centers))
            new_cluster_centers.append(extreme_periods[period_type]["profile"])
            extreme_periods[period_type]["newClusterNo"] = i + len(cluster_centers)

        for i, c_period in enumerate(new_cluster_order):
            cluster_dist = sum(
                (profiles_df.iloc[i].values - cluster_centers[c_period]) ** 2
            )
            for ii, extrem_period_type in enumerate(extreme_periods):
                is_other_extreme = False
                for other_ex_period in extreme_periods:
                    if (
                        i == extreme_periods[other_ex_period]["stepNo"]
                        and other_ex_period != extrem_period_type
                    ):
                        is_other_extreme = True
                extperiod_dist = sum(
                    (
                        profiles_df.iloc[i].values
                        - extreme_periods[extrem_period_type]["profile"]
                    )
                    ** 2
                )
                if extperiod_dist < cluster_dist and not is_other_extreme:
                    new_cluster_order[i] = extreme_periods[extrem_period_type][
                        "newClusterNo"
                    ]

    elif extreme_method == "replace_cluster_center":
        new_cluster_centers = list(cluster_centers)
        for period_type in extreme_periods:
            index = profiles_df.columns.get_loc(extreme_periods[period_type]["column"])
            new_cluster_centers[extreme_periods[period_type]["clusterNo"]][index] = (
                extreme_periods[period_type]["profile"][index]
            )
            if extreme_periods[period_type]["clusterNo"] not in extreme_cluster_idx:
                extreme_cluster_idx.append(extreme_periods[period_type]["clusterNo"])

    return new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods
