"""Extreme period counting and integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tsam.config import ExtremeConfig


def _append_col_with(column, append_with: str = " max."):
    """Append a string to the column name. For MultiIndexes, only last level is changed."""
    if isinstance(column, str):
        return column + append_with
    elif isinstance(column, tuple):
        col = list(column)
        col[-1] = col[-1] + append_with
        return tuple(col)


def add_extreme_periods(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extremes: ExtremeConfig,
    columns: list[str],
) -> tuple[list, list | np.ndarray, list[int], dict]:
    """Add extreme periods to clustered data.

    Replicates _add_extreme_periods (monolith lines 729-918).

    Returns (new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods_info).
    """
    extreme_periods: dict = {}
    extreme_period_no: list = []

    cc_list = [center.tolist() for center in cluster_centers]

    # Check which extreme periods exist
    for column in columns:
        if column in extremes.max_value:
            step_no = profiles_df[column].max(axis=1).idxmax()  # type: ignore[arg-type]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                max_col = _append_col_with(column, " max.")
                extreme_periods[max_col] = {
                    "step_no": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

        if column in extremes.min_value:
            step_no = profiles_df[column].min(axis=1).idxmin()  # type: ignore[arg-type]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                min_col = _append_col_with(column, " min.")
                extreme_periods[min_col] = {
                    "step_no": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

        if column in extremes.max_period:
            step_no = profiles_df[column].mean(axis=1).idxmax()  # type: ignore[call-overload]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                mean_max_col = _append_col_with(column, " daily max.")
                extreme_periods[mean_max_col] = {
                    "step_no": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

        if column in extremes.min_period:
            step_no = profiles_df[column].mean(axis=1).idxmin()  # type: ignore[call-overload]
            if (
                step_no not in extreme_period_no
                and profiles_df.loc[step_no, :].values.tolist() not in cc_list
            ):
                mean_min_col = _append_col_with(column, " daily min.")
                extreme_periods[mean_min_col] = {
                    "step_no": step_no,
                    "profile": profiles_df.loc[step_no, :].values,
                    "column": column,
                }
                extreme_period_no.append(step_no)

    # Get current related clusters of extreme periods
    for period_type in extreme_periods:
        extreme_periods[period_type]["cluster_no"] = cluster_order[
            extreme_periods[period_type]["step_no"]
        ]

    new_cluster_centers: list = []
    new_cluster_order = list(cluster_order)
    extreme_cluster_idx: list[int] = []

    if extremes.method == "append":
        for cluster_center in cluster_centers:
            new_cluster_centers.append(cluster_center)
        for i, period_type in enumerate(extreme_periods):
            extreme_cluster_idx.append(len(new_cluster_centers))
            new_cluster_centers.append(extreme_periods[period_type]["profile"])
            new_cluster_order[extreme_periods[period_type]["step_no"]] = i + len(
                cluster_centers
            )

    elif extremes.method == "new_cluster":
        for cluster_center in cluster_centers:
            new_cluster_centers.append(cluster_center)
        for i, period_type in enumerate(extreme_periods):
            extreme_cluster_idx.append(len(new_cluster_centers))
            new_cluster_centers.append(extreme_periods[period_type]["profile"])
            extreme_periods[period_type]["new_cluster_no"] = i + len(cluster_centers)

        for i, c_period in enumerate(new_cluster_order):
            cluster_dist = sum(
                (profiles_df.iloc[i].values - cluster_centers[c_period]) ** 2
            )
            for ii, extrem_period_type in enumerate(extreme_periods):
                is_other_extreme = False
                for other_ex_period in extreme_periods:
                    if (
                        i == extreme_periods[other_ex_period]["step_no"]
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
                        "new_cluster_no"
                    ]

    elif extremes.method == "replace":
        new_cluster_centers = list(cluster_centers)
        for period_type in extreme_periods:
            index = profiles_df.columns.get_loc(extreme_periods[period_type]["column"])
            new_cluster_centers[extreme_periods[period_type]["cluster_no"]][index] = (
                extreme_periods[period_type]["profile"][index]
            )
            if extreme_periods[period_type]["cluster_no"] not in extreme_cluster_idx:
                extreme_cluster_idx.append(extreme_periods[period_type]["cluster_no"])

    return new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods
