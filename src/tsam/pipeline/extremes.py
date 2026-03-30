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


def _detect_extreme(
    profiles_df: pd.DataFrame,
    column,
    series: pd.Series,
    suffix: str,
    extreme_period_no: list,
    cc_list: list,
    extreme_periods: dict,
) -> None:
    """Detect a single extreme period and add it to extreme_periods if unique."""
    step_no = series  # already reduced to a scalar index
    if (
        step_no not in extreme_period_no
        and profiles_df.loc[step_no, :].values.tolist() not in cc_list
    ):
        key = _append_col_with(column, suffix)
        extreme_periods[key] = {
            "step_no": step_no,
            "profile": profiles_df.loc[step_no, :].values,
            "column": column,
        }
        extreme_period_no.append(step_no)


def add_extreme_periods(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extremes: ExtremeConfig,
) -> tuple[list, list | np.ndarray, list[int], dict]:
    """Add extreme periods to clustered data.

    Returns (new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods_info).
    """
    columns = profiles_df.columns.get_level_values(0).unique().tolist()
    extreme_periods: dict = {}
    extreme_period_no: list = []

    cc_list = [center.tolist() for center in cluster_centers]

    # Detect extreme periods for each column
    _CHECKS: list[tuple[list[str], str, str]] = [
        (extremes.max_value, "max", " max."),
        (extremes.min_value, "min", " min."),
        (extremes.max_period, "mean_max", " daily max."),
        (extremes.min_period, "mean_min", " daily min."),
    ]
    for column in columns:
        for config_list, kind, suffix in _CHECKS:
            if column not in config_list:
                continue
            if kind == "max":
                step_no = profiles_df[column].max(axis=1).idxmax()  # type: ignore[arg-type]
            elif kind == "min":
                step_no = profiles_df[column].min(axis=1).idxmin()  # type: ignore[arg-type]
            elif kind == "mean_max":
                step_no = profiles_df[column].mean(axis=1).idxmax()  # type: ignore[call-overload]
            else:  # mean_min
                step_no = profiles_df[column].mean(axis=1).idxmin()  # type: ignore[call-overload]
            _detect_extreme(
                profiles_df,
                column,
                step_no,
                suffix,
                extreme_period_no,
                cc_list,
                extreme_periods,
            )

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

        # Build set of extreme period step numbers for quick lookup
        extreme_step_nos = {extreme_periods[pt]["step_no"] for pt in extreme_periods}

        for i, c_period in enumerate(new_cluster_order):
            # Skip periods that are themselves an extreme period for a different type
            if i in extreme_step_nos:
                # Only reassign if this period IS the extreme for exactly one type
                own_types = [
                    pt for pt in extreme_periods if extreme_periods[pt]["step_no"] == i
                ]
                if len(own_types) == 1:
                    new_cluster_order[i] = extreme_periods[own_types[0]][
                        "new_cluster_no"
                    ]
                continue

            cluster_dist = sum(
                (profiles_df.iloc[i].values - cluster_centers[c_period]) ** 2
            )
            # Find the closest extreme period (deterministic: first match with smallest distance)
            best_extreme = None
            best_dist = cluster_dist
            for extrem_period_type in extreme_periods:
                extperiod_dist = sum(
                    (
                        profiles_df.iloc[i].values
                        - extreme_periods[extrem_period_type]["profile"]
                    )
                    ** 2
                )
                if extperiod_dist < best_dist:
                    best_dist = extperiod_dist
                    best_extreme = extrem_period_type

            if best_extreme is not None:
                new_cluster_order[i] = extreme_periods[best_extreme]["new_cluster_no"]

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
