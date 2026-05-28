"""Extreme period detection and integration into a clustering result."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import assert_never

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tsam.config import ExtremeConfig


ExtremeKind = Literal["max", "min", "mean_max", "mean_min"]

_KIND_SUFFIX: dict[ExtremeKind, str] = {
    "max": " max.",
    "min": " min.",
    "mean_max": " daily max.",
    "mean_min": " daily min.",
}


def add_extreme_periods(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extremes: ExtremeConfig,
) -> tuple[list, list | np.ndarray, list[int], dict]:
    """Inject extreme periods into a clustering result.

    Splits into two phases: detection (find which input periods carry the
    requested extremes — peak demand, minimum solar, etc.) and integration
    (decide what to do with them, via *extremes.method*).

    Returns
    -------
    new_cluster_centers
        Cluster center vectors after extreme integration.
    new_cluster_order
        Updated assignment of original periods to clusters.
    extreme_cluster_idx
        Cluster indices that hold an extreme period (and are excluded from
        rescaling downstream).
    extreme_periods
        Bookkeeping of which input period satisfied each detected extreme.
    """
    extreme_periods = _detect_extremes(profiles_df, extremes, cluster_centers)

    for period_type in extreme_periods:
        extreme_periods[period_type]["cluster_no"] = cluster_order[
            extreme_periods[period_type]["step_no"]
        ]

    if extremes.method == "append":
        return _apply_append(cluster_centers, cluster_order, extreme_periods)
    if extremes.method == "new_cluster":
        return _apply_new_cluster(
            profiles_df, cluster_centers, cluster_order, extreme_periods
        )
    if extremes.method == "replace":
        return _apply_replace(
            profiles_df, cluster_centers, cluster_order, extreme_periods
        )
    assert_never(extremes.method)


def _detect_extremes(
    profiles_df: pd.DataFrame,
    extremes: ExtremeConfig,
    cluster_centers: list,
) -> dict:
    """Find the input periods carrying each requested extreme.

    Walks the four kinds (max_value / min_value / max_period / min_period)
    times the columns named under each. Skips periods that already match an
    existing cluster center exactly (no duplicate clusters).
    """
    extreme_periods: dict = {}
    extreme_period_no: list = []
    cc_list = [center.tolist() for center in cluster_centers]

    columns = profiles_df.columns.get_level_values(0).unique().tolist()
    checks: list[tuple[list[str], ExtremeKind]] = [
        (extremes.max_value, "max"),
        (extremes.min_value, "min"),
        (extremes.max_period, "mean_max"),
        (extremes.min_period, "mean_min"),
    ]
    for column in columns:
        for config_list, kind in checks:
            if column not in config_list:
                continue
            step_no = _extreme_step_for(profiles_df, column, kind)
            _record_extreme(
                profiles_df,
                column,
                step_no,
                _KIND_SUFFIX[kind],
                extreme_period_no,
                cc_list,
                extreme_periods,
            )
    return extreme_periods


def _extreme_step_for(profiles_df: pd.DataFrame, column, kind: ExtremeKind):
    """Return the step index of the extreme of *kind* for *column*."""
    if kind == "max":
        return profiles_df[column].max(axis=1).idxmax()  # type: ignore[arg-type]
    if kind == "min":
        return profiles_df[column].min(axis=1).idxmin()  # type: ignore[arg-type]
    if kind == "mean_max":
        return profiles_df[column].mean(axis=1).idxmax()  # type: ignore[call-overload]
    if kind == "mean_min":
        return profiles_df[column].mean(axis=1).idxmin()  # type: ignore[call-overload]
    assert_never(kind)


def _record_extreme(
    profiles_df: pd.DataFrame,
    column,
    step_no,
    suffix: str,
    extreme_period_no: list,
    cc_list: list,
    extreme_periods: dict,
) -> None:
    """Add a single extreme to the bookkeeping if it's new and not a duplicate cluster."""
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


def _append_col_with(column, append_with: str):
    """Append a suffix to a column name. For MultiIndexes only the last level is changed."""
    if isinstance(column, str):
        return column + append_with
    if isinstance(column, tuple):
        col = list(column)
        col[-1] = col[-1] + append_with
        return tuple(col)
    return column


def _apply_append(
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extreme_periods: dict,
) -> tuple[list, list | np.ndarray, list[int], dict]:
    """Add each extreme as a new standalone cluster, no reassignment of other periods."""
    new_cluster_centers = list(cluster_centers)
    new_cluster_order = list(cluster_order)
    extreme_cluster_idx: list[int] = []

    for i, period_type in enumerate(extreme_periods):
        extreme_cluster_idx.append(len(new_cluster_centers))
        new_cluster_centers.append(extreme_periods[period_type]["profile"])
        new_cluster_order[extreme_periods[period_type]["step_no"]] = i + len(
            cluster_centers
        )

    return new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods


def _apply_new_cluster(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extreme_periods: dict,
) -> tuple[list, list | np.ndarray, list[int], dict]:
    """Add each extreme as a new cluster, then reassign nearby periods to it."""
    new_cluster_centers = list(cluster_centers)
    new_cluster_order = list(cluster_order)
    extreme_cluster_idx: list[int] = []

    for i, period_type in enumerate(extreme_periods):
        extreme_cluster_idx.append(len(new_cluster_centers))
        new_cluster_centers.append(extreme_periods[period_type]["profile"])
        extreme_periods[period_type]["new_cluster_no"] = i + len(cluster_centers)

    extreme_step_nos = {extreme_periods[pt]["step_no"] for pt in extreme_periods}

    for i, c_period in enumerate(new_cluster_order):
        if i in extreme_step_nos:
            # Periods satisfying multiple extreme types stay on their original
            # cluster — ambiguous which new cluster they belong to.
            own_types = [
                pt for pt in extreme_periods if extreme_periods[pt]["step_no"] == i
            ]
            if len(own_types) == 1:
                new_cluster_order[i] = extreme_periods[own_types[0]]["new_cluster_no"]
            continue

        cluster_dist = sum(
            (profiles_df.iloc[i].values - cluster_centers[c_period]) ** 2
        )
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

    return new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods


def _apply_replace(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extreme_periods: dict,
) -> tuple[list, list | np.ndarray, list[int], dict]:
    """Overwrite the affected column of the nearest cluster center with the extreme value.

    No new cluster is created and no period reassignment happens; only the
    relevant column slot of the existing nearest center is overwritten.
    cluster_order is returned unchanged.
    """
    new_cluster_centers = list(cluster_centers)
    extreme_cluster_idx: list[int] = []

    for period_type in extreme_periods:
        index = profiles_df.columns.get_loc(extreme_periods[period_type]["column"])
        new_cluster_centers[extreme_periods[period_type]["cluster_no"]][index] = (
            extreme_periods[period_type]["profile"][index]
        )
        if extreme_periods[period_type]["cluster_no"] not in extreme_cluster_idx:
            extreme_cluster_idx.append(extreme_periods[period_type]["cluster_no"])

    return (
        new_cluster_centers,
        list(cluster_order),
        extreme_cluster_idx,
        extreme_periods,
    )
