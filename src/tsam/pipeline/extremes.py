"""Extreme period counting and integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.pipeline.types import ExtremePeriod

if TYPE_CHECKING:
    import pandas as pd

    from tsam.config import ExtremeConfig
    from tsam.pipeline.types import ExtremeKind


def _detect_extreme(
    profiles_df: pd.DataFrame,
    column: str | tuple,
    kind: ExtremeKind,
    period_row_index: int,
    center_profiles: list,
    extreme_periods: list[ExtremePeriod],
) -> None:
    """Record one already-located extreme period, unless it is redundant.

    The caller has reduced a column to the single period that holds its
    extreme; this appends that period to *extreme_periods*. It is skipped in
    two cases: the period was already claimed as the extreme of another column,
    or its profile is identical to a cluster center that clustering produced
    anyway — in both cases keeping it would duplicate a typical period rather
    than preserve new information.

    Args:
        profiles_df: Period profiles the extreme profile is taken from.
        column: Column whose extreme this period is.
        kind: Which extreme of that column the period holds.
        period_row_index: Row index in *profiles_df* of the extreme period.
        center_profiles: Cluster centers as plain lists, for the duplicate
            check.
        extreme_periods: Extremes accepted so far; appended to in place.
    """
    profile = np.asarray(profiles_df.loc[period_row_index, :].values)
    already_taken = any(
        extreme.period_row_index == period_row_index for extreme in extreme_periods
    )
    if already_taken or profile.tolist() in center_profiles:
        return None
    extreme_periods.append(
        ExtremePeriod(
            column=column, kind=kind, period_row_index=period_row_index, profile=profile
        )
    )


def add_extreme_periods(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extremes: ExtremeConfig,
) -> tuple[list, list | np.ndarray, list[int], list[ExtremePeriod]]:
    """Ensure periods with extreme values survive into the typical-period set.

    Optional stage, configured by `ExtremeConfig`. Clustering tends to average
    away rare but important periods (peak demand, minimum solar). This stage
    detects such periods and explicitly represents them in the output.

    Extreme detection itself (per-column ``idxmax``/``idxmin``) is
    weight-invariant. When weights are active the ``profiles_df`` passed in is
    already weighted, so the ``new_cluster`` method's distance-based
    reassignment respects weights; the extracted profiles carry weights, which
    are stripped uniformly during the later unweighting step.

    **Extreme types** (which period is preserved):

    | Config field | What it preserves |
    |---|---|
    | `max_value=["demand"]` | The period containing the single highest demand value. |
    | `min_value=["solar"]` | The period containing the single lowest solar value. |
    | `max_period=["demand"]` | The period with the highest average demand. |
    | `min_period=["solar"]` | The period with the lowest average solar. |

    **Integration methods** (``ExtremeConfig.method``):

    | Method | Behavior |
    |---|---|
    | `"append"` | Add extreme periods as new clusters (increases `n_clusters`). Default. |
    | `"new_cluster"` | Like append, but also reassign nearby periods to the new cluster. |
    | `"replace"` | Overwrite the relevant column values in the nearest existing center. |

    Args:
        profiles_df: Period profiles (weighted if weights are active), searched
            for extremes.
        cluster_centers: Representatives produced by clustering, to be extended.
        cluster_order: Per-period cluster assignment, updated in place for the
            chosen method.
        extremes: Which extremes to preserve and how to integrate them.

    Returns:
        ``(new_cluster_centers, new_cluster_order, extreme_cluster_idx,
        extreme_periods)`` — the updated center list and assignment, the indices
        of the newly added extreme clusters, and one
        [`ExtremePeriod`][tsam.pipeline.types.ExtremePeriod] per preserved
        period, in the order they were added as clusters.

    Note:
        `cluster_periods` produces the clusters this stage augments, and
        `rescale_representatives` skips extreme clusters when correcting means.
    """
    columns = profiles_df.columns.get_level_values(0).unique().tolist()
    extreme_periods: list[ExtremePeriod] = []

    center_profiles = [center.tolist() for center in cluster_centers]

    # Detect extreme periods for each column
    # (lists_of_column_names_to_consider, extreme_variant)
    _CHECKS: list[tuple[list[str], ExtremeKind]] = [
        (extremes.max_value, "max"),
        (extremes.min_value, "min"),
        (extremes.max_period, "mean_max"),
        (extremes.min_period, "mean_min"),
    ]
    for column in columns:
        for config_list, kind in _CHECKS:
            if column not in config_list:
                continue
            if kind == "max":
                step_no = profiles_df[column].max(axis=1).idxmax()  # type: ignore[arg-type]
            elif kind == "min":
                step_no = profiles_df[column].min(axis=1).idxmin()  # type: ignore[arg-type]
            elif kind == "mean_max":
                step_no = profiles_df[column].mean(axis=1).idxmax()  # type: ignore[call-overload]
            elif kind == "mean_min":
                step_no = profiles_df[column].mean(axis=1).idxmin()  # type: ignore[call-overload]
            else:
                raise AssertionError(
                    f"unhandled kind: {kind!r}. Only {ExtremeKind!r} is handled."
                )

            _detect_extreme(
                profiles_df=profiles_df,
                column=column,
                kind=kind,
                period_row_index=step_no,
                center_profiles=center_profiles,
                extreme_periods=extreme_periods,
            )

    new_cluster_centers: list = []
    new_cluster_order = list(cluster_order)
    extreme_cluster_idx: list[int] = []

    if extremes.method == "append":
        new_cluster_centers = list(cluster_centers)
        for i, extreme in enumerate(extreme_periods):
            extreme_cluster_idx.append(len(new_cluster_centers))
            new_cluster_centers.append(extreme.profile)
            new_cluster_order[extreme.period_row_index] = i + len(cluster_centers)

    elif extremes.method == "new_cluster":
        new_cluster_centers = list(cluster_centers)
        for i, extreme in enumerate(extreme_periods):
            extreme_cluster_idx.append(len(new_cluster_centers))
            new_cluster_centers.append(extreme.profile)
            extreme.new_cluster_no = i + len(cluster_centers)

        # A period holds at most one extreme, so this is a 1:1 lookup.
        extreme_by_step = {
            extreme.period_row_index: extreme for extreme in extreme_periods
        }

        for i, c_period in enumerate(new_cluster_order):
            # A period that is itself an extreme joins its own new cluster.
            own_extreme = extreme_by_step.get(i)
            if own_extreme is not None:
                new_cluster_order[i] = own_extreme.new_cluster_no
                continue

            period_profile = profiles_df.iloc[i].values
            # Find the closest extreme period (deterministic: first match with smallest distance)
            best_extreme = None
            best_dist = sum((period_profile - cluster_centers[c_period]) ** 2)
            for extreme in extreme_periods:
                extreme_dist = sum((period_profile - extreme.profile) ** 2)
                if extreme_dist < best_dist:
                    best_dist = extreme_dist
                    best_extreme = extreme

            if best_extreme is not None:
                new_cluster_order[i] = best_extreme.new_cluster_no

    elif extremes.method == "replace":
        new_cluster_centers = list(cluster_centers)
        for extreme in extreme_periods:
            index = profiles_df.columns.get_loc(extreme.column)
            cluster_no = int(cluster_order[extreme.period_row_index])
            new_cluster_centers[cluster_no][index] = extreme.profile[index]
            if cluster_no not in extreme_cluster_idx:
                extreme_cluster_idx.append(cluster_no)

    return new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods
