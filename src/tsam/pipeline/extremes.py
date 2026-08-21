"""Extreme period counting and integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsam.pipeline.types import ExtremePeriod

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tsam.config import ExtremeConfig
    from tsam.pipeline.types import ExtremeKind


def detect_extreme_periods(
    profiles_df: pd.DataFrame,
    extremes: ExtremeConfig,
    cluster_centers: list | None = None,
) -> list[ExtremePeriod]:
    """Detect which periods carry the configured extremes, independent of clustering.

    Detection is a pure function of the period profiles (per-column
    ``idxmax``/``idxmin`` on the value or the period mean), so it can run before
    clustering or after it. Passing ``cluster_centers=None`` disables the "already
    a center" dedup, which the preserve_n_clusters path relies on: the detected
    periods are excluded from clustering, so they can never coincide with a
    regular center.

    Args:
        profiles_df: Period profiles (weighted if weights are active).
        extremes: Which extremes to preserve.
        cluster_centers: Existing representatives, used only for the "already a
            center" dedup. ``None`` disables that guard.

    Returns:
        One [`ExtremePeriod`][tsam.pipeline.types.ExtremePeriod] per extreme that
        survives the redundancy check, in detection order.
    """
    center_profiles = (
        [center.tolist() for center in cluster_centers]
        if cluster_centers is not None
        else []
    )
    # (columns to check, which extreme of them to look for)
    checks: list[tuple[list[str], ExtremeKind]] = [
        (extremes.max_value, "max"),
        (extremes.min_value, "min"),
        (extremes.max_period, "mean_max"),
        (extremes.min_period, "mean_min"),
    ]

    extreme_periods: list[ExtremePeriod] = []
    claimed: set[int] = set()
    for column in profiles_df.columns.get_level_values(0).unique():
        for configured_columns, kind in checks:
            if column not in configured_columns:
                continue
            candidate = ExtremePeriod.locate(profiles_df, column, kind)
            # Two ways a located period is not worth keeping: another column
            # already claimed it, or clustering produced its profile as a
            # center anyway. Either way it would duplicate a typical period
            # rather than preserve something new.
            if candidate.step_no in claimed:
                continue
            if candidate.profile.tolist() in center_profiles:
                continue
            claimed.add(candidate.step_no)
            extreme_periods.append(candidate)

    return extreme_periods


def add_extreme_periods(
    profiles_df: pd.DataFrame,
    cluster_centers: list,
    cluster_order: list | np.ndarray,
    extremes: ExtremeConfig,
    predetected: list[ExtremePeriod] | None = None,
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

    ``ExtremeConfig.preserve_n_clusters`` instead carves the extremes out of the
    budget so the total stays exactly ``n_clusters``; that path detects extremes
    up front and passes them in via ``predetected``.

    Args:
        profiles_df: Period profiles (weighted if weights are active), searched
            for extremes.
        cluster_centers: Representatives produced by clustering, to be extended.
        cluster_order: Per-period cluster assignment, updated in place for the
            chosen method.
        extremes: Which extremes to preserve and how to integrate them.
        predetected: Extremes already found by `detect_extreme_periods`, reused
            instead of re-detecting. The preserve_n_clusters path passes them so
            the same periods size the budget and are added back. When ``None``,
            detection runs here as before.

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
    if predetected is not None:
        extreme_periods = predetected
    else:
        extreme_periods = detect_extreme_periods(profiles_df, extremes, cluster_centers)

    new_cluster_centers = list(cluster_centers)
    new_cluster_order = list(cluster_order)
    extreme_cluster_idx: list[int] = []

    if extremes.method in ("append", "new_cluster"):
        # Both give every extreme a cluster of its own; they differ only in
        # whether the other periods are then allowed to migrate into it.
        for extreme in extreme_periods:
            extreme.new_cluster_no = len(new_cluster_centers)
            extreme_cluster_idx.append(extreme.new_cluster_no)
            new_cluster_centers.append(extreme.profile)
            new_cluster_order[extreme.step_no] = extreme.new_cluster_no

    if extremes.method == "new_cluster":
        extreme_steps = {extreme.step_no for extreme in extreme_periods}
        for i, c_period in enumerate(new_cluster_order):
            if i in extreme_steps:
                continue  # already sitting in its own cluster

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
        for extreme in extreme_periods:
            index = profiles_df.columns.get_loc(extreme.column)
            cluster_no = int(cluster_order[extreme.step_no])
            new_cluster_centers[cluster_no][index] = extreme.profile[index]
            if cluster_no not in extreme_cluster_idx:
                extreme_cluster_idx.append(cluster_no)

    return new_cluster_centers, new_cluster_order, extreme_cluster_idx, extreme_periods
