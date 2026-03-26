"""Clustering wrappers around period_aggregation and representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.period_aggregation import aggregate_periods
from tsam.representations import representations

if TYPE_CHECKING:
    from tsam.config import ClusterConfig, Distribution, MinMaxMean
    from tsam.pipeline.types import PeriodProfiles, PredefParams


def cluster_periods(
    candidates: np.ndarray,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
    n_feature_cols: int,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Run clustering via aggregate_periods.

    Candidates are already weighted (if weights exist). Period-sum columns
    (if any) are stripped for representation via n_feature_cols.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    # If period-sum features are appended, representations must run on
    # the non-augmented prefix so period-sum columns don't leak into
    # representatives.
    rep_candidates: np.ndarray | None = None
    if candidates.shape[1] != n_feature_cols:
        rep_candidates = candidates[:, :n_feature_cols]

    centers, center_indices, order = aggregate_periods(
        candidates,
        n_clusters=n_clusters,
        n_iter=100,
        solver=cluster.solver,
        cluster_method=cluster.method,
        representation_method=cluster.get_representation(),
        representation_dict=representation_dict,
        n_timesteps_per_period=n_timesteps_per_period,
        representation_candidates=rep_candidates,
    )
    return centers, center_indices, order


def cluster_sorted_periods(
    candidates: np.ndarray,
    period_profiles: PeriodProfiles,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Duration-curve clustering: sort descending, cluster, pick medoid from original.

    Candidates are already weighted (if weights exist). Medoids are picked
    from the (weighted) unsorted candidates.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    n_columns = period_profiles.n_columns

    # Sort each period's timesteps descending for all columns.
    # Use candidates (already weighted) so that clustering distance respects
    # column weights — matching v3 behaviour.
    n_periods, n_total = candidates.shape
    n_timesteps = n_total // n_columns

    values_3d = candidates.copy().reshape(n_periods, n_columns, n_timesteps)
    sorted_values = (-np.sort(-values_3d, axis=2, kind="stable")).reshape(n_periods, -1)

    _, center_indices, cluster_order = aggregate_periods(
        sorted_values,
        n_clusters=n_clusters,
        n_iter=30,
        solver=cluster.solver,
        cluster_method=cluster.method,
        representation_method=cluster.get_representation(),
        representation_dict=representation_dict,
        n_timesteps_per_period=n_timesteps_per_period,
    )

    # Pick medoid from unsorted candidates (already weighted).
    cluster_centers = []
    for cluster_num in np.unique(cluster_order):
        indices = np.where(cluster_order == cluster_num)[0]
        if len(indices) > 1:
            current_mean = sorted_values[indices].mean(axis=0)
            mindist_idx = np.argmin(
                np.square(sorted_values[indices] - current_mean).sum(axis=1)
            )
            cluster_centers.append(candidates[indices][mindist_idx])
        else:
            cluster_centers.append(candidates[indices][0])

    return cluster_centers, center_indices, cluster_order


def use_predefined_assignments(
    candidates: np.ndarray,
    predef: PredefParams,
    representation_method: str | Distribution | MinMaxMean | None,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
) -> tuple[list[np.ndarray] | np.ndarray, list[int] | None, list | np.ndarray]:
    """Skip clustering, compute representatives from predefined assignments.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    if predef.cluster_center_indices is not None:
        return (
            candidates[predef.cluster_center_indices],
            list(predef.cluster_center_indices),
            predef.cluster_order,
        )
    else:
        centers, computed_indices = representations(
            candidates,
            predef.cluster_order,  # type: ignore[arg-type]
            default="medoid",
            representation_method=representation_method,
            representation_dict=representation_dict,
            n_timesteps_per_period=n_timesteps_per_period,
        )
        return centers, computed_indices, predef.cluster_order
