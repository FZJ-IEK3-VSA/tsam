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
    weighted_candidates: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Run clustering via aggregate_periods.

    If weighted_candidates is provided, clustering uses weighted data for
    distance calculation but representations are computed from unweighted
    candidates.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    clustering_input = (
        weighted_candidates if weighted_candidates is not None else candidates
    )
    centers, center_indices, order = aggregate_periods(
        clustering_input,
        n_clusters=n_clusters,
        n_iter=100,
        solver=cluster.solver,
        cluster_method=cluster.method,
        representation_method=cluster.get_representation(),
        representation_dict=representation_dict,
        n_timesteps_per_period=n_timesteps_per_period,
        representation_candidates=candidates
        if weighted_candidates is not None
        else None,
    )
    return centers, center_indices, order


def cluster_sorted_periods(
    candidates: np.ndarray,
    period_profiles: PeriodProfiles,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
    weighted_candidates: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Duration-curve clustering: sort descending, cluster, pick medoid from original.

    If weighted_candidates is provided, the sorted profiles are weighted
    before clustering distance computation, but medoids are picked from
    the unweighted unsorted candidates.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    profiles_values = period_profiles.profiles_dataframe.values
    n_columns = period_profiles.n_columns

    # Sort each period's timesteps descending for all columns
    n_periods, n_total = profiles_values.shape
    n_timesteps = n_total // n_columns

    values_3d = profiles_values.copy().reshape(n_periods, n_columns, n_timesteps)
    sorted_values = (-np.sort(-values_3d, axis=2, kind="stable")).reshape(n_periods, -1)

    # If weights are active, also sort the weighted profiles and cluster on those
    if weighted_candidates is not None:
        w_3d = weighted_candidates.copy().reshape(n_periods, n_columns, n_timesteps)
        sorted_weighted = (-np.sort(-w_3d, axis=2, kind="stable")).reshape(
            n_periods, -1
        )
        clustering_input = sorted_weighted
    else:
        clustering_input = sorted_values

    _, center_indices, cluster_order = aggregate_periods(
        clustering_input,
        n_clusters=n_clusters,
        n_iter=30,
        solver=cluster.solver,
        cluster_method=cluster.method,
        representation_method=cluster.get_representation(),
        representation_dict=representation_dict,
        n_timesteps_per_period=n_timesteps_per_period,
        representation_candidates=sorted_values
        if weighted_candidates is not None
        else None,
    )

    # Pick medoid from original (unsorted, unweighted) candidates
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
