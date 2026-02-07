"""Clustering wrappers around period_aggregation and representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.period_aggregation import aggregate_periods
from tsam.representations import representations

if TYPE_CHECKING:
    from tsam.config import ClusterConfig
    from tsam.pipeline.types import PeriodProfiles, PredefParams


def cluster_periods(
    candidates: np.ndarray,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
) -> tuple[list, list | None, np.ndarray]:
    """Run clustering via aggregate_periods.

    Replicates monolith lines 1172-1188.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    centers, center_indices, order = aggregate_periods(
        candidates,
        n_clusters=n_clusters,
        n_iter=100,
        solver=cluster.solver,
        cluster_method=cluster.method,
        representation_method=cluster.get_representation(),
        representation_dict=representation_dict,
        n_timesteps_per_period=n_timesteps_per_period,
    )
    return centers, center_indices, order


def cluster_sorted_periods(
    candidates: np.ndarray,
    period_profiles: PeriodProfiles,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
) -> tuple[list, list | None, np.ndarray]:
    """Duration-curve clustering: sort descending, cluster, pick medoid from original.

    Replicates _cluster_sorted_periods (monolith lines 1029-1091).

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    profiles_values = period_profiles.profiles_dataframe.values
    n_columns = period_profiles.n_columns

    # Sort each period's timesteps descending for all columns
    n_periods, n_total = profiles_values.shape
    n_timesteps = n_total // n_columns

    values_3d = profiles_values.copy().reshape(n_periods, n_columns, n_timesteps)
    sorted_values = (-np.sort(-values_3d, axis=2, kind="stable")).reshape(n_periods, -1)

    _alt_centers, center_indices, cluster_order = aggregate_periods(
        sorted_values,
        n_clusters=n_clusters,
        n_iter=30,
        solver=cluster.solver,
        cluster_method=cluster.method,
        representation_method=cluster.get_representation(),
        representation_dict=representation_dict,
        n_timesteps_per_period=n_timesteps_per_period,
    )

    # Pick medoid from original (unsorted) candidates
    cluster_centers = []
    for cluster_num in np.unique(cluster_order):
        indice = np.where(cluster_order == cluster_num)[0]
        if len(indice) > 1:
            current_mean = sorted_values[indice].mean(axis=0)
            mindist_idx = np.argmin(
                np.square(sorted_values[indice] - current_mean).sum(axis=1)
            )
            cluster_centers.append(candidates[indice][mindist_idx])
        else:
            cluster_centers.append(candidates[indice][0])

    return cluster_centers, center_indices, cluster_order


def use_predefined_assignments(
    candidates: np.ndarray,
    predef: PredefParams,
    representation_method,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
) -> tuple[list | np.ndarray, list | None, list | np.ndarray]:
    """Skip clustering, compute representatives from predefined assignments.

    Replicates monolith lines 1154-1169.

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
            predef.cluster_order,
            default="medoid",
            representation_method=representation_method,
            representation_dict=representation_dict,
            n_timesteps_per_period=n_timesteps_per_period,
        )
        return centers, computed_indices, predef.cluster_order
