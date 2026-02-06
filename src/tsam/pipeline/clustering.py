"""Clustering wrappers around periodAggregation and representations."""

from __future__ import annotations

import numpy as np

from tsam.periodAggregation import aggregatePeriods
from tsam.representations import representations


def cluster_periods(
    candidates: np.ndarray,
    n_clusters: int,
    cluster_method: str,
    solver: str,
    representation_method: str | None,
    representation_dict: dict | None,
    distribution_period_wise: bool,
    n_timesteps_per_period: int,
) -> tuple[list, list | None, np.ndarray]:
    """Run clustering via aggregatePeriods.

    Replicates monolith lines 1172-1188.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    centers, center_indices, order = aggregatePeriods(
        candidates,
        n_clusters=n_clusters,
        n_iter=100,
        solver=solver,
        clusterMethod=cluster_method,
        representationMethod=representation_method,
        representationDict=representation_dict,
        distributionPeriodWise=distribution_period_wise,
        timeStepsPerPeriod=n_timesteps_per_period,
    )
    return centers, center_indices, order


def cluster_sorted_periods(
    candidates: np.ndarray,
    profiles_values: np.ndarray,
    n_columns: int,
    n_clusters: int,
    cluster_method: str,
    solver: str,
    representation_method: str | None,
    representation_dict: dict | None,
    distribution_period_wise: bool,
    n_timesteps_per_period: int,
) -> tuple[list, np.ndarray, list | None]:
    """Duration-curve clustering: sort descending, cluster, pick medoid from original.

    Replicates _clusterSortedPeriods (monolith lines 1029-1091).

    Returns (cluster_centers, cluster_order, cluster_center_indices).
    """
    # Sort each period's timesteps descending for all columns
    n_periods, n_total = profiles_values.shape
    n_timesteps = n_total // n_columns

    values_3d = profiles_values.copy().reshape(n_periods, n_columns, n_timesteps)
    sorted_values = (-np.sort(-values_3d, axis=2, kind="stable")).reshape(n_periods, -1)

    _alt_centers, center_indices, cluster_order = aggregatePeriods(
        sorted_values,
        n_clusters=n_clusters,
        n_iter=30,
        solver=solver,
        clusterMethod=cluster_method,
        representationMethod=representation_method,
        representationDict=representation_dict,
        distributionPeriodWise=distribution_period_wise,
        timeStepsPerPeriod=n_timesteps_per_period,
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

    return cluster_centers, cluster_order, center_indices


def use_predefined_assignments(
    candidates: np.ndarray,
    cluster_order: list | np.ndarray,
    center_indices: list | np.ndarray | None,
    representation_method: str | None,
    representation_dict: dict | None,
    distribution_period_wise: bool,
    n_timesteps_per_period: int,
) -> tuple[list | np.ndarray, list | None, list | np.ndarray]:
    """Skip clustering, compute representatives from predefined assignments.

    Replicates monolith lines 1154-1169.

    Returns (cluster_centers, cluster_center_indices, cluster_order).
    """
    if center_indices is not None:
        return candidates[center_indices], list(center_indices), cluster_order
    else:
        centers, computed_indices = representations(
            candidates,
            cluster_order,
            default="medoidRepresentation",
            representationMethod=representation_method,
            representationDict=representation_dict,
            distributionPeriodWise=distribution_period_wise,
            timeStepsPerPeriod=n_timesteps_per_period,
        )
        return centers, computed_indices, cluster_order
