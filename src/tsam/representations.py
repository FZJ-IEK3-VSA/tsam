from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from tsam.config import Distribution, MinMaxMean
from tsam.utils.duration_representation import duration_representation

# Aliases: old verbose names → new short names.
# The legacy wrapper sends old names; the pipeline sends new names.
_ALIASES = {
    "meanRepresentation": "mean",
    "medoidRepresentation": "medoid",
    "maxoidRepresentation": "maxoid",
    "minmaxmeanRepresentation": "minmax_mean",
    "durationRepresentation": "distribution",
    "distributionRepresentation": "distribution",
    "distributionAndMinMaxRepresentation": "distribution_minmax",
}


def representations(
    candidates: np.ndarray,
    cluster_order: np.ndarray,
    default: str,
    representation_method: str | Distribution | MinMaxMean | None = None,
    representation_dict: dict[str, str] | None = None,
    distribution_period_wise: bool = True,
    n_timesteps_per_period: int | None = None,
) -> tuple[list[np.ndarray], list[int] | None]:
    cluster_center_indices = None
    if representation_method is None:
        representation_method = default

    # --- Dispatch on Representation objects first ---
    if isinstance(representation_method, Distribution):
        period_wise = representation_method.scope == "cluster"
        cluster_centers = duration_representation(
            candidates,
            cluster_order,
            period_wise,
            n_timesteps_per_period,
            represent_min_max=representation_method.preserve_minmax,
        )
        return cluster_centers, cluster_center_indices

    if isinstance(representation_method, MinMaxMean):
        cluster_centers = minmax_mean_representation(
            candidates,
            cluster_order,
            representation_dict,  # type: ignore[arg-type]
            n_timesteps_per_period,  # type: ignore[arg-type]
        )
        return cluster_centers, cluster_center_indices

    # --- Fallback: string-based dispatch (legacy wrapper compat) ---
    # Normalize old names to new names
    representation_method = _ALIASES.get(representation_method, representation_method)
    if representation_method == "mean":
        cluster_centers = mean_representation(candidates, cluster_order)
    elif representation_method == "medoid":
        cluster_centers, cluster_center_indices = medoid_representation(
            candidates, cluster_order
        )
    elif representation_method == "maxoid":
        cluster_centers, cluster_center_indices = maxoid_representation(
            candidates, cluster_order
        )
    elif representation_method == "minmax_mean":
        cluster_centers = minmax_mean_representation(
            candidates,
            cluster_order,
            representation_dict,  # type: ignore[arg-type]
            n_timesteps_per_period,  # type: ignore[arg-type]
        )
    elif representation_method == "distribution":
        cluster_centers = duration_representation(
            candidates,
            cluster_order,
            distribution_period_wise,
            n_timesteps_per_period,
            represent_min_max=False,
        )
    elif representation_method == "distribution_minmax":
        cluster_centers = duration_representation(
            candidates,
            cluster_order,
            distribution_period_wise,
            n_timesteps_per_period,
            represent_min_max=True,
        )
    else:
        raise ValueError("Chosen 'representationMethod' does not exist.")

    return cluster_centers, cluster_center_indices


def maxoid_representation(
    candidates: np.ndarray,
    cluster_order: np.ndarray,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Represents the candidates of a given cluster group (cluster_order)
    by its maxoid, measured with the euclidean distance.
    """
    # set cluster member that is farthest away from the points of the other clusters as maxoid
    cluster_centers = []
    cluster_center_indices = []
    for cluster_num in np.unique(cluster_order):
        indices = np.where(cluster_order == cluster_num)
        inner_dist_matrix = euclidean_distances(candidates, candidates[indices])
        min_dist_idx = np.argmax(inner_dist_matrix.sum(axis=0))
        cluster_centers.append(candidates[indices][min_dist_idx])
        cluster_center_indices.append(indices[0][min_dist_idx])

    return cluster_centers, cluster_center_indices


def medoid_representation(
    candidates: np.ndarray,
    cluster_order: np.ndarray,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Represents the candidates of a given cluster group (cluster_order)
    by its medoid, measured with the euclidean distance.
    """
    # set cluster center as medoid
    cluster_centers = []
    cluster_center_indices = []
    for cluster_num in np.unique(cluster_order):
        indices = np.where(cluster_order == cluster_num)
        inner_dist_matrix = euclidean_distances(candidates[indices])
        min_dist_idx = np.argmin(inner_dist_matrix.sum(axis=0))
        cluster_centers.append(candidates[indices][min_dist_idx])
        cluster_center_indices.append(indices[0][min_dist_idx])

    return cluster_centers, cluster_center_indices


def mean_representation(
    candidates: np.ndarray,
    cluster_order: np.ndarray,
) -> list[np.ndarray]:
    """
    Represents the candidates of a given cluster group (cluster_order)
    by its mean.
    """
    # set cluster centers as means of the group candidates
    cluster_centers = []
    for cluster_num in np.unique(cluster_order):
        indices = np.where(cluster_order == cluster_num)
        current_mean = candidates[indices].mean(axis=0)
        cluster_centers.append(current_mean)
    return cluster_centers


def minmax_mean_representation(
    candidates: np.ndarray,
    cluster_order: np.ndarray,
    representation_dict: dict[str, str],
    n_timesteps_per_period: int,
) -> list[np.ndarray]:
    """
    Represents the candidates of a given cluster group (cluster_order)
    by either the minimum, the maximum or the mean values of each time step for
    all periods in that cluster depending on the command for each attribute.
    """
    cluster_centers = []
    rep_values = list(representation_dict.values())
    for cluster_num in np.unique(cluster_order):
        indices = np.where(cluster_order == cluster_num)
        current_cluster_center = np.zeros(
            len(representation_dict) * n_timesteps_per_period
        )
        for attribute_num, rep in enumerate(rep_values):
            start_idx = attribute_num * n_timesteps_per_period
            end_idx = (attribute_num + 1) * n_timesteps_per_period
            if rep == "min":
                current_cluster_center[start_idx:end_idx] = candidates[
                    indices, start_idx:end_idx
                ].min(axis=1)
            elif rep == "max":
                current_cluster_center[start_idx:end_idx] = candidates[
                    indices, start_idx:end_idx
                ].max(axis=1)
            elif rep == "mean":
                current_cluster_center[start_idx:end_idx] = candidates[
                    indices, start_idx:end_idx
                ].mean(axis=1)
            else:
                raise ValueError(
                    'At least one value in the representationDict is neither "min", "max" nor "mean".'
                )
        cluster_centers.append(current_cluster_center)
    return cluster_centers
