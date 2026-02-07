import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from tsam.config import Distribution, MinMaxMean
from tsam.utils.duration_representation import duration_representation

# Aliases: old verbose names → new short names.
# The monolith sends old names; the pipeline sends new names.
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
    candidates,
    cluster_order,
    default,
    representation_method=None,
    representation_dict=None,
    distribution_period_wise=True,
    n_timesteps_per_period=None,
):
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
            candidates, cluster_order, representation_dict, n_timesteps_per_period
        )
        return cluster_centers, cluster_center_indices

    # --- Fallback: string-based dispatch (monolith compat) ---
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
            candidates, cluster_order, representation_dict, n_timesteps_per_period
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


def maxoid_representation(candidates, cluster_order):
    """
    Represents the candidates of a given cluster group (cluster_order)
    by its medoid, measured with the euclidean distance.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param cluster_order: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type cluster_order: np.array
    """
    # set cluster member that is farthest away from the points of the other clusters as maxoid
    cluster_centers = []
    cluster_center_indices = []
    for cluster_num in np.unique(cluster_order):
        indice = np.where(cluster_order == cluster_num)
        inner_dist_matrix = euclidean_distances(candidates, candidates[indice])
        min_dist_idx = np.argmax(inner_dist_matrix.sum(axis=0))
        cluster_centers.append(candidates[indice][min_dist_idx])
        cluster_center_indices.append(indice[0][min_dist_idx])

    return cluster_centers, cluster_center_indices


def medoid_representation(candidates, cluster_order):
    """
    Represents the candidates of a given cluster group (cluster_order)
    by its medoid, measured with the euclidean distance.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param cluster_order: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type cluster_order: np.array
    """
    # set cluster center as medoid
    cluster_centers = []
    cluster_center_indices = []
    for cluster_num in np.unique(cluster_order):
        indice = np.where(cluster_order == cluster_num)
        inner_dist_matrix = euclidean_distances(candidates[indice])
        min_dist_idx = np.argmin(inner_dist_matrix.sum(axis=0))
        cluster_centers.append(candidates[indice][min_dist_idx])
        cluster_center_indices.append(indice[0][min_dist_idx])

    return cluster_centers, cluster_center_indices


def mean_representation(candidates, cluster_order):
    """
    Represents the candidates of a given cluster group (cluster_order)
    by its mean.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param cluster_order: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type cluster_order: np.array
    """
    # set cluster centers as means of the group candidates
    cluster_centers = []
    for cluster_num in np.unique(cluster_order):
        indice = np.where(cluster_order == cluster_num)
        current_mean = candidates[indice].mean(axis=0)
        cluster_centers.append(current_mean)
    return cluster_centers


def minmax_mean_representation(
    candidates, cluster_order, representation_dict, n_timesteps_per_period
):
    """
    Represents the candidates of a given cluster group (cluster_order)
    by either the minimum, the maximum or the mean values of each time step for
    all periods in that cluster depending on the command for each attribute.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param cluster_order: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type cluster_order: np.array

    :param representation_dict: A dictionary which defines for each attribute whether the typical
        period should be represented by the minimum or maximum values within each cluster.
        optional (default: None)
    :type representation_dict: dictionary

    :param n_timesteps_per_period: The number of discrete timesteps which describe one period. required
    :type n_timesteps_per_period: integer
    """
    # set cluster center depending of the representation_dict
    cluster_centers = []
    for cluster_num in np.unique(cluster_order):
        indice = np.where(cluster_order == cluster_num)
        current_cluster_center = np.zeros(
            len(representation_dict) * n_timesteps_per_period
        )
        for attribute_num in range(len(representation_dict)):
            start_idx = attribute_num * n_timesteps_per_period
            end_idx = (attribute_num + 1) * n_timesteps_per_period
            if list(representation_dict.values())[attribute_num] == "min":
                current_cluster_center[start_idx:end_idx] = candidates[
                    indice, start_idx:end_idx
                ].min(axis=1)
            elif list(representation_dict.values())[attribute_num] == "max":
                current_cluster_center[start_idx:end_idx] = candidates[
                    indice, start_idx:end_idx
                ].max(axis=1)
            elif list(representation_dict.values())[attribute_num] == "mean":
                current_cluster_center[start_idx:end_idx] = candidates[
                    indice, start_idx:end_idx
                ].mean(axis=1)
            else:
                raise ValueError(
                    'At least one value in the representationDict is neither "min", "max" nor "mean".'
                )
        cluster_centers.append(current_cluster_center)
    return cluster_centers
