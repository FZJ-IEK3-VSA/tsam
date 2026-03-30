from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.representations import representations

if TYPE_CHECKING:
    from tsam.config import Distribution, MinMaxMean

# Aliases: old verbose names → new short names.
# The legacy wrapper sends old names; the pipeline sends new names.
_METHOD_ALIASES = {
    "k_means": "kmeans",
    "k_medoids": "kmedoids",
    "k_maxoids": "kmaxoids",
    "adjacent_periods": "contiguous",
}


def aggregate_periods(
    candidates: np.ndarray,
    n_clusters: int = 8,
    n_iter: int = 100,
    cluster_method: str = "kmeans",
    solver: str = "highs",
    representation_method: str | Distribution | MinMaxMean | None = None,
    representation_dict: dict[str, str] | None = None,
    distribution_period_wise: bool = True,
    n_timesteps_per_period: int | None = None,
    representation_candidates: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """
    Clusters the data based on one of the cluster methods:
    'averaging', 'kmeans', 'kmedoids' or 'hierarchical'.
    """
    # Normalize old names to new names
    cluster_method = _METHOD_ALIASES.get(cluster_method, cluster_method)

    # Use separate candidates for representation if provided
    _rep_candidates = (
        representation_candidates
        if representation_candidates is not None
        else candidates
    )

    # cluster the data
    if cluster_method == "averaging":
        n_sets = len(candidates)
        cluster_size = n_sets // n_clusters
        order_lists = [[n_cluster] * cluster_size for n_cluster in range(n_clusters)]
        remainder = n_sets - cluster_size * n_clusters
        if remainder > 0:
            order_lists.append([n_clusters - 1] * remainder)
        cluster_order = np.hstack(np.array(order_lists, dtype=object))  # type: ignore[call-overload]
        cluster_centers, cluster_center_indices = representations(
            _rep_candidates,
            cluster_order,
            default="mean",
            representation_method=representation_method,
            representation_dict=representation_dict,
            distribution_period_wise=distribution_period_wise,
            n_timesteps_per_period=n_timesteps_per_period,
        )

    elif cluster_method == "kmeans":
        from sklearn.cluster import KMeans

        k_means = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=n_iter, tol=1e-4)

        cluster_order = k_means.fit_predict(candidates)
        # get with own mean representation to avoid numerical trouble caused by sklearn
        cluster_centers, cluster_center_indices = representations(
            _rep_candidates,
            cluster_order,
            default="mean",
            representation_method=representation_method,
            representation_dict=representation_dict,
            distribution_period_wise=distribution_period_wise,
            n_timesteps_per_period=n_timesteps_per_period,
        )

    elif cluster_method == "kmedoids":
        from tsam.utils.k_medoids_exact import KMedoids

        k_medoid = KMedoids(n_clusters=n_clusters, solver=solver)

        cluster_order = k_medoid.fit_predict(candidates)
        cluster_centers, cluster_center_indices = representations(
            _rep_candidates,
            cluster_order,
            default="medoid",
            representation_method=representation_method,
            representation_dict=representation_dict,
            distribution_period_wise=distribution_period_wise,
            n_timesteps_per_period=n_timesteps_per_period,
        )

    elif cluster_method == "kmaxoids":
        from tsam.utils.k_maxoids import KMaxoids

        k_maxoid = KMaxoids(n_clusters=n_clusters)

        cluster_order = k_maxoid.fit_predict(candidates)
        cluster_centers, cluster_center_indices = representations(
            _rep_candidates,
            cluster_order,
            default="maxoid",
            representation_method=representation_method,
            representation_dict=representation_dict,
            distribution_period_wise=distribution_period_wise,
            n_timesteps_per_period=n_timesteps_per_period,
        )

    elif cluster_method == "hierarchical" or cluster_method == "contiguous":
        if n_clusters == 1:
            cluster_order = np.asarray([0] * len(candidates))
        else:
            from sklearn.cluster import AgglomerativeClustering

            if cluster_method == "hierarchical":
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward"
                )
            elif cluster_method == "contiguous":
                adjacency_matrix = np.eye(len(candidates), k=1) + np.eye(
                    len(candidates), k=-1
                )
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward", connectivity=adjacency_matrix
                )
            cluster_order = clustering.fit_predict(candidates)
        # represent hierarchical aggregation with medoid
        cluster_centers, cluster_center_indices = representations(
            _rep_candidates,
            cluster_order,
            default="medoid",
            representation_method=representation_method,
            representation_dict=representation_dict,
            distribution_period_wise=distribution_period_wise,
            n_timesteps_per_period=n_timesteps_per_period,
        )

    else:
        raise ValueError(
            f"Unknown cluster_method '{cluster_method}'. "
            f"Valid options: 'averaging', 'kmeans', 'kmedoids', 'kmaxoids', 'hierarchical', 'contiguous'."
        )

    return cluster_centers, cluster_center_indices, cluster_order
