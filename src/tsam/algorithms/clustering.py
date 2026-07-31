from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.algorithms.representations import representations
from tsam.config import DEFAULT_REPRESENTATION

if TYPE_CHECKING:
    from tsam.config import (
        ClusterMethod,
        ClusterMethodName,
        Distribution,
        MinMaxMean,
    )


def assign_clusters(
    candidates: np.ndarray,
    n_clusters: int,
    cluster_method: ClusterMethod,
    n_iter: int = 100,
) -> np.ndarray:
    """Assign each period to a cluster.

    Pure clustering step: maps each period (row of ``candidates``) to a cluster
    id, without computing the cluster representatives. See
    :func:`cluster_and_represent` for the combined clustering + representation
    step.

    ``cluster_method`` is a method name ('averaging', 'kmeans', 'kmedoids',
    'kmaxoids', 'hierarchical', 'contiguous') or a method-config object —
    ``KMedoids`` carrying solver options, ``KMeans`` carrying the restart
    count.
    """
    from tsam.config import KMeans as KMeansConfig
    from tsam.config import KMedoids

    kmedoids_config = cluster_method if isinstance(cluster_method, KMedoids) else None
    kmeans_config = cluster_method if isinstance(cluster_method, KMeansConfig) else None
    if kmedoids_config is not None:
        cluster_method = "kmedoids"
    elif kmeans_config is not None:
        cluster_method = "kmeans"

    if cluster_method == "averaging":
        n_sets = len(candidates)
        cluster_size = n_sets // n_clusters
        order_lists = [[c] * cluster_size for c in range(n_clusters)]
        remainder = n_sets - cluster_size * n_clusters
        if remainder > 0:
            order_lists.append([n_clusters - 1] * remainder)
        order = np.hstack(np.array(order_lists, dtype=object))  # type: ignore[call-overload]
        return np.asarray(order)

    if cluster_method == "kmeans":
        from sklearn.cluster import KMeans

        # An explicit n_init on the config wins; otherwise the caller's default
        # stands, which differs between the ordinary and duration-curve paths.
        if kmeans_config is not None and kmeans_config.n_init is not None:
            n_iter = kmeans_config.n_init
        random_state = kmeans_config.random_state if kmeans_config else None
        k_means = KMeans(
            n_clusters=n_clusters,
            max_iter=1000,
            n_init=n_iter,
            tol=1e-4,
            random_state=random_state,
        )
        return np.asarray(k_means.fit_predict(candidates))

    if cluster_method == "kmedoids":
        from tsam.algorithms.k_medoids_exact import ExactKMedoids

        config = kmedoids_config if kmedoids_config is not None else KMedoids()
        kmedoids = ExactKMedoids(
            n_clusters=n_clusters,
            solver=config.solver,
            options=config.options,
        )
        return np.asarray(kmedoids.fit_predict(candidates))

    if cluster_method == "kmaxoids":
        from tsam.algorithms.k_maxoids import KMaxoids

        return np.asarray(KMaxoids(n_clusters=n_clusters).fit_predict(candidates))

    if cluster_method in ("hierarchical", "contiguous"):
        if n_clusters == 1:
            return np.asarray([0] * len(candidates))

        from sklearn.cluster import AgglomerativeClustering

        if cluster_method == "hierarchical":
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        else:  # contiguous: only adjacent periods may be merged
            adjacency_matrix = np.eye(len(candidates), k=1) + np.eye(
                len(candidates), k=-1
            )
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward", connectivity=adjacency_matrix
            )
        return np.asarray(clustering.fit_predict(candidates))

    raise ValueError(
        f"Unknown cluster_method '{cluster_method}'. "
        f"Valid options: 'averaging', 'kmeans', 'kmedoids', 'kmaxoids', "
        f"'hierarchical', 'contiguous'."
    )


def cluster_and_represent(
    candidates: np.ndarray,
    n_clusters: int = 8,
    n_iter: int = 100,
    cluster_method: ClusterMethod = "kmeans",
    representation_method: str | Distribution | MinMaxMean | None = None,
    representation_dict: dict[str, str] | None = None,
    distribution_period_wise: bool = True,
    n_timesteps_per_period: int | None = None,
    representation_candidates: np.ndarray | None = None,
    reference_attribute_idx: int | None = None,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Cluster ``candidates`` and compute the representative profile per cluster.

    Two steps: :func:`assign_clusters` assigns each period to a cluster, then
    :func:`~tsam.algorithms.representations.representations` derives the cluster
    representatives (using the per-method default unless ``representation_method``
    overrides it).
    """
    from tsam.config import KMeans as KMeansConfig
    from tsam.config import KMedoids

    method_name: ClusterMethodName
    if isinstance(cluster_method, KMedoids):
        method_name = "kmedoids"
    elif isinstance(cluster_method, KMeansConfig):
        method_name = "kmeans"
    else:
        method_name = cluster_method
    cluster_order = assign_clusters(
        candidates,
        n_clusters,
        cluster_method,
        n_iter=n_iter,
    )

    # Representatives may be drawn from a separate candidate set
    # (e.g. unweighted or with included period sums).
    rep_candidates = (
        representation_candidates
        if representation_candidates is not None
        else candidates
    )
    cluster_centers, cluster_center_indices = representations(
        rep_candidates,
        cluster_order,
        default=DEFAULT_REPRESENTATION[method_name],
        representation_method=representation_method,
        representation_dict=representation_dict,
        distribution_period_wise=distribution_period_wise,
        n_timesteps_per_period=n_timesteps_per_period,
        reference_attribute_idx=reference_attribute_idx,
    )
    return cluster_centers, cluster_center_indices, cluster_order
