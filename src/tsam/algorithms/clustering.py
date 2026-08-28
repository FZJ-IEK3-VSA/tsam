from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.algorithms.representations import representations
from tsam.config import DEFAULT_REPRESENTATION, KMedoids, method_name

if TYPE_CHECKING:
    from tsam.config import ClusterMethod, Distribution, MinMaxMean


def densify_labels(cluster_order: np.ndarray) -> np.ndarray:
    """Renumber cluster labels onto ``0..n-1``, closing gaps left by empty clusters.

    No clustering method promises to fill every cluster it was asked for.
    k-maxoids on data with fewer distinct period shapes than clusters is the
    plainest case: it leaves labels nobody uses, so the assignment comes back as
    e.g. ``[0, 1, 3]`` for ``n_clusters=8``.

    Cluster ids are positions everywhere downstream — representatives live in a
    list indexed by id — so an unused label puts every cluster above it out of
    step with its representative.
    :func:`~tsam.algorithms.representations.representations` refuses such a
    label space outright, which is why densifying happens here, before it is
    called. Nothing is lost by dropping the label: a cluster with no members has
    no periods to compute a representative from, and contributes nothing to the
    reconstruction.

    Renumbering preserves order, so a label space that is already dense — every
    method's normal outcome — comes back with its labels unchanged.
    """
    # The labels actually in use, ascending and without duplicates.
    used_labels = np.unique(cluster_order)
    # A label's position in that sorted list is its new id: the lowest label
    # becomes 0, the next becomes 1, and any gap between them closes.
    new_ids = np.searchsorted(used_labels, cluster_order)
    return np.asarray(new_ids, dtype=int)


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

    Valid ``cluster_method`` values: 'averaging', 'kmeans', 'kmedoids',
    'kmaxoids', 'hierarchical', 'contiguous'.

    Returns:
        The cluster id of each period, as a dense label space: the ids run
        ``0..n-1`` with no gaps, where ``n`` is the number of clusters that
        actually received periods. That can be fewer than *n_clusters* — see
        :func:`densify_labels` for why the gaps are closed rather than kept.
    """
    raw_cluster_order = _raw_assignment(
        candidates,
        n_clusters,
        cluster_method,
        n_iter=n_iter,
    )
    # Clustering configuration might yield non-dense cluster indices (e.g., when
    # fewer clusters are needed due to duplicated periods). We reassign indices
    # to be contiguous so downstream functions have a clear, well-defined data structure.
    cluster_order = densify_labels(cluster_order=raw_cluster_order)
    return cluster_order


def _raw_assignment(
    candidates: np.ndarray,
    n_clusters: int,
    cluster_method: ClusterMethod,
    n_iter: int = 100,
) -> np.ndarray:
    """Dispatch to the clustering method, returning its labels as it produced them."""

    # Split the method argument into the name to dispatch on and the k-medoids
    # config to run with: the bare string "kmedoids" simply means KMedoids().
    kmedoids_config = (
        cluster_method if isinstance(cluster_method, KMedoids) else KMedoids()
    )
    cluster_method = method_name(cluster_method)

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

        k_means = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=n_iter, tol=1e-4)
        return np.asarray(k_means.fit_predict(candidates))

    if cluster_method == "kmedoids":
        from tsam.algorithms.k_medoids_exact import ExactKMedoids

        kmedoids = ExactKMedoids(
            n_clusters=n_clusters,
            solver=kmedoids_config.solver,
            options=kmedoids_config.options,
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
        default=DEFAULT_REPRESENTATION[method_name(cluster_method)],
        representation_method=representation_method,
        representation_dict=representation_dict,
        distribution_period_wise=distribution_period_wise,
        n_timesteps_per_period=n_timesteps_per_period,
        reference_attribute_idx=reference_attribute_idx,
    )
    return cluster_centers, cluster_center_indices, cluster_order
