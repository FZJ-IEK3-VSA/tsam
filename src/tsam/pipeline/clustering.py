"""Clustering wrappers around period_aggregation and representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.utils.period_aggregation import aggregate_periods
from tsam.utils.representations import representations

if TYPE_CHECKING:
    from tsam.config import ClusterConfig, Distribution, MinMaxMean
    from tsam.pipeline.types import PredefParams


def cluster_periods(
    candidates: np.ndarray,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
    representation_candidates: np.ndarray | None,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Group period profiles into clusters and pick a representative for each.

    The core stage of the pipeline. It assigns every candidate period to one of
    ``n_clusters`` groups and selects (or computes) the representative period
    for each group.

    Candidates arriving here are already weighted (if weights were given), so
    the assignment reflects weighted importance; representatives are unweighted
    later during post-processing. If *representation_candidates* is given,
    representatives are computed from those columns instead of *candidates* —
    used when period-sum features were appended for clustering distance only.

    **Clustering methods** ([`ClusterConfig.method`][tsam.config.ClusterConfig]):

    | Method | Description |
    |---|---|
    | `"hierarchical"` | Agglomerative (Ward linkage). Default. Deterministic. |
    | `"kmeans"` | K-means. Fast but non-deterministic (set a seed externally). |
    | `"kmedoids"` | Exact k-medoids via MILP. Slow but optimal. Needs a solver. |
    | `"kmaxoids"` | K-maxoids heuristic. |
    | `"averaging"` | Simple period averaging (1 cluster = mean of all). |
    | `"contiguous"` | Adjacent periods only (preserves temporal order). |

    **Representation methods**
    ([`ClusterConfig.representation`][tsam.config.ClusterConfig]) — how each
    cluster's representative period is built:

    | Representation | Description |
    |---|---|
    | `"mean"` | Arithmetic mean of cluster members. |
    | `"medoid"` | The real period closest to the cluster center. Default. |
    | `"maxoid"` | The real period farthest from the center. |
    | `"distribution"` | Duration-curve fit: sorts values to preserve the distribution. |
    | `"distribution_minmax"` | Like `"distribution"` but also preserves extreme values. |
    | `"minmax_mean"` | Separate min/max/mean per column. |
    | `Distribution(...)` | Fine-grained control over distribution representation. |
    | `MinMaxMean(...)` | Fine-grained control over which columns get min/max treatment. |

    Parameters
    ----------
    candidates
        Candidate period matrix (possibly weighted / augmented).
    n_clusters
        Number of clusters (typical periods) to form.
    cluster
        Clustering configuration: ``method``, ``representation``, ``solver``.
    representation_dict
        Per-column representation overrides (e.g. for ``minmax_mean``).
    n_timesteps_per_period
        Timesteps per period, needed by distribution-style representations.
    representation_candidates
        Alternative columns to compute representatives from (when period-sum
        features were appended to ``candidates`` for distance only).

    Returns
    -------
    tuple[list[np.ndarray], list[int] | None, np.ndarray]
        ``(cluster_centers, cluster_center_indices, cluster_order)`` —
        representatives, the medoid period indices (if applicable), and the
        per-period cluster assignment.

    See Also
    --------
    cluster_sorted_periods : Duration-curve variant clustering on sorted values.
    use_predefined_assignments : Reuse stored assignments instead of clustering.
    add_extreme_periods : Inject extreme-value periods after clustering.
    rescale_representatives : Correct column means of the representatives.
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
        representation_candidates=representation_candidates,
    )
    return centers, center_indices, order


def cluster_sorted_periods(
    candidates: np.ndarray,
    n_columns: int,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Cluster periods by value distribution rather than temporal shape.

    Used when ``ClusterConfig.use_duration_curves=True``. Each period's values
    are sorted in descending order before clustering, so periods are grouped by
    their value distribution (duration curve) rather than the order in which
    values occur. Useful when intra-period ordering does not matter — e.g. for
    energy-storage optimization.

    Candidates are already weighted (if weights exist); the descending sort and
    the clustering distance therefore respect column weights, matching v3
    behaviour. The medoid representative is picked from the **unsorted**
    (weighted) candidates so the typical period keeps a realistic temporal
    shape.

    See `cluster_periods` for the available clustering methods and
    representations.

    Parameters
    ----------
    candidates
        Candidate period matrix (possibly weighted).
    n_columns
        Number of original columns, needed to reshape per-column before sorting.
    n_clusters
        Number of clusters to form.
    cluster
        Clustering configuration.
    representation_dict
        Per-column representation overrides.
    n_timesteps_per_period
        Timesteps per period.

    Returns
    -------
    tuple[list[np.ndarray], list[int] | None, np.ndarray]
        ``(cluster_centers, cluster_center_indices, cluster_order)``.

    See Also
    --------
    cluster_periods : Standard clustering on the temporal profile.
    """

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
    """Reuse stored cluster assignments instead of clustering from scratch.

    The transfer path, taken when ``ClusteringResult.apply()`` runs an existing
    clustering on new data. Clustering is skipped entirely: the stored
    ``cluster_order`` is reused, and representatives are either the stored
    medoid periods (if center indices were saved) or recomputed from the new
    candidates under the same assignment.

    Parameters
    ----------
    candidates
        Candidate period matrix for the new data.
    predef
        Predefined assignments (``cluster_order`` and optional center indices).
    representation_method
        Representation to apply when recomputing centers.
    representation_dict
        Per-column representation overrides.
    n_timesteps_per_period
        Timesteps per period.

    Returns
    -------
    tuple
        ``(cluster_centers, cluster_center_indices, cluster_order)``.

    See Also
    --------
    cluster_periods : The from-scratch clustering this path replaces.
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
