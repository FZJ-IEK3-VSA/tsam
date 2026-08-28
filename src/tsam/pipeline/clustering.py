"""Config-aware clustering stage: adapt ClusterConfig onto the algorithm kernels.

Thin wrappers over :mod:`tsam.algorithms.clustering` and
:mod:`tsam.algorithms.representations`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.algorithms.clustering import assign_clusters, cluster_and_represent
from tsam.algorithms.representations import representations
from tsam.algorithms.selection import deterministic_argmin
from tsam.config import DEFAULT_REPRESENTATION

if TYPE_CHECKING:
    from tsam.config import ClusterConfig
    from tsam.pipeline.types import PredefParams

# Restarts granted to non-deterministic clustering methods (k-means). The
# duration-curve path uses fewer, since it clusters sorted profiles whose
# distances are far better behaved.
_N_ITER = 100
_N_ITER_SORTED = 30


def cluster_periods(
    candidates: np.ndarray,
    n_clusters: int,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
    representation_candidates: np.ndarray | None,
    reference_attribute_idx: int | None = None,
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

    **Clustering methods** — how candidate periods are grouped
    ([`ClusterConfig.method`][tsam.config.ClusterConfig]):

    | Method | Description |
    |---|---|
    | `"hierarchical"` | Agglomerative (Ward linkage). Default. Deterministic. |
    | `"kmeans"` | K-means. Fast but non-deterministic (set a seed externally). |
    | `"kmedoids"` | Exact k-medoids via MILP. Slow but optimal. Needs a solver. |
    | `"kmaxoids"` | K-maxoids heuristic. |
    | `"averaging"` | Simple period averaging (1 cluster = mean of all). |
    | `"contiguous"` | Adjacent periods only (preserves temporal order). |

    **Representation methods** — how each cluster's representative period is built
    ([`ClusterConfig.representation`][tsam.config.ClusterConfig]):

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

    Args:
        candidates: Candidate period matrix (possibly weighted / augmented).
        n_clusters: Number of clusters (typical periods) to form.
        cluster: Clustering configuration: ``method``, ``representation``,
            ``solver``.
        representation_dict: Per-column representation overrides (e.g. for
            ``minmax_mean``).
        n_timesteps_per_period: Timesteps per period, needed by
            distribution-style representations.
        representation_candidates: Alternative columns to compute representatives
            from (when period-sum features were appended to ``candidates`` for
            distance only).
        reference_attribute_idx: Attribute (column) index used by the
            ``"reference"`` concurrency strategy of the distribution
            representations; ignored by all other representations.

    Returns:
        ``(cluster_centers, cluster_center_indices, cluster_order)`` —
        representatives, the medoid period indices (if applicable), and the
        per-period cluster assignment.

    Note:
        The body is a call into
        [`cluster_and_represent`][tsam.algorithms.clustering.cluster_and_represent],
        which knows nothing about `ClusterConfig`. This function is where the
        configuration is turned into that kernel's arguments and where the
        pipeline's iteration budget is set, so the algorithm layer stays free of
        config types and the orchestrator stays free of kernel arguments. It is
        also the stage the pipeline documentation points at, alongside its two
        siblings.

    Note:
        Related stages: `cluster_sorted_periods` (duration-curve variant on
        sorted values), `use_predefined_assignments` (reuse stored assignments
        instead of clustering), `add_extreme_periods` (inject extreme-value
        periods after clustering), and `rescale_representatives` (correct the
        column means of the representatives).
    """
    centers, center_indices, order = cluster_and_represent(
        candidates,
        n_clusters=n_clusters,
        n_iter=_N_ITER,
        cluster_method=cluster.method,
        representation_method=cluster.get_representation(),
        representation_dict=representation_dict,
        n_timesteps_per_period=n_timesteps_per_period,
        representation_candidates=representation_candidates,
        reference_attribute_idx=reference_attribute_idx,
    )
    return centers, center_indices, order


def cluster_sorted_periods(
    candidates: np.ndarray,
    n_columns: int,
    n_timesteps_per_period: int,
    n_clusters: int,
    cluster: ClusterConfig,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Cluster periods by value distribution rather than temporal shape.

    Used when ``ClusterConfig.use_duration_curves=True``. Each period's values
    are sorted in descending order before clustering, so periods are grouped by
    their value distribution (duration curve) rather than the order in which
    values occur. Useful when intra-period ordering does not matter — e.g. for
    energy-storage optimization.

    Candidates are already weighted (if weights exist); the descending sort and
    the clustering distance therefore respect column weights, matching v3
    behaviour.

    **``ClusterConfig.representation`` does not apply on this path.** Every
    cluster is represented by a real, *unsorted* period: the member closest to
    its cluster's centroid in sorted (duration-curve) space. A synthesized
    representative — a mean, or a re-sorted distribution profile — would carry
    a temporal shape that no period in the cluster ever had, which is precisely
    what the sorting was meant to abstract away. See `cluster_periods` for the
    path where the configured representation is honoured.

    Args:
        candidates: Candidate period matrix (possibly weighted), laid out as
            ``n_columns`` contiguous blocks of equal length. It must **not**
            carry the period-sum features that `ClusterConfig.include_period_sums`
            appends: those are not timesteps, so they would make the per-column
            reshape below misread every block boundary. Period sums therefore
            do not influence duration-curve clustering.
        n_columns: Number of original columns, needed to reshape per-column
            before sorting.
        n_timesteps_per_period: Length of each column's block. Passed in rather
            than derived as ``candidates.shape[1] // n_columns`` because that
            division cannot tell a wider block from an extra block, and so
            silently accepted the augmented matrix.
        n_clusters: Number of clusters to form.
        cluster: Clustering configuration; only ``method`` and ``solver`` are
            used here, for the reason given above.

    Raises:
        ValueError: If ``candidates`` does not split evenly into ``n_columns``
            blocks of the same length.

    Returns:
        ``(cluster_centers, cluster_center_indices, cluster_order)``. The
        indices identify the original period each center was taken from, so
        centers and indices always describe the same periods.

    Note:
        `cluster_periods` performs standard clustering on the temporal profile.
    """

    # Sort each period's timesteps descending for all columns.
    # Use candidates (already weighted) so that clustering distance respects
    # column weights — matching v3 behaviour.
    n_periods, n_total = candidates.shape
    expected = n_columns * n_timesteps_per_period
    if n_total != expected:
        raise ValueError(
            f"cluster_sorted_periods expects {n_columns} blocks of "
            f"{n_timesteps_per_period} timesteps ({expected} columns), but the "
            f"candidate matrix is {n_total} wide. Any appended period-sum "
            "features must be trimmed off before sorting."
        )

    values_3d = candidates.copy().reshape(n_periods, n_columns, n_timesteps_per_period)
    sorted_values = (-np.sort(-values_3d, axis=2, kind="stable")).reshape(n_periods, -1)

    cluster_order = assign_clusters(
        sorted_values,
        n_clusters,
        cluster.method,
        n_iter=_N_ITER_SORTED,
    )

    cluster_centers = []
    cluster_center_indices = []
    for cluster_num in np.unique(cluster_order):
        indices = np.where(cluster_order == cluster_num)[0]
        if len(indices) > 1:
            current_mean = sorted_values[indices].mean(axis=0)
            closest = deterministic_argmin(
                np.square(sorted_values[indices] - current_mean).sum(axis=1)
            )
        else:
            closest = 0
        # Take the profile from the unsorted candidates so the typical period
        # keeps a realistic temporal shape.
        cluster_centers.append(candidates[indices[closest]])
        cluster_center_indices.append(int(indices[closest]))

    return cluster_centers, cluster_center_indices, cluster_order


def use_predefined_assignments(
    candidates: np.ndarray,
    predef: PredefParams,
    cluster: ClusterConfig,
    representation_dict: dict | None,
    n_timesteps_per_period: int,
    reference_attribute_idx: int | None = None,
) -> tuple[list[np.ndarray] | np.ndarray, list[int] | None, list | np.ndarray]:
    """Reuse stored cluster assignments instead of clustering from scratch.

    The transfer path, taken when ``ClusteringResult.apply()`` runs an existing
    clustering on new data. Clustering is skipped entirely: the stored
    ``cluster_order`` is reused, and representatives are either the stored
    medoid periods (if center indices were saved) or recomputed from the new
    candidates under the same assignment.

    This is the one path that does not run `assign_clusters`, so the stored
    order is never densified. A label space with a gap is caught by
    `representations` when centers are recomputed, but goes unchecked when
    center indices were saved; `_drop_empty_clusters` closes it later.

    Args:
        candidates: Candidate period matrix for the new data.
        predef: Predefined assignments (``cluster_order`` and optional center
            indices).
        cluster: Clustering configuration; supplies the representation applied
            when recomputing centers.
        representation_dict: Per-column representation overrides.
        n_timesteps_per_period: Timesteps per period.
        reference_attribute_idx: Attribute (column) index used by the
            ``"reference"`` concurrency strategy of the distribution
            representations; ignored by all other representations.

    Returns:
        ``(cluster_centers, cluster_center_indices, cluster_order)``.

    Note:
        `cluster_periods` is the from-scratch clustering this path replaces.
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
            default=DEFAULT_REPRESENTATION[cluster.method_name],
            representation_method=cluster.get_representation(),
            representation_dict=representation_dict,
            n_timesteps_per_period=n_timesteps_per_period,
            reference_attribute_idx=reference_attribute_idx,
        )
        return centers, computed_indices, predef.cluster_order
