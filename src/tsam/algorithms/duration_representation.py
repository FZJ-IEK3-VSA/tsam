"""Orders a set of representation values to fit several candidate value sets."""

from __future__ import annotations

import warnings

import numpy as np

from tsam.algorithms.concurrency import compute_ordering
from tsam.commons import bounded_water_fill
from tsam.options import options

_MINMAX_INFEASIBLE_MSG = (
    "The cluster is too small to preserve the sum of the duration curve and "
    "additionally the min and max values of the original cluster members. The "
    "min max values of the cluster are not preserved. This does not necessarily "
    "mean that the min and max values of the original time series are not "
    "preserved."
)


def duration_representation(
    candidates: np.ndarray,
    cluster_order: np.ndarray,
    distribution_period_wise: bool,
    n_timesteps_per_period: int,
    represent_min_max: bool = False,
    concurrency_method: str | None = None,
    reference_attribute_idx: int | None = None,
) -> list[np.ndarray]:
    """Represent each cluster by a duration curve fitted per attribute.

    Orders the values of a cluster group so that, for every attribute, the
    number of time steps at each value level best fits the cluster members
    (the duration curve / distribution). Either the distribution of each
    cluster is preserved (``distribution_period_wise=True``) or the
    distribution of the whole time series is preserved.

    Args:
        candidates: Candidate matrix, one row per candidate period.
        cluster_order: Cluster label per candidate (index → group).
        distribution_period_wise: Preserve each cluster's distribution when
            True, otherwise preserve the global distribution only.
        n_timesteps_per_period: Time steps per period.
        represent_min_max: Whether to preserve each cluster attribute's
            minimum and maximum.
        concurrency_method: Strategy used to lay each attribute's duration-curve
            values onto the synthetic time axis, one of
            :data:`~tsam.algorithms.concurrency.CONCURRENCY_METHODS`. All
            strategies preserve every attribute's marginal distribution and
            differ only in how cross-attribute concurrency is preserved. Only
            supported with ``distribution_period_wise=True``. ``None`` uses
            ``"independent"``.
        reference_attribute_idx: Attribute index required by the ``"reference"``
            concurrency strategy; ignored otherwise.

    Returns:
        The cluster centers as rows of time steps × attributes.
    """

    num_attributes = int(candidates.shape[1] / n_timesteps_per_period)

    # There are two options for the duration representation. Either, the distribution of each cluster is preserved
    # (period_wise = True) or the distribution of the total time series is preserved only. In the latter case, the
    # inner-cluster variance is smaller and the variance across the typical periods' mean values is higher
    if distribution_period_wise:
        n_attrs = num_attributes

        # Reshape to 3D: (periods, attributes, timesteps)
        candidates_3d = candidates.reshape(-1, n_attrs, n_timesteps_per_period)

        cluster_centers = []
        for cluster_num in np.unique(cluster_order):
            indice = np.where(cluster_order == cluster_num)[0]
            n_cands = len(indice)
            if n_cands == 0:
                continue

            # (n_cands, n_attrs, timesteps) -> (n_attrs, n_cands, timesteps)
            cluster_data = candidates_3d[indice].transpose(1, 0, 2)

            # Sort all values per attribute, then reshape to duration curve
            flat = cluster_data.reshape(n_attrs, -1)
            flat.sort(axis=1, kind="stable")
            repr_values = flat.reshape(n_attrs, n_timesteps_per_period, n_cands).mean(
                axis=2
            )

            if represent_min_max:
                repr_values = _pin_min_max_preserve_sum(
                    repr_values,
                    flat[:, 0],
                    flat[:, -1],
                    options.minmax_max_passes,
                    options.minmax_tolerance,
                )

            means = np.round(cluster_data.mean(axis=1), 10)
            order = compute_ordering(
                cluster_data,
                means,
                repr_values,
                concurrency_method,
                reference_attribute_idx,
            )
            rows = np.arange(n_attrs)[:, None]
            final_repr = np.empty_like(repr_values)
            final_repr[rows, order] = repr_values

            cluster_centers.append(final_repr.ravel())

        return cluster_centers

    else:
        if concurrency_method is not None and concurrency_method != "independent":
            raise ValueError(
                "A concurrency-preserving duration representation "
                f"(concurrency_method={concurrency_method!r}) is only supported "
                "with distribution_period_wise=True (scope='cluster')."
            )
        candidates_3d = candidates.reshape(-1, num_attributes, n_timesteps_per_period)
        unique_clusters = np.unique(cluster_order)
        n_clusters = len(unique_clusters)
        member_masks = [cluster_order == c for c in unique_clusters]
        # each cluster's size, once per time step within the period
        weights = np.repeat(
            [int(mask.sum()) for mask in member_masks], n_timesteps_per_period
        )

        centers = np.empty((n_clusters, num_attributes, n_timesteps_per_period))
        for a in range(num_attributes):
            attr_values = candidates_3d[:, a, :]
            # per-cluster centroid value of every time step, flattened row-major
            cluster_means = np.concatenate(
                [np.round(attr_values[mask].mean(axis=0), 10) for mask in member_masks]
            )
            # sort all (cluster, time step) slots by centroid value
            order = np.argsort(cluster_means, kind="stable")
            sorted_weights = weights[order]
            # sort all values of the original time series
            sorted_attr = np.sort(attr_values.ravel(), kind="stable")
            # take mean of sections of the original duration curve according to
            # the cluster and its weight the respective section is assigned to
            representation_values = []
            counter = 0
            for j in sorted_weights:
                representation_values.append(sorted_attr[counter : counter + j].mean())
                counter += j
            # respect max and min of the attributes
            if represent_min_max:
                representation_values = _represent_min_max(
                    representation_values,
                    sorted_attr,
                    sorted_weights,
                    options.minmax_max_passes,
                    options.minmax_tolerance,
                )

            # arrange the representation values back into (cluster, time step) slots
            slots = np.empty(n_clusters * n_timesteps_per_period)
            slots[order] = representation_values
            centers[:, a, :] = slots.reshape(n_clusters, n_timesteps_per_period)

        return list(centers.reshape(n_clusters, -1))


def _pin_min_max_preserve_sum(
    repr_values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_passes: int,
    rel_tolerance: float,
) -> np.ndarray:
    """Pin each attribute's duration-curve endpoints to the cluster min/max.

    The plain duration curve already has the integral-preserving property that
    its per-attribute sum equals the average member sum. Naively overwriting the
    endpoints with the cluster min/max shifts that sum (and the integral, unless
    a later rescaling happens to repair it). Instead, the delta introduced at the
    endpoints is water-filled back across the interior points (clipped to the
    ``[min, max]`` envelope), so the sum is preserved and the result stays inside
    the envelope.

    With fewer than three points there is no interior mass to absorb the shift,
    so the sum cannot be preserved while pinning both endpoints. This is the case
    for the segment representation (``T == 1``): a single value cannot carry both
    the min and the max, and forcing it to the max inflates the integral. There
    the integral takes priority and the duration-curve values are returned
    unchanged.

    Args:
        repr_values: Ascending duration curves, shape (n_attrs, n_points).
        lower: Per-attribute cluster minimum, shape (n_attrs,).
        upper: Per-attribute cluster maximum, shape (n_attrs,).
        max_passes: Max water-fill passes.
        rel_tolerance: Relative tolerance for sum preservation.

    Returns:
        The duration curves with endpoints pinned and the integral preserved.
    """
    out = np.array(repr_values, dtype=float)
    _, n_points = out.shape
    if n_points < 3:
        return out

    infeasible = 0
    for a in range(out.shape[0]):
        lo, hi = float(lower[a]), float(upper[a])
        target_mid = float(out[a].sum()) - lo - hi
        out[a, 0] = lo
        out[a, -1] = hi
        mid, feasible, _ = bounded_water_fill(
            out[a, 1:-1],
            np.ones(n_points - 2),
            lo,
            hi,
            target_mid,
            rel_tolerance=rel_tolerance,
            max_passes=max_passes,
        )
        if not feasible:
            infeasible += 1
        out[a, 1:-1] = mid
    if infeasible:
        warnings.warn(
            "Could not preserve both the cluster min/max and the integral for "
            f"{infeasible} attribute(s) (the cluster is too small or its "
            "envelope too narrow). The integral of the affected attributes may "
            "deviate slightly."
        )
    return out


def _represent_min_max(
    representation_values: list[float],
    sorted_attr: np.ndarray,
    sorted_weights: np.ndarray,
    max_passes: int,
    rel_tolerance: float,
) -> list[float]:
    """Preserve an attribute's min and max in the duration-curve representation.

    Pins the first and last duration-curve values to the original attribute
    minimum and maximum, then redistributes the resulting delta across the
    interior points with a bounded water-fill so the sum (integral) is preserved
    and every value stays inside the ``[min, max]`` envelope.

    Args:
        representation_values: Duration-curve representation values, ascending.
        sorted_attr: The sorted original attribute values.
        sorted_weights: Per-section occurrence counts of the original series
            (the section weights).
        max_passes: Max water-fill passes.
        rel_tolerance: Relative tolerance for sum preservation.

    Returns:
        The duration-curve values with the min/max pinned and the sum preserved
        where feasible.
    """
    if np.any(np.array(representation_values) < 0):
        raise ValueError("Negative values in the duration curve representation")

    delta_max = sorted_attr.max() - representation_values[-1]
    appearance_max = sorted_weights[-1]
    delta_min = sorted_attr.min() - representation_values[0]
    appearance_min = sorted_weights[0]

    if delta_min == 0 and delta_max == 0:
        return representation_values

    delta_sum = delta_max * appearance_max + delta_min * appearance_min

    mid_weights = np.asarray(sorted_weights[1:-1], dtype=float)
    mid_orig = np.asarray(representation_values[1:-1], dtype=float)
    weighted_mid_sum = float(np.sum(mid_weights * mid_orig))
    correction_factor = -delta_sum / weighted_mid_sum if weighted_mid_sum != 0 else 0.0

    if correction_factor < -1 or correction_factor > 1:
        warnings.warn(_MINMAX_INFEASIBLE_MSG)
        return representation_values

    # Multiplicative warm start: preserves the weighted sum exactly and keeps
    # zero-valued segments at zero; matches legacy behaviour when no bound is hit.
    corrected = mid_orig * (1 + correction_factor)

    target_weighted_sum = weighted_mid_sum - delta_sum
    corrected, feasible, _ = bounded_water_fill(
        corrected,
        mid_weights,
        float(sorted_attr.min()),
        float(sorted_attr.max()),
        target_weighted_sum,
        rel_tolerance=rel_tolerance,
        max_passes=max_passes,
    )
    if not feasible:
        warnings.warn(_MINMAX_INFEASIBLE_MSG)

    representation_values[1:-1] = corrected

    representation_values[-1] += delta_max
    representation_values[0] += delta_min

    return representation_values
