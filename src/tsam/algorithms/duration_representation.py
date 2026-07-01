"""Orders a set of representation values to fit several candidate value sets"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

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

    Returns:
        The cluster centers as rows of time steps × attributes.
    """

    # make pd.DataFrame each row represents a candidate, and the columns are defined by two levels: the attributes and
    # the time steps inside the candidates.
    column_tuples = []
    num_attributes = int(candidates.shape[1] / n_timesteps_per_period)
    for i in range(num_attributes):
        for j in range(n_timesteps_per_period):
            column_tuples.append((i, j))
    candidates_df = pd.DataFrame(
        candidates, columns=pd.MultiIndex.from_tuples(column_tuples)
    )

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

            # Reorder each attribute's repr_values by its mean profile.
            # Round means before argsort to ensure identical tie-breaking
            # across platforms and numpy versions.
            means = np.round(cluster_data.mean(axis=1), 10)
            order = means.argsort(axis=1, kind="stable")
            rows = np.arange(n_attrs)[:, None]
            final_repr = np.empty_like(repr_values)
            final_repr[rows, order] = repr_values

            cluster_centers.append(final_repr.ravel())

        return cluster_centers

    else:
        cluster_centers_list = []
        for a in candidates_df.columns.get_level_values(0).unique():
            mean_vals = []
            cluster_lengths = []
            for cluster_num in np.unique(cluster_order):
                positions = np.where(cluster_order == cluster_num)
                n_candidates = len(positions[0])
                # get all the values of a certain attribute and cluster
                candidate_values = candidates_df.loc[positions[0], a]
                # calculate centroid of each cluster and append to list
                mean_vals.append(np.round(candidate_values.mean(), 10))
                # make a list of weights of each cluster for each time step within the period
                cluster_lengths.append(np.repeat(n_candidates, n_timesteps_per_period))
            # concat centroid values and cluster weights for all clusters
            means_and_weights = pd.concat(
                [
                    pd.DataFrame(np.array(mean_vals)).stack(
                        future_stack=True,
                    ),
                    pd.DataFrame(np.array(cluster_lengths)).stack(
                        future_stack=True,
                    ),
                ],
                axis=1,
            )
            # sort all values of all clusters according to the centroid values
            # (column label 0 is an int; pandas-stubs only type str labels)
            means_and_weights_sorted = means_and_weights.sort_values(
                0,  # type: ignore[call-overload]
                kind="stable",
            )
            # save order of the sorted centroid values across all clusters
            order = means_and_weights_sorted.index
            # sort all values of the original time series
            sorted_attr = (
                candidates_df.loc[:, a]
                .stack(
                    future_stack=True,
                )
                .sort_values(kind="stable")
                .values
            )
            # take mean of sections of the original duration curve according to the cluster and its weight the
            # respective section is assigned to
            representation_values = []
            counter = 0
            for i, j in enumerate(means_and_weights_sorted[1]):
                representation_values.append(sorted_attr[counter : counter + j].mean())
                counter += j
            # respect max and min of the attributes
            if represent_min_max:
                representation_values = _represent_min_max(
                    representation_values,
                    sorted_attr,
                    means_and_weights_sorted,
                    options.minmax_max_passes,
                    options.minmax_tolerance,
                )

            # transform all representation values to a data frame and arrange it
            # according to the order of the sorted centroid values
            representation_df = pd.DataFrame(np.array(representation_values))
            representation_df.index = order
            representation_df.sort_index(inplace=True)
            # append all cluster values attribute-wise to a list
            cluster_centers_list.append(representation_df.unstack())
        # rearrange so that rows are the cluster centers and columns are time steps x attributes
        stacked = np.array(pd.concat(cluster_centers_list, axis=1))
        return list(stacked)


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
        # Sum to preserve across the interior once the endpoints are pinned.
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
    means_and_weights_sorted: pd.DataFrame,
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
        means_and_weights_sorted: Per-section occurrence counts of the original
            series (the section weights).
        max_passes: Max water-fill passes.
        rel_tolerance: Relative tolerance for sum preservation.

    Returns:
        The duration-curve values with the min/max pinned and the sum preserved
        where feasible.
    """
    if np.any(np.array(representation_values) < 0):
        raise ValueError("Negative values in the duration curve representation")

    # Change of the endpoints toward the original min/max, and how often each
    # endpoint occurs in the original series.
    delta_max = sorted_attr.max() - representation_values[-1]
    appearance_max = means_and_weights_sorted[1].iloc[-1]
    delta_min = sorted_attr.min() - representation_values[0]
    appearance_min = means_and_weights_sorted[1].iloc[0]

    if delta_min == 0 and delta_max == 0:
        return representation_values

    # Sum shift introduced by moving the endpoints to the original min/max.
    delta_sum = delta_max * appearance_max + delta_min * appearance_min

    mid_weights = np.asarray(means_and_weights_sorted[1].iloc[1:-1].values, dtype=float)
    mid_orig = np.asarray(representation_values[1:-1], dtype=float)
    weighted_mid_sum = float(np.sum(mid_weights * mid_orig))
    # How much the interior values must change to preserve the mean.
    correction_factor = -delta_sum / weighted_mid_sum if weighted_mid_sum != 0 else 0.0

    if correction_factor < -1 or correction_factor > 1:
        warnings.warn(_MINMAX_INFEASIBLE_MSG)
        return representation_values

    # Initial multiplicative correction: preserves the weighted sum exactly and
    # keeps zero-valued segments at zero (matching the cluster distribution
    # shape). When no bound is violated this matches the legacy behaviour.
    corrected = mid_orig * (1 + correction_factor)

    # Clip to the cluster envelope and water-fill the clipped mass back into
    # segments with room, keeping the result envelope-safe while preserving the
    # sum up to feasibility.
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

    # Finally pin the endpoints to the original min/max.
    representation_values[-1] += delta_max
    representation_values[0] += delta_min

    return representation_values
