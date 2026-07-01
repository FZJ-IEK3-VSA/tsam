"""Orders a set of representation values to fit several candidate value sets"""

import warnings

import numpy as np
import pandas as pd

from tsam.commons import bounded_water_fill
from tsam.options import options


def duration_representation(
    candidates,
    cluster_order,
    distribution_period_wise,
    n_timesteps_per_period,
    represent_min_max=False,
):
    """
    Represents the candidates of a given cluster group (cluster_order)
    such that for every attribute the number of time steps is best fit.

    :param candidates: Dissimilarity matrix where each row represents a candidate
    :type candidates: np.ndarray

    :param cluster_order: Integer array where the index refers to the candidate and the Integer entry to the group
    :type cluster_order: np.array

    :param represent_min_max: If in every cluster the minimum and the maximum of the attribute should be represented
    :type represent_min_max: bool
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
                    repr_values, flat[:, 0], flat[:, -1]
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

    else:
        cluster_centers_list = []
        for a in candidates_df.columns.levels[0]:
            mean_vals = []
            cluster_lengths = []
            for cluster_num in np.unique(cluster_order):
                indice = np.where(cluster_order == cluster_num)
                n_candidates = len(indice[0])
                # get all the values of a certain attribute and cluster
                candidate_values = candidates_df.loc[indice[0], a]
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
            means_and_weights_sorted = means_and_weights.sort_values(0, kind="stable")
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
                    keep_sum=True,
                )

            # transform all representation values to a data frame and arrange it
            # according to the order of the sorted
            # centroid values
            representation_values = pd.DataFrame(np.array(representation_values))
            representation_values.index = order
            representation_values.sort_index(inplace=True)
            # append all cluster values attribute-wise to a list
            cluster_centers_list.append(representation_values.unstack())
        # rearrange so that rows are the cluster centers and columns are time steps x attributes
        cluster_centers = np.array(pd.concat(cluster_centers_list, axis=1))

    return cluster_centers


def _pin_min_max_preserve_sum(repr_values, lower, upper):
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

    :param repr_values: Ascending duration curves, shape (n_attrs, n_points)
    :type repr_values: np.ndarray
    :param lower: Per-attribute cluster minimum, shape (n_attrs,)
    :type lower: np.ndarray
    :param upper: Per-attribute cluster maximum, shape (n_attrs,)
    :type upper: np.ndarray
    """
    out = repr_values.astype(float).copy()
    _, n_points = out.shape
    if n_points < 3:
        return out

    for a in range(out.shape[0]):
        lo, hi = float(lower[a]), float(upper[a])
        # Sum to preserve across the interior once the endpoints are pinned.
        target_mid = float(out[a].sum()) - lo - hi
        out[a, 0] = lo
        out[a, -1] = hi
        tol = max(abs(target_mid), 1.0) * options.minmax_tolerance
        mid, feasible, _ = bounded_water_fill(
            out[a, 1:-1],
            np.ones(n_points - 2),
            lo,
            hi,
            target_mid,
            tolerance=tol,
            max_passes=options.minmax_max_passes,
        )
        if not feasible:
            warnings.warn(
                "Could not preserve both the cluster min/max and the integral "
                "for an attribute (the cluster is too small or its envelope too "
                "narrow). The integral of the affected attribute may deviate "
                "slightly."
            )
        out[a, 1:-1] = mid
    return out


def _represent_min_max(
    representation_values, sorted_attr, means_and_weights_sorted, keep_sum=True
):
    """
    Represents the the min and max values of the original time series in the
    duration curve representation such that the min and max values of the
    original time series are preserved.

    :param representation_values: The duration curve representation values
    :type representation_values: np.array

    :param sorted_attr: The sorted original time series
    :type sorted_attr: np.array

    :param means_and_weights_sorted: The number of occureance of
     the original time series.
    :type means_and_weights_sorted: pd.DataFrame

    :param keep_sum: If the sum of the duration curve should be preserved
    :type keep_sum: bool
    """

    if np.any(np.array(representation_values) < 0):
        raise ValueError("Negative values in the duration curve representation")

    # first retrieve the change of the values to the min and max values
    # of the original time series and their duration in the original
    # time series
    delta_max = sorted_attr.max() - representation_values[-1]
    appearance_max = means_and_weights_sorted[1].iloc[-1]
    delta_min = sorted_attr.min() - representation_values[0]
    appearance_min = means_and_weights_sorted[1].iloc[0]

    if delta_min == 0 and delta_max == 0:
        return representation_values

    if keep_sum:
        # now anticipate the shift of the sum of the time series
        # due to the change of the min and max values
        # of the duration curve
        delta_sum = delta_max * appearance_max + delta_min * appearance_min

        mid_weights = np.asarray(
            means_and_weights_sorted[1].iloc[1:-1].values, dtype=float
        )
        mid_orig = np.asarray(representation_values[1:-1], dtype=float)
        weighted_mid_sum = float(np.sum(mid_weights * mid_orig))
        # and derive how much the other values have to be changed to preserve
        # the mean of the duration curve
        correction_factor = (
            -delta_sum / weighted_mid_sum if weighted_mid_sum != 0 else 0.0
        )

        if correction_factor < -1 or correction_factor > 1:
            warnings.warn(
                "The cluster is too small to preserve the sum of the duration curve and additionally the min and max values of the original cluster members. The min max values of the cluster are not preserved. This does not necessarily mean that the min and max values of the original time series are not preserved."
            )
            return representation_values

        # Initial multiplicative correction: preserves the weighted sum exactly
        # and keeps zero-valued segments at zero (matching the distribution
        # shape of the cluster). When no bound is violated, no further work is
        # needed and this matches the legacy behaviour.
        corrected = mid_orig * (1 + correction_factor)

        # Clip to the cluster envelope and water-fill the mass that got clipped
        # back into segments that still have room, so the result is
        # envelope-safe while preserving the sum up to feasibility.
        target_weighted_sum = weighted_mid_sum - delta_sum
        tol = max(abs(target_weighted_sum), 1.0) * options.minmax_tolerance
        corrected, feasible, _ = bounded_water_fill(
            corrected,
            mid_weights,
            float(sorted_attr.min()),
            float(sorted_attr.max()),
            target_weighted_sum,
            tolerance=tol,
            max_passes=options.minmax_max_passes,
        )
        if not feasible:
            warnings.warn(
                "The cluster is too small to preserve the sum of the duration curve and additionally the min and max values of the original cluster members. The min max values of the cluster are not preserved. This does not necessarily mean that the min and max values of the original time series are not preserved."
            )

        representation_values[1:-1] = corrected

    # change the values of the duration curve such that the min and max
    # values are preserved
    representation_values[-1] += delta_max
    representation_values[0] += delta_min

    return representation_values
