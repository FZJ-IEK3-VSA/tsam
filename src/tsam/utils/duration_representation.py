"""Orders a set of representation values to fit several candidate value sets"""

import warnings

import numpy as np
import pandas as pd


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
        # Vectorized implementation using numpy 3D arrays instead of pandas MultiIndex
        n_periods = candidates.shape[0]
        n_attrs = num_attributes

        # Reshape to 3D: (periods, attributes, timesteps)
        candidates_3d = candidates.reshape(n_periods, n_attrs, n_timesteps_per_period)

        cluster_centers = []
        for cluster_num in np.unique(cluster_order):
            indice = np.where(cluster_order == cluster_num)[0]
            n_cands = len(indice)

            # Skip empty clusters
            if n_cands == 0:
                continue

            # Get all candidates for this cluster: (n_cands, n_attrs, timesteps)
            cluster_data = candidates_3d[indice]

            # Process all attributes at once using vectorized operations
            # Reshape to (n_attrs, n_cands * timesteps) for sorting
            flat_per_attr = cluster_data.transpose(1, 0, 2).reshape(n_attrs, -1)

            # Sort each attribute's values (stable sort for deterministic tie-breaking)
            sorted_flat = np.sort(flat_per_attr, axis=1, kind="stable")

            # Reshape and mean: (n_attrs, timesteps, n_cands) -> mean -> (n_attrs, timesteps)
            sorted_reshaped = sorted_flat.reshape(
                n_attrs, n_timesteps_per_period, n_cands
            )
            repr_values = sorted_reshaped.mean(axis=2)

            # Respect max and min of the attributes
            if represent_min_max:
                repr_values[:, 0] = sorted_flat[:, 0]
                repr_values[:, -1] = sorted_flat[:, -1]

            # Get mean profile order for each attribute (stable sort for deterministic tie-breaking)
            mean_profiles = cluster_data.mean(axis=0)  # (n_attrs, timesteps)
            orders = np.argsort(
                mean_profiles, axis=1, kind="stable"
            )  # (n_attrs, timesteps)

            # Reorder repr_values according to orders
            final_repr = np.empty_like(repr_values)
            for a in range(n_attrs):
                final_repr[a, orders[a]] = repr_values[a]

            # Flatten to (n_attrs * timesteps,)
            cluster_centers.append(final_repr.flatten())

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
                mean_vals.append(candidate_values.mean())
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
            means_and_weights_sorted = means_and_weights.sort_values(0)
            # save order of the sorted centroid values across all clusters
            order = means_and_weights_sorted.index
            # sort all values of the original time series
            sorted_attr = (
                candidates_df.loc[:, a]
                .stack(
                    future_stack=True,
                )
                .sort_values()
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
        # and derive how much the other values have to be changed to preserve
        # the mean of the duration curve
        correction_factor = (
            -delta_sum
            / (
                means_and_weights_sorted[1].iloc[1:-1] * representation_values[1:-1]
            ).sum()
        )

        if correction_factor < -1 or correction_factor > 1:
            warnings.warn(
                "The cluster is too small to preserve the sum of the duration curve and additionally the min and max values of the original cluster members. The min max values of the cluster are not preserved. This does not necessarily mean that the min and max values of the original time series are not preserved."
            )
            return representation_values

        # correct the values of the duration curve such
        # that the mean of the duration curve is preserved
        # since the min and max values are changed
        representation_values[1:-1] = np.multiply(
            representation_values[1:-1], (1 + correction_factor)
        )

    # change the values of the duration curve such that the min and max
    # values are preserved
    representation_values[-1] += delta_max
    representation_values[0] += delta_min

    return representation_values
