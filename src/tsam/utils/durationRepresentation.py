"""Orders a set of representation values to fit several candidate value sets"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Ordering strategies that derive the synthetic time axis of the duration
# representation. They differ only in how the per-attribute ordering of the
# duration-curve values is computed; the duration-curve values themselves
# (and therefore every attribute's marginal distribution) are identical.
CONCURRENCY_METHODS = (
    "independent",
    "reference",
    "medoid",
    "consensus",
    "assignment",
)


def _orderingForCluster(
    cluster_data, means, repr_values, concurrencyMethod, referenceAttributeIdx
):
    """
    Derive the ``order`` array (shape ``n_attrs x timeStepsPerPeriod``) that
    places each attribute's sorted duration-curve values onto the synthetic
    time axis of a single cluster.

    The duration-curve values (``repr_values``) and hence each attribute's
    marginal distribution are independent of this choice; the strategies only
    differ in how strongly the attributes' co-incidence (concurrency) across a
    common time axis is preserved.

    :param cluster_data: Cluster members reshaped to (n_attrs, n_cands, timesteps)
    :type cluster_data: np.ndarray

    :param means: Per-attribute mean profile over the cluster members,
        shape (n_attrs, timesteps), already rounded for stable tie-breaking
    :type means: np.ndarray

    :param repr_values: Per-attribute duration curve (sorted, ascending),
        shape (n_attrs, timesteps)
    :type repr_values: np.ndarray

    :param concurrencyMethod: One of :data:`CONCURRENCY_METHODS`
    :type concurrencyMethod: str

    :param referenceAttributeIdx: Reference attribute index, only used by the
        ``"reference"`` method
    :type referenceAttributeIdx: int or None
    """
    n_attrs, timeStepsPerPeriod = means.shape

    if concurrencyMethod == "independent":
        # Each attribute is reordered by its OWN mean profile. Fits each
        # attribute's distribution best, but the resulting time axes differ
        # between attributes, so concurrency is lost.
        return means.argsort(axis=1, kind="stable")

    if concurrencyMethod == "reference":
        # Derive a single temporal ordering from the reference attribute's mean
        # profile and apply it to every attribute. All attributes share one time
        # axis, so their co-incidence with the reference attribute is preserved.
        ref_order = means[referenceAttributeIdx].argsort(kind="stable")
        return np.broadcast_to(ref_order, (n_attrs, timeStepsPerPeriod))

    if concurrencyMethod == "medoid":
        # Order each attribute by the cluster medoid's own per-attribute ranks.
        # The medoid is a real, physically consistent multivariate period, so
        # reproducing its per-attribute rank pattern preserves its joint
        # co-occurrence (empirical copula) across all attribute pairs, while
        # each attribute still takes the values of its own duration curve.
        # cluster_data: (n_attrs, n_cands, timesteps) -> flat members (n_cands, n_attrs*timesteps)
        flat_members = cluster_data.transpose(1, 0, 2).reshape(
            cluster_data.shape[1], -1
        )
        distMatrix = np.linalg.norm(
            flat_members[:, None, :] - flat_members[None, :, :], axis=2
        )
        medoidIdx = np.argmin(distMatrix.sum(axis=0))
        medoidProfile = cluster_data[:, medoidIdx, :]  # (n_attrs, timesteps)
        return np.round(medoidProfile, 10).argsort(axis=1, kind="stable")

    if concurrencyMethod == "consensus":
        # Derive one shared time axis from a consensus of all attributes' mean
        # profiles via the first principal component, then broadcast it to every
        # attribute. Balances concurrency across all attributes instead of
        # privileging one, without needing a manual reference choice.
        std = means.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        z = (means - means.mean(axis=1, keepdims=True)) / std
        # First principal component of the (timesteps x n_attrs) profile matrix.
        _, _, vt = np.linalg.svd(z.T, full_matrices=False)
        scores = z.T @ vt[0]
        shared_order = np.round(scores, 10).argsort(kind="stable")
        return np.broadcast_to(shared_order, (n_attrs, timeStepsPerPeriod))

    if concurrencyMethod == "assignment":
        # Optimal single shared ordering: assign duration-curve ranks to
        # timesteps to minimise the total squared deviation from the cluster's
        # mean profile across all attributes simultaneously.
        # cost[r, t] = sum_attr (repr_values[attr, r] - means[attr, t])^2
        cost = ((repr_values[:, :, None] - means[:, None, :]) ** 2).sum(
            axis=0
        )  # (timesteps_rank, timesteps_time)
        rank_idx, time_idx = linear_sum_assignment(cost)
        shared_order = np.empty(timeStepsPerPeriod, dtype=int)
        shared_order[rank_idx] = time_idx
        return np.broadcast_to(shared_order, (n_attrs, timeStepsPerPeriod))

    raise ValueError(
        f"Unknown concurrencyMethod '{concurrencyMethod}'. "
        f"Expected one of {CONCURRENCY_METHODS}."
    )


def _pinMinMaxPreserveSum(repr_values, lo, hi):
    """
    Pin the first/last value of every attribute's duration curve to the cluster
    minimum/maximum while keeping each attribute's total (the integral) unchanged.

    The plain duration curve (``repr_values``) already has the property that its
    per-attribute sum equals the average member sum, which is what makes the
    aggregation integral-preserving. Naively overwriting the endpoints with the
    cluster min/max shifts that sum (and hence the integral, unless a later
    rescaling step happens to repair it). Instead, the delta introduced at the
    endpoints is redistributed across the interior points with a bounded
    water-fill (clipped to ``[lo, hi]``), so the sum is preserved and the result
    stays inside the cluster envelope.

    :param repr_values: Duration curves, shape (n_attrs, timeStepsPerPeriod),
        ascending per attribute
    :type repr_values: np.ndarray

    :param lo: Per-attribute cluster minimum, shape (n_attrs,)
    :type lo: np.ndarray

    :param hi: Per-attribute cluster maximum, shape (n_attrs,)
    :type hi: np.ndarray
    """
    out = repr_values.astype(float).copy()
    n_attrs, T = out.shape
    if T < 3:
        # With fewer than three points there is no interior mass to absorb the
        # shift introduced by pinning the endpoints, so min/max pinning cannot
        # preserve the per-attribute sum (the integral). This is in particular
        # the case for the segment representation, which represents each segment
        # by a single value (``T == 1``): a single value cannot carry both the
        # min and the max, and forcing it to ``hi`` inflates the
        # duration-weighted integral (a ~1-2% deviation in the reconstructed
        # time series). Preserving the integral takes priority over pinning the
        # envelope here, so the sum-preserving duration-curve values are
        # returned unchanged.
        return out

    target = out.sum(axis=1)  # the sum we must preserve, per attribute
    out[:, 0] = lo
    out[:, -1] = hi
    target_mid = target - lo - hi
    mid = out[:, 1:-1].copy()
    lo_c = lo[:, None]
    hi_c = hi[:, None]
    tol = np.maximum(np.abs(target_mid), 1.0) * 1e-13

    for _ in range(50):
        resid = target_mid - mid.sum(axis=1)
        if np.all(np.abs(resid) <= tol):
            break
        room = np.where(resid[:, None] > 0, hi_c - mid, mid - lo_c)
        total = room.sum(axis=1)
        active = total > 1e-15
        if not active.any():
            break
        step = np.minimum(np.abs(resid), total)
        scale = np.where(
            active, np.sign(resid) * step / np.where(active, total, 1.0), 0.0
        )
        mid = mid + scale[:, None] * room
        np.clip(mid, lo_c, hi_c, out=mid)

    if np.any(
        np.abs(target_mid - mid.sum(axis=1))
        > np.maximum(np.abs(target_mid), 1.0) * 1e-6
    ):
        warnings.warn(
            "Could not preserve both the cluster min/max and the integral for "
            "every attribute (the cluster is too small or its envelope too "
            "narrow). The integral of the affected attributes may deviate "
            "slightly."
        )
    out[:, 1:-1] = mid
    return out


def durationRepresentation(
    candidates,
    clusterOrder,
    distributionPeriodWise,
    timeStepsPerPeriod,
    representMinMax=False,
    referenceAttributeIdx=None,
    concurrencyMethod=None,
):
    """
    Represents the candidates of a given cluster group (clusterOrder)
    such that for every attribute the number of time steps is best fit.

    :param candidates: Dissimilarity matrix where each row represents a candidate
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the Integer entry to the group
    :type clusterOrder: np.array

    :param representMinMax: If in every cluster the minimum and the maximum of the attribute should be represented
    :type representMinMax: bool

    :param referenceAttributeIdx: If given, the integer index of the attribute
        whose mean profile defines a single temporal ordering that is applied to
        all attributes. This keeps the attributes aligned on a common time axis
        (preserving their co-incidence/concurrency with the reference attribute)
        while every attribute still takes the values of its own fitted duration
        curve. If None (default), each attribute is reordered by its own mean
        profile, which fits every distribution best but breaks concurrency
        across attributes. Only supported with distributionPeriodWise=True.
        Equivalent to ``concurrencyMethod="reference"``.
    :type referenceAttributeIdx: int or None

    :param concurrencyMethod: Strategy used to derive the synthetic time axis of
        the duration representation, one of :data:`CONCURRENCY_METHODS`. All
        strategies preserve every attribute's marginal distribution (duration
        curve) and differ only in how the attributes' concurrency across a
        common time axis is preserved:

        - ``"independent"`` (default): each attribute ordered by its own mean
          profile; best marginal fit, concurrency across attributes is lost.
        - ``"reference"``: single ordering from ``referenceAttributeIdx``'s mean
          profile, broadcast to all attributes.
        - ``"medoid"``: each attribute ordered by the cluster medoid's own ranks,
          reproducing a real period's joint co-occurrence across all attributes.
        - ``"consensus"``: single ordering from the first principal component of
          all attributes' mean profiles.
        - ``"assignment"``: optimal single ordering minimising total squared
          deviation from the cluster mean profile across all attributes.

        Only supported with ``distributionPeriodWise=True``. If None, falls back
        to ``"reference"`` when ``referenceAttributeIdx`` is given, otherwise
        ``"independent"``.
    :type concurrencyMethod: str or None
    """
    # Resolve the ordering strategy, keeping referenceAttributeIdx working as the
    # back-compatible way of requesting the "reference" strategy.
    if concurrencyMethod is None:
        concurrencyMethod = (
            "reference" if referenceAttributeIdx is not None else "independent"
        )
    if concurrencyMethod == "reference" and referenceAttributeIdx is None:
        raise ValueError(
            "concurrencyMethod='reference' requires a referenceAttributeIdx."
        )

    # make pd.DataFrame each row represents a candidate, and the columns are defined by two levels: the attributes and
    # the time steps inside the candidates.
    columnTuples = []
    num_attributes = int(candidates.shape[1] / timeStepsPerPeriod)
    for i in range(num_attributes):
        for j in range(timeStepsPerPeriod):
            columnTuples.append((i, j))
    candidates_df = pd.DataFrame(
        candidates, columns=pd.MultiIndex.from_tuples(columnTuples)
    )

    # There are two options for the duration representation. Either, the distribution of each cluster is preserved
    # (periodWise = True) or the distribution of the total time series is preserved only. In the latter case, the
    # inner-cluster variance is smaller and the variance across the typical periods' mean values is higher
    if distributionPeriodWise:
        n_attrs = num_attributes

        # Reshape to 3D: (periods, attributes, timesteps)
        candidates_3d = candidates.reshape(-1, n_attrs, timeStepsPerPeriod)

        clusterCenters = []
        for clusterNum in np.unique(clusterOrder):
            indice = np.where(clusterOrder == clusterNum)[0]
            n_cands = len(indice)
            if n_cands == 0:
                continue

            # (n_cands, n_attrs, timesteps) -> (n_attrs, n_cands, timesteps)
            cluster_data = candidates_3d[indice].transpose(1, 0, 2)

            # Sort all values per attribute, then reshape to duration curve
            flat = cluster_data.reshape(n_attrs, -1)
            flat = np.sort(flat, axis=1, kind="stable")
            repr_values = flat.reshape(n_attrs, timeStepsPerPeriod, n_cands).mean(
                axis=2
            )

            if representMinMax:
                repr_values = _pinMinMaxPreserveSum(
                    repr_values, flat[:, 0], flat[:, -1]
                )

            # Reorder each attribute's repr_values by a mean profile.
            # Round means before argsort to ensure identical tie-breaking
            # across platforms and numpy versions.
            means = np.round(cluster_data.mean(axis=1), 10)
            order = _orderingForCluster(
                cluster_data,
                means,
                repr_values,
                concurrencyMethod,
                referenceAttributeIdx,
            )
            rows = np.arange(n_attrs)[:, None]
            final_repr = np.empty_like(repr_values)
            final_repr[rows, order] = repr_values

            clusterCenters.append(final_repr.ravel())

    else:
        if concurrencyMethod != "independent":
            raise ValueError(
                "A concurrency-preserving duration representation "
                f"(concurrencyMethod='{concurrencyMethod}') is only supported "
                "with distributionPeriodWise=True."
            )
        clusterCentersList = []
        for a in candidates_df.columns.levels[0]:
            meanVals = []
            clusterLengths = []
            for clusterNum in np.unique(clusterOrder):
                indice = np.where(clusterOrder == clusterNum)
                noCandidates = len(indice[0])
                # get all the values of a certain attribute and cluster
                candidateValues = candidates_df.loc[indice[0], a]
                # calculate centroid of each cluster and append to list
                meanVals.append(np.round(candidateValues.mean(), 10))
                # make a list of weights of each cluster for each time step within the period
                clusterLengths.append(np.repeat(noCandidates, timeStepsPerPeriod))
            # concat centroid values and cluster weights for all clusters
            meansAndWeights = pd.concat(
                [
                    pd.DataFrame(np.array(meanVals)).stack(
                        future_stack=True,
                    ),
                    pd.DataFrame(np.array(clusterLengths)).stack(
                        future_stack=True,
                    ),
                ],
                axis=1,
            )
            # sort all values of all clusters according to the centroid values
            meansAndWeightsSorted = meansAndWeights.sort_values(0, kind="stable")
            # save order of the sorted centroid values across all clusters
            order = meansAndWeightsSorted.index
            # sort all values of the original time series
            sortedAttr = (
                candidates_df.loc[:, a]
                .stack(
                    future_stack=True,
                )
                .sort_values(kind="stable")
                .values
            )
            # take mean of sections of the original duration curve according to the cluster and its weight the
            # respective section is assigned to
            representationValues = []
            counter = 0
            for i, j in enumerate(meansAndWeightsSorted[1]):
                representationValues.append(sortedAttr[counter : counter + j].mean())
                counter += j
            # respect max and min of the attributes
            if representMinMax:
                representationValues = _representMinMax(
                    representationValues,
                    sortedAttr,
                    meansAndWeightsSorted,
                    keepSum=True,
                )

            # transform all representation values to a data frame and arrange it
            # according to the order of the sorted
            # centroid values
            representationValues = pd.DataFrame(np.array(representationValues))
            representationValues.index = order
            representationValues.sort_index(inplace=True)
            # append all cluster values attribute-wise to a list
            clusterCentersList.append(representationValues.unstack())
        # rearrange so that rows are the cluster centers and columns are time steps x attributes
        clusterCenters = np.array(pd.concat(clusterCentersList, axis=1))

    return clusterCenters


def _representMinMax(
    representationValues, sortedAttr, meansAndWeightsSorted, keepSum=True
):
    """
    Represents the the min and max values of the original time series in the
    duration curve representation such that the min and max values of the
    original time series are preserved.

    :param representationValues: The duration curve representation values
    :type representationValues: np.array

    :param sortedAttr: The sorted original time series
    :type sortedAttr: np.array

    :param meansAndWeightsSorted: The number of occureance of
     the original time series.
    :type meansAndWeightsSorted: pd.DataFrame

    :param keepSum: If the sum of the duration curve should be preserved
    :type keepSum: bool
    """

    if np.any(np.array(representationValues) < 0):
        raise ValueError("Negative values in the duration curve representation")

    # first retrieve the change of the values to the min and max values
    # of the original time series and their duration in the original
    # time series
    delta_max = sortedAttr.max() - representationValues[-1]
    appearance_max = meansAndWeightsSorted[1].iloc[-1]
    delta_min = sortedAttr.min() - representationValues[0]
    appearance_min = meansAndWeightsSorted[1].iloc[0]

    if delta_min == 0 and delta_max == 0:
        return representationValues

    if keepSum:
        # now anticipate the shift of the sum of the time series
        # due to the change of the min and max values
        # of the duration curve
        delta_sum = delta_max * appearance_max + delta_min * appearance_min
        mid_weights = np.asarray(
            meansAndWeightsSorted[1].iloc[1:-1].values, dtype=float
        )
        mid_orig = np.asarray(representationValues[1:-1], dtype=float)
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
            return representationValues

        # Initial multiplicative correction: preserves the weighted sum exactly
        # and keeps zero-valued segments at zero (matching the distribution
        # shape of the cluster). When no bound is violated, no further work
        # is needed and this matches the legacy behaviour.
        corrected = mid_orig * (1 + correction_factor)

        # Clip to the cluster envelope and redistribute the mass that got
        # clipped to segments that still have room. This makes the result
        # envelope-safe while restoring sum preservation up to feasibility.
        lower = float(sortedAttr.min())
        upper = float(sortedAttr.max())
        target_weighted_sum = weighted_mid_sum - delta_sum
        tol = max(abs(target_weighted_sum), 1.0) * 1e-12
        for _ in range(20):
            np.clip(corrected, lower, upper, out=corrected)
            residual = target_weighted_sum - float(np.sum(mid_weights * corrected))
            if abs(residual) <= tol:
                break
            if residual > 0:
                room = upper - corrected
            else:
                room = corrected - lower
            weighted_room = mid_weights * room
            total = float(weighted_room.sum())
            if total <= tol:
                # remaining mass has nowhere feasible to go
                warnings.warn(
                    "The cluster is too small to preserve the sum of the duration curve and additionally the min and max values of the original cluster members. The min max values of the cluster are not preserved. This does not necessarily mean that the min and max values of the original time series are not preserved."
                )
                break
            step = min(abs(residual), total)
            direction = 1.0 if residual > 0 else -1.0
            corrected = corrected + direction * step * room / total

        representationValues[1:-1] = corrected

    # change the values of the duration curve such that the min and max
    # values are preserved
    representationValues[-1] += delta_max
    representationValues[0] += delta_min

    return representationValues
