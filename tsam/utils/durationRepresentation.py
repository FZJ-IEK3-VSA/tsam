# -*- coding: utf-8 -*-
"""Orders a set of representation values to fit several candidate value sets"""

import warnings

import numpy as np
import pandas as pd


def durationRepresentation(
    candidates,
    clusterOrder,
    distributionPeriodWise,
    timeStepsPerPeriod,
    representMinMax=False,
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
    """

    # make pd.DataFrame each row represents a candidate, and the columns are defined by two levels: the attributes and
    # the time steps inside the candidates.
    columnTuples = []
    for i in range(int(candidates.shape[1] / timeStepsPerPeriod)):
        for j in range(timeStepsPerPeriod):
            columnTuples.append((i, j))
    candidates = pd.DataFrame(
        candidates, columns=pd.MultiIndex.from_tuples(columnTuples)
    )

    # There are two options for the duration representation. Either, the distribution of each cluster is preserved
    # (periodWise = True) or the distribution of the total time series is preserved only. In the latter case, the
    # inner-cluster variance is smaller and the variance across the typical periods' mean values is higher
    if distributionPeriodWise:
        clusterCenters = []
        for clusterNum in np.unique(clusterOrder):
            indice = np.where(clusterOrder == clusterNum)
            noCandidates = len(indice[0])
            clean_index = []

            clusterCenter = []
            # get a clean index depending on the size
            for y in candidates.columns.levels[1]:
                for x in range(noCandidates):
                    clean_index.append((x, y))
            for a in candidates.columns.levels[0]:
                # get all the values of a certain attribute and cluster
                candidateValues = candidates.loc[indice[0], a]
                # sort all values
                sortedAttr = candidateValues.stack().sort_values()
                # reindex and arrange such that every sorted segment gets represented by its mean
                sortedAttr.index = pd.MultiIndex.from_tuples(clean_index)
                representationValues = sortedAttr.unstack(level=0).mean(axis=1)
                # respect max and min of the attributes
                if representMinMax:
                    representationValues.loc[0] = sortedAttr.values[0]
                    representationValues.loc[
                        representationValues.index[-1]
                    ] = sortedAttr.values[-1]


                # get the order of the representation values such that euclidean distance to the candidates is minimized
                order = candidateValues.mean().sort_values().index
                # arrange
                representationValues.index = order
                representationValues.sort_index(inplace=True)

                # add to cluster center
                clusterCenter = np.append(clusterCenter, representationValues.values)

            clusterCenters.append(clusterCenter)

    else:
        clusterCentersList = []
        for a in candidates.columns.levels[0]:
            meanVals = []
            clusterLengths = []
            for clusterNum in np.unique(clusterOrder):
                indice = np.where(clusterOrder == clusterNum)
                noCandidates = len(indice[0])
                # get all the values of a certain attribute and cluster
                candidateValues = candidates.loc[indice[0], a]
                # calculate centroid of each cluster and append to list
                meanVals.append(candidateValues.mean())
                # make a list of weights of each cluster for each time step within the period
                clusterLengths.append(np.repeat(noCandidates, timeStepsPerPeriod))
            # concat centroid values and cluster weights for all clusters
            meansAndWeights = pd.concat(
                [
                    pd.DataFrame(np.array(meanVals)).stack(),
                    pd.DataFrame(np.array(clusterLengths)).stack(),
                ],
                axis=1,
            )
            # sort all values of all clusters according to the centroid values
            meansAndWeightsSorted = meansAndWeights.sort_values(0)
            # save order of the sorted centroid values across all clusters
            order = meansAndWeightsSorted.index
            # sort all values of the original time series
            sortedAttr = candidates.loc[:, a].stack().sort_values().values
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



def _representMinMax(representationValues, sortedAttr, meansAndWeightsSorted, 
                     keepSum=True):
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
        # and derive how much the other values have to be changed to preserve 
        # the mean of the duration curve
        correction_factor = - delta_sum / (meansAndWeightsSorted[1].iloc[1:-1] 
                                            * representationValues[1:-1]).sum()
        
        if correction_factor < -1 or correction_factor > 1:
            warnings.warn("The cluster is to small to preserve the sum of the duration curve and additionally the min and max values of the original cluster members. The min max values of the cluster are not preserved. This does not necessarily mean that the min and max values of the original time series are not preserved.")
            return representationValues

        # correct the values of the duration curve such 
        # that the mean of the duration curve is preserved
        # since the min and max values are changed
        representationValues[1:-1] = np.multiply(representationValues[1:-1], (
            1+ correction_factor))
        
    # change the values of the duration curve such that the min and max 
    # values are preserved
    representationValues[-1] += delta_max
    representationValues[0] += delta_min
    
    return representationValues
