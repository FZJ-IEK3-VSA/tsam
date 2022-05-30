# -*- coding: utf-8 -*-
"""Orders a set of representation values to fit several candidate value sets"""

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
                representationValues[-1] = sortedAttr.max()
                representationValues[0] = sortedAttr.min()
            # transform all representation values to a data frame and arrange it according to the order of the sorted
            # centroid values
            representationValues = pd.DataFrame(np.array(representationValues))
            representationValues.index = order
            representationValues.sort_index(inplace=True)
            # append all cluster values attribute-wise to a list
            clusterCentersList.append(representationValues.unstack())
        # rearrange so that rows are the cluster centers and columns are time steps x attributes
        clusterCenters = np.array(pd.concat(clusterCentersList, axis=1))

    return clusterCenters
