# -*- coding: utf-8 -*-
"""Orders a set of representation values to fit several candidate value sets"""

import numpy as np
import pandas as pd


def durationRepresentation(candidates, clusterOrder, timeStepsPerPeriod, representMinMax=False):
    '''
    Represents the candidates of a given cluster group (clusterOrder)
    such that for every attribute the number of time steps is best fit.

    :param candidates: Dissimilarity matrix where each row represents a candidate
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the Integer entry to the group
    :type clusterOrder: np.array

    :param representMinMax: If in every cluster the minimum and the maximum of the attribute should be represented
    :type representMinMax: bool
    '''

    # make pd.DataFrame each row represents a candidate, and the columns are defined by two levels: the attributes and
    # the time steps inside the candidates.
    columnTuples = []
    for i in range(int(candidates.shape[1] / timeStepsPerPeriod)):
        for j in range(timeStepsPerPeriod):
            columnTuples.append((i, j))
    candidates = pd.DataFrame(candidates, columns=pd.MultiIndex.from_tuples(columnTuples))

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
            # reindex and arange such that every sorted segment gets represented by its mean
            sortedAttr.index = pd.MultiIndex.from_tuples(clean_index)
            representationValues = sortedAttr.unstack(level=0).mean(axis=1)
            # respect max and min of the attributes
            if representMinMax:
                representationValues.loc[0] = sortedAttr.values[0]
                representationValues.loc[representationValues.index[-1]] = sortedAttr.values[-1]
            # get the order of the representation values such that euclidean distance to the candidates is minimized
            order = candidateValues.mean().sort_values().index
            # arrange
            representationValues.index = order
            representationValues.sort_index(inplace=True)

            # add to cluster center
            clusterCenter = np.append(clusterCenter, representationValues.values)

        clusterCenters.append(clusterCenter)
    return clusterCenters
