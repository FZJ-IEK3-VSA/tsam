# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def representations(candidates, clusterOrder, default, representationMethod=None, representationDict=None,
                    timeStepsPerPeriod=None):
    clusterCenterIndices = None
    if representationMethod is None:
        representationMethod = default
    if representationMethod == 'meanRepresentation':
        clusterCenters = meanRepresentation(candidates, clusterOrder)
    elif representationMethod == 'medoidRepresentation':
        clusterCenters, clusterCenterIndices = medoidRepresentation(candidates, clusterOrder)
    elif representationMethod == 'minmaxmeanRepresentation':
        clusterCenters = minmaxmeanRepresentation(candidates, clusterOrder, representationDict, timeStepsPerPeriod)
    return clusterCenters, clusterCenterIndices

def medoidRepresentation(candidates, clusterOrder):
    '''
    Represents the candidates of a given cluster group (clusterOrder)
    by its medoid, measured with the euclidean distance.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    '''
    # set cluster center as medoid
    clusterCenters = []
    clusterCenterIndices = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        innerDistMatrix = euclidean_distances(candidates[indice])
        mindistIdx = np.argmin(innerDistMatrix.sum(axis=0))
        clusterCenters.append(candidates[indice][mindistIdx])
        clusterCenterIndices.append(indice[0][mindistIdx])

    return clusterCenters, clusterCenterIndices


def meanRepresentation(candidates, clusterOrder):
    '''
    Represents the candidates of a given cluster group (clusterOrder)
    by its mean.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    '''
    # set cluster centers as means of the group candidates
    clusterCenters = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        currentMean = candidates[indice].mean(axis=0)
        clusterCenters.append(currentMean)
    return clusterCenters

def minmaxmeanRepresentation(candidates, clusterOrder, representationDict, timeStepsPerPeriod):
    '''
    Represents the candidates of a given cluster group (clusterOrder)
    by either the minimum, the maximum or the mean values of each time step for
    all periods in that cluster depending on the command for each attribute.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array

    :param representationDict: A dictionary which defines for each attribute whether the typical
        period should be represented by the minimum or maximum values within each cluster.
        optional (default: None)
    :type representationDict: dictionary

    :param timeStepsPerPeriod: The number of discrete timesteps which describe one period. required
    :type timeStepsPerPeriod: integer
    '''
    # set cluster center depending of the representationDict
    clusterCenters = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        currentClusterCenter = np.zeros(len(representationDict) * timeStepsPerPeriod)
        for attributeNum in range(len(representationDict)):
            startIdx = attributeNum * timeStepsPerPeriod
            endIdx = (attributeNum + 1) * timeStepsPerPeriod
            if list(representationDict.values())[attributeNum] == 'min':
                currentClusterCenter[startIdx:endIdx] = candidates[indice, startIdx:endIdx].min(axis=1)
            elif list(representationDict.values())[attributeNum] == 'max':
                currentClusterCenter[startIdx:endIdx] = candidates[indice, startIdx:endIdx].max(axis=1)
            elif list(representationDict.values())[attributeNum] == 'mean':
                currentClusterCenter[startIdx:endIdx] = candidates[indice, startIdx:endIdx].mean(axis=1)
            else:
                raise ValueError('At least one value in the representationDict is neither "min", "max" nor "mean".')
        clusterCenters.append(currentClusterCenter)
    return clusterCenters
