import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from tsam.utils.durationRepresentation import durationRepresentation

# Aliases: old verbose names → new short names.
# The monolith sends old names; the pipeline sends new names.
_ALIASES = {
    "meanRepresentation": "mean",
    "medoidRepresentation": "medoid",
    "maxoidRepresentation": "maxoid",
    "minmaxmeanRepresentation": "minmax_mean",
    "durationRepresentation": "distribution",
    "distributionRepresentation": "distribution",
    "distributionAndMinMaxRepresentation": "distribution_minmax",
}


def representations(
    candidates,
    clusterOrder,
    default,
    representationMethod=None,
    representationDict=None,
    distributionPeriodWise=True,
    timeStepsPerPeriod=None,
):
    clusterCenterIndices = None
    if representationMethod is None:
        representationMethod = default
    # Normalize old names to new names
    representationMethod = _ALIASES.get(representationMethod, representationMethod)
    if representationMethod == "mean":
        clusterCenters = meanRepresentation(candidates, clusterOrder)
    elif representationMethod == "medoid":
        clusterCenters, clusterCenterIndices = medoidRepresentation(
            candidates, clusterOrder
        )
    elif representationMethod == "maxoid":
        clusterCenters, clusterCenterIndices = maxoidRepresentation(
            candidates, clusterOrder
        )
    elif representationMethod == "minmax_mean":
        clusterCenters = minmaxmeanRepresentation(
            candidates, clusterOrder, representationDict, timeStepsPerPeriod
        )
    elif representationMethod == "distribution":
        clusterCenters = durationRepresentation(
            candidates,
            clusterOrder,
            distributionPeriodWise,
            timeStepsPerPeriod,
            representMinMax=False,
        )
    elif representationMethod == "distribution_minmax":
        clusterCenters = durationRepresentation(
            candidates,
            clusterOrder,
            distributionPeriodWise,
            timeStepsPerPeriod,
            representMinMax=True,
        )
    else:
        raise ValueError("Chosen 'representationMethod' does not exist.")

    return clusterCenters, clusterCenterIndices


def maxoidRepresentation(candidates, clusterOrder):
    """
    Represents the candidates of a given cluster group (clusterOrder)
    by its medoid, measured with the euclidean distance.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    """
    # set cluster member that is farthest away from the points of the other clusters as maxoid
    clusterCenters = []
    clusterCenterIndices = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        innerDistMatrix = euclidean_distances(candidates, candidates[indice])
        mindistIdx = np.argmax(innerDistMatrix.sum(axis=0))
        clusterCenters.append(candidates[indice][mindistIdx])
        clusterCenterIndices.append(indice[0][mindistIdx])

    return clusterCenters, clusterCenterIndices


def medoidRepresentation(candidates, clusterOrder):
    """
    Represents the candidates of a given cluster group (clusterOrder)
    by its medoid, measured with the euclidean distance.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    """
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
    """
    Represents the candidates of a given cluster group (clusterOrder)
    by its mean.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    """
    # set cluster centers as means of the group candidates
    clusterCenters = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        currentMean = candidates[indice].mean(axis=0)
        clusterCenters.append(currentMean)
    return clusterCenters


def minmaxmeanRepresentation(
    candidates, clusterOrder, representationDict, timeStepsPerPeriod
):
    """
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
    """
    # set cluster center depending of the representationDict
    clusterCenters = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        currentClusterCenter = np.zeros(len(representationDict) * timeStepsPerPeriod)
        for attributeNum in range(len(representationDict)):
            startIdx = attributeNum * timeStepsPerPeriod
            endIdx = (attributeNum + 1) * timeStepsPerPeriod
            if list(representationDict.values())[attributeNum] == "min":
                currentClusterCenter[startIdx:endIdx] = candidates[
                    indice, startIdx:endIdx
                ].min(axis=1)
            elif list(representationDict.values())[attributeNum] == "max":
                currentClusterCenter[startIdx:endIdx] = candidates[
                    indice, startIdx:endIdx
                ].max(axis=1)
            elif list(representationDict.values())[attributeNum] == "mean":
                currentClusterCenter[startIdx:endIdx] = candidates[
                    indice, startIdx:endIdx
                ].mean(axis=1)
            else:
                raise ValueError(
                    'At least one value in the representationDict is neither "min", "max" nor "mean".'
                )
        clusterCenters.append(currentClusterCenter)
    return clusterCenters
