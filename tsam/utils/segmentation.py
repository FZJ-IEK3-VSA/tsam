# -*- coding: utf-8 -*-
"""Further irregular feature-based segmentation of typical periods"""


import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tsam.timeseriesaggregation import meanRepresentation


def segmentation(normalizedTypicalPeriods, noSegments, timeStepsPerPeriod):

    segmentedNormalizedTypicalPeriodsList = []
    predictedSegmentedNormalizedTypicalPeriodsList = []
    # Do for each typical period
    for i in normalizedTypicalPeriods.index.get_level_values(0).unique():
        # Make numpy array with rows containing the segmenatation candidates (time steps)
        # and columns as dimensions of the
        segmentationCandidates = np.asarray(normalizedTypicalPeriods.loc[i,:])
        # Produce adjacency matrix: Each time step is only connected to its preceding and succeeding one
        adjacencyMatrix = np.eye(timeStepsPerPeriod, k=1) + np.eye(timeStepsPerPeriod, k=-1)
        # Execute clustering of adjacent time steps
        clustering = AgglomerativeClustering(n_clusters=noSegments, linkage='ward', connectivity=adjacencyMatrix)
        clusterOrder = clustering.fit_predict(segmentationCandidates)
        # Determine the indices where the segments change and the number of time steps in each segment
        indices, segmentNoOccur = np.unique(clusterOrder, return_index=True, return_counts=True)[1:3]
        clusterOrderUnique = [clusterOrder[index] for index in sorted(indices)]
        # Determine the segments's values
        clusterCenters = meanRepresentation(segmentationCandidates, clusterOrder)

        predictedSegmentedNormalizedTypicalPeriods = pd.DataFrame(
            clusterCenters,
            columns=normalizedTypicalPeriods.columns).reindex(clusterOrder).reset_index(drop=True)
        segmentedNormalizedTypicalPeriods = pd.DataFrame(
            clusterCenters,
            columns=normalizedTypicalPeriods.columns).reindex(clusterOrderUnique)
        segmentDuration = pd.DataFrame(segmentNoOccur, columns=['Segment Duration']).reindex(clusterOrderUnique)

        result = pd.concat([segmentedNormalizedTypicalPeriods, segmentDuration], axis=1).set_index(np.sort(indices))

        predictedSegmentedNormalizedTypicalPeriodsList.append(predictedSegmentedNormalizedTypicalPeriods)
        segmentedNormalizedTypicalPeriodsList.append(result)

    predictedSegmentedNormalizedTypicalPeriods = pd.concat(predictedSegmentedNormalizedTypicalPeriodsList, keys=normalizedTypicalPeriods.index.get_level_values(0).unique()).rename_axis(['','TimeStep'])
    segmentedNormalizedTypicalPeriods = pd.concat(segmentedNormalizedTypicalPeriodsList, keys=normalizedTypicalPeriods.index.get_level_values(0).unique()).rename_axis(['','TimeStep'])
    # print(segmentedNormalizedTypicalPeriods.drop(columns='Segment Duration'))
    # print(predictedSegmentedNormalizedTypicalPeriods)
    # print(normalizedTypicalPeriods)

    return predictedSegmentedNormalizedTypicalPeriods