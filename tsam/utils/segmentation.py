# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tsam.representations import representations


def segmentation(normalizedTypicalPeriods, noSegments, timeStepsPerPeriod, representationMethod=None,
                 representationDict=None):
    '''
    Agglomerative clustering of adjacent time steps within a set of typical periods in order to further reduce the
    temporal resolution within typical periods and to further reduce complexity of input data.

    :param normalizedTypicalPeriods: MultiIndex DataFrame containing the typical periods as first index, the time steps
        within the periods as second index and the attributes as columns.
    :type normalizedTypicalPeriods: pandas DataFrame

    :param noSegments: Number of segments in which the typical periods shoul be subdivided - equivalent to the number of
        inner-period clusters.
    :type noSegments: integer

    :param timeStepsPerPeriod: Number of time steps per period
    :type timeStepsPerPeriod: integer

    :returns:     - **segmentedNormalizedTypicalPeriods** (pandas DataFrame) --  MultiIndex DataFrame similar to
                    normalizedTypicalPeriods but with segments instead of time steps. Moreover, two additional index
                    levels define the length of each segment and the time step index at which each segment starts.
                  - **predictedSegmentedNormalizedTypicalPeriods** (pandas DataFrame) -- MultiIndex DataFrame with the same
                    shape of normalizedTypicalPeriods, but with overwritten values derived from segmentation used for
                    prediction of the original periods and accuracy indicators.
    '''
    # Initialize lists for predicted and segmented DataFrame
    segmentedNormalizedTypicalPeriodsList = []
    predictedSegmentedNormalizedTypicalPeriodsList = []
    # do for each typical period
    for i in normalizedTypicalPeriods.index.get_level_values(0).unique():
        # make numpy array with rows containing the segmenatation candidates (time steps)
        # and columns as dimensions of the
        segmentationCandidates = np.asarray(normalizedTypicalPeriods.loc[i,:])
        # produce adjacency matrix: Each time step is only connected to its preceding and succeeding one
        adjacencyMatrix = np.eye(timeStepsPerPeriod, k=1) + np.eye(timeStepsPerPeriod, k=-1)
        # execute clustering of adjacent time steps
        if noSegments==1:
            clusterOrder = np.asarray([0] * len(segmentationCandidates))
        else:
            clustering = AgglomerativeClustering(n_clusters=noSegments, linkage='ward', connectivity=adjacencyMatrix)
            clusterOrder = clustering.fit_predict(segmentationCandidates)
        # determine the indices where the segments change and the number of time steps in each segment
        segNo, indices, segmentNoOccur = np.unique(clusterOrder, return_index=True, return_counts=True)
        clusterOrderUnique = [clusterOrder[index] for index in sorted(indices)]
        # determine the segments' values
        clusterCenters, clusterCenterIndices = representations(segmentationCandidates, clusterOrder,
                                                               default='meanRepresentation',
                                                               representationMethod=representationMethod,
                                                               representationDict=representationDict,
                                                               timeStepsPerPeriod=1)
        #clusterCenters = meanRepresentation(segmentationCandidates, clusterOrder)
        # predict each time step of the period by representing it with the corresponding segment's values
        predictedSegmentedNormalizedTypicalPeriods = pd.DataFrame(
            clusterCenters,
            columns=normalizedTypicalPeriods.columns).reindex(clusterOrder).reset_index(drop=True)
        # represent the period by the segments in the right order only instead of each time step
        segmentedNormalizedTypicalPeriods = pd.DataFrame(
            clusterCenters,
            columns=normalizedTypicalPeriods.columns).reindex(clusterOrderUnique).set_index(np.sort(indices))
        # keep additional information on the lengths of the segments in the right order
        segmentDuration = pd.DataFrame(segmentNoOccur, columns=['Segment Duration']).reindex(clusterOrderUnique).set_index(np.sort(indices))
        # create DataFrame with reduced number of segments together with three indices per period:
        # 1. The segment number
        # 2. The segment duration
        # 3. The index of the original time step, at which the segment starts
        result=segmentedNormalizedTypicalPeriods.set_index([pd.Index(segNo, name='Segment Step'), segmentDuration['Segment Duration'], pd.Index(np.sort(indices), name='Original Start Step')])
        # append predicted and segmented DataFrame to list to create a big DataFrame for all periods
        predictedSegmentedNormalizedTypicalPeriodsList.append(predictedSegmentedNormalizedTypicalPeriods)
        segmentedNormalizedTypicalPeriodsList.append(result)
    # create a big DataFrame for all periods for predicted segmented time steps and segments and return
    predictedSegmentedNormalizedTypicalPeriods = pd.concat(predictedSegmentedNormalizedTypicalPeriodsList, keys=normalizedTypicalPeriods.index.get_level_values(0).unique()).rename_axis(['','TimeStep'])
    segmentedNormalizedTypicalPeriods = pd.concat(segmentedNormalizedTypicalPeriodsList, keys=normalizedTypicalPeriods.index.get_level_values(0).unique())
    return segmentedNormalizedTypicalPeriods, predictedSegmentedNormalizedTypicalPeriods