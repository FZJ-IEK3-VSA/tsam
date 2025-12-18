import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from tsam.representations import representations


def segmentation(
    normalizedTypicalPeriods,
    noSegments,
    timeStepsPerPeriod,
    representationMethod=None,
    representationDict=None,
    distributionPeriodWise=True,
    predefSegmentOrder=None,
    predefSegmentDurations=None,
    predefSegmentCenters=None,
):
    """
    Agglomerative clustering of adjacent time steps within a set of typical periods in order to further reduce the
    temporal resolution within typical periods and to further reduce complexity of input data.

    :param normalizedTypicalPeriods: MultiIndex DataFrame containing the typical periods as first index, the time steps
        within the periods as second index and the attributes as columns.
    :type normalizedTypicalPeriods: pandas DataFrame

    :param noSegments: Number of segments in which the typical periods should be subdivided - equivalent to the number of
        inner-period clusters.
    :type noSegments: integer

    :param timeStepsPerPeriod: Number of time steps per period
    :type timeStepsPerPeriod: integer

    :param predefSegmentOrder: Predefined segment assignments per timestep, per typical period.
        If provided, skips clustering and uses these assignments directly.
        List of lists/arrays, one per typical period.
    :type predefSegmentOrder: list or None

    :param predefSegmentDurations: Predefined durations per segment, per typical period.
        Required if predefSegmentOrder is provided.
        List of lists/arrays, one per typical period.
    :type predefSegmentDurations: list or None

    :param predefSegmentCenters: Predefined center indices per segment, per typical period.
        If provided with predefSegmentOrder, uses these as segment centers
        instead of calculating representations.
        List of lists/arrays, one per typical period.
    :type predefSegmentCenters: list or None

    :returns:     - **segmentedNormalizedTypicalPeriods** (pandas DataFrame) --  MultiIndex DataFrame similar to
                    normalizedTypicalPeriods but with segments instead of time steps. Moreover, two additional index
                    levels define the length of each segment and the time step index at which each segment starts.
                  - **predictedSegmentedNormalizedTypicalPeriods** (pandas DataFrame) -- MultiIndex DataFrame with the same
                    shape of normalizedTypicalPeriods, but with overwritten values derived from segmentation used for
                    prediction of the original periods and accuracy indicators.
    """
    # Initialize lists for predicted and segmented DataFrame
    segmentedNormalizedTypicalPeriodsList = []
    predictedSegmentedNormalizedTypicalPeriodsList = []

    # Get unique period indices
    period_indices = normalizedTypicalPeriods.index.get_level_values(0).unique()

    # do for each typical period
    for period_i, period_label in enumerate(period_indices):
        # make numpy array with rows containing the segmentation candidates (time steps)
        # and columns as dimensions of the
        segmentationCandidates = np.asarray(
            normalizedTypicalPeriods.loc[period_label, :]
        )

        # Check if using predefined segments for this period
        if predefSegmentOrder is not None:
            # Use predefined segment order
            clusterOrder = np.asarray(predefSegmentOrder[period_i])

            # Get predefined durations
            segmentNoOccur = np.asarray(predefSegmentDurations[period_i])

            # Calculate segment numbers and start indices from durations
            segNo = np.arange(noSegments)
            indices = np.concatenate([[0], np.cumsum(segmentNoOccur)[:-1]])

            # The unique cluster order is just 0, 1, 2, ..., n_segments-1 in order
            clusterOrderUnique = list(range(noSegments))

            # Determine segment values
            if predefSegmentCenters is not None:
                # Use predefined centers directly
                center_indices = predefSegmentCenters[period_i]
                clusterCenters = segmentationCandidates[center_indices]
            else:
                # Calculate representations from predefined order
                clusterCenters, _clusterCenterIndices = representations(
                    segmentationCandidates,
                    clusterOrder,
                    default="meanRepresentation",
                    representationMethod=representationMethod,
                    representationDict=representationDict,
                    distributionPeriodWise=distributionPeriodWise,
                    timeStepsPerPeriod=1,
                )
        else:
            # Original clustering logic
            # produce adjacency matrix: Each time step is only connected to its preceding and succeeding one
            adjacencyMatrix = np.eye(timeStepsPerPeriod, k=1) + np.eye(
                timeStepsPerPeriod, k=-1
            )
            # execute clustering of adjacent time steps
            if noSegments == 1:
                clusterOrder = np.asarray([0] * len(segmentationCandidates))
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=noSegments, linkage="ward", connectivity=adjacencyMatrix
                )
                clusterOrder = clustering.fit_predict(segmentationCandidates)
            # determine the indices where the segments change and the number of time steps in each segment
            segNo, indices, segmentNoOccur = np.unique(
                clusterOrder, return_index=True, return_counts=True
            )
            clusterOrderUnique = [clusterOrder[index] for index in sorted(indices)]
            # determine the segments' values
            clusterCenters, _clusterCenterIndices = representations(
                segmentationCandidates,
                clusterOrder,
                default="meanRepresentation",
                representationMethod=representationMethod,
                representationDict=representationDict,
                distributionPeriodWise=distributionPeriodWise,
                timeStepsPerPeriod=1,
            )

        # predict each time step of the period by representing it with the corresponding segment's values
        predictedSegmentedNormalizedTypicalPeriods = (
            pd.DataFrame(clusterCenters, columns=normalizedTypicalPeriods.columns)
            .reindex(clusterOrder)
            .reset_index(drop=True)
        )
        # represent the period by the segments in the right order only instead of each time step
        segmentedNormalizedTypicalPeriods = (
            pd.DataFrame(clusterCenters, columns=normalizedTypicalPeriods.columns)
            .reindex(clusterOrderUnique)
            .set_index(np.sort(indices))
        )
        # keep additional information on the lengths of the segments in the right order
        segmentDuration = (
            pd.DataFrame(segmentNoOccur, columns=["Segment Duration"])
            .reindex(clusterOrderUnique)
            .set_index(np.sort(indices))
        )
        # create DataFrame with reduced number of segments together with three indices per period:
        # 1. The segment number
        # 2. The segment duration
        # 3. The index of the original time step, at which the segment starts
        result = segmentedNormalizedTypicalPeriods.set_index(
            [
                pd.Index(segNo, name="Segment Step"),
                segmentDuration["Segment Duration"],
                pd.Index(np.sort(indices), name="Original Start Step"),
            ]
        )
        # append predicted and segmented DataFrame to list to create a big DataFrame for all periods
        predictedSegmentedNormalizedTypicalPeriodsList.append(
            predictedSegmentedNormalizedTypicalPeriods
        )
        segmentedNormalizedTypicalPeriodsList.append(result)

    # create a big DataFrame for all periods for predicted segmented time steps and segments and return
    predictedSegmentedNormalizedTypicalPeriods = pd.concat(
        predictedSegmentedNormalizedTypicalPeriodsList,
        keys=period_indices,
    ).rename_axis(["", "TimeStep"])
    segmentedNormalizedTypicalPeriods = pd.concat(
        segmentedNormalizedTypicalPeriodsList,
        keys=period_indices,
    )
    return segmentedNormalizedTypicalPeriods, predictedSegmentedNormalizedTypicalPeriods
