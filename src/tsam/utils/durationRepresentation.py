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

    # Convert candidates to numpy array at the beginning if it's a DataFrame
    if isinstance(candidates, pd.DataFrame):
        candidates_array = candidates.values
    else:
        candidates_array = candidates
    
    # Create a pandas DataFrame only when necessary
    columnTuples = [(i, j) for i in range(int(candidates_array.shape[1] / timeStepsPerPeriod)) 
                   for j in range(timeStepsPerPeriod)]
    
    candidates_df = pd.DataFrame(
        candidates_array, columns=pd.MultiIndex.from_tuples(columnTuples)
    )
    
    if distributionPeriodWise:
        clusterCenters = []
        unique_clusters = np.unique(clusterOrder)
        
        for clusterNum in unique_clusters:
            indice = np.where(clusterOrder == clusterNum)[0]
            noCandidates = len(indice)
            
            # Pre-allocate the full cluster center array
            cluster_values_count = noCandidates * timeStepsPerPeriod * len(candidates_df.columns.levels[0])
            clusterCenter = np.zeros(cluster_values_count)
            current_idx = 0
            
            for a in candidates_df.columns.levels[0]:
                # Get values using numpy indexing when possible
                candidateValues = candidates_df.loc[indice, a].values
                
                # Reshape to more easily work with numpy
                candidateValues_reshaped = candidateValues.reshape(-1)
                
                # Sort values using numpy
                sorted_values = np.sort(candidateValues_reshaped)
                
                # Calculate representative values directly
                values_per_timestep = noCandidates
                representation_values = np.zeros(timeStepsPerPeriod)
                
                for t in range(timeStepsPerPeriod):
                    start_idx = t * values_per_timestep
                    end_idx = start_idx + values_per_timestep
                    representation_values[t] = np.mean(sorted_values[start_idx:end_idx])
                
                # Handle min/max representation if needed
                if representMinMax:
                    representation_values[0] = sorted_values[0]
                    representation_values[-1] = sorted_values[-1]
                
                # Re-order values based on the mean of candidate values
                mean_values = np.mean(candidateValues, axis=0)
                order_indices = np.argsort(mean_values)
                
                # Reorder representation values
                representation_values_ordered = representation_values[order_indices]
                
                # Add to cluster center
                clusterCenter[current_idx:current_idx+len(representation_values)] = representation_values_ordered
                current_idx += len(representation_values)
                
            clusterCenters.append(clusterCenter[:current_idx])  # Trim if we didn't use the whole pre-allocation
    
    else:
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
                meanVals.append(candidateValues.mean())
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
            meansAndWeightsSorted = meansAndWeights.sort_values(0)
            # save order of the sorted centroid values across all clusters
            order = meansAndWeightsSorted.index
            # sort all values of the original time series
            sortedAttr = (
                candidates_df.loc[:, a]
                .stack(
                    future_stack=True,
                )
                .sort_values()
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
        # and derive how much the other values have to be changed to preserve
        # the mean of the duration curve
        correction_factor = (
            -delta_sum
            / (meansAndWeightsSorted[1].iloc[1:-1] * representationValues[1:-1]).sum()
        )

        if correction_factor < -1 or correction_factor > 1:
            warnings.warn(
                "The cluster is too small to preserve the sum of the duration curve and additionally the min and max values of the original cluster members. The min max values of the cluster are not preserved. This does not necessarily mean that the min and max values of the original time series are not preserved."
            )
            return representationValues

        # correct the values of the duration curve such
        # that the mean of the duration curve is preserved
        # since the min and max values are changed
        representationValues[1:-1] = np.multiply(
            representationValues[1:-1], (1 + correction_factor)
        )

    # change the values of the duration curve such that the min and max
    # values are preserved
    representationValues[-1] += delta_max
    representationValues[0] += delta_min

    return representationValues
