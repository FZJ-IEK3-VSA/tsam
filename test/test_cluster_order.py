import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_cluster_order():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)

    raw_wind = raw.loc[:, 'Wind'].to_frame()

    orig_raw_predefClusterOrder = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','results','testperiods_predefClusterOrder.csv'), index_col = [0,1])

    orig_raw_predefClusterOrderAndClusterCenters = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'results', 'testperiods_predefClusterOrderAndClusterCenters.csv'),index_col=[0, 1])

    starttime = time.time()

    aggregation_wind = tsam.TimeSeriesAggregation(raw_wind, noTypicalPeriods = 8, hoursPerPeriod = 24,
                                            clusterMethod = 'hierarchical', representationMethod='meanRepresentation')

    typPeriods_wind = aggregation_wind.createTypicalPeriods()

    aggregation_predefClusterOrder = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=8, hoursPerPeriod=24,
                                                                clusterMethod='hierarchical',
                                                                representationMethod='meanRepresentation',
                                                                predefClusterOrder=aggregation_wind.clusterOrder)

    typPeriods_predefClusterOrder = aggregation_predefClusterOrder.createTypicalPeriods()

    aggregation_predefClusterOrderAndClusterCenters = tsam.TimeSeriesAggregation(raw,
                                                                                 noTypicalPeriods=8, hoursPerPeriod=24,
                                                                                 clusterMethod='hierarchical',
                                                                                 representationMethod='meanRepresentation',
                                                                                 predefClusterOrder=aggregation_wind.clusterOrder,
                                                                                 predefClusterCenterIndices=aggregation_wind.clusterCenterIndices)

    typPeriods_predefClusterOrderAndClusterCenters = aggregation_predefClusterOrderAndClusterCenters.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))


    # sort the typical days in order to avoid error assertion due to different order
    sortedDaysOrig1 = orig_raw_predefClusterOrder.sum(axis=0,level=0).sort_values('GHI').index
    sortedDaysTest1 = typPeriods_predefClusterOrder.sum(axis=0,level=0).sort_values('GHI').index

    sortedDaysOrig2 = orig_raw_predefClusterOrderAndClusterCenters.sum(axis=0,level=0).sort_values('GHI').index
    sortedDaysTest2 = typPeriods_predefClusterOrderAndClusterCenters.sum(axis=0,level=0).sort_values('GHI').index

    # rearange their order
    orig1 = orig_raw_predefClusterOrder[typPeriods_predefClusterOrder.columns].unstack().loc[sortedDaysOrig1,:].stack()
    test1 = typPeriods_predefClusterOrder.unstack().loc[sortedDaysTest1,:].stack()
    orig2 = orig_raw_predefClusterOrderAndClusterCenters[typPeriods_predefClusterOrderAndClusterCenters.columns].unstack().loc[sortedDaysOrig2,:].stack()
    test2 = typPeriods_predefClusterOrderAndClusterCenters.unstack().loc[sortedDaysTest2,:].stack()

    np.testing.assert_array_almost_equal(orig1.values, test1[orig1.columns].values,decimal=4)
    np.testing.assert_array_almost_equal(orig2.values, test2[orig2.columns].values, decimal=4)

if __name__ == "__main__":
    test_cluster_order()