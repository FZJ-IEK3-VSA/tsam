import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_extremePeriods():

    hoursPerPeriod = 24

    noTypicalPeriods = 8

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)

    aggregation1 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', rescaleClusterPeriods=False,
                                              extremePeriodMethod='new_cluster_center', addPeakMax=['GHI'])

    aggregation2 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', rescaleClusterPeriods=False,
                                              extremePeriodMethod='append', addPeakMax=['GHI'])

    aggregation3 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', rescaleClusterPeriods=False,
                                              extremePeriodMethod='replace_cluster_center', addPeakMax=['GHI'])

    # make sure that the RMSE for new cluster centers (reassigning points to the exxtreme point if the distance to it is
    # smaller)is bigger than for appending just one extreme period
    np.testing.assert_array_less(aggregation1.accuracyIndicators().loc['GHI', 'RMSE'],
                                 aggregation2.accuracyIndicators().loc['GHI', 'RMSE'])

    # make sure that the RMSE for appending the extreme period is smaller than for replacing the cluster center by the
    # extreme period (conservative assumption)
    np.testing.assert_array_less(aggregation2.accuracyIndicators().loc['GHI', 'RMSE'],
                                 aggregation3.accuracyIndicators().loc['GHI', 'RMSE'])

    # check if addMeanMax and addMeanMin are working
    aggregation4 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', rescaleClusterPeriods=False,
                                              extremePeriodMethod='append', addMeanMax=['GHI'], addMeanMin=['GHI'])

    origData = aggregation4.predictOriginalData()

    np.testing.assert_array_almost_equal(raw.groupby(np.arange(len(raw)) // 24).mean().max().loc['GHI'],
                                         origData.groupby(np.arange(len(origData)) // 24).mean().max().loc['GHI'],
                                         decimal=6)

    np.testing.assert_array_almost_equal(raw.groupby(np.arange(len(raw)) // 24).mean().min().loc['GHI'],
                                         origData.groupby(np.arange(len(origData)) // 24).mean().min().loc['GHI'],
                                         decimal=6)

if __name__ == "__main__":
    test_extremePeriods()