import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_weightingFactors():

    hoursPerPeriod = 24

    noTypicalPeriods = 8

    weightDict1 = {'GHI': 1, 'T': 1, 'Wind': 1, 'Load': 1}

    weightDict2 = {'GHI': 2, 'T': 2, 'Wind': 2, 'Load': 2}

    weightDict3 = {'GHI': 2, 'T': 1, 'Wind': 1, 'Load': 1}

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)

    aggregation1 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', weightDict=weightDict1)

    aggregation2 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', weightDict=weightDict2)

    aggregation3 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', weightDict=weightDict3)

    # make sure that the accuracy indicators stay the same when the different attributes are equally overweighted
    np.testing.assert_almost_equal(aggregation1.accuracyIndicators().values, aggregation2.accuracyIndicators().values,
                                   decimal=6)

    # make sure that the RMSE of GHI is less while the other RMSEs are bigger, when GHI is overweighted
    np.testing.assert_array_less(aggregation3.accuracyIndicators().loc['GHI', 'RMSE'],
                                 aggregation1.accuracyIndicators().loc['GHI', 'RMSE'])
    np.testing.assert_array_less(aggregation1.accuracyIndicators().loc[['Load', 'T', 'Wind'], 'RMSE'],
                                 aggregation3.accuracyIndicators().loc[['Load', 'T', 'Wind'], 'RMSE'])

if __name__ == "__main__":
    test_weightingFactors()