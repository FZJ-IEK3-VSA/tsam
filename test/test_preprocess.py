import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_preprocess():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)

    raw_wind = raw.loc[:, 'Wind'].to_frame()

    aggregation_wind = tsam.TimeSeriesAggregation(raw_wind, noTypicalPeriods = 8, hoursPerPeriod = 24,
                                            clusterMethod = 'hierarchical')

    aggregation_wind._preProcessTimeSeries()

    test = aggregation_wind.normalizedPeriodlyProfiles

    orig = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','results','preprocessed_wind.csv'), index_col = [0], header = [0,1])

    np.testing.assert_array_almost_equal(test.values, orig.values,decimal=15)

    
if __name__ == "__main__":
    test_preprocess()