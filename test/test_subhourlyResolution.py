import os
import time

import pandas as pd
import numpy as np
import copy

import tsam.timeseriesaggregation as tsam



def test_subhourlyResolution():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'testdata.csv'), index_col=0)

    rawSubhourlyInndex = copy.deepcopy(raw)

    # reset index of new dataframe to 15 min. intervals
    rawSubhourlyInndex.index = pd.date_range('2050-01-01 00:30:00', periods=8760, freq=(str(0.25) + 'H'),
                                             tz='Europe/Berlin')

    starttime = time.time()

    aggregation1 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=8, hoursPerPeriod=24,
                                              clusterMethod='hierarchical')

    typPeriods1 = aggregation1.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))

    starttime = time.time()

    # cluster dataframe with 15 min. intervals to six hours per period, which equals 24 time steps per period
    aggregation2 = tsam.TimeSeriesAggregation(rawSubhourlyInndex, noTypicalPeriods=8, hoursPerPeriod=6,
                                              clusterMethod='hierarchical')

    typPeriods2 = aggregation2.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))

    # check that the results from both aggregations are the same with respect to the clustered values
    np.testing.assert_almost_equal(typPeriods1.values, typPeriods2.values, decimal=6)

if __name__ == "__main__":
    test_hierarchical()