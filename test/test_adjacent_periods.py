import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam

def test_adjacent_periods():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'testdata.csv'), index_col=0)

    noTypicalPeriods = 8

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(raw, noTypicalPeriods = noTypicalPeriods, hoursPerPeriod = 24,
                                             clusterMethod = 'adjacent_periods',
                                             representationMethod='meanRepresentation')

    typPeriods = aggregation.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))

    # check whether the clusterOrder consists of noTypicalPeriods blocks of the same number
    np.testing.assert_array_almost_equal(np.size(np.where(np.diff(aggregation.clusterOrder) != 0)), noTypicalPeriods-1,
                                         decimal=4)

if __name__ == "__main__":
    test_adjacent_periods()