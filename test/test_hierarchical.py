import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_hierarchical():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)

    results = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','results','testperiods_hierarchical.csv'), index_col = [0,1])

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(raw, noTypicalPeriods = 8, hoursPerPeriod = 24, 
                                            clusterMethod = 'hierarchical', 
                                            extremePeriodMethod = 'new_cluster_center',
                                        addPeakMin = ['T'], addPeakMax = ['Load'] )

    typPeriods = aggregation.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))


    np.testing.assert_array_almost_equal(typPeriods.values, results.values,decimal=4)


if __name__ == "__main__":
    test_hierarchical()