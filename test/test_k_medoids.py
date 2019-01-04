import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_hierarchical():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)

    results = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','results','testperiods_kmedoids.csv'), index_col = [0,1])

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(raw, noTypicalPeriods = 8, hoursPerPeriod = 24*7, 
                                            clusterMethod = 'k_medoids', )

    typPeriods = aggregation.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))
    
    sortedTypPeriods = typPeriods.reindex(typPeriods['GHI'].unstack().sum(axis=1).sort_values().index,level=0)

    np.testing.assert_array_almost_equal(sortedTypPeriods.values, results.values,decimal=4)


if __name__ == "__main__":
    test_hierarchical()