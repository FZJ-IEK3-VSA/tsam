import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_indexMatching():

    hoursPerPeriod = 24

    noSegments = 8

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=8, hoursPerPeriod=hoursPerPeriod,
                                             clusterMethod='k_means', segmentation=True, noSegments=noSegments)

    typPeriods = aggregation.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))

    indexTable = aggregation.indexMatching()

    # make sure that the PeriodNum column equals the clusterOrder
    np.testing.assert_array_almost_equal(indexTable.loc[::24,'PeriodNum'].values, aggregation.clusterOrder, decimal=4)

    # make sure that the TimeStep indices equal the number of hoursPerPeriod arranged as array
    np.testing.assert_array_almost_equal(pd.unique(indexTable.loc[:, 'TimeStep']),
                                         np.arange(hoursPerPeriod, dtype='int64'), decimal=4)

    # make sure that the SegmentIndex indices equal the number of noSegments arranged as array
    np.testing.assert_array_almost_equal(pd.unique(indexTable.loc[:, 'SegmentIndex']),
                                         np.arange(noSegments, dtype='int64'), decimal=4)

if __name__ == "__main__":
    test_indexMatching()