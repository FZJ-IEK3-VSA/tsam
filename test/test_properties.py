import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_properties():

    hoursPerPeriod = 24

    noSegments = 8

    noTypicalPeriods = 8

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','testdata.csv'), index_col = 0)



    starttime = time.time()

    aggregation1 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', segmentation=True, noSegments=noSegments)

    print('Clustering took ' + str(time.time() - starttime))

    np.testing.assert_array_almost_equal(aggregation1.stepIdx, np.arange(noSegments), decimal=4)



    starttime = time.time()

    aggregation2 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical')

    print('Clustering took ' + str(time.time() - starttime))

    np.testing.assert_array_almost_equal(aggregation2.stepIdx, np.arange(hoursPerPeriod), decimal=4)



    starttime = time.time()

    aggregation3 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical')

    print('Clustering took ' + str(time.time() - starttime))

    np.testing.assert_array_almost_equal(aggregation3.clusterPeriodIdx, np.arange(noTypicalPeriods), decimal=4)



    starttime = time.time()

    aggregation4 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', segmentation=True, noSegments=noSegments)

    print('Clustering took ' + str(time.time() - starttime))

    appearances = np.unique(aggregation4.clusterOrder, return_counts=True)[1].tolist()

    occurrenceDict = {i: j for i, j in enumerate(appearances)}

    # make sure that the clusterPeriodNoOccur equals the number of appearances in the clusterOrder
    np.testing.assert_array_almost_equal(list(aggregation4.clusterPeriodNoOccur.values()),
                                         list(occurrenceDict.values()), decimal=4)



    starttime = time.time()

    aggregation5 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', segmentation=True, noSegments=noSegments)

    print('Clustering took ' + str(time.time() - starttime))

    # make sure that the values of the clusterPeriodDict equal those from the typicalPeriods-dataframe
    np.testing.assert_array_almost_equal(pd.DataFrame.from_dict(data=aggregation5.clusterPeriodDict).values,
                                         aggregation5.createTypicalPeriods().values, decimal=4)



    starttime = time.time()

    aggregation6 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical')

    print('Clustering took ' + str(time.time() - starttime))

    # make sure that the sum of all segment durations in each period equals the hours per period
    for i in range(noTypicalPeriods):
        np.testing.assert_array_almost_equal(pd.DataFrame.from_dict(aggregation6.segmentDurationDict).loc[i].sum()[0],
                                             hoursPerPeriod, decimal=4)



    starttime = time.time()

    aggregation7 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', segmentation=True, noSegments=noSegments)

    print('Clustering took ' + str(time.time() - starttime))

    # make sure that the sum of all segment durations in each period equals the hours per period
    for i in range(noTypicalPeriods):
        np.testing.assert_array_almost_equal(pd.DataFrame.from_dict(aggregation7.segmentDurationDict).loc[i].sum()[0],
                                             hoursPerPeriod, decimal=4)



    starttime = time.time()

    aggregation8 = tsam.TimeSeriesAggregation(raw, noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
                                              clusterMethod='hierarchical', segmentation=True, noSegments=noSegments)

    print('Clustering took ' + str(time.time() - starttime))

    indexTable = aggregation8.indexMatching()

    # make sure that the PeriodNum column equals the clusterOrder
    np.testing.assert_array_almost_equal(indexTable.loc[::24,'PeriodNum'].values, aggregation8.clusterOrder, decimal=4)

    # make sure that the TimeStep indices equal the number of hoursPerPeriod arranged as array
    np.testing.assert_array_almost_equal(pd.unique(indexTable.loc[:, 'TimeStep']),
                                         np.arange(hoursPerPeriod, dtype='int64'), decimal=4)

    # make sure that the SegmentIndex indices equal the number of noSegments arranged as array
    np.testing.assert_array_almost_equal(pd.unique(indexTable.loc[:, 'SegmentIndex']),
                                         np.arange(noSegments, dtype='int64'), decimal=4)

if __name__ == "__main__":
    test_properties()