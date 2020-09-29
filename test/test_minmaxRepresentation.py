import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam

def test_minmaxRepresentation():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'testdata.csv'), index_col=0)

    noTypicalPeriods = 8

    hoursPerPeriod = 24

    representationDict = {'GHI': 'max', 'T': 'min', 'Wind': 'mean', 'Load': 'min'}

    starttime = time.time()

    print(raw.columns)

    aggregation = tsam.TimeSeriesAggregation(raw, noTypicalPeriods = noTypicalPeriods, hoursPerPeriod = hoursPerPeriod,
                                             clusterMethod = 'hierarchical', rescaleClusterPeriods=False,
                                             representationMethod='minmaxmeanRepresentation',
                                             representationDict=representationDict)

    typPeriods = aggregation.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))

    for i in range(noTypicalPeriods):
        for j in representationDict:
            if representationDict[j] == 'min':
                calculated = tsam.unstackToPeriods(raw, hoursPerPeriod)[0].loc[
                    np.where(aggregation.clusterOrder == i)[0], j].min().values
            elif representationDict[j] == 'max':
                calculated = tsam.unstackToPeriods(raw, hoursPerPeriod)[0].loc[
                    np.where(aggregation.clusterOrder == i)[0], j].max().values
            elif representationDict[j] == 'mean':
                calculated = tsam.unstackToPeriods(raw, hoursPerPeriod)[0].loc[
                    np.where(aggregation.clusterOrder == i)[0], j].mean().values
            algorithmResult = typPeriods.loc[i, :].loc[:, j].values
            # print(calculated,algorithmResult)
            np.testing.assert_array_almost_equal(calculated, algorithmResult, decimal=4)

if __name__ == "__main__":
    test_minmaxRepresentation()