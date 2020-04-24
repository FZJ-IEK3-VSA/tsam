import os
import time
import copy

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_durationCurve():

    # do everything for one attribute only to make sure that scaling does not play a role
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'testdata.csv'), index_col=0)['GHI']\
        .to_frame()

    noTypicalPeriods = 8

    hoursPerPeriod = 24

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(raw, noTypicalPeriods = noTypicalPeriods, hoursPerPeriod = hoursPerPeriod,
                                             clusterMethod = 'hierarchical', sortValues=True,
                                             rescaleClusterPeriods=False)

    typPeriods = aggregation.createTypicalPeriods()

    print('Clustering took ' + str(time.time() - starttime))

    # sort every attribute in every period in descending order for both, the found typical period and the days
    # that belong to the corresponding cluster
    for i in range(noTypicalPeriods):
        calculated = tsam.unstackToPeriods(raw, hoursPerPeriod)[0].loc[
                     np.where(aggregation.clusterOrder == i)[0], :]
        calculatedSorted = copy.deepcopy(calculated)
        algorithmResult = tsam.unstackToPeriods(typPeriods.loc[i], hoursPerPeriod)[0]
        for j in raw.columns:
            dfR = algorithmResult[j]
            dfR[dfR.columns] = np.sort(dfR)[:, ::-1]
            algorithmResult[j] = dfR
            df = calculatedSorted[j]
            df[df.columns] = np.sort(df)[:, ::-1]
            calculatedSorted[j] = df

        # make sure that the found typical period is always the one that is closest to the clusters centroid
        currentMeant = calculatedSorted.mean(axis=0)
        minIdx = np.square(calculatedSorted - currentMeant).sum(axis=1).idxmin()
        np.testing.assert_array_almost_equal(calculatedSorted.loc[minIdx], algorithmResult.iloc[0], decimal=4)

if __name__ == "__main__":
    test_durationCurve()