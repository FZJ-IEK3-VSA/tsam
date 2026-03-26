import copy
import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_durationCurve():
    # do everything for one attribute only to make sure that scaling does not play a role
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)["GHI"].to_frame()

    no_typical_periods = 8

    hours_per_period = 24

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        sort_values=True,
        rescale_cluster_periods=False,
    )

    typPeriods = aggregation.create_typical_periods()

    print("Clustering took " + str(time.time() - starttime))

    # sort every attribute in every period in descending order for both, the found typical period and the days
    # that belong to the corresponding cluster
    for i in range(no_typical_periods):
        calculated = tsam.unstack_to_periods(raw, hours_per_period)[0].loc[
            np.where(aggregation.cluster_order == i)[0], :
        ]
        calculatedSorted = copy.deepcopy(calculated)
        algorithmResult = tsam.unstack_to_periods(typPeriods.loc[i], hours_per_period)[
            0
        ]
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
        np.testing.assert_array_almost_equal(
            calculatedSorted.loc[minIdx], algorithmResult.iloc[0], decimal=4
        )


if __name__ == "__main__":
    test_durationCurve()
