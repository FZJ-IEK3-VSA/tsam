import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam


def test_k_maxoids():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    starttime = time.time()

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod="k_means",
        rescaleClusterPeriods=False,
    )

    predictedPeriods1 = aggregation1.predictOriginalData()

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod="k_maxoids",
        rescaleClusterPeriods=False,
    )

    predictedPeriods2 = aggregation2.predictOriginalData()

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the maximum values of the time series predicted by k-maxoids are bigger than those predicted by
    # k-means
    np.testing.assert_array_less(predictedPeriods1.max(), predictedPeriods2.max())

    # make sure that the minimum values of the time series predicted by k-maxoids are smaller than those predicted by
    # k-means except for those of GHI since the minimum value of 0 during night time is found by both algorithms
    np.testing.assert_array_less(
        predictedPeriods2.min()[1:], predictedPeriods1.min()[1:]
    )


if __name__ == "__main__":
    test_k_maxoids()
