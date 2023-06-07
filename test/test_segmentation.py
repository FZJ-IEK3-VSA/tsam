import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam


def test_segmentation():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    orig_raw = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "examples",
            "results",
            "testperiods_segmentation.csv",
        ),
        index_col=[0, 1, 2],
    )

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=20,
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        segmentation=True,
        noSegments=12,
    )

    typPeriods = aggregation.createTypicalPeriods()

    print("Clustering took " + str(time.time() - starttime))

    # sort the typical days in order to avoid error assertion due to different order
    sortedDaysOrig = orig_raw.groupby(level=0).sum().sort_values("GHI").index
    sortedDaysTest = typPeriods.groupby(level=0).sum().sort_values("GHI").index

    # rearange their order
    orig = orig_raw[typPeriods.columns].unstack().loc[sortedDaysOrig, :].stack()
    test = typPeriods.unstack().loc[sortedDaysTest, :].stack()

    np.testing.assert_array_almost_equal(orig.values, test.values, decimal=4)


if __name__ == "__main__":
    test_segmentation()
