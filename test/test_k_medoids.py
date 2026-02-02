import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import RESULTS_DIR, TESTDATA_CSV


def test_k_medoids():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    orig_raw = pd.read_csv(
        RESULTS_DIR / "testperiods_kmedoids.csv",
        index_col=[0, 1],
    )

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24 * 7,
        clusterMethod="k_medoids",
    )

    typPeriods = aggregation.createTypicalPeriods()

    print("Clustering took " + str(time.time() - starttime))

    # sort the typical days in order to avoid error assertion due to different order
    sortedDaysOrig = orig_raw.groupby(level=0).sum().sort_values("GHI").index
    sortedDaysTest = typPeriods.groupby(level=0).sum().sort_values("GHI").index

    # rearange their order
    orig = (
        orig_raw[typPeriods.columns]
        .unstack()
        .loc[sortedDaysOrig, :]
        .stack(
            future_stack=True,
        )
    )
    test = (
        typPeriods.unstack()
        .loc[sortedDaysTest, :]
        .stack(
            future_stack=True,
        )
    )

    np.testing.assert_array_almost_equal(orig.values, test.values, decimal=4)


if __name__ == "__main__":
    test_k_medoids()
