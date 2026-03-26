import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_adjacent_periods():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    no_typical_periods = 8

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=24,
        cluster_method="adjacent_periods",
        representation_method="meanRepresentation",
    )

    typPeriods = aggregation.create_typical_periods()

    print("Clustering took " + str(time.time() - starttime))

    # check whether the cluster_order consists of no_typical_periods blocks of the same number
    np.testing.assert_array_almost_equal(
        np.size(np.where(np.diff(aggregation.cluster_order) != 0)),
        no_typical_periods - 1,
        decimal=4,
    )


if __name__ == "__main__":
    test_adjacent_periods()
