import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_averaging():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    no_typical_periods = 8

    hours_per_period = 24

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="averaging",
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

    # check whether the cluster centers are in line with the average of the candidates assigned to the different
    # clusters
    for i in range(no_typical_periods):
        calc = (
            tsam.unstack_to_periods(raw, hours_per_period)[0]
            .loc[np.where(aggregation.cluster_order == i)]
            .mean(axis=0)
            .to_frame()
            .values
        )
        orig = tsam.unstack_to_periods(typPeriods.loc[i], hours_per_period)[0].T.values
        np.testing.assert_array_almost_equal(calc, orig, decimal=4)


if __name__ == "__main__":
    test_averaging()
