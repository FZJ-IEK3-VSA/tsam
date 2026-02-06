import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_minmaxRepresentation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    no_typical_periods = 8

    hours_per_period = 24

    representationDict = {"GHI": "max", "T": "min", "Wind": "mean", "Load": "min"}

    starttime = time.time()

    print(raw.columns)

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        rescale_cluster_periods=False,
        representation_method="minmaxmeanRepresentation",
        representation_dict=representationDict,
    )

    typPeriods = aggregation.create_typical_periods()

    print("Clustering took " + str(time.time() - starttime))

    for i in range(no_typical_periods):
        for j in representationDict:
            if representationDict[j] == "min":
                calculated = (
                    tsam.unstack_to_periods(raw, hours_per_period)[0]
                    .loc[np.where(aggregation.cluster_order == i)[0], j]
                    .min()
                    .values
                )
            elif representationDict[j] == "max":
                calculated = (
                    tsam.unstack_to_periods(raw, hours_per_period)[0]
                    .loc[np.where(aggregation.cluster_order == i)[0], j]
                    .max()
                    .values
                )
            elif representationDict[j] == "mean":
                calculated = (
                    tsam.unstack_to_periods(raw, hours_per_period)[0]
                    .loc[np.where(aggregation.cluster_order == i)[0], j]
                    .mean()
                    .values
                )
            algorithmResult = typPeriods.loc[i, :].loc[:, j].values
            # print(calculated,algorithmResult)
            np.testing.assert_array_almost_equal(calculated, algorithmResult, decimal=4)


if __name__ == "__main__":
    test_minmaxRepresentation()
