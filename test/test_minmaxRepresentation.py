import time

import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, MinMaxMean, aggregate, unstack_to_periods


def test_minmaxRepresentation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 8

    period_duration = 24

    # GHI -> max, T -> min, Load -> min, Wind -> mean (the default for unlisted columns)
    representationDict = {"GHI": "max", "T": "min", "Wind": "mean", "Load": "min"}

    starttime = time.time()

    print(raw.columns)

    result = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(
            method="hierarchical",
            representation=MinMaxMean(max_columns=["GHI"], min_columns=["T", "Load"]),
        ),
        preserve_column_means=False,
    )

    typPeriods = result.cluster_representatives

    print("Clustering took " + str(time.time() - starttime))

    for i in range(n_clusters):
        for j in representationDict:
            if representationDict[j] == "min":
                calculated = (
                    unstack_to_periods(raw, period_duration)
                    .loc[np.where(result.cluster_assignments == i)[0], j]
                    .min()
                    .values
                )
            elif representationDict[j] == "max":
                calculated = (
                    unstack_to_periods(raw, period_duration)
                    .loc[np.where(result.cluster_assignments == i)[0], j]
                    .max()
                    .values
                )
            elif representationDict[j] == "mean":
                calculated = (
                    unstack_to_periods(raw, period_duration)
                    .loc[np.where(result.cluster_assignments == i)[0], j]
                    .mean()
                    .values
                )
            algorithmResult = typPeriods.loc[i, :].loc[:, j].values
            # print(calculated,algorithmResult)
            np.testing.assert_array_almost_equal(calculated, algorithmResult, decimal=4)


if __name__ == "__main__":
    test_minmaxRepresentation()
