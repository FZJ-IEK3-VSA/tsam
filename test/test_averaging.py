import time

import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate, unstack_to_periods


def test_averaging():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 8

    period_duration = 24

    starttime = time.time()

    result = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="averaging", representation="mean"),
    )

    typPeriods = result.cluster_representatives

    print("Clustering took " + str(time.time() - starttime))

    # check whether the cluster_assignments consists of n_clusters blocks of the same number
    np.testing.assert_array_almost_equal(
        np.size(np.where(np.diff(result.cluster_assignments) != 0)),
        n_clusters - 1,
        decimal=4,
    )

    # check whether the cluster centers are in line with the average of the candidates assigned to the different
    # clusters
    for i in range(n_clusters):
        calc = (
            unstack_to_periods(raw, period_duration)
            .loc[np.where(result.cluster_assignments == i)]
            .mean(axis=0)
            .to_frame()
            .values
        )
        orig = unstack_to_periods(typPeriods.loc[i], period_duration).T.values
        np.testing.assert_array_almost_equal(calc, orig, decimal=4)


if __name__ == "__main__":
    test_averaging()
