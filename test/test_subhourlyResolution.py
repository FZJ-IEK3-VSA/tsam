import copy
import time

import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate


def test_subhourlyResolution():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    rawSubhourlyInndex = copy.deepcopy(raw)

    # reset index of new dataframe to 15 min. intervals
    rawSubhourlyInndex.index = pd.date_range(
        "2050-01-01 00:30:00", periods=8760, freq=(str(0.25) + "h"), tz="Europe/Berlin"
    )

    starttime = time.time()

    aggregation1 = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="hierarchical"),
    )

    typPeriods1 = aggregation1.cluster_representatives

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    # cluster dataframe with 15 min. intervals to six hours per period, which equals 24 time steps per period
    aggregation2 = aggregate(
        rawSubhourlyInndex,
        n_clusters=8,
        period_duration=6,
        cluster=ClusterConfig(method="hierarchical"),
    )

    typPeriods2 = aggregation2.cluster_representatives

    print("Clustering took " + str(time.time() - starttime))

    # check that the results from both aggregations are the same with respect to the clustered values
    np.testing.assert_almost_equal(typPeriods1.values, typPeriods2.values, decimal=6)


if __name__ == "__main__":
    test_subhourlyResolution()
