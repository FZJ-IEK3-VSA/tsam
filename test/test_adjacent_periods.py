import time

import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate


def test_adjacent_periods():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    noTypicalPeriods = 8

    starttime = time.time()

    result = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=24,
        cluster=ClusterConfig(method="contiguous", representation="mean"),
    )

    typPeriods = result.cluster_representatives

    print("Clustering took " + str(time.time() - starttime))

    # check whether the clusterOrder consists of noTypicalPeriods blocks of the same number
    np.testing.assert_array_almost_equal(
        np.size(np.where(np.diff(result.cluster_assignments) != 0)),
        noTypicalPeriods - 1,
        decimal=4,
    )


if __name__ == "__main__":
    test_adjacent_periods()
