import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import RESULTS_DIR


def test_aggregate_hiearchical():
    normalizedPeriodlyProfiles = pd.read_csv(
        RESULTS_DIR / "preprocessed_wind.csv",
        index_col=[0],
        header=[0, 1],
    )

    _clusterCenters, _clusterCenterIndices, clusterOrder = tsam.aggregatePeriods(
        normalizedPeriodlyProfiles.values,
        n_clusters=8,
        clusterMethod="hierarchical",
    )

    orig = [
        1,
        2,
        3,
        1,
        5,
        3,
        2,
        0,
        0,
        2,
        2,
        2,
        6,
        3,
        2,
        2,
        0,
        2,
        0,
        0,
        4,
        6,
        4,
        7,
        2,
        2,
        0,
        4,
        1,
        6,
        7,
        2,
        2,
        3,
        2,
        0,
        0,
        2,
        0,
        0,
        2,
        0,
        5,
        7,
        3,
        2,
        0,
        0,
        2,
        1,
        2,
        2,
        7,
        0,
        0,
        2,
        0,
        0,
        2,
        5,
        4,
        1,
        2,
        2,
        2,
        2,
        4,
        3,
        0,
        7,
        0,
        0,
        2,
        2,
        2,
        0,
        0,
        0,
        2,
        0,
        0,
        2,
        3,
        0,
    ]

    np.testing.assert_array_almost_equal(orig, clusterOrder, decimal=4)


if __name__ == "__main__":
    test_aggregate_hiearchical()
