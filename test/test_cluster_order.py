import dataclasses
import time

import numpy as np
import pandas as pd

from conftest import RESULTS_DIR, TESTDATA_CSV
from tsam import ClusterConfig, aggregate


def test_cluster_order():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    raw_wind = raw.loc[:, "Wind"].to_frame()

    orig_raw_predefClusterOrder = pd.read_csv(
        RESULTS_DIR / "testperiods_predefClusterOrder.csv",
        index_col=[0, 1],
    )

    orig_raw_predefClusterOrderAndClusterCenters = pd.read_csv(
        RESULTS_DIR / "testperiods_predefClusterOrderAndClusterCenters.csv",
        index_col=[0, 1],
    )

    starttime = time.time()

    # Derive a clustering on a single attribute, then transfer it to the full data.
    aggregation_wind = aggregate(
        raw_wind,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="hierarchical", representation="mean"),
    )
    clustering_wind = aggregation_wind.clustering

    # Transfer the cluster assignments only (cluster centers are recomputed).
    typPeriods_predefClusterOrder = (
        dataclasses.replace(clustering_wind, cluster_centers=None)
        .apply(raw)
        .cluster_representatives
    )

    # Transfer cluster assignments together with the cluster centers.
    typPeriods_predefClusterOrderAndClusterCenters = clustering_wind.apply(
        raw
    ).cluster_representatives

    print("Clustering took " + str(time.time() - starttime))

    # sort the typical days in order to avoid error assertion due to different order
    sortedDaysOrig1 = (
        orig_raw_predefClusterOrder.groupby(level=0).sum().sort_values("GHI").index
    )
    sortedDaysTest1 = (
        typPeriods_predefClusterOrder.groupby(level=0).sum().sort_values("GHI").index
    )

    sortedDaysOrig2 = (
        orig_raw_predefClusterOrderAndClusterCenters.groupby(level=0)
        .sum()
        .sort_values("GHI")
        .index
    )
    sortedDaysTest2 = (
        typPeriods_predefClusterOrderAndClusterCenters.groupby(level=0)
        .sum()
        .sort_values("GHI")
        .index
    )

    # rearange their order
    orig1 = (
        orig_raw_predefClusterOrder[typPeriods_predefClusterOrder.columns]
        .unstack()
        .loc[sortedDaysOrig1, :]
        .stack(
            future_stack=True,
        )
    )
    test1 = (
        typPeriods_predefClusterOrder.unstack()
        .loc[sortedDaysTest1, :]
        .stack(
            future_stack=True,
        )
    )
    orig2 = (
        orig_raw_predefClusterOrderAndClusterCenters[
            typPeriods_predefClusterOrderAndClusterCenters.columns
        ]
        .unstack()
        .loc[sortedDaysOrig2, :]
        .stack(
            future_stack=True,
        )
    )
    test2 = (
        typPeriods_predefClusterOrderAndClusterCenters.unstack()
        .loc[sortedDaysTest2, :]
        .stack(
            future_stack=True,
        )
    )

    np.testing.assert_array_almost_equal(
        orig1.values, test1[orig1.columns].values, decimal=4
    )
    np.testing.assert_array_almost_equal(
        orig2.values, test2[orig2.columns].values, decimal=4
    )


if __name__ == "__main__":
    test_cluster_order()
