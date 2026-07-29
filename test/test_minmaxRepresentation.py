import time

import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import (
    ClusterConfig,
    MinMaxMean,
    SegmentConfig,
    aggregate,
    unstack_to_periods,
)


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


def test_segment_and_cluster_minmax_are_independent():
    """A MinMaxMean on SegmentConfig steers segments, not cluster centers.

    v3 kept a single global ``representationDict`` and let the segment
    representation overwrite the cluster one, so the two stages could not name
    different columns. They are resolved separately now.
    """
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    common = {
        "n_clusters": 4,
        "period_duration": 24,
        "preserve_column_means": False,
    }
    cluster_minmax = ClusterConfig(
        method="hierarchical",
        representation=MinMaxMean(max_columns=["GHI"], min_columns=["T"]),
    )

    # Same cluster representation, different segment representations: if the
    # segment dict were still shared with the cluster stage, these would be
    # indistinguishable.
    max_load = aggregate(
        raw,
        cluster=cluster_minmax,
        segments=SegmentConfig(
            n_segments=6, representation=MinMaxMean(max_columns=["Load"])
        ),
        **common,
    )
    min_load = aggregate(
        raw,
        cluster=cluster_minmax,
        segments=SegmentConfig(
            n_segments=6, representation=MinMaxMean(min_columns=["Load"])
        ),
        **common,
    )

    assert (
        max_load.cluster_representatives["Load"].sum()
        > min_load.cluster_representatives["Load"].sum()
    ), "the segment representation did not reach the segmentation stage"

    # The cluster representation is untouched by either: with no segmentation,
    # GHI is still the per-cluster maximum.
    unsegmented = aggregate(raw, cluster=cluster_minmax, **common)
    periods = unstack_to_periods(raw, 24)
    for cluster_no in range(4):
        members = np.where(unsegmented.cluster_assignments == cluster_no)[0]
        np.testing.assert_array_almost_equal(
            periods.loc[members, "GHI"].max().values,
            unsegmented.cluster_representatives.loc[cluster_no, "GHI"].values,
            decimal=4,
        )


if __name__ == "__main__":
    test_minmaxRepresentation()
