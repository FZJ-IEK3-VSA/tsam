import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, aggregate


def test_extremePeriods():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
        preserve_column_means=False,
        extremes=ExtremeConfig(method="new_cluster", max_value=["GHI"]),
    )

    aggregation2 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
        preserve_column_means=False,
        extremes=ExtremeConfig(method="append", max_value=["GHI"]),
    )

    aggregation3 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
        preserve_column_means=False,
        extremes=ExtremeConfig(method="replace", max_value=["GHI"]),
    )

    # new_cluster reassigns periods that are closer to the extreme than to their
    # cluster's centroid, so it keeps the extreme as a cluster of its own like
    # append does. Its RMSE is not ordered against append's: the reassignment
    # baseline is the centroid clustering minimised (#492), not the
    # representation a period is reconstructed from.
    assert aggregation1.n_clusters == aggregation2.n_clusters == noTypicalPeriods + 1
    assert aggregation1.cluster_counts[noTypicalPeriods] >= 1

    # make sure that the RMSE for appending the extreme period is smaller than for replacing the cluster center by the
    # extreme period (conservative assumption)
    np.testing.assert_array_less(
        aggregation2.accuracy.rmse["GHI"],
        aggregation3.accuracy.rmse["GHI"],
    )

    # check if addMeanMax and addMeanMin are working
    aggregation4 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
        preserve_column_means=False,
        extremes=ExtremeConfig(method="append", max_period=["GHI"], min_period=["GHI"]),
    )

    origData = aggregation4.reconstructed

    np.testing.assert_array_almost_equal(
        raw.groupby(np.arange(len(raw)) // 24).mean().max().loc["GHI"],
        origData.groupby(np.arange(len(origData)) // 24).mean().max().loc["GHI"],
        decimal=6,
    )

    np.testing.assert_array_almost_equal(
        raw.groupby(np.arange(len(raw)) // 24).mean().min().loc["GHI"],
        origData.groupby(np.arange(len(origData)) // 24).mean().min().loc["GHI"],
        decimal=6,
    )


def test_new_cluster_compares_against_centroid_not_representation():
    """`new_cluster` measures the incumbent distance to the cluster centroid.

    With a ``distribution_minmax`` representation the cluster center is a
    duration curve, not chronologically aligned with real periods, so measuring
    against it made every member look far away. Here both extremes land in the
    same 29-period cluster and all 29 used to flip, emptying it (#492, and the
    rescaling crash of #478). Against the centroid only 11 flip, so all eight
    regular clusters survive. Dropping an emptied cluster is still covered in
    test_empty_clusters.py.
    """
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method="hierarchical", representation="distribution_minmax"
        ),
        extremes=ExtremeConfig(
            method="new_cluster", max_value=["Load"], min_value=["T"]
        ),
    )

    # Two extremes on top of 8 clusters: no regular cluster may be emptied.
    assert aggregation.n_clusters == 10
    counts = aggregation.cluster_counts
    assert all(counts[c] >= 1 for c in range(10))
    assert sum(counts.values()) == 365
    # 29 - 11 periods stay in the cluster that holds both extremes.
    assert sorted(counts[c] for c in range(8))[0] == 18

    reconstructed = aggregation.reconstructed
    np.testing.assert_array_almost_equal(
        raw.mean().values, reconstructed.mean().values, decimal=6
    )


if __name__ == "__main__":
    test_extremePeriods()
