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

    # make sure that the RMSE for new cluster centers (reassigning points to the exxtreme point if the distance to it is
    # smaller)is bigger than for appending just one extreme period
    np.testing.assert_array_less(
        aggregation1.accuracy.rmse["GHI"],
        aggregation2.accuracy.rmse["GHI"],
    )

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


if __name__ == "__main__":
    test_extremePeriods()
