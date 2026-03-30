import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam_legacy
from conftest import TESTDATA_CSV


def test_extremePeriods():
    hours_per_period = 24

    no_typical_periods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = tsam_legacy.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        rescale_cluster_periods=False,
        extreme_period_method="new_cluster_center",
        add_peak_max=["GHI"],
    )

    aggregation2 = tsam_legacy.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        rescale_cluster_periods=False,
        extreme_period_method="append",
        add_peak_max=["GHI"],
    )

    aggregation3 = tsam_legacy.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        rescale_cluster_periods=False,
        extreme_period_method="replace_cluster_center",
        add_peak_max=["GHI"],
    )

    # make sure that the RMSE for new cluster centers (reassigning points to the exxtreme point if the distance to it is
    # smaller)is bigger than for appending just one extreme period
    np.testing.assert_array_less(
        aggregation1.accuracy_indicators().loc["GHI", "RMSE"],
        aggregation2.accuracy_indicators().loc["GHI", "RMSE"],
    )

    # make sure that the RMSE for appending the extreme period is smaller than for replacing the cluster center by the
    # extreme period (conservative assumption)
    np.testing.assert_array_less(
        aggregation2.accuracy_indicators().loc["GHI", "RMSE"],
        aggregation3.accuracy_indicators().loc["GHI", "RMSE"],
    )

    # check if add_mean_max and add_mean_min are working
    aggregation4 = tsam_legacy.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        rescale_cluster_periods=False,
        extreme_period_method="append",
        add_mean_max=["GHI"],
        add_mean_min=["GHI"],
    )

    origData = aggregation4.predict_original_data()

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
