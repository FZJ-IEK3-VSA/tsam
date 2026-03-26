import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_accuracyIndicators():
    hours_per_period = 24

    no_typical_periods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
    )

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        sort_values=True,
    )

    # make sure that the sum of the attribute specific RMSEs is smaller for the normal time series clustering than for
    # the duration curve clustering
    np.testing.assert_array_less(
        aggregation1.accuracy_indicators().loc[:, "RMSE"].sum(),
        aggregation2.accuracy_indicators().loc[:, "RMSE"].sum(),
    )

    # make sure that the sum of the attribute specific duration curve RMSEs is smaller for the duration curve
    # clustering than for the normal time series clustering
    np.testing.assert_array_less(
        aggregation2.accuracy_indicators().loc[:, "RMSE_duration"].sum(),
        aggregation1.accuracy_indicators().loc[:, "RMSE_duration"].sum(),
    )

    # make sure that the same accounts for the total accuracy indicator
    np.testing.assert_array_less(
        aggregation1.total_accuracy_indicators()["RMSE"],
        aggregation2.total_accuracy_indicators()["RMSE"],
    )
    # make sure that the same accounts for the total accuracy indicator
    np.testing.assert_array_less(
        aggregation2.total_accuracy_indicators()["RMSE_duration"],
        aggregation1.total_accuracy_indicators()["RMSE_duration"],
    )


if __name__ == "__main__":
    test_accuracyIndicators()
