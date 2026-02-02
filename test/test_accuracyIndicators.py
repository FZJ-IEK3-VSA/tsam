import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_accuracyIndicators():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
    )

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        sortValues=True,
    )

    # make sure that the sum of the attribute specific RMSEs is smaller for the normal time series clustering than for
    # the duration curve clustering
    np.testing.assert_array_less(
        aggregation1.accuracyIndicators().loc[:, "RMSE"].sum(),
        aggregation2.accuracyIndicators().loc[:, "RMSE"].sum(),
    )

    # make sure that the sum of the attribute specific duration curve RMSEs is smaller for the duration curve
    # clustering than for the normal time series clustering
    np.testing.assert_array_less(
        aggregation2.accuracyIndicators().loc[:, "RMSE_duration"].sum(),
        aggregation1.accuracyIndicators().loc[:, "RMSE_duration"].sum(),
    )

    # make sure that the same accounts for the total accuracy indicator
    np.testing.assert_array_less(
        aggregation1.totalAccuracyIndicators()["RMSE"],
        aggregation2.totalAccuracyIndicators()["RMSE"],
    )
    # make sure that the same accounts for the total accuracy indicator
    np.testing.assert_array_less(
        aggregation2.totalAccuracyIndicators()["RMSE_duration"],
        aggregation1.totalAccuracyIndicators()["RMSE_duration"],
    )


if __name__ == "__main__":
    test_accuracyIndicators()
