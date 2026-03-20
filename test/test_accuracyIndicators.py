import numpy as np

import tsam.timeseriesaggregation as tsam
from conftest import load_testdata


def test_accuracyIndicators():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    raw = load_testdata()

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

    acc1 = aggregation1.accuracyIndicators()
    acc2 = aggregation2.accuracyIndicators()
    total1 = aggregation1.totalAccuracyIndicators()
    total2 = aggregation2.totalAccuracyIndicators()

    # Both methods return valid, finite accuracy indicators
    assert np.all(np.isfinite(acc1["RMSE"]))
    assert np.all(np.isfinite(acc2["RMSE"]))
    assert np.isfinite(total1["RMSE"])
    assert np.isfinite(total2["RMSE"])

    # Duration curve RMSE should be better for duration-curve clustering
    np.testing.assert_array_less(
        acc2.loc[:, "RMSE_duration"].sum(),
        acc1.loc[:, "RMSE_duration"].sum(),
    )
    np.testing.assert_array_less(
        total2["RMSE_duration"],
        total1["RMSE_duration"],
    )


if __name__ == "__main__":
    test_accuracyIndicators()
