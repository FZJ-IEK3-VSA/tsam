import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate


def test_accuracyIndicators():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
    )

    aggregation2 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical", use_duration_curves=True),
    )

    # make sure that the sum of the attribute specific RMSEs is smaller for the normal time series clustering than for
    # the duration curve clustering
    np.testing.assert_array_less(
        aggregation1.accuracy.rmse.sum(),
        aggregation2.accuracy.rmse.sum(),
    )

    # make sure that the sum of the attribute specific duration curve RMSEs is smaller for the duration curve
    # clustering than for the normal time series clustering
    np.testing.assert_array_less(
        aggregation2.accuracy.rmse_duration.sum(),
        aggregation1.accuracy.rmse_duration.sum(),
    )

    # make sure that the same accounts for the total accuracy indicator
    np.testing.assert_array_less(
        aggregation1.accuracy.weighted_rmse,
        aggregation2.accuracy.weighted_rmse,
    )
    # make sure that the same accounts for the total accuracy indicator
    np.testing.assert_array_less(
        aggregation2.accuracy.weighted_rmse_duration,
        aggregation1.accuracy.weighted_rmse_duration,
    )


def test_accuracyIndicators_partial_weights():
    # Regression: GH #276. accuracyIndicators raised KeyError when the data
    # had columns absent from weightDict.
    hoursPerPeriod = 24
    noTypicalPeriods = 8
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    partial_weights = {raw.columns[0]: 2.0}

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        weightDict=partial_weights,
    )

    indicators = aggregation.accuracyIndicators()
    assert set(indicators.index) == set(raw.columns)
