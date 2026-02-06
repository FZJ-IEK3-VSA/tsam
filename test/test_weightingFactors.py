import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_weightingFactors():
    hours_per_period = 24

    no_typical_periods = 8

    weightDict1 = {"GHI": 1, "T": 1, "Wind": 1, "Load": 1}

    weightDict2 = {"GHI": 2, "T": 2, "Wind": 2, "Load": 2}

    weightDict3 = {"GHI": 2, "T": 1, "Wind": 1, "Load": 1}

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        weight_dict=weightDict1,
    )

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        weight_dict=weightDict2,
    )

    aggregation3 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        weight_dict=weightDict3,
    )

    # make sure that the accuracy indicators stay the same when the different attributes are equally overweighted
    np.testing.assert_almost_equal(
        aggregation1.accuracy_indicators().values,
        aggregation2.accuracy_indicators().values,
        decimal=6,
    )

    # make sure that the RMSE of GHI is less while the other RMSEs are bigger, when GHI is overweighted
    np.testing.assert_array_less(
        aggregation3.accuracy_indicators().loc["GHI", "RMSE"],
        aggregation1.accuracy_indicators().loc["GHI", "RMSE"],
    )
    np.testing.assert_array_less(
        aggregation1.accuracy_indicators().loc[["Load", "T", "Wind"], "RMSE"],
        aggregation3.accuracy_indicators().loc[["Load", "T", "Wind"], "RMSE"],
    )


if __name__ == "__main__":
    test_weightingFactors()
