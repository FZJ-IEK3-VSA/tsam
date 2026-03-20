import numpy as np
import pytest

import tsam.timeseriesaggregation as tsam
from conftest import load_testdata

pytestmark = pytest.mark.filterwarnings("ignore::tsam.exceptions.LegacyAPIWarning")


def test_weightingFactors():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    weightDict1 = {"GHI": 1, "T": 1, "Wind": 1, "Load": 1}

    weightDict2 = {"GHI": 2, "T": 2, "Wind": 2, "Load": 2}

    weightDict3 = {"GHI": 2, "T": 1, "Wind": 1, "Load": 1}

    raw = load_testdata()

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        weightDict=weightDict1,
    )

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        weightDict=weightDict2,
    )

    aggregation3 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        weightDict=weightDict3,
    )

    # make sure that the accuracy indicators stay the same when the different attributes are equally overweighted
    np.testing.assert_almost_equal(
        aggregation1.accuracyIndicators().values,
        aggregation2.accuracyIndicators().values,
        decimal=6,
    )

    # make sure that the RMSE of GHI is less or equal while the other RMSEs are on average bigger or equal,
    # when GHI is overweighted
    assert (
        aggregation3.accuracyIndicators().loc["GHI", "RMSE"]
        <= aggregation1.accuracyIndicators().loc["GHI", "RMSE"]
    )
    assert (
        aggregation1.accuracyIndicators().loc[["Load", "T", "Wind"], "RMSE"].mean()
        <= aggregation3.accuracyIndicators().loc[["Load", "T", "Wind"], "RMSE"].mean()
    )


if __name__ == "__main__":
    test_weightingFactors()
