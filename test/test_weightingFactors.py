import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV

pytestmark = pytest.mark.filterwarnings("ignore::tsam.exceptions.LegacyAPIWarning")


def test_weightingFactors():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    weightDict1 = {"GHI": 1, "T": 1, "Wind": 1, "Load": 1}

    weightDict2 = {"GHI": 2, "T": 2, "Wind": 2, "Load": 2}

    weightDict3 = {"GHI": 2, "T": 1, "Wind": 1, "Load": 1}

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

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

    # make sure that the RMSE of GHI is less while the other RMSEs are bigger, when GHI is overweighted
    np.testing.assert_array_less(
        aggregation3.accuracyIndicators().loc["GHI", "RMSE"],
        aggregation1.accuracyIndicators().loc["GHI", "RMSE"],
    )
    np.testing.assert_array_less(
        aggregation1.accuracyIndicators().loc[["Load", "T", "Wind"], "RMSE"],
        aggregation3.accuracyIndicators().loc[["Load", "T", "Wind"], "RMSE"],
    )


def test_uniform_weights_equal_no_weights():
    """Uniform scaling produces identical typicalPeriods and predictOriginalData()."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    agg_none = tsam.TimeSeriesAggregation(
        raw.copy(), noTypicalPeriods=8, hoursPerPeriod=24, clusterMethod="hierarchical"
    )
    agg_uniform = tsam.TimeSeriesAggregation(
        raw.copy(),
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        weightDict={"GHI": 1.0, "T": 1.0, "Wind": 1.0, "Load": 1.0},
    )

    tp_none = agg_none.createTypicalPeriods()
    tp_uniform = agg_uniform.createTypicalPeriods()

    pd.testing.assert_frame_equal(tp_none, tp_uniform)
    pd.testing.assert_frame_equal(
        agg_none.predictOriginalData(), agg_uniform.predictOriginalData()
    )


def test_reconstructed_within_original_range():
    """Reconstructed values stay within original data range with non-uniform weights."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    agg = tsam.TimeSeriesAggregation(
        raw.copy(),
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        weightDict={"GHI": 5.0, "T": 1.0, "Wind": 1.0, "Load": 1.0},
    )
    agg.createTypicalPeriods()
    reconstructed = agg.predictOriginalData()

    for col in raw.columns:
        assert reconstructed[col].min() >= raw[col].min() - 1e-6, (
            f"{col} reconstructed min below original"
        )
        assert reconstructed[col].max() <= raw[col].max() + 1e-6, (
            f"{col} reconstructed max above original"
        )


if __name__ == "__main__":
    test_weightingFactors()
