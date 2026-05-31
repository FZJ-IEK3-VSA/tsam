import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate


def test_weightingFactors():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    weightDict1 = {"GHI": 1, "T": 1, "Wind": 1, "Load": 1}

    weightDict2 = {"GHI": 2, "T": 2, "Wind": 2, "Load": 2}

    weightDict3 = {"GHI": 2, "T": 1, "Wind": 1, "Load": 1}

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
        weights=weightDict1,
    )

    aggregation2 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
        weights=weightDict2,
    )

    aggregation3 = aggregate(
        raw,
        n_clusters=noTypicalPeriods,
        period_duration=hoursPerPeriod,
        cluster=ClusterConfig(method="hierarchical"),
        weights=weightDict3,
    )

    # make sure that the accuracy indicators stay the same when the different attributes are equally overweighted
    np.testing.assert_almost_equal(
        aggregation1.accuracy.rmse.values,
        aggregation2.accuracy.rmse.values,
        decimal=6,
    )
    np.testing.assert_almost_equal(
        aggregation1.accuracy.mae.values,
        aggregation2.accuracy.mae.values,
        decimal=6,
    )
    np.testing.assert_almost_equal(
        aggregation1.accuracy.rmse_duration.values,
        aggregation2.accuracy.rmse_duration.values,
        decimal=6,
    )

    # make sure that the RMSE of GHI is less while the other RMSEs are bigger, when GHI is overweighted
    np.testing.assert_array_less(
        aggregation3.accuracy.rmse["GHI"],
        aggregation1.accuracy.rmse["GHI"],
    )
    np.testing.assert_array_less(
        aggregation1.accuracy.rmse[["Load", "T", "Wind"]],
        aggregation3.accuracy.rmse[["Load", "T", "Wind"]],
    )


def test_uniform_weights_equal_no_weights():
    """Uniform scaling produces identical typicalPeriods and reconstructed data."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    agg_none = aggregate(
        raw.copy(),
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="hierarchical"),
    )
    agg_uniform = aggregate(
        raw.copy(),
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="hierarchical"),
        weights={"GHI": 1.0, "T": 1.0, "Wind": 1.0, "Load": 1.0},
    )

    tp_none = agg_none.cluster_representatives
    tp_uniform = agg_uniform.cluster_representatives

    pd.testing.assert_frame_equal(tp_none, tp_uniform)
    pd.testing.assert_frame_equal(agg_none.reconstructed, agg_uniform.reconstructed)


def test_reconstructed_within_original_range():
    """Reconstructed values stay within original data range with non-uniform weights."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    agg = aggregate(
        raw.copy(),
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="hierarchical"),
        weights={"GHI": 5.0, "T": 1.0, "Wind": 1.0, "Load": 1.0},
    )
    reconstructed = agg.reconstructed

    for col in raw.columns:
        assert reconstructed[col].min() >= raw[col].min() - 1e-6, (
            f"{col} reconstructed min below original"
        )
        assert reconstructed[col].max() <= raw[col].max() + 1e-6, (
            f"{col} reconstructed max above original"
        )


if __name__ == "__main__":
    test_weightingFactors()
