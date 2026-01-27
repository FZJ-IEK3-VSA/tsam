import numpy as np
import pandas as pd
import pytest

import tsam
import tsam.timeseriesaggregation as tsam_legacy
from conftest import TESTDATA_CSV
from tsam.config import ExtremeConfig


def test_extremePeriods():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="new_cluster_center",
        addPeakMax=["GHI"],
    )

    aggregation2 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="append",
        addPeakMax=["GHI"],
    )

    aggregation3 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="replace_cluster_center",
        addPeakMax=["GHI"],
    )

    # make sure that the RMSE for new cluster centers (reassigning points to the exxtreme point if the distance to it is
    # smaller)is bigger than for appending just one extreme period
    np.testing.assert_array_less(
        aggregation1.accuracyIndicators().loc["GHI", "RMSE"],
        aggregation2.accuracyIndicators().loc["GHI", "RMSE"],
    )

    # make sure that the RMSE for appending the extreme period is smaller than for replacing the cluster center by the
    # extreme period (conservative assumption)
    np.testing.assert_array_less(
        aggregation2.accuracyIndicators().loc["GHI", "RMSE"],
        aggregation3.accuracyIndicators().loc["GHI", "RMSE"],
    )

    # check if addMeanMax and addMeanMin are working
    aggregation4 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="append",
        addMeanMax=["GHI"],
        addMeanMin=["GHI"],
    )

    origData = aggregation4.predictOriginalData()

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


def test_include_in_count_exact_clusters_append():
    """Final n_clusters equals requested when include_in_count=True with append method."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 10
    result = tsam.aggregate(
        raw,
        n_clusters=n_clusters,
        extremes=ExtremeConfig(
            method="append",
            max_value=["GHI"],
            min_value=["T"],
            include_in_count=True,
        ),
    )

    # With include_in_count=True, final cluster count should equal n_clusters
    assert len(result.cluster_weights) == n_clusters


def test_include_in_count_exact_clusters_new_cluster():
    """Final n_clusters equals requested when include_in_count=True with new_cluster method."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 10
    result = tsam.aggregate(
        raw,
        n_clusters=n_clusters,
        extremes=ExtremeConfig(
            method="new_cluster",
            max_value=["GHI"],
            include_in_count=True,
        ),
    )

    # With include_in_count=True, final cluster count should equal n_clusters
    assert len(result.cluster_weights) == n_clusters


def test_include_in_count_false_adds_extra_clusters():
    """Default behavior: extremes are added on top of n_clusters."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 10
    result = tsam.aggregate(
        raw,
        n_clusters=n_clusters,
        extremes=ExtremeConfig(
            method="append",
            max_value=["GHI"],
            min_value=["T"],
            include_in_count=False,  # Default
        ),
    )

    # With include_in_count=False (default), extremes are added on top
    # So final count should be > n_clusters (n_clusters + n_extremes)
    assert len(result.cluster_weights) > n_clusters


def test_include_in_count_validation_error():
    """Error if n_clusters <= n_extremes when include_in_count=True."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    with pytest.raises(ValueError, match="must be greater than"):
        tsam.aggregate(
            raw,
            n_clusters=2,
            extremes=ExtremeConfig(
                max_value=["GHI", "T", "Wind"],  # 3 extremes
                include_in_count=True,
            ),
        )


def test_include_in_count_warns_with_replace():
    """include_in_count=True with replace method emits a warning."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    with pytest.warns(UserWarning, match="has no effect"):
        tsam.aggregate(
            raw,
            n_clusters=10,
            extremes=ExtremeConfig(
                method="replace",
                max_value=["GHI"],
                include_in_count=True,
            ),
        )


def test_include_in_count_preserves_extremes():
    """Extreme values are still preserved with include_in_count=True."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = tsam.aggregate(
        raw,
        n_clusters=10,
        extremes=ExtremeConfig(
            method="append",
            max_value=["GHI"],
            include_in_count=True,
        ),
        preserve_column_means=False,  # Don't rescale to check raw extreme preservation
    )

    # The maximum GHI value should be preserved in the typical periods
    orig_max = raw["GHI"].max()
    typical_max = result.cluster_representatives["GHI"].max()

    np.testing.assert_almost_equal(orig_max, typical_max, decimal=5)


def test_include_in_count_serialization():
    """ExtremeConfig with include_in_count serializes correctly."""
    config = ExtremeConfig(
        method="append",
        max_value=["Load"],
        include_in_count=True,
    )

    d = config.to_dict()
    assert d["include_in_count"] is True

    config2 = ExtremeConfig.from_dict(d)
    assert config2.include_in_count is True


def test_include_in_count_default_false():
    """Default value of include_in_count is False."""
    config = ExtremeConfig(max_value=["Load"])
    assert config.include_in_count is False

    # to_dict should not include it when False
    d = config.to_dict()
    assert "include_in_count" not in d


if __name__ == "__main__":
    test_extremePeriods()
