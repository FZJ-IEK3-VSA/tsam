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


def test_replace_method_transfer():
    """Test that 'replace' extreme method transfers correctly with apply()."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # Aggregate with replace method
    result = tsam.aggregate(
        raw,
        n_clusters=8,
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI"],
        ),
        preserve_column_means=False,
    )

    # Check that extreme_replacements is stored
    assert result.clustering.extreme_replacements is not None
    assert len(result.clustering.extreme_replacements) > 0

    # Verify the replacement tuple structure: (cluster_idx, column_name, period_idx)
    for replacement in result.clustering.extreme_replacements:
        assert len(replacement) == 3
        cluster_idx, column_name, period_idx = replacement
        assert isinstance(cluster_idx, int)
        assert isinstance(column_name, str)
        assert isinstance(period_idx, int)
        assert column_name == "GHI"

    # Apply to the same data - should get identical results
    transferred = result.clustering.apply(raw)

    # The extreme values should be preserved
    orig_max = raw["GHI"].max()
    typical_max = transferred.cluster_representatives["GHI"].max()
    np.testing.assert_almost_equal(orig_max, typical_max, decimal=5)


def test_replace_method_transfer_to_new_data():
    """Test that 'replace' method correctly transfers to different data."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # Aggregate with replace method
    result = tsam.aggregate(
        raw,
        n_clusters=8,
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI"],
        ),
        preserve_column_means=False,
    )

    # Create modified data with different values
    raw_modified = raw.copy()
    raw_modified["GHI"] = raw_modified["GHI"] * 1.5

    # Apply to modified data
    transferred = result.clustering.apply(raw_modified)

    # The extreme values should come from the modified data
    # (the extreme period's GHI values should be 1.5x the original)
    orig_max = raw_modified["GHI"].max()
    typical_max = transferred.cluster_representatives["GHI"].max()
    np.testing.assert_almost_equal(orig_max, typical_max, decimal=5)


def test_replace_method_missing_column_warning():
    """Test that missing columns during transfer emit a warning."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # Aggregate with replace method
    result = tsam.aggregate(
        raw,
        n_clusters=8,
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI"],
        ),
        preserve_column_means=False,
    )

    # Create data without the GHI column
    raw_no_ghi = raw.drop(columns=["GHI"])

    # Apply should warn about missing column
    with pytest.warns(UserWarning, match="Column 'GHI' not found"):
        result.clustering.apply(raw_no_ghi)


def test_extreme_replacements_serialization():
    """Test that extreme_replacements serializes and deserializes correctly."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = tsam.aggregate(
        raw,
        n_clusters=8,
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI"],
        ),
    )

    # Serialize and deserialize
    d = result.clustering.to_dict()
    assert "extreme_replacements" in d
    assert len(d["extreme_replacements"]) > 0

    # Check structure
    for r in d["extreme_replacements"]:
        assert len(r) == 3

    # Deserialize
    from tsam.config import ClusteringResult

    restored = ClusteringResult.from_dict(d)

    assert restored.extreme_replacements == result.clustering.extreme_replacements


def test_replace_multiple_columns():
    """Test replace method with multiple extreme columns."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = tsam.aggregate(
        raw,
        n_clusters=8,
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI", "T"],
            min_value=["Wind"],
        ),
        preserve_column_means=False,
    )

    # Should have replacements for each extreme
    assert result.clustering.extreme_replacements is not None

    columns_replaced = {r[1] for r in result.clustering.extreme_replacements}
    assert (
        "GHI" in columns_replaced
        or "T" in columns_replaced
        or "Wind" in columns_replaced
    )

    # Apply and verify extremes are preserved
    transferred = result.clustering.apply(raw)

    # Check that max GHI is preserved
    np.testing.assert_almost_equal(
        raw["GHI"].max(),
        transferred.cluster_representatives["GHI"].max(),
        decimal=5,
    )


if __name__ == "__main__":
    test_extremePeriods()
