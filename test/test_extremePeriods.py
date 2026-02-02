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
    """Test that 'replace' extreme method transfers correctly with apply().

    Applying to the same data must reproduce the original aggregation exactly.
    """
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = tsam.aggregate(
        raw,
        n_clusters=8,
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI"],
        ),
        preserve_column_means=False,
    )

    # Check that extreme_replacements is stored with correct structure
    assert result.clustering.extreme_replacements is not None
    assert len(result.clustering.extreme_replacements) > 0
    for replacement in result.clustering.extreme_replacements:
        assert len(replacement) == 3
        cluster_idx, column_name, period_idx = replacement
        assert isinstance(cluster_idx, int)
        assert isinstance(column_name, str)
        assert isinstance(period_idx, int)
        assert column_name == "GHI"

    # Apply to same data — must reproduce original cluster_representatives entirely
    transferred = result.clustering.apply(raw)
    pd.testing.assert_frame_equal(
        result.cluster_representatives,
        transferred.cluster_representatives,
    )


def test_replace_method_transfer_to_new_data():
    """Test that 'replace' method correctly transfers to different data."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

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

    transferred = result.clustering.apply(raw_modified)

    # For each replacement, the full profile must match the source period from raw_modified
    hours_per_period = 24
    for cluster_idx, col, period_idx in result.clustering.extreme_replacements:
        start = period_idx * hours_per_period
        end = start + hours_per_period
        expected = raw_modified[col].iloc[start:end].values
        actual = transferred.cluster_representatives.loc[cluster_idx, col].values
        np.testing.assert_array_almost_equal(actual, expected, decimal=5)


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

    assert result.clustering.extreme_replacements is not None

    # All requested extreme columns must appear in replacements
    columns_replaced = {r[1] for r in result.clustering.extreme_replacements}
    assert "GHI" in columns_replaced
    assert "T" in columns_replaced
    assert "Wind" in columns_replaced

    # Apply and verify each extreme is preserved
    transferred = result.clustering.apply(raw)

    np.testing.assert_almost_equal(
        raw["GHI"].max(),
        transferred.cluster_representatives["GHI"].max(),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        raw["T"].max(),
        transferred.cluster_representatives["T"].max(),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        raw["Wind"].min(),
        transferred.cluster_representatives["Wind"].min(),
        decimal=5,
    )


def test_replace_with_weights():
    """Test replace method with non-default weights exercises the weight fix."""
    from tsam.config import ClusterConfig

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = tsam.aggregate(
        raw,
        n_clusters=8,
        cluster=ClusterConfig(weights={"GHI": 5.0, "T": 0.5, "Wind": 1.0, "Load": 1.0}),
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI"],
        ),
        preserve_column_means=False,
    )

    # Apply to same data — must reproduce original cluster_representatives
    transferred = result.clustering.apply(raw)
    pd.testing.assert_frame_equal(
        result.cluster_representatives,
        transferred.cluster_representatives,
    )


def test_replace_preserves_full_extreme_profile():
    """Verify the entire extreme period profile matches raw input values."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = tsam.aggregate(
        raw,
        n_clusters=8,
        extremes=ExtremeConfig(
            method="replace",
            max_value=["GHI"],
        ),
        preserve_column_means=False,
    )

    hours_per_period = 24
    for cluster_idx, col, period_idx in result.clustering.extreme_replacements:
        start = period_idx * hours_per_period
        end = start + hours_per_period
        expected = raw[col].iloc[start:end].values
        actual = result.cluster_representatives.loc[cluster_idx, col].values
        np.testing.assert_array_almost_equal(actual, expected, decimal=5)


if __name__ == "__main__":
    test_extremePeriods()
