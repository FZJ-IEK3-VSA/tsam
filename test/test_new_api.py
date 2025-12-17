"""Tests for the new simplified API."""

import pandas as pd
import pytest

import tsam
from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, SegmentConfig, aggregate


@pytest.fixture
def sample_data():
    """Load sample time series data."""
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


class TestAggregate:
    """Tests for the aggregate() function."""

    def test_basic_aggregation(self, sample_data):
        """Test basic aggregation with minimal parameters."""
        result = aggregate(sample_data, n_periods=8)

        assert result.typical_periods is not None
        assert result.n_periods == 8
        assert len(result.cluster_weights) == 8
        assert result.accuracy is not None

    def test_with_cluster_config(self, sample_data):
        """Test aggregation with custom cluster configuration."""
        result = aggregate(
            sample_data,
            n_periods=8,
            cluster=ClusterConfig(
                method="kmeans",
                representation="mean",
            ),
        )

        assert result.n_periods == 8
        assert result.typical_periods is not None

    def test_with_segmentation(self, sample_data):
        """Test aggregation with segmentation."""
        result = aggregate(
            sample_data,
            n_periods=8,
            segments=SegmentConfig(n_segments=12),
        )

        assert result.n_segments == 12
        assert result.segment_durations is not None

    def test_with_extremes(self, sample_data):
        """Test aggregation with extreme period preservation."""
        # Get first column name for extreme config
        col = sample_data.columns[0]

        result = aggregate(
            sample_data,
            n_periods=8,
            extremes=ExtremeConfig(
                method="append",
                max_timesteps=[col],
            ),
        )

        # With append, we should have more periods
        assert result.n_periods >= 8

    def test_result_reconstruct(self, sample_data):
        """Test that reconstruction works."""
        result = aggregate(sample_data, n_periods=8)
        reconstructed = result.reconstruct()

        assert reconstructed.shape == sample_data.shape

    def test_result_to_dict(self, sample_data):
        """Test that to_dict works."""
        result = aggregate(sample_data, n_periods=8)
        data = result.to_dict()

        assert "typical_periods" in data
        assert "cluster_assignments" in data
        assert "accuracy" in data

    def test_accuracy_metrics(self, sample_data):
        """Test accuracy metrics."""
        result = aggregate(sample_data, n_periods=8)

        assert hasattr(result.accuracy, "rmse")
        assert hasattr(result.accuracy, "mae")
        assert hasattr(result.accuracy, "rmse_duration")
        assert all(result.accuracy.rmse >= 0)


class TestValidation:
    """Tests for input validation."""

    def test_invalid_n_periods(self, sample_data):
        """Test that invalid n_periods raises error."""
        with pytest.raises(ValueError, match="n_periods"):
            aggregate(sample_data, n_periods=0)

        with pytest.raises(ValueError, match="n_periods"):
            aggregate(sample_data, n_periods=-1)

    def test_invalid_data_type(self):
        """Test that non-DataFrame data raises error."""
        with pytest.raises(TypeError, match="DataFrame"):
            aggregate([1, 2, 3], n_periods=8)

    def test_invalid_extreme_columns(self, sample_data):
        """Test that invalid extreme columns raise error."""
        with pytest.raises(ValueError, match="not found"):
            aggregate(
                sample_data,
                n_periods=8,
                extremes=ExtremeConfig(max_timesteps=["nonexistent_column"]),
            )

    def test_invalid_weight_columns(self, sample_data):
        """Test that invalid weight columns raise error."""
        with pytest.raises(ValueError, match="not found"):
            aggregate(
                sample_data,
                n_periods=8,
                cluster=ClusterConfig(weights={"nonexistent": 1.0}),
            )

    def test_segments_exceeds_timesteps(self, sample_data):
        """Test that too many segments raises error."""
        with pytest.raises(ValueError, match="n_segments"):
            aggregate(
                sample_data,
                n_periods=8,
                period_hours=24,
                segments=SegmentConfig(n_segments=100),
            )


class TestClusterConfig:
    """Tests for ClusterConfig."""

    def test_default_representation(self):
        """Test that default representation is set correctly per method."""
        assert ClusterConfig(method="kmeans").get_representation() == "mean"
        assert ClusterConfig(method="hierarchical").get_representation() == "medoid"
        assert ClusterConfig(method="kmaxoids").get_representation() == "maxoid"

    def test_explicit_representation(self):
        """Test that explicit representation overrides default."""
        config = ClusterConfig(method="kmeans", representation="medoid")
        assert config.get_representation() == "medoid"


class TestImports:
    """Test that imports work correctly."""

    def test_top_level_imports(self):
        """Test that all exports are accessible from tsam."""
        assert hasattr(tsam, "aggregate")
        assert hasattr(tsam, "ClusterConfig")
        assert hasattr(tsam, "SegmentConfig")
        assert hasattr(tsam, "ExtremeConfig")
        assert hasattr(tsam, "AggregationResult")
        assert hasattr(tsam, "AccuracyMetrics")
        # Legacy
        assert hasattr(tsam, "TimeSeriesAggregation")

    def test_version(self):
        """Test that version is defined."""
        assert hasattr(tsam, "__version__")
        assert tsam.__version__ == "3.0.0"
