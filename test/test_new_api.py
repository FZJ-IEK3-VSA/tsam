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
        result = aggregate(sample_data, n_clusters=8)

        assert result.cluster_representatives is not None
        assert result.n_clusters == 8
        assert len(result.cluster_weights) == 8
        assert result.accuracy is not None

    def test_with_cluster_config(self, sample_data):
        """Test aggregation with custom cluster configuration."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            cluster=ClusterConfig(
                method="kmeans",
                representation="mean",
            ),
        )

        assert result.n_clusters == 8
        assert result.cluster_representatives is not None

    def test_with_segmentation(self, sample_data):
        """Test aggregation with segmentation."""
        result = aggregate(
            sample_data,
            n_clusters=8,
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
            n_clusters=8,
            extremes=ExtremeConfig(
                method="append",
                max_value=[col],
            ),
        )

        # With append, we should have more periods
        assert result.n_clusters >= 8

    def test_result_reconstruct(self, sample_data):
        """Test that reconstruction works."""
        result = aggregate(sample_data, n_clusters=8)
        reconstructed = result.reconstruct()

        assert reconstructed.shape == sample_data.shape

    def test_result_to_dict(self, sample_data):
        """Test that to_dict works."""
        result = aggregate(sample_data, n_clusters=8)
        data = result.to_dict()

        assert "cluster_representatives" in data
        assert "cluster_assignments" in data
        assert "accuracy" in data

    def test_accuracy_metrics(self, sample_data):
        """Test accuracy metrics."""
        result = aggregate(sample_data, n_clusters=8)

        assert hasattr(result.accuracy, "rmse")
        assert hasattr(result.accuracy, "mae")
        assert hasattr(result.accuracy, "rmse_duration")
        assert all(result.accuracy.rmse >= 0)


class TestValidation:
    """Tests for input validation."""

    def test_invalid_n_clusters(self, sample_data):
        """Test that invalid n_clusters raises error."""
        with pytest.raises(ValueError, match="n_clusters"):
            aggregate(sample_data, n_clusters=0)

        with pytest.raises(ValueError, match="n_clusters"):
            aggregate(sample_data, n_clusters=-1)

    def test_invalid_data_type(self):
        """Test that non-DataFrame data raises error."""
        with pytest.raises(TypeError, match="DataFrame"):
            aggregate([1, 2, 3], n_clusters=8)

    def test_invalid_extreme_columns(self, sample_data):
        """Test that invalid extreme columns raise error."""
        with pytest.raises(ValueError, match="not found"):
            aggregate(
                sample_data,
                n_clusters=8,
                extremes=ExtremeConfig(max_value=["nonexistent_column"]),
            )

    def test_invalid_weight_columns(self, sample_data):
        """Test that invalid weight columns raise error."""
        with pytest.raises(ValueError, match="not found"):
            aggregate(
                sample_data,
                n_clusters=8,
                cluster=ClusterConfig(weights={"nonexistent": 1.0}),
            )

    def test_segments_exceeds_timesteps(self, sample_data):
        """Test that too many segments raises error."""
        with pytest.raises(ValueError, match="n_segments"):
            aggregate(
                sample_data,
                n_clusters=8,
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


class TestAssignments:
    """Tests for the assignments property."""

    def test_assignments_basic(self, sample_data):
        """Test basic assignments DataFrame structure."""
        result = aggregate(sample_data, n_clusters=8)
        assignments = result.assignments

        # Should have same length as original data
        assert len(assignments) == len(sample_data)

        # Should have required columns
        assert "period_idx" in assignments.columns
        assert "timestep_idx" in assignments.columns
        assert "cluster_idx" in assignments.columns

        # Should not have segment_idx without segmentation
        assert "segment_idx" not in assignments.columns

        # Should have same index as original data
        assert assignments.index.equals(sample_data.index)

    def test_assignments_with_segmentation(self, sample_data):
        """Test assignments includes segment_idx when segmentation is used."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )
        assignments = result.assignments

        # Should have segment_idx column
        assert "segment_idx" in assignments.columns

        # Segment indices should be in valid range
        assert assignments["segment_idx"].min() >= 0
        assert assignments["segment_idx"].max() < 6

    def test_assignments_values_valid(self, sample_data):
        """Test that assignment values are within expected ranges."""
        result = aggregate(sample_data, n_clusters=8)
        assignments = result.assignments

        # Period indices should be sequential
        n_timesteps_per_period = result.n_timesteps_per_period
        n_original_periods = len(sample_data) // n_timesteps_per_period

        assert assignments["period_idx"].min() >= 0
        assert assignments["period_idx"].max() < n_original_periods

        # Timestep indices should be within period
        assert assignments["timestep_idx"].min() == 0
        assert assignments["timestep_idx"].max() == n_timesteps_per_period - 1

        # Cluster indices should be in valid range
        assert assignments["cluster_idx"].min() >= 0
        assert assignments["cluster_idx"].max() < result.n_clusters


class TestSegmentTransfer:
    """Tests for predefined segment transfer."""

    def test_segment_assignments_property(self, sample_data):
        """Test that segment_assignments property works."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        seg_assignments = result.segment_assignments

        # Should not be None when segmentation is used
        assert seg_assignments is not None

        # Should have one tuple per typical period
        assert len(seg_assignments) == result.n_clusters

        # Each inner tuple should sum to timesteps per period
        for period_assignments in seg_assignments:
            assert len(period_assignments) == result.n_timesteps_per_period

    def test_segment_durations_property(self, sample_data):
        """Test that segment_durations property works."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        seg_durations = result.segment_durations

        # Should not be None when segmentation is used
        assert seg_durations is not None

        # Should have one tuple per typical period
        assert len(seg_durations) == result.n_clusters

        # Each inner tuple should have n_segments elements
        for period_durations in seg_durations:
            assert len(period_durations) == 6

        # Durations should sum to timesteps per period
        for period_durations in seg_durations:
            assert sum(period_durations) == result.n_timesteps_per_period

    def test_segment_transfer(self, sample_data):
        """Test that segment assignments can be transferred to reproduce results."""
        # First aggregation
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        # Get segment assignments for transfer
        seg_assignments = result1.segment_assignments
        seg_durations = result1.segment_durations

        # Second aggregation with predefined segments and clusters
        result2 = aggregate(
            sample_data,
            n_clusters=8,
            cluster=ClusterConfig(
                predef_cluster_assignments=tuple(result1.cluster_assignments),
                predef_cluster_centers=tuple(result1.cluster_centers),
            ),
            segments=SegmentConfig(
                n_segments=6,
                predef_segment_assignments=seg_assignments,
                predef_segment_durations=seg_durations,
            ),
        )

        # Results should match
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_segment_properties_none_without_segmentation(self, sample_data):
        """Test that segment properties return None without segmentation."""
        result = aggregate(sample_data, n_clusters=8)

        assert result.segment_assignments is None
        assert result.segment_durations is None


class TestPredef:
    """Tests for the predefined parameter and PredefinedConfig."""

    def test_predef_property(self, sample_data):
        """Test that predefined property returns PredefinedConfig."""
        from tsam import PredefinedConfig

        result = aggregate(sample_data, n_clusters=8)
        predefined = result.predefined

        assert isinstance(predefined, PredefinedConfig)
        assert len(predefined.cluster_assignments) == len(result.cluster_assignments)

    def test_predef_transfer(self, sample_data):
        """Test transferring with predefined parameter."""
        result1 = aggregate(sample_data, n_clusters=8)

        # Transfer using predefined
        result2 = aggregate(sample_data, n_clusters=8, predefined=result1.predefined)

        # Results should match
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_predef_with_segments(self, sample_data):
        """Test predefined transfer with segmentation."""
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        # Transfer using predefined
        result2 = aggregate(sample_data, n_clusters=8, predefined=result1.predefined)

        # Should automatically apply segmentation
        assert result2.n_segments == 6
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_predef_from_dict(self, sample_data):
        """Test predefined transfer via dict (for JSON serialization)."""
        from tsam import PredefinedConfig

        result1 = aggregate(sample_data, n_clusters=8)

        # Convert to dict and back (simulates JSON save/load)
        predef_dict = result1.predefined.to_dict()
        predefined = PredefinedConfig.from_dict(predef_dict)

        result2 = aggregate(sample_data, n_clusters=8, predefined=predefined)

        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_predef_dict_directly(self, sample_data):
        """Test passing dict directly to predefined parameter."""
        result1 = aggregate(sample_data, n_clusters=8)

        # Pass dict directly (API accepts both)
        result2 = aggregate(
            sample_data, n_clusters=8, predefined=result1.predefined.to_dict()
        )

        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )


class TestSegmentConfigValidation:
    """Tests for SegmentConfig validation."""

    def test_predef_segment_assignments_requires_durations(self):
        """Test that predef_segment_assignments requires predef_segment_durations."""
        with pytest.raises(ValueError, match="predef_segment_durations"):
            SegmentConfig(
                n_segments=6,
                predef_segment_assignments=((0, 0, 1, 1, 2, 2),),
            )

    def test_predef_segment_durations_requires_order(self):
        """Test that predef_segment_durations requires predef_segment_assignments."""
        with pytest.raises(ValueError, match="predef_segment_assignments"):
            SegmentConfig(
                n_segments=6,
                predef_segment_durations=((2, 2, 2),),
            )

    def test_predef_segment_centers_requires_order(self):
        """Test that predef_segment_centers requires predef_segment_assignments."""
        with pytest.raises(ValueError, match="predef_segment_assignments"):
            SegmentConfig(
                n_segments=6,
                predef_segment_centers=((0, 2, 4),),
            )

    def test_valid_predef_segment_config(self):
        """Test that valid predefined segment config doesn't raise."""
        config = SegmentConfig(
            n_segments=3,
            predef_segment_assignments=((0, 0, 1, 1, 2, 2),),
            predef_segment_durations=((2, 2, 2),),
        )
        assert config.predef_segment_assignments is not None
        assert config.predef_segment_durations is not None
