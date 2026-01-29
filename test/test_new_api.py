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

    def test_result_reconstructed(self, sample_data):
        """Test that reconstruction works."""
        result = aggregate(sample_data, n_clusters=8)
        reconstructed = result.reconstructed

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
                period_duration=24,
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

    def test_segment_assignments_and_durations_in_clustering(self, sample_data):
        """Test that segment_assignments and segment_durations are available via clustering."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        # Segment assignments
        seg_assignments = result.clustering.segment_assignments
        assert seg_assignments is not None
        assert len(seg_assignments) == 8
        for period_assignments in seg_assignments:
            assert len(period_assignments) == result.n_timesteps_per_period

        # Segment durations
        seg_durations = result.clustering.segment_durations
        assert seg_durations is not None
        assert len(seg_durations) == 8
        for period_durations in seg_durations:
            assert len(period_durations) == 6
            assert sum(period_durations) == result.n_timesteps_per_period

    def test_segment_transfer(self, sample_data):
        """Test that segment assignments can be transferred to reproduce results."""
        # First aggregation
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        # Transfer using ClusteringResult.apply()
        result2 = result1.clustering.apply(sample_data)

        # Results should match
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_segment_properties_none_without_segmentation(self, sample_data):
        """Test that segment properties return None without segmentation."""
        result = aggregate(sample_data, n_clusters=8)

        assert result.clustering.segment_assignments is None
        assert result.clustering.segment_durations is None


class TestClusteringResult:
    """Tests for ClusteringResult and apply()."""

    def test_clustering_property_and_apply(self, sample_data):
        """Test that clustering property returns ClusteringResult and can be applied."""
        from tsam import ClusteringResult

        result = aggregate(sample_data, n_clusters=8)
        clustering = result.clustering

        assert isinstance(clustering, ClusteringResult)
        assert len(clustering.cluster_assignments) == len(result.cluster_assignments)
        assert clustering.n_clusters == result.n_clusters

        # Apply clustering to same data
        result2 = clustering.apply(sample_data)
        pd.testing.assert_frame_equal(
            result.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_clustering_apply_with_segments(self, sample_data):
        """Test clustering apply with segmentation."""
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        # Apply clustering (includes segment info)
        result2 = result1.clustering.apply(sample_data)

        # Should automatically apply segmentation
        assert result2.n_segments == 6
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_clustering_from_dict(self, sample_data):
        """Test clustering transfer via dict (for JSON serialization)."""
        from tsam import ClusteringResult

        result1 = aggregate(sample_data, n_clusters=8)

        # Convert to dict and back (simulates JSON save/load)
        clustering_dict = result1.clustering.to_dict()
        clustering = ClusteringResult.from_dict(clustering_dict)

        result2 = clustering.apply(sample_data)

        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_clustering_json_roundtrip(self, sample_data, tmp_path):
        """Test saving and loading clustering to/from JSON file."""
        from tsam import ClusteringResult

        result1 = aggregate(sample_data, n_clusters=8)

        # Save to file
        json_path = tmp_path / "clustering.json"
        result1.clustering.to_json(str(json_path))

        # Load and apply
        clustering = ClusteringResult.from_json(str(json_path))
        result2 = clustering.apply(sample_data)

        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_clustering_includes_period_duration(self, sample_data):
        """Test that clustering includes period_duration."""
        result = aggregate(sample_data, n_clusters=8, period_duration=24)
        clustering = result.clustering

        assert clustering.period_duration == 24
        assert clustering.n_clusters == 8
        assert clustering.n_original_periods == len(result.cluster_assignments)

    def test_clustering_period_duration_preserved_in_json(self, sample_data, tmp_path):
        """Test that period_duration is preserved through JSON serialization."""
        from tsam import ClusteringResult

        result = aggregate(sample_data, n_clusters=8, period_duration=24)

        # Save and load
        json_path = tmp_path / "clustering.json"
        result.clustering.to_json(str(json_path))
        clustering = ClusteringResult.from_json(str(json_path))

        assert clustering.period_duration == 24


class TestDeterministicPreservation:
    """Tests for deterministic clustering preservation through save/load cycles."""

    def test_transfer_fields_preserved_in_json(self, sample_data, tmp_path):
        """Test that all transfer fields are preserved through JSON roundtrip."""
        from tsam import ClusteringResult

        result = aggregate(
            sample_data,
            n_clusters=8,
            cluster=ClusterConfig(method="kmeans", representation="mean"),
            segments=SegmentConfig(n_segments=6, representation="medoid"),
            preserve_column_means=False,
        )

        # Save and load
        json_path = tmp_path / "clustering.json"
        result.clustering.to_json(str(json_path))
        clustering = ClusteringResult.from_json(str(json_path))

        # Verify transfer fields
        assert clustering.preserve_column_means is False
        assert clustering.representation == "mean"
        assert clustering.segment_representation == "medoid"
        assert clustering.period_duration == 24
        assert len(clustering.cluster_assignments) == len(result.cluster_assignments)
        assert clustering.segment_assignments is not None
        assert clustering.segment_durations is not None

    def test_representation_method_deterministic(self, sample_data, tmp_path):
        """Test that representation method produces same results when reapplied."""
        from tsam import ClusteringResult

        # Test with mean representation
        result_mean = aggregate(
            sample_data,
            n_clusters=8,
            cluster=ClusterConfig(representation="mean"),
        )

        # Save, load, and reapply
        json_path = tmp_path / "clustering_mean.json"
        result_mean.clustering.to_json(str(json_path))
        clustering = ClusteringResult.from_json(str(json_path))
        result_reapplied = clustering.apply(sample_data)

        # Results should be identical
        pd.testing.assert_frame_equal(
            result_mean.cluster_representatives,
            result_reapplied.cluster_representatives,
        )

    def test_apply_to_different_data(self, sample_data):
        """Test applying clustering from subset to full data."""
        # Cluster on single column
        wind_only = sample_data[["Wind"]]
        result_wind = aggregate(wind_only, n_clusters=8)

        # Apply to full data
        result_full = result_wind.clustering.apply(sample_data)

        # Cluster assignments should be identical
        assert list(result_wind.cluster_assignments) == list(
            result_full.cluster_assignments
        )

        # Full result should have all columns
        assert list(result_full.cluster_representatives.columns) == list(
            sample_data.columns
        )

    def test_segmentation_preserved_through_json(self, sample_data, tmp_path):
        """Test that segmentation is fully preserved through JSON roundtrip."""
        from tsam import ClusteringResult

        result1 = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        # Save and load
        json_path = tmp_path / "clustering_seg.json"
        result1.clustering.to_json(str(json_path))
        clustering = ClusteringResult.from_json(str(json_path))

        # Apply to same data
        result2 = clustering.apply(sample_data)

        # Segmentation structure should match
        assert result2.n_segments == result1.n_segments
        assert result2.segment_durations == result1.segment_durations

        # Typical periods should be identical
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_preserve_column_means_setting_preserved(self, sample_data, tmp_path):
        """Test that preserve_column_means=False produces different results than preserve_column_means=True."""
        from tsam import ClusteringResult

        # With preserve_column_means=False
        result_no_preserve_column_means = aggregate(
            sample_data, n_clusters=8, preserve_column_means=False
        )

        # Save, load, apply
        json_path = tmp_path / "clustering_no_preserve_column_means.json"
        result_no_preserve_column_means.clustering.to_json(str(json_path))
        clustering = ClusteringResult.from_json(str(json_path))
        result_reapplied = clustering.apply(sample_data)

        # Should preserve preserve_column_means=False behavior
        assert clustering.preserve_column_means is False
        pd.testing.assert_frame_equal(
            result_no_preserve_column_means.cluster_representatives,
            result_reapplied.cluster_representatives,
        )


class TestSegmentConfigValidation:
    """Tests for SegmentConfig validation."""

    def test_n_segments_must_be_positive(self):
        """Test that n_segments must be positive."""
        with pytest.raises(ValueError, match="n_segments must be positive"):
            SegmentConfig(n_segments=0)

        with pytest.raises(ValueError, match="n_segments must be positive"):
            SegmentConfig(n_segments=-1)

    def test_valid_segment_config(self):
        """Test that valid segment config doesn't raise."""
        config = SegmentConfig(n_segments=6)
        assert config.n_segments == 6
        assert config.representation == "mean"


class TestSegmentCenters:
    """Tests for segment center preservation."""

    def test_segment_centers_with_medoid(self, sample_data):
        """Test that segment centers are captured with medoid representation."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6, representation="medoid"),
        )

        segment_centers = result.clustering.segment_centers

        # Should not be None for medoid representation
        assert segment_centers is not None

        # Should have one tuple per typical period
        assert len(segment_centers) == result.n_clusters

        # Each inner tuple should have n_segments elements
        for period_centers in segment_centers:
            assert len(period_centers) == 6

        # All center indices should be valid (within timesteps per period)
        for period_centers in segment_centers:
            for idx in period_centers:
                assert 0 <= idx < result.n_timesteps_per_period

    def test_segment_centers_none_with_mean(self, sample_data):
        """Test that segment centers are None with mean representation."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6, representation="mean"),
        )

        # Mean representation doesn't have center indices
        assert result.clustering.segment_centers is None

    def test_segment_centers_preserved_in_json(self, sample_data, tmp_path):
        """Test that segment centers are preserved through JSON roundtrip."""
        from tsam import ClusteringResult

        result1 = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6, representation="medoid"),
        )

        # Save and load
        json_path = tmp_path / "clustering_seg_centers.json"
        result1.clustering.to_json(str(json_path))
        clustering = ClusteringResult.from_json(str(json_path))

        # Segment centers should be preserved
        assert clustering.segment_centers is not None
        assert clustering.segment_centers == result1.clustering.segment_centers

        # Apply and verify results are identical
        result2 = clustering.apply(sample_data)
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )

    def test_segment_centers_deterministic_transfer(self, sample_data):
        """Test that segment centers produce deterministic results when reapplied."""
        # First aggregation with medoid segments
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6, representation="medoid"),
        )

        # Apply clustering (which uses predefined segment centers)
        result2 = result1.clustering.apply(sample_data)

        # Results should be identical
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
        )


class TestDurationParsing:
    """Tests for pandas Timedelta string parsing in duration parameters."""

    def test_period_duration_string(self, sample_data):
        """Test that period_duration accepts pandas Timedelta strings."""
        # All these should produce equivalent results
        result_int = aggregate(sample_data, n_clusters=8, period_duration=24)
        result_float = aggregate(sample_data, n_clusters=8, period_duration=24.0)
        result_hours = aggregate(sample_data, n_clusters=8, period_duration="24h")
        result_day = aggregate(sample_data, n_clusters=8, period_duration="1d")

        # All should produce same typical periods
        pd.testing.assert_frame_equal(
            result_int.cluster_representatives,
            result_hours.cluster_representatives,
        )
        pd.testing.assert_frame_equal(
            result_int.cluster_representatives,
            result_day.cluster_representatives,
        )
        pd.testing.assert_frame_equal(
            result_int.cluster_representatives,
            result_float.cluster_representatives,
        )

    def test_temporal_resolution_string(self, sample_data):
        """Test that temporal_resolution accepts pandas Timedelta strings."""
        # Should be equivalent: 1.0 hours and '1h'
        result_float = aggregate(
            sample_data, n_clusters=8, period_duration=24, temporal_resolution=1.0
        )
        result_str = aggregate(
            sample_data, n_clusters=8, period_duration="24h", temporal_resolution="1h"
        )

        pd.testing.assert_frame_equal(
            result_float.cluster_representatives,
            result_str.cluster_representatives,
        )

    def test_invalid_duration_string(self, sample_data):
        """Test that invalid duration strings raise clear errors."""
        with pytest.raises(ValueError, match="period_duration"):
            aggregate(sample_data, n_clusters=8, period_duration="invalid")

    def test_invalid_duration_type(self, sample_data):
        """Test that wrong types raise TypeError."""
        with pytest.raises(TypeError, match="period_duration"):
            aggregate(sample_data, n_clusters=8, period_duration=[24])

    def test_negative_duration(self, sample_data):
        """Test that negative durations raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            aggregate(sample_data, n_clusters=8, period_duration=-24)
