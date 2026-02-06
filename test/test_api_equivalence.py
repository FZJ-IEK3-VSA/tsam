"""Tests to verify new API produces identical results to old API."""

import numpy as np
import pandas as pd
import pytest

# Old API
import tsam.hyperparametertuning as old_tune
import tsam.timeseriesaggregation as old_tsam
from conftest import TESTDATA_CSV

# New API
from tsam import (
    ClusterConfig,
    ClusteringResult,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    SegmentConfig,
    aggregate,
)
from tsam.tuning import (
    find_clusters_for_reduction,
    find_optimal_combination,
    find_pareto_front,
    find_segments_for_reduction,
)


@pytest.fixture
def sample_data():
    """Load sample time series data."""
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


@pytest.fixture
def small_data(sample_data):
    """Small dataset for faster tests."""
    return sample_data.iloc[:240, :]


class TestAggregateEquivalence:
    """Tests that new aggregate() produces identical results to TimeSeriesAggregation."""

    def test_hierarchical_default(self, sample_data):
        """Test hierarchical clustering with default settings."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
        )

        # Compare typical periods (check_names=False: old uses 'TimeStep', new uses 'timestep')
        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

        # Compare cluster assignments
        np.testing.assert_array_equal(
            old_agg.cluster_order, new_result.cluster_assignments
        )

        # Compare accuracy
        old_accuracy = old_agg.accuracy_indicators()
        np.testing.assert_allclose(
            old_accuracy["RMSE"].values,
            new_result.accuracy.rmse.values,
            rtol=1e-10,
        )

    def test_kmeans(self, sample_data):
        """Test k-means clustering."""
        # Set seed for deterministic k-means results
        np.random.seed(42)
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="k_means",
        )
        old_result = old_agg.create_typical_periods()

        # Reset seed to get same random state for new API
        np.random.seed(42)
        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="kmeans"),
        )

        # With same seed, results should be identical
        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

        old_accuracy = old_agg.accuracy_indicators()
        np.testing.assert_allclose(
            old_accuracy["RMSE"].values,
            new_result.accuracy.rmse.values,
            rtol=1e-5,
        )

    def test_hierarchical_with_medoid(self, sample_data):
        """Test hierarchical with medoid representation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
            representation_method="medoidRepresentation",
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", representation="medoid"),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_with_weights(self, sample_data):
        """Test weighted clustering."""
        weights = {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0}

        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
            weight_dict=weights,
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", weights=weights),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_with_segmentation(self, sample_data):
        """Test with segmentation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
            segmentation=True,
            no_segments=12,
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
            segments=SegmentConfig(n_segments=12),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_with_duration_curves(self, sample_data):
        """Test duration curve representation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
            representation_method="durationRepresentation",
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", representation="distribution"),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_with_extremes_append(self, sample_data):
        """Test extreme period handling with append method."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
            extreme_period_method="append",
            add_peak_max=["Load"],
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
            extremes=ExtremeConfig(method="append", max_value=["Load"]),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_contiguous_clustering(self, sample_data):
        """Test contiguous (adjacent periods) clustering."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="adjacent_periods",
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="contiguous"),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_rescale_off(self, sample_data):
        """Test with rescaling disabled."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
            rescale_cluster_periods=False,
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
            preserve_column_means=False,
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_distribution_minmax_representation(self, sample_data):
        """Test distribution with min/max representation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
            representation_method="distributionAndMinMaxRepresentation",
        )
        old_result = old_agg.create_typical_periods()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(
                method="hierarchical", representation="distribution_minmax"
            ),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )


class TestRepresentationObjects:
    """Tests for typed representation objects (Distribution, MinMaxMean)."""

    def test_distribution_global_equivalence(self, sample_data):
        """Test Distribution(scope='global') matches old distributionPeriodWise=False."""
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="distributionRepresentation",
            distributionPeriodWise=False,
            rescaleClusterPeriods=False,
        )
        old_result = old_agg.createTypicalPeriods()

        new_result = aggregate(
            sample_data,
            n_clusters=8,
            preserve_column_means=False,
            cluster=ClusterConfig(
                method="hierarchical",
                representation=Distribution(scope="global"),
            ),
        )

        pd.testing.assert_frame_equal(
            old_result, new_result.cluster_representatives, check_names=False
        )

    def test_distribution_cluster_equivalence(self, sample_data):
        """Test Distribution(scope='cluster') matches old distributionPeriodWise=True."""
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="distributionRepresentation",
            distributionPeriodWise=True,
            rescaleClusterPeriods=False,
        )
        old_result = old_agg.createTypicalPeriods()

        new_result = aggregate(
            sample_data,
            n_clusters=8,
            preserve_column_means=False,
            cluster=ClusterConfig(
                method="hierarchical",
                representation=Distribution(scope="cluster"),
            ),
        )

        pd.testing.assert_frame_equal(
            old_result, new_result.cluster_representatives, check_names=False
        )

    def test_distribution_minmax_global_equivalence(self, sample_data):
        """Test Distribution(scope='global', preserve_minmax=True) matches old API."""
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="distributionAndMinMaxRepresentation",
            distributionPeriodWise=False,
            rescaleClusterPeriods=False,
        )
        old_result = old_agg.createTypicalPeriods()

        new_result = aggregate(
            sample_data,
            n_clusters=8,
            preserve_column_means=False,
            cluster=ClusterConfig(
                method="hierarchical",
                representation=Distribution(scope="global", preserve_minmax=True),
            ),
        )

        pd.testing.assert_frame_equal(
            old_result, new_result.cluster_representatives, check_names=False
        )

    def test_minmaxmean_equivalence(self, sample_data):
        """Test MinMaxMean matches old representationDict."""
        rep_dict = {"GHI": "max", "T": "min", "Wind": "mean", "Load": "min"}
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="minmaxmeanRepresentation",
            representationDict=rep_dict,
            rescaleClusterPeriods=False,
        )
        old_result = old_agg.createTypicalPeriods()

        new_result = aggregate(
            sample_data,
            n_clusters=8,
            preserve_column_means=False,
            cluster=ClusterConfig(
                method="hierarchical",
                representation=MinMaxMean(
                    max_columns=["GHI"], min_columns=["T", "Load"]
                ),
            ),
        )

        pd.testing.assert_frame_equal(
            old_result, new_result.cluster_representatives, check_names=False
        )

    def test_segment_distribution_global_equivalence(self, sample_data):
        """Test Distribution(scope='global') for segments matches old API."""
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="medoidRepresentation",
            segmentation=True,
            noSegments=8,
            segmentRepresentationMethod="distributionRepresentation",
            distributionPeriodWise=False,
            rescaleClusterPeriods=False,
        )
        old_result = old_agg.createTypicalPeriods()

        new_result = aggregate(
            sample_data,
            n_clusters=8,
            preserve_column_means=False,
            cluster=ClusterConfig(method="hierarchical", representation="medoid"),
            segments=SegmentConfig(
                n_segments=8, representation=Distribution(scope="global")
            ),
        )

        pd.testing.assert_frame_equal(
            old_result, new_result.cluster_representatives, check_names=False
        )

    def test_segment_distribution_global_roundtrip(self, sample_data, tmp_path):
        """Test Distribution(scope='global') segment transfer via JSON roundtrip."""
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            preserve_column_means=False,
            cluster=ClusterConfig(method="hierarchical", representation="medoid"),
            segments=SegmentConfig(
                n_segments=8, representation=Distribution(scope="global")
            ),
        )

        # JSON roundtrip
        json_path = tmp_path / "clustering.json"
        result1.clustering.to_json(str(json_path))
        loaded = ClusteringResult.from_json(str(json_path))
        result2 = loaded.apply(sample_data)

        pd.testing.assert_frame_equal(
            result1.cluster_representatives, result2.cluster_representatives
        )

    def test_representation_object_json_roundtrip(self, sample_data, tmp_path):
        """Test that typed representation objects survive JSON roundtrip."""
        result = aggregate(
            sample_data,
            n_clusters=8,
            preserve_column_means=False,
            cluster=ClusterConfig(
                method="hierarchical",
                representation=MinMaxMean(
                    max_columns=["GHI"], min_columns=["T", "Load"]
                ),
            ),
        )

        json_path = tmp_path / "clustering.json"
        result.clustering.to_json(str(json_path))
        loaded = ClusteringResult.from_json(str(json_path))

        assert isinstance(loaded.representation, MinMaxMean)
        assert loaded.representation.max_columns == ["GHI"]
        assert loaded.representation.min_columns == ["T", "Load"]

        result2 = loaded.apply(sample_data)
        pd.testing.assert_frame_equal(
            result.cluster_representatives, result2.cluster_representatives
        )


class TestTuningEquivalence:
    """Tests that new tuning functions produce equivalent results to old API."""

    def test_find_clusters_for_reduction(self):
        """Test find_clusters_for_reduction matches old function."""
        test_cases = [
            (100, 10, 0.5),
            (101, 10, 0.5),
            (101, 2, 0.5),
            (8760, 24, 0.01),
        ]

        for n_timesteps, n_segments, data_reduction in test_cases:
            old_result = old_tune.get_no_periods_for_data_reduction(
                n_timesteps, n_segments, data_reduction
            )
            new_result = find_clusters_for_reduction(
                n_timesteps, n_segments, data_reduction
            )
            assert old_result == new_result, (
                f"Mismatch for ({n_timesteps}, {n_segments}, {data_reduction}): "
                f"old={old_result}, new={new_result}"
            )

    def test_find_segments_for_reduction(self):
        """Test find_segments_for_reduction matches old function."""
        test_cases = [
            (100, 10, 0.5),
            (8760, 8, 0.01),
        ]

        for n_timesteps, n_clusters, data_reduction in test_cases:
            old_result = old_tune.get_no_segments_for_data_reduction(
                n_timesteps, n_clusters, data_reduction
            )
            new_result = find_segments_for_reduction(
                n_timesteps, n_clusters, data_reduction
            )
            assert old_result == new_result

    def test_find_optimal_combination(self, sample_data):
        """Test find_optimal_combination matches old identify_optimal_segment_period_combination."""
        data_reduction = 0.01
        col = "Wind"
        data = sample_data[[col]]

        # Old API with default settings to match new API defaults
        old_tuner = old_tune.HyperTunedAggregations(
            old_tsam.TimeSeriesAggregation(
                data,
                hours_per_period=24,
                cluster_method="hierarchical",
                representation_method="durationRepresentation",
                # Use defaults: distribution_period_wise=True, rescale_cluster_periods=True
                segmentation=True,
            )
        )
        old_segments, old_periods, old_rmse = (
            old_tuner.identify_optimal_segment_period_combination(
                data_reduction=data_reduction
            )
        )

        # New API
        new_result = find_optimal_combination(
            data,
            data_reduction=data_reduction,
            period_duration=24,
            cluster=ClusterConfig(
                method="hierarchical",
                representation="distribution",
            ),
            show_progress=False,
        )

        # Results should match
        assert new_result.n_clusters == old_periods
        assert new_result.n_segments == old_segments
        np.testing.assert_allclose(new_result.rmse, old_rmse, rtol=1e-5)

    def test_find_pareto_front(self, small_data):
        """Test find_pareto_front produces decreasing RMSE like old API."""
        # Old API
        old_tuner = old_tune.HyperTunedAggregations(
            old_tsam.TimeSeriesAggregation(
                small_data,
                hours_per_period=12,
                cluster_method="hierarchical",
                representation_method="meanRepresentation",
                distribution_period_wise=False,
                rescale_cluster_periods=False,
                segmentation=True,
            )
        )
        old_tuner.identify_pareto_optimal_aggregation()
        old_rmse_history = old_tuner._rmse_history

        # New API
        new_results = find_pareto_front(
            small_data,
            period_duration=12,
            cluster=ClusterConfig(method="hierarchical", representation="mean"),
            show_progress=False,
        )

        # Get RMSE history from summary
        new_rmse_history = new_results.summary["rmse"].tolist()

        # RMSE histories should match between old and new API
        np.testing.assert_allclose(new_rmse_history, old_rmse_history, rtol=1e-10)

        # RMSE should be monotonically decreasing (or equal)
        for i in range(1, len(new_rmse_history)):
            assert new_rmse_history[i] <= new_rmse_history[i - 1] + 1e-10

        # Last RMSE should be 0 (full resolution)
        assert new_rmse_history[-1] < 1e-10

    def test_find_optimal_combination_save_all_results(self, small_data):
        """Test that find_optimal_combination with save_all_results stores all AggregationResults."""
        result = find_optimal_combination(
            small_data,
            data_reduction=0.1,
            period_duration=12,
            show_progress=False,
            save_all_results=True,
        )

        # all_results should have same length as history
        assert len(result.all_results) == len(result.history)

        # Each result should be a valid AggregationResult
        for r in result.all_results:
            assert r.cluster_representatives is not None
            assert r.accuracy is not None


class TestSubhourlyResolution:
    """Test that new API handles sub-hourly resolution correctly."""

    def test_15min_resolution(self):
        """Test with 15-minute resolution data."""
        # Create 15-min data for 7 days
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=7 * 96, freq="15min")
        data = pd.DataFrame(
            {
                "x": np.sin(np.linspace(0, 14 * np.pi, len(dates)))
                + np.random.rand(len(dates)) * 0.1,
            },
            index=dates,
        )

        # Old API with explicit resolution
        old_agg = old_tsam.TimeSeriesAggregation(
            data,
            no_typical_periods=4,
            hours_per_period=24,
            resolution=0.25,  # 15 minutes = 0.25 hours
            cluster_method="hierarchical",
        )
        old_result = old_agg.create_typical_periods()

        # New API (should infer resolution)
        new_result = aggregate(
            data,
            n_clusters=4,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
        )

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    def test_tuning_with_15min_resolution(self):
        """Test tuning functions with 15-minute resolution."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=30 * 96, freq="15min")
        data = pd.DataFrame(
            {
                "x": np.sin(np.linspace(0, 60 * np.pi, len(dates)))
                + np.random.rand(len(dates)) * 0.1,
            },
            index=dates,
        )

        # Should not raise and should find valid configuration
        result = find_optimal_combination(
            data,
            data_reduction=0.1,
            period_duration=24,
            show_progress=False,
        )

        assert result.n_clusters > 0
        assert result.n_segments > 0
        # With 96 timesteps per period, we can have up to 96 segments
        assert result.n_segments <= 96


class TestReconstructionEquivalence:
    """Test that reconstruction produces identical results."""

    def test_reconstruct_matches_old_predict(self, sample_data):
        """Test that reconstructed() matches predict_original_data()."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            no_typical_periods=8,
            hours_per_period=24,
            cluster_method="hierarchical",
        )
        old_agg.create_typical_periods()
        old_reconstructed = old_agg.predict_original_data()

        # New API
        new_result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
        )
        new_reconstructed = new_result.reconstructed

        pd.testing.assert_frame_equal(
            old_reconstructed,
            new_reconstructed,
            check_names=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
