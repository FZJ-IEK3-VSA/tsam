"""Tests to verify new API produces identical results to old API."""

import os

import numpy as np
import pandas as pd
import pytest

# Old API
import tsam.hyperparametertuning as old_tune
import tsam.timeseriesaggregation as old_tsam

# New API
from tsam import ClusterConfig, ExtremeConfig, SegmentConfig, aggregate
from tsam.tuning import (
    find_optimal_combination,
    find_pareto_front,
    periods_for_reduction,
    segments_for_reduction,
)


@pytest.fixture
def sample_data():
    """Load sample time series data."""
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


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
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical"),
        )

        # Compare typical periods
        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

        # Compare cluster assignments
        np.testing.assert_array_equal(
            old_agg.clusterOrder, new_result.cluster_assignments
        )

        # Compare accuracy
        old_accuracy = old_agg.accuracyIndicators()
        np.testing.assert_allclose(
            old_accuracy["RMSE"].values,
            new_result.accuracy.rmse.values,
            rtol=1e-10,
        )

    def test_kmeans(self, sample_data):
        """Test k-means clustering."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="k_means",
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="kmeans"),
        )

        # K-means may have some randomness, but results should be very close
        old_accuracy = old_agg.accuracyIndicators()
        np.testing.assert_allclose(
            old_accuracy["RMSE"].values,
            new_result.accuracy.rmse.values,
            rtol=0.1,  # Allow 10% tolerance for k-means randomness
        )

    def test_hierarchical_with_medoid(self, sample_data):
        """Test hierarchical with medoid representation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="medoidRepresentation",
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical", representation="medoid"),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_with_weights(self, sample_data):
        """Test weighted clustering."""
        weights = {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0}

        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            weightDict=weights,
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical", weights=weights),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_with_segmentation(self, sample_data):
        """Test with segmentation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            segmentation=True,
            noSegments=12,
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical"),
            segments=SegmentConfig(n_segments=12),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_with_duration_curves(self, sample_data):
        """Test duration curve representation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="durationRepresentation",
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical", representation="duration"),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_with_extremes_append(self, sample_data):
        """Test extreme period handling with append method."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            extremePeriodMethod="append",
            addPeakMax=["Load"],
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical"),
            extremes=ExtremeConfig(method="append", max_timesteps=["Load"]),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_contiguous_clustering(self, sample_data):
        """Test contiguous (adjacent periods) clustering."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="adjacent_periods",
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="contiguous"),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_rescale_off(self, sample_data):
        """Test with rescaling disabled."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            rescaleClusterPeriods=False,
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical"),
            rescale=False,
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_distribution_minmax_representation(self, sample_data):
        """Test distribution with min/max representation."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="distributionAndMinMaxRepresentation",
        )
        old_result = old_agg.createTypicalPeriods()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(
                method="hierarchical", representation="distribution_minmax"
            ),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )


class TestTuningEquivalence:
    """Tests that new tuning functions produce equivalent results to old API."""

    def test_periods_for_reduction(self):
        """Test periods_for_reduction matches old function."""
        test_cases = [
            (100, 10, 0.5),
            (101, 10, 0.5),
            (101, 2, 0.5),
            (8760, 24, 0.01),
        ]

        for n_timesteps, n_segments, data_reduction in test_cases:
            old_result = old_tune.getNoPeriodsForDataReduction(
                n_timesteps, n_segments, data_reduction
            )
            new_result = periods_for_reduction(n_timesteps, n_segments, data_reduction)
            assert old_result == new_result, (
                f"Mismatch for ({n_timesteps}, {n_segments}, {data_reduction}): "
                f"old={old_result}, new={new_result}"
            )

    def test_segments_for_reduction(self):
        """Test segments_for_reduction matches old function."""
        test_cases = [
            (100, 10, 0.5),
            (8760, 8, 0.01),
        ]

        for n_timesteps, n_periods, data_reduction in test_cases:
            old_result = old_tune.getNoSegmentsForDataReduction(
                n_timesteps, n_periods, data_reduction
            )
            new_result = segments_for_reduction(n_timesteps, n_periods, data_reduction)
            assert old_result == new_result

    def test_find_optimal_combination(self, sample_data):
        """Test find_optimal_combination matches old identifyOptimalSegmentPeriodCombination."""
        data_reduction = 0.01
        col = "Wind"
        data = sample_data[[col]]

        # Old API with default settings to match new API defaults
        old_tuner = old_tune.HyperTunedAggregations(
            old_tsam.TimeSeriesAggregation(
                data,
                hoursPerPeriod=24,
                clusterMethod="hierarchical",
                representationMethod="durationRepresentation",
                # Use defaults: distributionPeriodWise=True, rescaleClusterPeriods=True
                segmentation=True,
            )
        )
        old_segments, old_periods, old_rmse = (
            old_tuner.identifyOptimalSegmentPeriodCombination(
                dataReduction=data_reduction
            )
        )

        # New API
        new_result = find_optimal_combination(
            data,
            data_reduction=data_reduction,
            period_hours=24,
            cluster=ClusterConfig(
                method="hierarchical",
                representation="duration",
            ),
            show_progress=False,
        )

        # Results should match
        assert new_result.optimal_n_periods == old_periods
        assert new_result.optimal_n_segments == old_segments
        np.testing.assert_allclose(new_result.optimal_rmse, old_rmse, rtol=1e-5)

    def test_find_pareto_front(self, small_data):
        """Test find_pareto_front produces decreasing RMSE like old API."""
        # Old API
        old_tuner = old_tune.HyperTunedAggregations(
            old_tsam.TimeSeriesAggregation(
                small_data,
                hoursPerPeriod=12,
                clusterMethod="hierarchical",
                representationMethod="meanRepresentation",
                distributionPeriodWise=False,
                rescaleClusterPeriods=False,
                segmentation=True,
            )
        )
        old_tuner.identifyParetoOptimalAggregation()
        old_rmse_history = old_tuner._RMSEHistory

        # New API
        new_results = find_pareto_front(
            small_data,
            period_hours=12,
            cluster=ClusterConfig(method="hierarchical", representation="mean"),
            show_progress=False,
        )

        # Both should have decreasing RMSE
        new_rmse_history = [r.optimal_rmse for r in new_results]

        # RMSE should be monotonically decreasing (or equal)
        for i in range(1, len(new_rmse_history)):
            assert new_rmse_history[i] <= new_rmse_history[i - 1] + 1e-10

        # Last RMSE should be 0 (full resolution)
        assert new_rmse_history[-1] < 1e-10

    def test_save_all_results(self, small_data):
        """Test that save_all_results stores all AggregationResults."""
        result = find_optimal_combination(
            small_data,
            data_reduction=0.1,
            period_hours=12,
            show_progress=False,
            save_all_results=True,
        )

        # all_results should have same length as history
        assert len(result.all_results) == len(result.history)

        # Each result should be a valid AggregationResult
        for r in result.all_results:
            assert r.typical_periods is not None
            assert r.accuracy is not None


class TestSubhourlyResolution:
    """Test that new API handles sub-hourly resolution correctly."""

    def test_15min_resolution(self):
        """Test with 15-minute resolution data."""
        # Create 15-min data for 7 days
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
            noTypicalPeriods=4,
            hoursPerPeriod=24,
            resolution=0.25,  # 15 minutes = 0.25 hours
            clusterMethod="hierarchical",
        )
        old_result = old_agg.createTypicalPeriods()

        # New API (should infer resolution)
        new_result = aggregate(
            data,
            n_periods=4,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical"),
        )

        pd.testing.assert_frame_equal(
            old_result.reset_index(drop=True),
            new_result.typical_periods.reset_index(drop=True),
            check_names=False,
        )

    def test_tuning_with_15min_resolution(self):
        """Test tuning functions with 15-minute resolution."""
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
            period_hours=24,
            show_progress=False,
        )

        assert result.optimal_n_periods > 0
        assert result.optimal_n_segments > 0
        # With 96 timesteps per period, we can have up to 96 segments
        assert result.optimal_n_segments <= 96


class TestReconstructionEquivalence:
    """Test that reconstruction produces identical results."""

    def test_reconstruct_matches_old_predict(self, sample_data):
        """Test that reconstruct() matches predictOriginalData()."""
        # Old API
        old_agg = old_tsam.TimeSeriesAggregation(
            sample_data,
            noTypicalPeriods=8,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
        )
        old_agg.createTypicalPeriods()
        old_reconstructed = old_agg.predictOriginalData()

        # New API
        new_result = aggregate(
            sample_data,
            n_periods=8,
            period_hours=24,
            cluster=ClusterConfig(method="hierarchical"),
        )
        new_reconstructed = new_result.reconstruct()

        pd.testing.assert_frame_equal(
            old_reconstructed.reset_index(drop=True),
            new_reconstructed.reset_index(drop=True),
            check_names=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
