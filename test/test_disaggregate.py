"""Tests for disaggregate functionality."""

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusteringResult, SegmentConfig, aggregate


@pytest.fixture
def sample_data():
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


@pytest.fixture
def result(sample_data):
    return aggregate(sample_data, n_clusters=8)


@pytest.fixture
def result_segmented(sample_data):
    return aggregate(sample_data, n_clusters=8, segments=SegmentConfig(n_segments=4))


class TestAggregationResultDisaggregate:
    """Tests for AggregationResult.disaggregate."""

    def test_shape_matches_original(self, result):
        expanded = result.disaggregate(result.cluster_representatives)
        assert expanded.shape == result.original.shape

    def test_datetime_index_restored(self, result):
        expanded = result.disaggregate(result.cluster_representatives)
        pd.testing.assert_index_equal(expanded.index, result.original.index)

    def test_matches_reconstructed(self, result):
        expanded = result.disaggregate(result.cluster_representatives)
        np.testing.assert_allclose(
            expanded.values, result.reconstructed.values, rtol=1e-10
        )

    def test_columns_preserved(self, result):
        expanded = result.disaggregate(result.cluster_representatives)
        assert list(expanded.columns) == list(result.cluster_representatives.columns)

    def test_segmented_shape_matches_original(self, result_segmented):
        expanded = result_segmented.disaggregate(
            result_segmented.cluster_representatives
        )
        assert expanded.shape == result_segmented.original.shape

    def test_segmented_datetime_index(self, result_segmented):
        expanded = result_segmented.disaggregate(
            result_segmented.cluster_representatives
        )
        pd.testing.assert_index_equal(expanded.index, result_segmented.original.index)

    def test_segmented_has_nan(self, result_segmented):
        """Segmented disaggregate leaves NaN for non-segment-start timesteps."""
        expanded = result_segmented.disaggregate(
            result_segmented.cluster_representatives
        )
        assert expanded.isna().any().any()

    def test_segmented_ffill_removes_nan(self, result_segmented):
        """User can ffill to get step function."""
        expanded = result_segmented.disaggregate(
            result_segmented.cluster_representatives
        )
        filled = expanded.ffill()
        assert not filled.isna().any().any()

    def test_arbitrary_data(self, result):
        """Disaggregate works with different column values."""
        modified = result.cluster_representatives * 2 + 1
        expanded = result.disaggregate(modified)
        original_expanded = result.disaggregate(result.cluster_representatives)
        np.testing.assert_allclose(
            expanded.values, original_expanded.values * 2 + 1, rtol=1e-10
        )

    def test_subset_columns(self, result):
        """Disaggregate works with a subset of columns."""
        subset = result.cluster_representatives.iloc[:, :2]
        expanded = result.disaggregate(subset)
        assert expanded.shape == (len(result.original), 2)

    def test_multiindex_columns(self, result):
        """Disaggregate works correctly with MultiIndex columns."""
        reps = result.cluster_representatives.copy()
        reps.columns = pd.MultiIndex.from_tuples(
            [(col, "north") for col in reps.columns]
        )
        expanded = result.clustering.disaggregate(reps)
        assert expanded.shape == (
            result.clustering.n_original_periods
            * result.clustering.n_timesteps_per_period,
            len(reps.columns),
        )
        assert isinstance(expanded.columns, pd.MultiIndex)
        assert not expanded.isna().any().any()


class TestClusteringResultDisaggregate:
    """Tests for ClusteringResult.disaggregate."""

    def test_datetime_index_restored(self, result):
        expanded = result.clustering.disaggregate(result.cluster_representatives)
        if result.clustering.time_index is not None:
            assert isinstance(expanded.index, pd.DatetimeIndex)
            assert expanded.index.equals(result.clustering.time_index)
        else:
            assert isinstance(expanded.index, pd.RangeIndex)

    def test_integer_index_when_no_time_index(self, result):
        """Without time_index, disaggregate returns a RangeIndex."""
        from dataclasses import replace

        clustering_no_ti = replace(result.clustering, time_index=None)
        expanded = clustering_no_ti.disaggregate(result.cluster_representatives)
        assert isinstance(expanded.index, pd.RangeIndex)

    def test_shape(self, result):
        expanded = result.clustering.disaggregate(result.cluster_representatives)
        n_periods = result.clustering.n_original_periods
        n_ts = result.clustering.n_timesteps_per_period
        assert len(expanded) == n_periods * n_ts

    def test_segmented(self, result_segmented):
        expanded = result_segmented.clustering.disaggregate(
            result_segmented.cluster_representatives
        )
        n_periods = result_segmented.clustering.n_original_periods
        n_ts = result_segmented.clustering.n_timesteps_per_period
        assert len(expanded) == n_periods * n_ts

    def test_segmented_nan_count(self, result_segmented):
        """Each segment places one value; rest are NaN."""
        expanded = result_segmented.clustering.disaggregate(
            result_segmented.cluster_representatives
        )
        n_cols = len(result_segmented.cluster_representatives.columns)
        n_segments = result_segmented.n_segments
        n_periods = result_segmented.clustering.n_original_periods
        expected_values = n_segments * n_periods * n_cols
        assert expanded.notna().sum().sum() == expected_values

    def test_timestep_input_on_segmented_clustering_raises(self, result_segmented):
        """Passing (cluster, timestep) data to a segmented clustering raises."""
        n_clusters = result_segmented.n_clusters
        n_ts = result_segmented.clustering.n_timesteps_per_period
        idx = pd.MultiIndex.from_product([range(n_clusters), range(n_ts)])
        data = pd.DataFrame(
            np.ones((n_clusters * n_ts, len(result_segmented.original.columns))),
            index=idx,
            columns=result_segmented.original.columns,
        )
        with pytest.raises(ValueError, match="segmentation"):
            result_segmented.clustering.disaggregate(data)

    def test_segmented_input_on_nonsegmented_clustering_raises(self, result):
        """Passing segment-level data to a non-segmented clustering raises."""
        n_clusters = result.n_clusters
        idx = pd.MultiIndex.from_tuples(
            [(c, s, d) for c in range(n_clusters) for s, d in [(0, 5), (1, 7)]],
        )
        data = pd.DataFrame(
            np.ones((len(idx), len(result.original.columns))),
            index=idx,
            columns=result.original.columns,
        )
        with pytest.raises(ValueError, match="segmentation"):
            result.clustering.disaggregate(data)

    def test_io_roundtrip(self, result, tmp_path):
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        loaded = ClusteringResult.from_json(str(path))

        original = result.clustering.disaggregate(result.cluster_representatives)
        restored = loaded.disaggregate(result.cluster_representatives)
        np.testing.assert_array_equal(original.values, restored.values)

    def test_io_roundtrip_preserves_time_index(self, result, tmp_path):
        """JSON round-trip preserves the original DatetimeIndex."""
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        loaded = ClusteringResult.from_json(str(path))

        assert loaded.time_index is not None
        assert loaded.time_index.equals(result.clustering.time_index)

        restored = loaded.disaggregate(result.cluster_representatives)
        assert isinstance(restored.index, pd.DatetimeIndex)
        assert restored.index.equals(result.clustering.time_index)

    def test_io_roundtrip_no_time_index(self, tmp_path):
        """Old serialized files without time_index still work."""

        cr = ClusteringResult(
            period_duration=24.0,
            cluster_assignments=(0, 1, 0, 1),
            n_timesteps_per_period=24,
        )
        path = tmp_path / "clustering.json"
        cr.to_json(str(path))
        loaded = ClusteringResult.from_json(str(path))
        assert loaded.time_index is None

    def test_io_roundtrip_segmented(self, result_segmented, tmp_path):
        path = tmp_path / "clustering.json"
        result_segmented.clustering.to_json(str(path))
        loaded = ClusteringResult.from_json(str(path))

        original = result_segmented.clustering.disaggregate(
            result_segmented.cluster_representatives
        )
        restored = loaded.disaggregate(result_segmented.cluster_representatives)
        # Compare NaN-aware
        np.testing.assert_array_equal(
            np.isnan(original.values), np.isnan(restored.values)
        )
        mask = ~np.isnan(original.values)
        np.testing.assert_array_equal(original.values[mask], restored.values[mask])


class TestDisaggregateEdgeCases:
    """Edge cases and robustness tests."""

    def test_padded_last_period(self, sample_data):
        """Data whose length isn't divisible by period length gets trimmed correctly."""
        # 100 rows with hourly data = 4 full days + 4 hours → last period is padded
        short = sample_data.iloc[:100]
        result = aggregate(short, n_clusters=3)
        expanded = result.disaggregate(result.cluster_representatives)
        assert len(expanded) == len(short)
        pd.testing.assert_index_equal(expanded.index, short.index)

    def test_single_cluster(self, sample_data):
        """Degenerate case: n_clusters=1."""
        result = aggregate(sample_data, n_clusters=1)
        expanded = result.disaggregate(result.cluster_representatives)
        assert expanded.shape == result.original.shape
        np.testing.assert_allclose(
            expanded.values, result.reconstructed.values, rtol=1e-10
        )

    def test_extreme_periods_append(self, sample_data):
        """Extreme periods with 'append' add extra clusters."""
        from tsam import ExtremeConfig

        result = aggregate(
            sample_data,
            n_clusters=8,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
            ),
        )
        expanded = result.disaggregate(result.cluster_representatives)
        assert expanded.shape == result.original.shape
        np.testing.assert_allclose(
            expanded.values, result.reconstructed.values, rtol=1e-10
        )

    def test_extreme_periods_new_cluster(self, sample_data):
        """Extreme periods with 'new_cluster' add extra clusters."""
        from tsam import ExtremeConfig

        result = aggregate(
            sample_data,
            n_clusters=8,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI"],
            ),
        )
        expanded = result.disaggregate(result.cluster_representatives)
        assert expanded.shape == result.original.shape
        np.testing.assert_allclose(
            expanded.values, result.reconstructed.values, rtol=1e-10
        )

    def test_kmeans(self, sample_data):
        """Works with kmeans clustering."""
        from tsam import ClusterConfig

        result = aggregate(
            sample_data,
            n_clusters=8,
            cluster=ClusterConfig(method="kmeans"),
        )
        expanded = result.disaggregate(result.cluster_representatives)
        assert expanded.shape == result.original.shape
        np.testing.assert_allclose(
            expanded.values, result.reconstructed.values, rtol=1e-10
        )

    def test_segmented_nan_at_correct_positions(self, result_segmented):
        """NaN values are between segment starts, not at segment starts."""
        expanded = result_segmented.clustering.disaggregate(
            result_segmented.cluster_representatives
        )
        durations = result_segmented.clustering.segment_durations
        assignments = result_segmented.clustering.cluster_assignments
        n_ts = result_segmented.clustering.n_timesteps_per_period

        # Check first few original periods
        for period_idx in range(min(3, len(assignments))):
            cluster = assignments[period_idx]
            cluster_durations = durations[cluster]
            start = period_idx * n_ts
            pos = 0
            for d in cluster_durations:
                # Segment start should have values
                assert expanded.iloc[start + pos].notna().all(), (
                    f"period {period_idx}, segment start at {pos} should have values"
                )
                # Positions after start should be NaN (if duration > 1)
                for offset in range(1, d):
                    assert expanded.iloc[start + pos + offset].isna().all(), (
                        f"period {period_idx}, offset {pos + offset} should be NaN"
                    )
                pos += d

    def test_reconstructed_unchanged_by_refactor(self, sample_data):
        """predictOriginalData still matches via the shared _expand_periods helper."""
        import warnings

        from tsam.timeseriesaggregation import TimeSeriesAggregation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg = TimeSeriesAggregation(sample_data, noTypicalPeriods=8)
            agg.createTypicalPeriods()
            reconstructed_old_api = agg.predictOriginalData()

        result = aggregate(sample_data, n_clusters=8)
        np.testing.assert_allclose(
            result.reconstructed.values, reconstructed_old_api.values, rtol=1e-10
        )


class TestDisaggregateValidation:
    """Tests for disaggregate input validation."""

    def test_flat_index_raises(self, result):
        with pytest.raises(ValueError, match="MultiIndex"):
            result.clustering.disaggregate(pd.DataFrame({"a": [1, 2, 3]}))

    def test_wrong_clusters_raises(self, result):
        wrong = result.cluster_representatives.copy()
        wrong.index = pd.MultiIndex.from_tuples([(c + 100, t) for c, t in wrong.index])
        with pytest.raises(ValueError, match="Cluster IDs"):
            result.clustering.disaggregate(wrong)

    def test_missing_cluster_raises(self, result):
        """Dropping a cluster from the input raises."""
        first_cluster = result.cluster_representatives.index.get_level_values(0)[0]
        partial = result.cluster_representatives.drop(first_cluster, level=0)
        with pytest.raises(ValueError, match="missing clusters"):
            result.clustering.disaggregate(partial)

    def test_wrong_timesteps_raises(self, result):
        """Wrong number of timesteps per period raises."""
        wrong = result.cluster_representatives.iloc[::2]
        with pytest.raises(ValueError, match="timesteps"):
            result.clustering.disaggregate(wrong)
