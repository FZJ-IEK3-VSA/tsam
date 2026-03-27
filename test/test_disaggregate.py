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


class TestClusteringResultDisaggregate:
    """Tests for ClusteringResult.disaggregate."""

    def test_integer_index(self, result):
        expanded = result.clustering.disaggregate(result.cluster_representatives)
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

    def test_timestep_input_on_segmented_clustering(self, result_segmented):
        """Passing (cluster, timestep) data to a segmented clustering skips segment expansion."""
        # Build a (cluster, timestep) DataFrame manually from the segmented clustering
        n_clusters = result_segmented.n_clusters
        n_ts = result_segmented.clustering.n_timesteps_per_period
        idx = pd.MultiIndex.from_product([range(n_clusters), range(n_ts)])
        data = pd.DataFrame(
            np.ones((n_clusters * n_ts, len(result_segmented.original.columns))),
            index=idx,
            columns=result_segmented.original.columns,
        )
        expanded = result_segmented.clustering.disaggregate(data)
        n_periods = result_segmented.clustering.n_original_periods
        assert len(expanded) == n_periods * n_ts
        assert not expanded.isna().any().any()  # No NaN — no segment expansion

    def test_io_roundtrip(self, result, tmp_path):
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        loaded = ClusteringResult.from_json(str(path))

        original = result.clustering.disaggregate(result.cluster_representatives)
        restored = loaded.disaggregate(result.cluster_representatives)
        np.testing.assert_array_equal(original.values, restored.values)

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
        with pytest.raises(ValueError, match="timesteps per period"):
            result.clustering.disaggregate(wrong)
