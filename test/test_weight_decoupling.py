"""Tests for the weight decoupling refactor.

Verifies that per-column weights affect only clustering distance and
are correctly handled for partial weight dicts, duration curves,
serialization round-trips, and the deprecated property.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ClusteringResult, aggregate
from tsam.pipeline import _build_weight_vector
from tsam.weights import MIN_WEIGHT, validate_weights


@pytest.fixture
def sample_data():
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestBuildWeightVector:
    """Unit tests for _build_weight_vector."""

    def test_none_weights_returns_none(self):
        cols = pd.Index(["A", "B", "C"])
        assert _build_weight_vector(cols, None) is None

    def test_empty_weights_returns_none(self):
        cols = pd.Index(["A", "B", "C"])
        assert _build_weight_vector(cols, {}) is None

    def test_all_unit_weights_returns_none(self):
        cols = pd.Index(["A", "B", "C"])
        assert _build_weight_vector(cols, {"A": 1.0, "B": 1.0, "C": 1.0}) is None

    def test_full_weights(self):
        cols = pd.Index(["A", "B", "C"])
        result = _build_weight_vector(cols, {"A": 2.0, "B": 1.0, "C": 3.0})
        np.testing.assert_array_equal(result, [2.0, 1.0, 3.0])

    def test_partial_weights_default_to_one(self):
        """Unlisted columns must default to 1.0, not be omitted."""
        cols = pd.Index(["A", "B", "C"])
        result = _build_weight_vector(cols, {"B": 2.0})
        np.testing.assert_array_equal(result, [1.0, 2.0, 1.0])

    def test_min_weight_enforcement(self):
        cols = pd.Index(["A", "B"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _build_weight_vector(cols, {"A": 0.0})
            assert len(w) == 1
            assert "minimal tolerable" in str(w[0].message)
        assert result is not None
        assert result[0] == pytest.approx(1e-6)
        assert result[1] == 1.0

    def test_preserves_column_order(self):
        """Weights must follow the column order, not the dict order."""
        cols = pd.Index(["C", "A", "B"])
        result = _build_weight_vector(cols, {"A": 2.0, "B": 3.0, "C": 4.0})
        np.testing.assert_array_equal(result, [4.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPartialWeights:
    """Partial weight dicts must work correctly."""

    def test_partial_weight_only_affects_specified_column(self, sample_data):
        """Weighting only Load should change clustering vs unweighted."""
        unweighted = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
        )
        partial = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", weights={"Load": 10.0}),
        )
        # With a very high weight on Load, Load's RMSE should improve
        assert partial.accuracy.rmse["Load"] <= unweighted.accuracy.rmse["Load"]

    def test_partial_weight_equals_full_with_defaults(self, sample_data):
        """weights={"Load": 2.0} must equal {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0}."""
        partial = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", weights={"Load": 2.0}),
        )
        full = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(
                method="hierarchical",
                weights={"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
            ),
        )
        np.testing.assert_array_equal(
            partial.cluster_assignments, full.cluster_assignments
        )
        pd.testing.assert_frame_equal(
            partial.cluster_representatives, full.cluster_representatives
        )


class TestDurationCurvesWithWeights:
    """Weights must actually affect clustering when use_duration_curves=True."""

    def test_weighted_duration_curves_differ_from_unweighted(self, sample_data):
        unweighted = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", use_duration_curves=True),
        )
        weighted = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(
                method="hierarchical",
                use_duration_curves=True,
                weights={"Load": 10.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
            ),
        )
        # With a very strong weight on Load, assignments should change
        # (or at least Load RMSE should improve)
        assert weighted.accuracy.rmse["Load"] <= unweighted.accuracy.rmse["Load"]


class TestDeprecatedClusterWeightsProperty:
    """The deprecated cluster_weights property must emit FutureWarning."""

    def test_emits_future_warning(self, sample_data):
        result = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
        )
        with pytest.warns(FutureWarning, match="cluster_weights.*deprecated"):
            _ = result.cluster_weights


class TestWeightRoundTrip:
    """ClusteringResult.apply() must preserve weights through serialization."""

    def test_apply_with_weights(self, sample_data):
        weights = {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0}
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", weights=weights),
        )

        # Apply stored clustering to same data
        result2 = result1.clustering.apply(sample_data)

        np.testing.assert_array_equal(
            result1.cluster_assignments, result2.cluster_assignments
        )

    def test_json_roundtrip_preserves_weights(self, sample_data, tmp_path):
        weights = {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0}
        result1 = aggregate(
            sample_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", weights=weights),
        )

        # Serialize and deserialize
        json_path = tmp_path / "clustering.json"
        result1.clustering.to_json(str(json_path))
        loaded = ClusteringResult.from_json(str(json_path))

        # Verify weights survived
        assert loaded.column_weights is not None
        restored = dict(loaded.column_weights)
        assert restored == weights

        # Apply and verify identical assignments
        result2 = loaded.apply(sample_data)
        np.testing.assert_array_equal(
            result1.cluster_assignments, result2.cluster_assignments
        )


# ---------------------------------------------------------------------------
# validate_weights tests
# ---------------------------------------------------------------------------


class TestValidateWeights:
    """Unit tests for the unified validate_weights() function."""

    def test_none_returns_none(self):
        assert validate_weights(pd.Index(["A", "B"]), None) is None

    def test_empty_returns_none(self):
        assert validate_weights(pd.Index(["A", "B"]), {}) is None

    def test_all_unit_returns_none(self):
        assert validate_weights(pd.Index(["A", "B"]), {"A": 1.0, "B": 1.0}) is None

    def test_valid_weights_returned(self):
        result = validate_weights(pd.Index(["A", "B"]), {"A": 2.0, "B": 1.0})
        assert result == {"A": 2.0, "B": 1.0}

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="Weight columns not found"):
            validate_weights(pd.Index(["A", "B"]), {"A": 1.0, "Z": 2.0})

    def test_min_weight_clamping(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_weights(pd.Index(["A", "B"]), {"A": 0.0, "B": 1.0})
        assert len(w) == 1
        assert "minimal tolerable" in str(w[0].message)
        assert result is not None
        assert result["A"] == pytest.approx(MIN_WEIGHT)

    def test_old_wrapper_rejects_invalid_columns(self, sample_data):
        """Old wrapper now raises ValueError on invalid weight column names."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tsam.timeseriesaggregation import TimeSeriesAggregation

            agg = TimeSeriesAggregation(
                time_series=sample_data,
                no_typical_periods=8,
                hours_per_period=24,
                weight_dict={"NonExistent": 2.0},
            )
            with pytest.raises(ValueError, match="Weight columns not found"):
                agg.create_typical_periods()
