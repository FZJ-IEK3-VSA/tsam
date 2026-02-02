"""Test for predictOriginalData bug with sameMean + segmentation.

Regression test for the bug where using normalize_column_means=True together
with segmentation caused incorrect denormalization in predictOriginalData().

The bug was: reconstructed values were ~1.5-2x larger than they should be
because the data was incorrectly divided by _normalizedMean when using
segmentation (predictedSegmentedNormalizedTypicalPeriods wasn't modified
in-place like normalizedTypicalPeriods was).
"""

import numpy as np
import pandas as pd
import pytest

import tsam
from tsam import ClusterConfig, SegmentConfig
from tsam.result import AggregationResult


@pytest.fixture
def test_data():
    """Create test data with different scales."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=365 * 24, freq="h")
    return pd.DataFrame(
        {
            "Load": np.sin(np.linspace(0, 50 * np.pi, len(dates))) * 100 + 500,
            "Temp": np.sin(np.linspace(0, 50 * np.pi, len(dates)) + 1) * 15 + 20,
            "Price": np.random.rand(len(dates)) * 50 + 10,
        },
        index=dates,
    )


class TestReconstructSameMeanSegmentation:
    """Test reconstruction with normalize_column_means + segmentation."""

    def _check_reconstruction_bounds(
        self, result: AggregationResult, max_ratio: float = 1.5
    ):
        """Check that reconstructed values are within reasonable bounds."""
        original = result._aggregation.timeSeries
        reconstructed = result.reconstructed

        orig_max = original.max()
        recon_max = reconstructed.max()

        for col in original.columns:
            ratio = recon_max[col] / orig_max[col]
            assert abs(ratio) <= max_ratio, (
                f"Column {col}: reconstructed max ({recon_max[col]:.2f}) is "
                f"{ratio:.2f}x original max ({orig_max[col]:.2f})"
            )

    def test_segments_only(self, test_data):
        """Segmentation alone should work correctly."""
        result = tsam.aggregate(
            test_data,
            n_clusters=8,
            period_duration=24,
            segments=SegmentConfig(n_segments=4),
        )
        self._check_reconstruction_bounds(result)

    def test_normalize_column_means_only(self, test_data):
        """normalize_column_means alone should work correctly."""
        result = tsam.aggregate(
            test_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(normalize_column_means=True),
        )
        self._check_reconstruction_bounds(result)

    def test_normalize_column_means_with_segments(self, test_data):
        """normalize_column_means + segmentation should work correctly.

        This is the bug case - previously reconstructed values were ~1.6x too large.
        """
        result = tsam.aggregate(
            test_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(normalize_column_means=True),
            segments=SegmentConfig(n_segments=4),
        )
        self._check_reconstruction_bounds(result)

    def test_normalize_with_segments_mean_repr(self, test_data):
        """normalize_column_means + segmentation + mean representation."""
        result = tsam.aggregate(
            test_data,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(normalize_column_means=True, representation="mean"),
            segments=SegmentConfig(n_segments=4),
        )
        self._check_reconstruction_bounds(result)

    def test_normalize_with_different_segment_counts(self, test_data):
        """Test various segment counts with normalize_column_means."""
        for n_segments in [2, 4, 8]:
            result = tsam.aggregate(
                test_data,
                n_clusters=8,
                period_duration=24,
                cluster=ClusterConfig(normalize_column_means=True),
                segments=SegmentConfig(n_segments=n_segments),
            )
            self._check_reconstruction_bounds(result)
