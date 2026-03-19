"""Test for segmentation + weights bug (GitHub issue #178).

When segmentation is used together with weights, predictOriginalData()
returned values scaled by the weight factor (e.g. weight=100 → output 100×
too large). The root cause: the segmentation path in predictOriginalData()
uses predictedSegmentedNormalizedTypicalPeriods which still has weights
baked in, but _postProcessTimeSeries was called with applyWeighting=False.
"""

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as tsam

pytestmark = pytest.mark.filterwarnings("ignore::tsam.exceptions.LegacyAPIWarning")


def _make_data():
    """Small synthetic dataset: 30 days × 24h."""
    rng = np.random.default_rng(42)
    n = 30 * 24
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "Solar": np.clip(
                np.sin(np.linspace(0, 30 * 2 * np.pi, n)) * 500 + rng.normal(0, 50, n),
                0,
                None,
            ),
            "Wind": rng.uniform(0, 12, n),
            "Load": 400
            + 100 * np.sin(np.linspace(0, 30 * 2 * np.pi, n))
            + rng.normal(0, 20, n),
        },
        index=index,
    )


COMMON = {
    "noTypicalPeriods": 4,
    "hoursPerPeriod": 24,
    "clusterMethod": "hierarchical",
    "segmentation": True,
    "noSegments": 4,
}


class TestSegmentationWeightLeak:
    """Weights must not leak into reconstructed data when segmentation is used."""

    def test_uniform_weights_equal_no_weights(self):
        """Uniform weights + segmentation must produce the same
        reconstructed data as no weights at all."""
        data = _make_data()
        agg_none = tsam.TimeSeriesAggregation(data.copy(), **COMMON)
        agg_none.createTypicalPeriods()

        agg_uniform = tsam.TimeSeriesAggregation(
            data.copy(),
            **COMMON,
            weightDict={"Solar": 100, "Wind": 100, "Load": 100},
        )
        agg_uniform.createTypicalPeriods()

        pd.testing.assert_frame_equal(
            agg_none.predictOriginalData(),
            agg_uniform.predictOriginalData(),
            atol=1e-6,
        )

    def test_reconstructed_means_not_scaled_by_weight(self):
        """Column means of reconstructed data must stay close to the
        original means, not be multiplied by the weight."""
        data = _make_data()
        agg = tsam.TimeSeriesAggregation(
            data.copy(),
            **COMMON,
            weightDict={"Solar": 100, "Wind": 1, "Load": 1},
        )
        agg.createTypicalPeriods()
        pred = agg.predictOriginalData()

        for col in data.columns:
            orig_mean = data[col].mean()
            pred_mean = pred[col].mean()
            if orig_mean == 0:
                continue
            rel_error = abs(pred_mean - orig_mean) / abs(orig_mean)
            assert rel_error < 0.5, (
                f"{col}: reconstructed mean {pred_mean:.2f} vs original "
                f"{orig_mean:.2f} — rel error {rel_error:.1%} suggests "
                f"weights leaked into output"
            )

    def test_reconstructed_values_within_original_range(self):
        """Reconstructed values must stay within the original data range,
        not be inflated by weight factors."""
        data = _make_data()
        agg = tsam.TimeSeriesAggregation(
            data.copy(),
            **COMMON,
            weightDict={"Solar": 50, "Wind": 1, "Load": 1},
        )
        agg.createTypicalPeriods()
        pred = agg.predictOriginalData()

        for col in data.columns:
            assert pred[col].max() <= data[col].max() + 1e-6, (
                f"{col}: pred max {pred[col].max():.2f} > "
                f"data max {data[col].max():.2f}"
            )
            assert pred[col].min() >= data[col].min() - 1e-6, (
                f"{col}: pred min {pred[col].min():.2f} < "
                f"data min {data[col].min():.2f}"
            )

    def test_segmentation_samemean_weights(self):
        """sameMean + segmentation + weights must not produce scaled output."""
        data = _make_data()
        agg = tsam.TimeSeriesAggregation(
            data.copy(),
            **COMMON,
            sameMean=True,
            weightDict={"Solar": 100, "Wind": 1, "Load": 1},
        )
        agg.createTypicalPeriods()
        pred = agg.predictOriginalData()

        for col in data.columns:
            assert pred[col].max() <= data[col].max() + 1e-6
            assert pred[col].min() >= data[col].min() - 1e-6
