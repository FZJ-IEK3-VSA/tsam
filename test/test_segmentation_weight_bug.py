"""Test for segmentation + weights bug (GitHub issue #178).

When segmentation is used together with weights, the reconstructed data
returned values scaled by the weight factor (e.g. weight=100 → output 100×
too large). The root cause: the segmentation reconstruction path used
predicted segmented normalized typical periods which still had weights
baked in, but the post-processing was applied without removing the
weighting.
"""

import numpy as np
import pandas as pd

from tsam import ClusterConfig, SegmentConfig, aggregate


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


def _aggregate(data, weights=None, scale_by_column_means=False):
    return aggregate(
        data,
        n_clusters=4,
        period_duration=24,
        cluster=ClusterConfig(
            method="hierarchical",
            scale_by_column_means=scale_by_column_means,
        ),
        segments=SegmentConfig(n_segments=4),
        weights=weights,
    )


class TestSegmentationWeightLeak:
    """Weights must not leak into reconstructed data when segmentation is used."""

    def test_uniform_weights_equal_no_weights(self):
        """Uniform weights + segmentation must produce the same
        reconstructed data as no weights at all."""
        data = _make_data()
        result_none = _aggregate(data.copy())

        result_uniform = _aggregate(
            data.copy(),
            weights={"Solar": 100, "Wind": 100, "Load": 100},
        )

        pd.testing.assert_frame_equal(
            result_none.reconstructed,
            result_uniform.reconstructed,
            atol=1e-6,
        )

    def test_reconstructed_means_not_scaled_by_weight(self):
        """Column means of reconstructed data must stay close to the
        original means, not be multiplied by the weight."""
        data = _make_data()
        result = _aggregate(
            data.copy(),
            weights={"Solar": 100, "Wind": 1, "Load": 1},
        )
        pred = result.reconstructed

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
        result = _aggregate(
            data.copy(),
            weights={"Solar": 50, "Wind": 1, "Load": 1},
        )
        pred = result.reconstructed

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
        result = _aggregate(
            data.copy(),
            weights={"Solar": 100, "Wind": 1, "Load": 1},
            scale_by_column_means=True,
        )
        pred = result.reconstructed

        for col in data.columns:
            assert pred[col].max() <= data[col].max() + 1e-6
            assert pred[col].min() >= data[col].min() - 1e-6
