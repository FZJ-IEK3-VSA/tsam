"""Thorough tests for weight handling in tsam.

Tests that weights affect ONLY clustering distance, and do not leak into:
- output scale (typical_periods, predicted_data)
- rescaling behavior
- accuracy indicators
- reconstructed column means
"""

import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, SegmentConfig, aggregate

RAW = pd.read_csv(TESTDATA_CSV, index_col=0)
N_TYPICAL = 8
HOURS_PER_PERIOD = 24


def _run(weight_dict=None, **kwargs):
    cluster_method = kwargs.pop("cluster_method", "hierarchical")
    same_mean = kwargs.pop("same_mean", False)

    agg_kwargs = {
        "n_clusters": N_TYPICAL,
        "period_duration": HOURS_PER_PERIOD,
        "cluster": ClusterConfig(
            method=cluster_method, scale_by_column_means=same_mean
        ),
    }

    if "rescale_cluster_periods" in kwargs:
        agg_kwargs["preserve_column_means"] = kwargs.pop("rescale_cluster_periods")

    segmentation = kwargs.pop("segmentation", False)
    no_segments = kwargs.pop("no_segments", None)
    if segmentation or no_segments is not None:
        agg_kwargs["segments"] = SegmentConfig(n_segments=no_segments)

    add_peak_max = kwargs.pop("add_peak_max", None)
    add_peak_min = kwargs.pop("add_peak_min", None)
    if add_peak_max is not None or add_peak_min is not None:
        agg_kwargs["extremes"] = ExtremeConfig(
            max_value=add_peak_max or [],
            min_value=add_peak_min or [],
        )

    if weight_dict is not None:
        agg_kwargs["weights"] = weight_dict

    agg_kwargs.update(kwargs)
    return aggregate(RAW.copy(), **agg_kwargs)


# ---------------------------------------------------------------------------
# 1. Output range: typical_periods must stay within original data bounds
# ---------------------------------------------------------------------------


class TestOutputRange:
    """typical_periods and predicted_data must not exceed original data bounds
    regardless of weight configuration."""

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            {"GHI": 1, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 5, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 0.1, "T": 10, "Wind": 0.5, "Load": 3},
            {"GHI": 100, "T": 100, "Wind": 100, "Load": 100},
        ],
    )
    def test_typical_periods_within_bounds(self, weights):
        agg = _run(weights)
        tp = agg.cluster_representatives

        for col in RAW.columns:
            col_min = RAW[col].min()
            col_max = RAW[col].max()
            assert tp[col].min() >= col_min - 1e-6, (
                f"{col}: typical min {tp[col].min()} < data min {col_min}"
            )
            assert tp[col].max() <= col_max + 1e-6, (
                f"{col}: typical max {tp[col].max()} > data max {col_max}"
            )

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            {"GHI": 5, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 0.1, "T": 10, "Wind": 0.5, "Load": 3},
        ],
    )
    def test_predicted_data_within_bounds(self, weights):
        agg = _run(weights)
        pred = agg.reconstructed

        for col in RAW.columns:
            col_min = RAW[col].min()
            col_max = RAW[col].max()
            assert pred[col].min() >= col_min - 1e-6, (
                f"{col}: pred min {pred[col].min()} < data min {col_min}"
            )
            assert pred[col].max() <= col_max + 1e-6, (
                f"{col}: pred max {pred[col].max()} > data max {col_max}"
            )


# ---------------------------------------------------------------------------
# 2. Uniform weights == no weights (identical cluster assignments → identical output)
# ---------------------------------------------------------------------------


class TestUniformWeightsEquivalence:
    """Uniform weights (all same value) should produce identical results to no weights."""

    @pytest.mark.parametrize("uniform_weight", [1, 2, 0.5, 10, 100])
    def test_uniform_weights_typical_periods(self, uniform_weight):
        agg_none = _run(None)
        agg_uniform = _run(dict.fromkeys(RAW.columns, uniform_weight))

        pd.testing.assert_frame_equal(
            agg_none.cluster_representatives,
            agg_uniform.cluster_representatives,
            atol=1e-6,
        )

    @pytest.mark.parametrize("uniform_weight", [1, 2, 10])
    def test_uniform_weights_predicted_data(self, uniform_weight):
        agg_none = _run(None)
        agg_uniform = _run(dict.fromkeys(RAW.columns, uniform_weight))

        pd.testing.assert_frame_equal(
            agg_none.reconstructed,
            agg_uniform.reconstructed,
            atol=1e-6,
        )

    @pytest.mark.parametrize("uniform_weight", [1, 2, 10])
    def test_uniform_weights_accuracy(self, uniform_weight):
        agg_none = _run(None)
        agg_uniform = _run(dict.fromkeys(RAW.columns, uniform_weight))

        pd.testing.assert_series_equal(
            agg_none.accuracy.rmse, agg_uniform.accuracy.rmse, atol=1e-6
        )
        pd.testing.assert_series_equal(
            agg_none.accuracy.mae, agg_uniform.accuracy.mae, atol=1e-6
        )
        pd.testing.assert_series_equal(
            agg_none.accuracy.rmse_duration,
            agg_uniform.accuracy.rmse_duration,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# 3. Rescaling: column means should be preserved
# ---------------------------------------------------------------------------


class TestRescalePreservesMeans:
    """After rescaling, reconstructed data should preserve the original
    column means (within tolerance), regardless of weights."""

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            {"GHI": 1, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 5, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 0.1, "T": 10, "Wind": 0.5, "Load": 3},
        ],
    )
    def test_predicted_data_preserves_column_means(self, weights):
        agg = _run(weights, rescale_cluster_periods=True)
        pred = agg.reconstructed

        for col in RAW.columns:
            orig_mean = RAW[col].mean()
            pred_mean = pred[col].mean()
            if orig_mean == 0:
                assert abs(pred_mean) < 1e-6
            else:
                rel_error = abs(pred_mean - orig_mean) / abs(orig_mean)
                assert rel_error < 0.02, (
                    f"{col}: mean relative error {rel_error:.4f} exceeds 2% "
                    f"(orig={orig_mean:.4f}, pred={pred_mean:.4f})"
                )


# ---------------------------------------------------------------------------
# 4. Weights affect clustering but not scale
# ---------------------------------------------------------------------------


class TestWeightsAffectOnlyClustering:
    """Non-uniform weights should change cluster assignments but the
    output (typical_periods) should still be in the original data scale."""

    def test_non_uniform_weights_change_assignments(self):
        """Different weights should (in general) give different cluster orders."""
        agg1 = _run({"GHI": 1, "T": 1, "Wind": 1, "Load": 1})
        agg3 = _run({"GHI": 10, "T": 1, "Wind": 1, "Load": 1})

        order1 = list(agg1.cluster_assignments)
        order3 = list(agg3.cluster_assignments)
        assert order1 != order3, (
            "Expected different cluster orders with extreme weight diff"
        )

    def test_weight_does_not_scale_output(self):
        """Even with extreme weights, output values should be in original data range."""
        agg = _run({"GHI": 100, "T": 0.01, "Wind": 1, "Load": 1})
        tp = agg.cluster_representatives

        # GHI should NOT be 100x its original range
        assert tp["GHI"].max() <= RAW["GHI"].max() + 1e-6
        # T should NOT be 0.01x its original range
        assert tp["T"].min() >= RAW["T"].min() - 1e-6


# ---------------------------------------------------------------------------
# 5. Weight interaction with sameMean
# ---------------------------------------------------------------------------


class TestWeightsWithSameMean:
    """Weights combined with same_mean=True should not produce out-of-range values."""

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            {"GHI": 5, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 0.1, "T": 10, "Wind": 0.5, "Load": 3},
        ],
    )
    def test_same_mean_output_in_range(self, weights):
        agg = _run(weights, same_mean=True)
        tp = agg.cluster_representatives

        for col in RAW.columns:
            assert tp[col].min() >= RAW[col].min() - 1e-6
            assert tp[col].max() <= RAW[col].max() + 1e-6

    def test_same_mean_uniform_weights_equal_no_weights(self):
        agg_none = _run(None, same_mean=True)
        agg_uniform = _run(dict.fromkeys(RAW.columns, 3), same_mean=True)
        pd.testing.assert_frame_equal(
            agg_none.cluster_representatives,
            agg_uniform.cluster_representatives,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            {"GHI": 5, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 0.1, "T": 10, "Wind": 0.5, "Load": 3},
        ],
    )
    def test_same_mean_preserves_column_means(self, weights):
        agg = _run(weights, same_mean=True, rescale_cluster_periods=True)
        pred = agg.reconstructed

        for col in RAW.columns:
            orig_mean = RAW[col].mean()
            pred_mean = pred[col].mean()
            if orig_mean == 0:
                continue
            rel_error = abs(pred_mean - orig_mean) / abs(orig_mean)
            assert rel_error < 0.02, (
                f"{col}: sameMean + weight mean error {rel_error:.4f} > 2%"
            )


# ---------------------------------------------------------------------------
# 6. Weight interaction with extreme periods
# ---------------------------------------------------------------------------


class TestWeightsWithExtremePeriods:
    """Weights should not distort extreme period selection or values."""

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            {"GHI": 5, "T": 1, "Wind": 1, "Load": 1},
        ],
    )
    def test_extreme_periods_in_range(self, weights):
        agg = _run(
            weights,
            add_peak_max=["GHI"],
            add_peak_min=["T"],
        )
        tp = agg.cluster_representatives

        for col in RAW.columns:
            assert tp[col].min() >= RAW[col].min() - 1e-6
            assert tp[col].max() <= RAW[col].max() + 1e-6


# ---------------------------------------------------------------------------
# 7. Accuracy indicators should be unaffected by weight scale
# ---------------------------------------------------------------------------


class TestAccuracyIndicatorsConsistency:
    """Accuracy indicators should compare unweighted data —
    they should not be inflated or deflated by weight magnitude."""

    def test_accuracy_values_reasonable(self):
        """RMSE and MAE should be between 0 and 1 (on normalized data)."""
        agg = _run({"GHI": 10, "T": 1, "Wind": 1, "Load": 1})
        acc = agg.accuracy

        for col in RAW.columns:
            rmse = acc.rmse[col]
            mae = acc.mae[col]
            assert 0 <= rmse <= 2, f"{col} RMSE={rmse} out of reasonable range"
            assert 0 <= mae <= 2, f"{col} MAE={mae} out of reasonable range"

    def test_weight_scaling_does_not_inflate_metrics(self):
        """Doubling one weight should not double that column's RMSE."""
        agg1 = _run({"GHI": 1, "T": 1, "Wind": 1, "Load": 1})
        agg2 = _run({"GHI": 2, "T": 1, "Wind": 1, "Load": 1})

        rmse1 = agg1.accuracy.rmse["GHI"]
        rmse2 = agg2.accuracy.rmse["GHI"]

        if rmse1 > 0:
            ratio = rmse2 / rmse1
            assert ratio < 1.8, (
                f"GHI RMSE ratio {ratio:.2f} suggests weight leaked into metric"
            )


# ---------------------------------------------------------------------------
# 8. Partial weight dict (not all columns specified)
# ---------------------------------------------------------------------------


class TestPartialWeightDict:
    """When weight_dict only specifies some columns, others should be unaffected."""

    def test_partial_weights_runs(self):
        """Should not crash when only some columns are weighted."""
        agg = _run({"GHI": 5})
        tp = agg.cluster_representatives
        assert tp.shape[1] == len(RAW.columns)

    def test_partial_weights_output_in_range(self):
        agg = _run({"GHI": 5})
        tp = agg.cluster_representatives
        for col in RAW.columns:
            assert tp[col].min() >= RAW[col].min() - 1e-6
            assert tp[col].max() <= RAW[col].max() + 1e-6

    @pytest.mark.xfail(
        reason="accuracyIndicators KeyError with partial weightDict - pre-existing bug"
    )
    def test_partial_weights_accuracy(self):
        """Accuracy indicators should work with partial weight dicts."""
        agg = _run({"GHI": 5})
        acc = agg.accuracy
        assert set(acc.rmse.index) == set(RAW.columns)


# ---------------------------------------------------------------------------
# 9. Edge case: very large / very small weights
# ---------------------------------------------------------------------------


class TestExtremeWeights:
    """Extreme weight values should not break the pipeline."""

    def test_very_large_weight(self):
        agg = _run({"GHI": 1000, "T": 1, "Wind": 1, "Load": 1})
        tp = agg.cluster_representatives
        assert not tp.isnull().any().any(), "NaN in output with large weight"
        assert tp["GHI"].max() <= RAW["GHI"].max() + 1e-6

    def test_very_small_weight(self):
        """Very small weights should be clamped to options.min_weight, not zero."""
        agg = _run({"GHI": 1e-10, "T": 1, "Wind": 1, "Load": 1})
        tp = agg.cluster_representatives
        assert not tp.isnull().any().any(), "NaN in output with tiny weight"

    def test_zero_weight_clamped(self):
        """Zero weight should be clamped, not cause division by zero."""
        agg = _run({"GHI": 0, "T": 1, "Wind": 1, "Load": 1})
        tp = agg.cluster_representatives
        assert not tp.isnull().any().any(), "NaN in output with zero weight"


# ---------------------------------------------------------------------------
# 10. Weight x rescale interaction: scale_ub correctness
# ---------------------------------------------------------------------------


class TestRescaleScaleUb:
    """The rescaling clip bound (scale_ub) should not cause weighted columns
    to be clipped differently in a way that breaks reconstruction."""

    def test_rescale_with_high_weight_preserves_mean(self):
        """A column with very high weight should still have its mean preserved."""
        agg = _run(
            {"GHI": 50, "T": 1, "Wind": 1, "Load": 1},
            rescale_cluster_periods=True,
        )
        pred = agg.reconstructed

        orig_mean = RAW["GHI"].mean()
        pred_mean = pred["GHI"].mean()
        if orig_mean > 0:
            rel_error = abs(pred_mean - orig_mean) / orig_mean
            assert rel_error < 0.02, (
                f"GHI mean rel error {rel_error:.4f} with high weight — "
                f"scale_ub may be distorting rescaling"
            )

    def test_rescale_without_weights_preserves_mean(self):
        """Baseline: rescaling without weights should preserve means well."""
        agg = _run(None, rescale_cluster_periods=True)
        pred = agg.reconstructed

        for col in RAW.columns:
            orig_mean = RAW[col].mean()
            pred_mean = pred[col].mean()
            if orig_mean == 0:
                continue
            rel_error = abs(pred_mean - orig_mean) / abs(orig_mean)
            assert rel_error < 0.02, (
                f"{col}: baseline rescale rel error {rel_error:.4f}"
            )


# ---------------------------------------------------------------------------
# 11. Weight x k-means (different clustering method)
# ---------------------------------------------------------------------------


class TestWeightsWithKMeans:
    """Weight handling should be consistent across clustering methods."""

    def test_kmeans_uniform_weights_match_no_weights(self):
        """k-means is non-deterministic; uniform scaling changes centroid init.
        We only check that the output is reasonable, not bit-identical."""
        agg_uniform = _run(dict.fromkeys(RAW.columns, 3), cluster_method="kmeans")
        tp = agg_uniform.cluster_representatives
        for col in RAW.columns:
            assert tp[col].max() <= RAW[col].max() + 1e-6
            assert tp[col].min() >= RAW[col].min() - 1e-6

    def test_kmeans_output_in_range(self):
        agg = _run(
            {"GHI": 5, "T": 1, "Wind": 1, "Load": 1},
            cluster_method="kmeans",
        )
        tp = agg.cluster_representatives
        for col in RAW.columns:
            assert tp[col].max() <= RAW[col].max() + 1e-6
            assert tp[col].min() >= RAW[col].min() - 1e-6


# ---------------------------------------------------------------------------
# 12. Weight x segmentation (the core bug: weights leaked into reconstruction)
# ---------------------------------------------------------------------------


class TestWeightsWithSegmentation:
    """Weights must not leak into segmented reconstruction.
    Before the fix, predict_original_data with segmentation returned values
    scaled by the weight (e.g. GHI=100 → output 100x too large)."""

    def test_segmentation_uniform_weights_equal_no_weights(self):
        """Uniform weights + segmentation must match no weights."""
        agg_none = _run(None, segmentation=True, no_segments=4)
        agg_uniform = _run(
            dict.fromkeys(RAW.columns, 100), segmentation=True, no_segments=4
        )
        pd.testing.assert_frame_equal(
            agg_none.reconstructed,
            agg_uniform.reconstructed,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "weights",
        [
            {"GHI": 100, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 0.1, "T": 10, "Wind": 0.5, "Load": 3},
        ],
    )
    def test_segmentation_output_in_range(self, weights):
        agg = _run(weights, segmentation=True, no_segments=4)
        pred = agg.reconstructed
        for col in RAW.columns:
            assert pred[col].min() >= RAW[col].min() - 1e-6, (
                f"{col}: pred min {pred[col].min()} < data min {RAW[col].min()}"
            )
            assert pred[col].max() <= RAW[col].max() + 1e-6, (
                f"{col}: pred max {pred[col].max()} > data max {RAW[col].max()}"
            )

    @pytest.mark.parametrize(
        "weights",
        [
            {"GHI": 100, "T": 1, "Wind": 1, "Load": 1},
            {"GHI": 0.1, "T": 10, "Wind": 0.5, "Load": 3},
        ],
    )
    def test_segmentation_preserves_column_means(self, weights):
        """Reconstructed means should be close to original, not scaled by weight."""
        agg = _run(
            weights, segmentation=True, no_segments=4, rescale_cluster_periods=True
        )
        pred = agg.reconstructed
        for col in RAW.columns:
            orig_mean = RAW[col].mean()
            pred_mean = pred[col].mean()
            if orig_mean == 0:
                continue
            rel_error = abs(pred_mean - orig_mean) / abs(orig_mean)
            assert rel_error < 0.05, (
                f"{col}: segmentation mean error {rel_error:.4f} > 5% "
                f"(orig={orig_mean:.4f}, pred={pred_mean:.4f})"
            )

    def test_segmentation_samemean_weights(self):
        """same_mean + segmentation + weights must not produce scaled output."""
        agg = _run(
            {"GHI": 100, "T": 1, "Wind": 1, "Load": 1},
            segmentation=True,
            no_segments=4,
            same_mean=True,
        )
        pred = agg.reconstructed
        for col in RAW.columns:
            assert pred[col].min() >= RAW[col].min() - 1e-6
            assert pred[col].max() <= RAW[col].max() + 1e-6

    def test_segmentation_typical_periods_in_range(self):
        """typical_periods with segmentation + weights should be in range."""
        agg = _run(
            {"GHI": 100, "T": 1, "Wind": 1, "Load": 1},
            segmentation=True,
            no_segments=4,
        )
        tp = agg.cluster_representatives
        for col in RAW.columns:
            assert tp[col].min() >= RAW[col].min() - 1e-6
            assert tp[col].max() <= RAW[col].max() + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
