"""Reusable comparison tables keep physical statistics and normalized errors distinct."""

import numpy as np
import pandas as pd
import pytest

import tsam
from tsam import ClusterConfig
from tsam.metrics import aggregation_summary, series_statistics


def test_series_statistics_preserves_units_order_and_unequal_lengths():
    original = pd.DataFrame({"load": [400, 600, 800], "solar": [0, 20, 100]})
    representative = pd.DataFrame({"solar": [0, 60], "load": [500, 700]})
    summary = series_statistics(
        {"original": original, "representative": representative}
    )
    assert list(summary.index) == ["original", "representative"]
    assert summary.columns.names == ["attribute", "statistic"]
    np.testing.assert_allclose(summary.loc["original", "load"], [400, 600, 800])
    np.testing.assert_allclose(summary.loc["representative", "solar"], [0, 30, 60])
    assert summary.loc["original", ("solar", "mean")] == 40


def test_series_statistics_column_subset_ignores_unselected_metadata():
    frame = pd.DataFrame({"load": [2, 4], "description": ["day", "night"]})
    summary = series_statistics({"raw": frame}, columns=["load"])
    assert list(summary.columns.get_level_values(0).unique()) == ["load"]
    assert summary.loc["raw", ("load", "mean")] == 3


def test_aggregation_summary_uses_normalized_weighted_errors():
    data = pd.DataFrame(
        {"a": [0, 1, 2, 0, 0, 5, 6, 0], "b": [400, 600, 800, 300, 500, 500, 700, 450]}
    )
    result = tsam.aggregate(
        data,
        n_clusters=1,
        period_duration=4,
        cluster=ClusterConfig(representation="mean"),
        weights={"a": 3, "b": 1},
        preserve_column_means=False,
    )
    summary = aggregation_summary({"mean": result})
    residuals = (data - result.reconstructed) / (data.max() - data.min())
    per_column_mse = (residuals**2).mean()
    expected = np.sqrt(np.average(per_column_mse, weights=[3, 1]))
    assert summary.loc["mean", "rmse"] == pytest.approx(expected)
    assert (
        summary.loc["mean", "rmse_duration"] == result.accuracy.weighted_rmse_duration
    )
    assert (
        summary.loc["mean", "correlation_error"] == result.concurrency.correlation_error
    )
    assert (
        summary.loc["mean", "rank_correlation_error"]
        == result.concurrency.rank_correlation_error
    )


def test_single_attribute_correlation_errors_remain_nan():
    result = tsam.aggregate(
        pd.DataFrame({"a": [0, 1, 2, 0]}), n_clusters=1, period_duration=4
    )
    summary = aggregation_summary({"one": result})
    assert (
        summary.loc["one", ["correlation_error", "rank_correlation_error"]].isna().all()
    )


@pytest.mark.parametrize(
    "series, columns, message",
    [
        ({}, None, "at least one"),
        ({"a": pd.DataFrame({"load": []})}, None, "empty"),
        ({"a": pd.DataFrame({"load": [1]})}, [], "at least one column"),
        ({"a": pd.DataFrame({"load": [1]})}, ["solar"], "missing columns"),
        ({"a": pd.DataFrame({"load": [np.nan]})}, None, "non-finite"),
        ({"a": pd.DataFrame({"load": [np.inf]})}, None, "non-finite"),
    ],
)
def test_invalid_series_fail_with_context(series, columns, message):
    with pytest.raises(ValueError, match=message):
        series_statistics(series, columns=columns)


def test_empty_aggregation_comparison_is_rejected():
    with pytest.raises(ValueError, match="at least one aggregation result"):
        aggregation_summary({})
