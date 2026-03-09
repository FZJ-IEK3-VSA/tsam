"""Smoke tests for tsam.plot — every plot method returns a go.Figure."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

import tsam
from conftest import TESTDATA_CSV
from tsam.plot import ResultPlotAccessor, _validate_columns


@pytest.fixture(scope="module")
def sample_data() -> pd.DataFrame:
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


@pytest.fixture(scope="module")
def result(sample_data) -> tsam.AggregationResult:
    return tsam.aggregate(sample_data, n_clusters=8)


@pytest.fixture(scope="module")
def result_segmented(sample_data) -> tsam.AggregationResult:
    return tsam.aggregate(
        sample_data, n_clusters=8, segments=tsam.SegmentConfig(n_segments=4)
    )


# ---- _validate_columns ---------------------------------------------------


class TestValidateColumns:
    def test_none_returns_all(self):
        assert _validate_columns(None, ["a", "b"]) == ["a", "b"]

    def test_valid_subset(self):
        assert _validate_columns(["b"], ["a", "b", "c"]) == ["b"]

    def test_invalid_warns(self):
        with pytest.warns(UserWarning, match="not found"):
            result = _validate_columns(["a", "x"], ["a", "b"])
        assert result == ["a"]

    def test_all_invalid_raises(self):
        with pytest.raises(ValueError, match="None of the requested"):
            _validate_columns(["x", "y"], ["a", "b"])


# ---- Accessor access ------------------------------------------------------


class TestAccessor:
    def test_plot_returns_accessor(self, result):
        assert isinstance(result.plot, ResultPlotAccessor)


# ---- cluster_representatives ----------------------------------------------


class TestClusterRepresentatives:
    def test_returns_figure(self, result):
        fig = result.plot.cluster_representatives()
        assert isinstance(fig, go.Figure)

    def test_with_columns(self, result):
        col = result.original.columns[0]
        fig = result.plot.cluster_representatives(columns=[col])
        assert isinstance(fig, go.Figure)


# ---- cluster_members -------------------------------------------------------


class TestClusterMembers:
    def test_returns_figure(self, result):
        fig = result.plot.cluster_members()
        assert isinstance(fig, go.Figure)
        assert fig.frames  # should have animation frames

    def test_single_column(self, result):
        col = result.original.columns[0]
        fig = result.plot.cluster_members(columns=[col])
        assert isinstance(fig, go.Figure)

    def test_specific_clusters(self, result):
        fig = result.plot.cluster_members(clusters=[0, 1])
        assert isinstance(fig, go.Figure)

    def test_animate_column(self, result):
        fig = result.plot.cluster_members(animate="Column")
        assert isinstance(fig, go.Figure)
        assert fig.frames

    def test_invalid_animate_raises(self, result):
        with pytest.raises(ValueError, match="animate must be"):
            result.plot.cluster_members(animate="invalid")

    def test_invalid_clusters_warns(self, result):
        with pytest.warns(UserWarning, match="not found"):
            fig = result.plot.cluster_members(clusters=[0, 9999])
        assert isinstance(fig, go.Figure)

    def test_all_invalid_clusters_raises(self, result):
        with pytest.raises(ValueError, match="None of the requested"):
            result.plot.cluster_members(clusters=[9999])

    def test_with_segmentation(self, result_segmented):
        fig = result_segmented.plot.cluster_members()
        assert isinstance(fig, go.Figure)


# ---- cluster_weights -------------------------------------------------------


class TestClusterWeights:
    def test_returns_figure(self, result):
        fig = result.plot.cluster_weights()
        assert isinstance(fig, go.Figure)


# ---- accuracy --------------------------------------------------------------


class TestAccuracy:
    def test_returns_figure(self, result):
        fig = result.plot.accuracy()
        assert isinstance(fig, go.Figure)


# ---- segment_durations ----------------------------------------------------


class TestSegmentDurations:
    def test_returns_figure(self, result_segmented):
        fig = result_segmented.plot.segment_durations()
        assert isinstance(fig, go.Figure)

    def test_raises_without_segmentation(self, result):
        with pytest.raises(ValueError, match="No segmentation"):
            result.plot.segment_durations()


# ---- compare ---------------------------------------------------------------


class TestCompare:
    def test_overlay(self, result):
        fig = result.plot.compare()
        assert isinstance(fig, go.Figure)

    def test_side_by_side(self, result):
        fig = result.plot.compare(mode="side_by_side")
        assert isinstance(fig, go.Figure)

    def test_duration_curve(self, result):
        fig = result.plot.compare(mode="duration_curve")
        assert isinstance(fig, go.Figure)

    def test_with_columns(self, result):
        col = result.original.columns[0]
        fig = result.plot.compare(columns=[col])
        assert isinstance(fig, go.Figure)

    def test_invalid_mode_raises(self, result):
        with pytest.raises(ValueError, match="Unknown mode"):
            result.plot.compare(mode="invalid")


# ---- residuals -------------------------------------------------------------


class TestResiduals:
    def test_time_series(self, result):
        fig = result.plot.residuals()
        assert isinstance(fig, go.Figure)

    def test_histogram(self, result):
        fig = result.plot.residuals(mode="histogram")
        assert isinstance(fig, go.Figure)

    def test_by_period(self, result):
        fig = result.plot.residuals(mode="by_period")
        assert isinstance(fig, go.Figure)

    def test_by_timestep(self, result):
        fig = result.plot.residuals(mode="by_timestep")
        assert isinstance(fig, go.Figure)

    def test_with_columns(self, result):
        col = result.original.columns[0]
        fig = result.plot.residuals(columns=[col])
        assert isinstance(fig, go.Figure)

    def test_invalid_mode_raises(self, result):
        with pytest.raises(ValueError, match="Unknown mode"):
            result.plot.residuals(mode="invalid")
