"""Smoke tests for tsam.plot — every plot method returns a go.Figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import tsam
from conftest import TESTDATA_CSV
from tsam.plot import (
    AttributeSpace,
    ResultPlotAccessor,
    _validate_columns,
    compare_partitions,
)


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

    def test_with_segmentation(self, result_segmented):
        # Segmented results have a 3-level representatives index
        # (period, segment step, segment duration); plotting must not crash.
        fig = result_segmented.plot.cluster_representatives()
        assert isinstance(fig, go.Figure)

    def test_segmentation_trace_matches_data(self, result_segmented):
        """Segmented representative trace contains the segment values."""
        col = result_segmented.original.columns[0]
        cluster_id = sorted(set(result_segmented.cluster_assignments))[0]
        fig = result_segmented.plot.cluster_representatives(columns=[col])

        expected = result_segmented.cluster_representatives.loc[cluster_id, col].values
        trace = next(t for t in fig.data if f"Period {cluster_id} " in t.name)
        np.testing.assert_array_almost_equal(np.asarray(trace.y, dtype=float), expected)


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

    def test_slider_column(self, result):
        fig = result.plot.cluster_members(slider="column")
        assert isinstance(fig, go.Figure)
        assert fig.frames

    def test_invalid_slider_raises(self, result):
        with pytest.raises(ValueError, match="slider must be"):
            result.plot.cluster_members(slider="invalid")

    def test_invalid_clusters_warns(self, result):
        with pytest.warns(UserWarning, match="not found"):
            fig = result.plot.cluster_members(clusters=[0, 9999])
        assert isinstance(fig, go.Figure)

    def test_all_invalid_clusters_raises(self, result):
        with pytest.raises(ValueError, match="None of the requested"):
            result.plot.cluster_members(clusters=[9999])

    def test_representative_trace_matches_data(self, result):
        """Verify the representative trace contains correct values from cluster_representatives."""
        col = result.original.columns[0]
        cluster_id = sorted(set(result.cluster_assignments))[0]
        fig = result.plot.cluster_members(columns=[col], clusters=[cluster_id])

        # With single column + single cluster: traces are [member, representative]
        rep_trace = fig.data[1]
        expected = result.cluster_representatives.loc[cluster_id, col].values
        np.testing.assert_array_almost_equal(rep_trace.y, expected)

    def test_member_trace_contains_all_members(self, result):
        """Verify member trace has data for every period assigned to the cluster."""
        col = result.original.columns[0]
        cluster_id = sorted(set(result.cluster_assignments))[0]
        n_members = int((result.cluster_assignments == cluster_id).sum())
        n_ts = result.n_timesteps_per_period
        fig = result.plot.cluster_members(columns=[col], clusters=[cluster_id])

        member_trace = fig.data[0]
        # NaN-separated: n_members segments of n_ts values, with n_members-1 NaN separators
        y = np.array(member_trace.y, dtype=float)
        expected_len = n_members * n_ts + (n_members - 1)
        assert len(y) == expected_len
        # Count NaN separators
        assert np.isnan(y).sum() == n_members - 1

    def test_with_segmentation(self, result_segmented):
        fig = result_segmented.plot.cluster_members()
        assert isinstance(fig, go.Figure)

    def test_segmented_representative_expanded(self, result_segmented):
        """Verify segmented representative is expanded to full timestep length."""
        col = result_segmented.original.columns[0]
        cluster_id = sorted(set(result_segmented.cluster_assignments))[0]
        fig = result_segmented.plot.cluster_members(
            columns=[col], clusters=[cluster_id]
        )
        rep_trace = fig.data[1]
        # Should be expanded to n_timesteps_per_period, not n_segments
        assert len(rep_trace.y) == result_segmented.n_timesteps_per_period


# ---- clusters_over_time ----------------------------------------------------


class TestClustersOverTime:
    def test_returns_figure(self, result):
        fig = result.plot.clusters_over_time()
        assert isinstance(fig, go.Figure)

    def test_single_column(self, result):
        col = result.original.columns[0]
        fig = result.plot.clusters_over_time(columns=[col])
        assert isinstance(fig, go.Figure)

    def test_reconstructed_with_overlay(self, result):
        col = result.original.columns[0]
        fig = result.plot.clusters_over_time(
            columns=[col], reconstructed=True, overlay_original=True
        )
        assert isinstance(fig, go.Figure)

    def test_consistent_cluster_colors(self, result):
        """Legend colours match the shared map used by the other cluster plots."""
        from tsam.plot import _PERIOD_SHADE_OPACITY, _cluster_color_map, _to_rgba

        cmap = _cluster_color_map(result.cluster_assignments)
        fig = result.plot.clusters_over_time(columns=[result.original.columns[0]])
        legend_colors = {
            tr.name: tr.marker.color
            for tr in fig.data
            if tr.name and tr.name.startswith("cluster ")
        }
        for cid, color in cmap.items():
            expected = _to_rgba(color, _PERIOD_SHADE_OPACITY)
            assert legend_colors[f"cluster {cid}"] == expected

    def test_with_segmentation(self, result_segmented):
        col = result_segmented.original.columns[0]
        fig = result_segmented.plot.clusters_over_time(
            columns=[col], reconstructed=True
        )
        assert isinstance(fig, go.Figure)

    def test_one_shape_per_period_per_column(self, result):
        """Each period is shaded once in every column subplot."""
        cols = list(result.original.columns)
        fig = result.plot.clusters_over_time(columns=cols)
        n_periods = len(result.cluster_assignments)
        assert len(fig.layout.shapes) == n_periods * len(cols)

    def test_period_shading_is_not_quadratic(self, result):
        """Regression guard: shapes are assigned in one batch, not per-period.

        Adding each period rectangle with ``fig.add_vrect`` re-validates every
        shape already on the figure, so a full year over several columns took
        minutes. The batched assignment is ~5000x faster; this asserts the whole
        call stays well under a second even for all columns at full resolution.
        """
        import time

        cols = list(result.original.columns)
        start = time.perf_counter()
        result.plot.clusters_over_time(columns=cols)
        elapsed = time.perf_counter() - start
        # The old path took ~20 minutes here; a generous ceiling still catches a
        # regression to per-shape validation without being flaky on a slow CI box.
        assert elapsed < 15.0, f"clusters_over_time took {elapsed:.1f}s"


# ---- cluster_counts --------------------------------------------------------


class TestClusterCounts:
    def test_returns_figure(self, result):
        fig = result.plot.cluster_counts()
        assert isinstance(fig, go.Figure)

    def test_cluster_weights_alias_deprecated(self, result):
        with pytest.warns(FutureWarning, match="cluster_weights.*deprecated"):
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

    @pytest.mark.parametrize("mode", ["overlay", "side_by_side", "duration_curve"])
    def test_time_slice(self, result, mode):
        col = result.original.columns[0]
        window = slice("2010-01-11", "2010-01-17")
        full = result.plot.compare(columns=[col], mode=mode)
        sliced = result.plot.compare(columns=[col], mode=mode, time_slice=window)
        assert isinstance(sliced, go.Figure)
        full_points = sum(len(t.x) for t in full.data if t.x is not None)
        sliced_points = sum(len(t.x) for t in sliced.data if t.x is not None)
        assert sliced_points < full_points

    @pytest.mark.parametrize("mode", ["overlay", "side_by_side", "duration_curve"])
    @pytest.mark.parametrize("color", ["column", "source"])
    def test_color_dimension(self, result, mode, color):
        fig = result.plot.compare(columns=["GHI", "Load"], mode=mode, color=color)
        assert isinstance(fig, go.Figure)

    def test_color_source_swaps_legend(self, result):
        # color drives the legend group; "column" vs "source" must differ.
        by_col = result.plot.compare(columns=["Load"], color="column")
        by_src = result.plot.compare(columns=["Load"], color="source")
        col_groups = {t.legendgroup for t in by_col.data}
        src_groups = {t.legendgroup for t in by_src.data}
        assert col_groups != src_groups

    def test_invalid_color_raises(self, result):
        with pytest.raises(ValueError, match="color must be"):
            result.plot.compare(color="invalid")


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


# ---- AttributeSpace --------------------------------------------------------


class TestAttributeSpace:
    def test_returns_figure(self):
        space = AttributeSpace("solar", "load")
        space.add_path([0.0, 1.0, 0.5], [3.0, 4.0, 3.5], name="day 0")
        assert isinstance(space.figure, go.Figure)

    def test_axis_labels_use_units(self):
        space = AttributeSpace("solar", "load", units={"solar": "W/m²"})
        assert space.figure.layout.xaxis.title.text == "solar [W/m²]"
        # an attribute without a unit is labelled bare
        assert space.figure.layout.yaxis.title.text == "load"

    def test_add_path_is_chainable(self):
        space = AttributeSpace("a", "b")
        returned = space.add_path([0, 1], [0, 1], name="one").add_path(
            [1, 2], [1, 2], name="two"
        )
        assert returned is space

    def test_arrows_add_a_hidden_trace(self):
        """Each path is one visible trace, plus an arrowhead trace when arrows=True."""
        with_arrows = AttributeSpace("a", "b")
        with_arrows.add_path([0, 1, 2], [0, 1, 0], name="p", arrows=True)
        without = AttributeSpace("a", "b")
        without.add_path([0, 1, 2], [0, 1, 0], name="p", arrows=False)

        assert len(with_arrows.figure.data) == 2
        assert len(without.figure.data) == 1
        arrow_trace = with_arrows.figure.data[1]
        assert arrow_trace.marker.symbol == "arrow"
        assert arrow_trace.showlegend is False
        # the first point has no incoming segment, so it carries no arrowhead
        assert arrow_trace.marker.size[0] == 0

    def test_single_point_path_has_no_arrows(self):
        space = AttributeSpace("a", "b")
        space.add_path([0.0], [0.0], name="p", arrows=True)
        assert len(space.figure.data) == 1

    def test_symbols_cycle_so_paths_differ_without_colour(self):
        space = AttributeSpace("a", "b")
        for i in range(3):
            space.add_path([0, 1], [i, i], name=f"p{i}", arrows=False)
        symbols = [trace.marker.symbol for trace in space.figure.data]
        assert len(set(symbols)) == 3

    def test_explicit_symbol_and_colour_are_respected(self):
        space = AttributeSpace("a", "b")
        space.add_path(
            [0, 1], [0, 1], name="p", symbol="square", color="#123456", arrows=False
        )
        assert space.figure.data[0].marker.symbol == "square"
        assert space.figure.data[0].marker.color == "#123456"

    def test_mismatched_lengths_raise(self):
        space = AttributeSpace("a", "b")
        with pytest.raises(ValueError, match="same shape"):
            space.add_path([0, 1, 2], [0, 1], name="p")


# ---- feature_space ---------------------------------------------------------


@pytest.fixture(scope="module")
def designed_days() -> pd.DataFrame:
    """The 12 designed days used by the clustering tutorial.

    Five sunny days are identical to each other and two cloudy pairs sit almost
    on top of each other, which is exactly the overplotting the marker merging
    has to survive.
    """
    return pd.read_csv(
        TESTDATA_CSV.parent / "comparison_days.csv", index_col=0, parse_dates=True
    )


ARCHETYPES = [
    "sunny",
    "cloudy",
    "high-load",
    "sunny",
    "cloudy",
    "storm",
    "cloudy",
    "sunny",
    "high-load",
    "sunny",
    "cloudy",
    "sunny",
]


@pytest.fixture(scope="module")
def designed_result(designed_days) -> tsam.AggregationResult:
    return tsam.aggregate(
        designed_days,
        n_clusters=3,
        period_duration="1D",
        preserve_column_means=False,
    )


class TestFeatureSpace:
    def test_returns_figure(self, designed_result):
        assert isinstance(designed_result.plot.feature_space(), go.Figure)

    def test_labels_become_marker_traces(self, designed_result):
        fig = designed_result.plot.feature_space(labels=ARCHETYPES)
        names = {trace.name for trace in fig.data}
        assert {"sunny", "cloudy", "high-load", "storm"} <= names

    def test_coincident_periods_merge_into_one_counted_marker(self, designed_result):
        """The five sunny days share a cluster and a location: one marker, "x5"."""
        fig = designed_result.plot.feature_space(labels=ARCHETYPES)
        sunny = next(t for t in fig.data if t.name == "sunny")
        assert len(sunny.x) == 1
        assert sunny.text[0].endswith("5")

    def test_merge_tolerance_zero_separates_merely_nearby_periods(
        self, designed_result
    ):
        """Tolerance 0 merges only exact duplicates, not near-neighbours.

        The five sunny days sit at two distinct spots (three exactly on one,
        two on another), so dropping the tolerance must split them back into
        two markers while still collapsing each exact duplicate group.
        """
        fig = designed_result.plot.feature_space(labels=ARCHETYPES, merge_tolerance=0.0)
        sunny = next(t for t in fig.data if t.name == "sunny")
        assert len(sunny.x) == 2
        assert sorted(t[-1] for t in sunny.text) == ["2", "3"]

    def test_axes_locked_to_equal_aspect(self, designed_result):
        """Distance is the subject of the plot, so it must not be distorted."""
        fig = designed_result.plot.feature_space()
        assert fig.layout.yaxis.scaleanchor == "x"
        assert fig.layout.yaxis.scaleratio == 1

    def test_medoid_centers_are_marked_as_real_periods(self, designed_days):
        """A medoid centre is an observed period; the legend must say so."""
        medoid = tsam.aggregate(
            designed_days,
            n_clusters=3,
            period_duration="1D",
            cluster=tsam.ClusterConfig(method="kmedoids"),
            preserve_column_means=False,
        )
        names = " ".join(t.name or "" for t in medoid.plot.feature_space().data)
        assert "real period" in names

    def test_mean_centers_are_marked_as_synthetic(self, designed_days):
        kmeans = tsam.aggregate(
            designed_days,
            n_clusters=3,
            period_duration="1D",
            cluster=tsam.ClusterConfig(method="kmeans"),
            preserve_column_means=False,
        )
        names = " ".join(t.name or "" for t in kmeans.plot.feature_space().data)
        assert "(mean)" in names

    def test_wrong_label_count_raises(self, designed_result):
        with pytest.raises(ValueError, match="one label per period"):
            designed_result.plot.feature_space(labels=["only", "two"])

    def test_hiding_centers_and_assignments(self, designed_result):
        bare = designed_result.plot.feature_space(
            show_centers=False, show_assignments=False
        )
        full = designed_result.plot.feature_space()
        assert len(bare.data) < len(full.data)


# ---- compare_partitions ----------------------------------------------------


class TestComparePartitions:
    def test_returns_figure(self, designed_result):
        fig = compare_partitions({"run": designed_result})
        assert isinstance(fig, go.Figure)

    def test_identical_groupings_look_identical_after_canonicalisation(self):
        """Same partition, different ids — the whole point of canonical=True."""
        fig = compare_partitions({"a": [2, 2, 0, 0], "b": [7, 7, 3, 3]})
        rows = np.asarray(fig.data[0].z)
        assert np.array_equal(rows[0], rows[1])

    def test_raw_ids_survive_when_canonical_is_off(self):
        fig = compare_partitions({"a": [2, 2, 0, 0]}, canonical=False)
        assert list(np.asarray(fig.data[0].z)[0]) == [2, 2, 0, 0]

    def test_accepts_results_and_raw_sequences_together(self, designed_result):
        fig = compare_partitions(
            {"result": designed_result, "manual": [0] * 12},
            labels=ARCHETYPES,
        )
        assert np.asarray(fig.data[0].z).shape == (2, 12)

    def test_labels_annotate_the_period_axis(self, designed_result):
        fig = compare_partitions({"run": designed_result}, labels=ARCHETYPES)
        assert any("storm" in tick for tick in fig.layout.xaxis.ticktext)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one clustering"):
            compare_partitions({})

    def test_ragged_partitions_raise(self):
        with pytest.raises(ValueError, match="same number of periods"):
            compare_partitions({"a": [0, 1], "b": [0, 1, 2]})

    def test_wrong_label_count_raises(self, designed_result):
        with pytest.raises(ValueError, match="labels has"):
            compare_partitions({"run": designed_result}, labels=["one"])
