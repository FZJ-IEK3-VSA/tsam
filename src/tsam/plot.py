"""Plotting accessor for tsam aggregation results.

Provides convenient plotting methods directly on the result object for
validation and visualization of aggregation quality.

Usage:
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> result.plot.compare()  # Compare original vs reconstructed
    >>> result.plot.residuals()  # View reconstruction errors
    >>> result.plot.cluster_representatives()
    >>> result.plot.cluster_members()  # All periods per cluster
    >>> result.plot.cluster_weights()
    >>> result.plot.accuracy()

For exploring raw data before aggregation, use plotly directly with
``tsam.unstack_to_periods()`` to reshape data for heatmaps:
    >>> import plotly.express as px
    >>> unstacked = tsam.unstack_to_periods(df, period_duration=24)
    >>> px.imshow(unstacked["Load"].values.T)

Note: This module requires the 'plotly' optional dependency.
Install with: pip install tsam[plot]
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "The tsam.plot module requires plotly. Install it with: pip install tsam[plot]"
    ) from e

if TYPE_CHECKING:
    from tsam.result import AggregationResult


def _validate_columns(
    requested: list[str] | None,
    available: list[str],
    context: str = "data",
) -> list[str]:
    """Validate and filter column names, warning about invalid ones.

    Parameters
    ----------
    requested : list[str] | None
        Columns requested by user. If None, returns all available.
    available : list[str]
        Columns available in the data.
    context : str
        Description for error messages (e.g., "original data").

    Returns
    -------
    list[str]
        Valid columns to use.

    Raises
    ------
    ValueError
        If no valid columns remain after filtering.
    """
    if requested is None:
        return available

    valid = [c for c in requested if c in available]
    invalid = [c for c in requested if c not in available]

    if invalid:
        warnings.warn(
            f"Columns not found in {context} and will be ignored: {invalid}. "
            f"Available columns: {available}",
            UserWarning,
            stacklevel=3,
        )

    if not valid:
        raise ValueError(
            f"None of the requested columns {requested} exist in {context}. "
            f"Available columns: {available}"
        )

    return valid


def _duration_curve_figure(
    results: dict[str, pd.DataFrame],
    columns: list[str],
    title: str | None = None,
) -> go.Figure:
    """Create duration curve comparison figure (internal helper)."""
    frames = []
    for name, data in results.items():
        for col in columns:
            sorted_vals = data[col].sort_values(ascending=False).reset_index(drop=True)
            frames.append(
                pd.DataFrame(
                    {
                        "Hour": range(len(sorted_vals)),
                        "Value": sorted_vals.values,
                        "Method": name,
                        "Column": col,
                    }
                )
            )
    long_df = pd.concat(frames, ignore_index=True)
    return px.line(
        long_df,
        x="Hour",
        y="Value",
        color="Column",
        line_dash="Method",
        title=title or "Duration Curve Comparison",
    )


class ResultPlotAccessor:
    """Plotting accessor for AggregationResult.

    Provides convenient plotting methods directly on the result object.

    Examples
    --------
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> result.plot.compare()  # Compare original vs reconstructed
    >>> result.plot.residuals()  # View reconstruction errors
    >>> result.plot.cluster_representatives()
    >>> result.plot.cluster_members()
    >>> result.plot.cluster_weights()
    """

    def __init__(self, result: AggregationResult):
        self._result = result

    def cluster_representatives(
        self,
        columns: list[str] | None = None,
        title: str = "Cluster Representatives",
    ) -> go.Figure:
        """Plot all cluster representatives (typical periods).

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot.
        title : str, default "Cluster Representatives"
            Plot title.

        Returns
        -------
        go.Figure
        """
        typ = self._result.cluster_representatives
        weights = self._result.cluster_weights

        available_columns = [c for c in typ.columns if c not in ["cluster", "timestep"]]
        columns = _validate_columns(
            columns, available_columns, "cluster_representatives"
        )

        # Reset index to get period/timestep as columns
        df = typ[columns].reset_index()
        df.columns = pd.Index(["Period", "Timestep", *columns])

        # Map period IDs to labels with weights
        df["Period"] = df["Period"].map(lambda p: f"Period {p} (n={weights.get(p, 1)})")

        long_df = df.melt(
            id_vars=["Period", "Timestep"],
            var_name="Column",
            value_name="Value",
        )

        fig = px.line(
            long_df,
            x="Timestep",
            y="Value",
            color="Period",
            facet_col="Column" if len(columns) > 1 else None,
            title=title,
        )

        return fig

    def cluster_members(
        self,
        columns: list[str] | None = None,
        clusters: list[int] | None = None,
        animate: str = "Cluster",
        title: str | None = None,
    ) -> go.Figure:
        """Plot all original periods grouped by cluster with representative highlighted.

        Shows individual member periods as faint lines and the cluster
        representative as a bold line. A slider lets you flip through
        either clusters or columns.

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot. If None, plots all columns.
        clusters : list[int], optional
            Cluster indices to include. If None, includes all clusters.
        animate : str, default "Cluster"
            Which dimension to put on the animation slider.
            The other dimension becomes ``facet_col``.

            - ``"Cluster"``: slider flips through clusters, columns are facets.
            - ``"Column"``: slider flips through columns, clusters are facets.
        title : str, optional
            Plot title. Defaults to "Cluster Members".

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.cluster_members(columns=["Load"])
        >>> result.plot.cluster_members(clusters=[0, 3])  # specific clusters
        >>> result.plot.cluster_members(animate="Column")  # flip through columns
        """
        from plotly.subplots import make_subplots

        from tsam.api import unstack_to_periods

        result = self._result
        columns = _validate_columns(
            columns, list(result.original.columns), "original data"
        )
        n_ts = result.n_timesteps_per_period
        idx = result.original.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
            timestep_hours = (idx[1] - idx[0]).total_seconds() / 3600
        else:
            timestep_hours = 1.0
        unstacked = unstack_to_periods(result.original, n_ts * timestep_hours)
        assignments = result.cluster_assignments
        representatives = result.cluster_representatives
        weights = result.cluster_weights
        timesteps = np.arange(n_ts)

        all_cluster_ids = sorted(set(assignments))
        if clusters is not None:
            invalid = [c for c in clusters if c not in all_cluster_ids]
            if invalid:
                warnings.warn(
                    f"Cluster indices not found and will be ignored: {invalid}. "
                    f"Available clusters: {all_cluster_ids}",
                    UserWarning,
                    stacklevel=2,
                )
            cluster_ids = [c for c in clusters if c in all_cluster_ids]
            if not cluster_ids:
                raise ValueError(
                    f"None of the requested clusters {clusters} exist. "
                    f"Available clusters: {all_cluster_ids}"
                )
        else:
            cluster_ids = all_cluster_ids
        members_by_cluster = {
            cid: np.where(assignments == cid)[0] for cid in cluster_ids
        }

        def _rep_values(cluster_id: int, col: str) -> np.ndarray:
            """Get representative values expanded to full timesteps."""
            rep = representatives.loc[cluster_id]
            if result.n_segments is not None:
                durations = rep.index.get_level_values("Segment Duration").astype(int)
                return np.repeat(rep[col].values, durations)
            return rep[col].values  # type: ignore[no-any-return]

        if animate not in ("Cluster", "Column"):
            raise ValueError(f"animate must be 'Cluster' or 'Column', got {animate!r}")

        # Pre-extract member data as numpy arrays for fast access.
        # member_arrays[cid][col] = 2D array (n_members, n_ts)
        member_arrays: dict[int, dict[str, np.ndarray]] = {}
        for cid in cluster_ids:
            members = members_by_cluster[cid]
            member_arrays[cid] = {
                col: np.asarray(unstacked[col].iloc[members].values) for col in columns
            }

        cluster_labels = {
            cid: f"Cluster {cid} (n={weights.get(cid, 1)})" for cid in cluster_ids
        }

        # Determine which dimension is animated vs faceted.
        anim_keys: list[int | str]
        if animate == "Cluster":
            anim_keys = list(cluster_ids)
            anim_labels = [cluster_labels[c] for c in cluster_ids]
            facet_labels = columns
        else:
            anim_keys = list(columns)
            anim_labels = list(columns)
            facet_labels = [cluster_labels[c] for c in cluster_ids]

        n_facets = len(facet_labels)
        traces_per_facet = 2  # one bundled member trace + one representative
        MEMBER = {"color": "rgba(99, 110, 250, 0.3)"}
        REP = {"color": "#EF553B", "width": 3}

        # Precompute NaN-separated x-arrays (one per unique member count).
        # Each member's timesteps are separated by a NaN to break the line.
        _member_x: dict[int, np.ndarray] = {}
        for cid in cluster_ids:
            n_m = len(members_by_cluster[cid])
            if n_m not in _member_x:
                tile = np.empty(n_ts + 1)
                tile[:n_ts] = timesteps
                tile[n_ts] = np.nan
                _member_x[n_m] = np.tile(tile, n_m)[:-1]

        def _member_y(cid: int, col: str) -> np.ndarray:
            """All members as NaN-separated y-values (vectorized)."""
            data = member_arrays[cid][col]  # (n_members, n_ts)
            padded = np.column_stack([data, np.full(data.shape[0], np.nan)])
            return padded.ravel()[:-1]

        def _frame_traces(anim_key: int | str) -> list[go.Scatter]:
            """Build Scatter traces for one animation frame."""
            out: list[go.Scatter] = []
            first_member = True
            first_rep = True
            for facet_idx in range(n_facets):
                if animate == "Cluster":
                    cid, col = cast("int", anim_key), columns[facet_idx]
                else:
                    cid, col = cluster_ids[facet_idx], cast("str", anim_key)

                n_m = len(members_by_cluster[cid])
                out.append(
                    go.Scatter(
                        x=_member_x[n_m],
                        y=_member_y(cid, col),
                        mode="lines",
                        line=MEMBER,
                        name="Member",
                        legendgroup="Member",
                        showlegend=first_member,
                    )
                )
                first_member = False

                out.append(
                    go.Scatter(
                        x=timesteps,
                        y=_rep_values(cid, col),
                        mode="lines",
                        line=REP,
                        name="Representative",
                        legendgroup="Representative",
                        showlegend=first_rep,
                    )
                )
                first_rep = False

            return out

        # Build figure with subplots for facets.
        if n_facets > 1:
            fig = make_subplots(rows=1, cols=n_facets, subplot_titles=facet_labels)
        else:
            fig = go.Figure()

        # Initial traces (first animation frame).
        initial = _frame_traces(anim_keys[0])
        if n_facets > 1:
            rows = [1] * len(initial)
            cols_idx = [i // traces_per_facet + 1 for i in range(len(initial))]
            fig.add_traces(initial, rows=rows, cols=cols_idx)
        else:
            fig.add_traces(initial)

        # Animation frames.
        fig.frames = [
            go.Frame(data=_frame_traces(key), name=label)
            for key, label in zip(anim_keys, anim_labels)
        ]

        # Slider.
        steps = [
            {
                "args": [
                    [f.name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                    },
                ],
                "label": f.name,
                "method": "animate",
            }
            for f in fig.frames
        ]
        fig.update_layout(
            sliders=[{"active": 0, "steps": steps}],
            title=title or "Cluster Members",
        )

        # Y-axis scaling.
        if animate == "Cluster":
            # Facets are columns (different units) — independent y-axes,
            # fixed across all cluster frames.
            if n_facets > 1:
                fig.update_yaxes(matches=None, showticklabels=True)
            for i, col in enumerate(columns):
                vals = np.concatenate(
                    [member_arrays[cid][col].ravel() for cid in cluster_ids]
                )
                ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
                margin = (ymax - ymin) * 0.05
                key = "yaxis" if i == 0 else f"yaxis{i + 1}"
                fig.layout[key].range = [ymin - margin, ymax + margin]
        else:
            # Facets are clusters (same column) — y-axis range adapts per
            # column frame.
            for frame_idx, col in enumerate(columns):
                vals = np.concatenate(
                    [member_arrays[cid][col].ravel() for cid in cluster_ids]
                )
                ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
                margin = (ymax - ymin) * 0.05
                n_axes = max(n_facets, 1)
                axis_ranges = {}
                for i in range(n_axes):
                    key = "yaxis" if i == 0 else f"yaxis{i + 1}"
                    axis_ranges[key] = {"range": [ymin - margin, ymax + margin]}
                fig.frames[frame_idx].layout = go.Layout(**axis_ranges)
            if fig.frames:
                for key, val in fig.frames[0].layout.to_plotly_json().items():
                    if key.startswith("yaxis"):
                        fig.layout[key].range = val["range"]

        return fig

    def cluster_weights(self, title: str = "Cluster Weights") -> go.Figure:
        """Plot cluster weight distribution.

        Parameters
        ----------
        title : str, default "Cluster Weights"
            Plot title.

        Returns
        -------
        go.Figure
        """
        weights = self._result.cluster_weights
        df = pd.DataFrame(
            {
                "Period": [f"Period {p}" for p in weights],
                "Count": list(weights.values()),
            }
        )

        fig = px.bar(
            df,
            x="Period",
            y="Count",
            title=title,
            text="Count",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(textposition="auto")
        fig.update_layout(showlegend=False)

        return fig

    def accuracy(self, title: str = "Accuracy Metrics") -> go.Figure:
        """Plot accuracy metrics by column.

        Parameters
        ----------
        title : str, default "Accuracy Metrics"
            Plot title.

        Returns
        -------
        go.Figure
        """
        acc = self._result.accuracy
        columns = list(acc.rmse.index)

        records = []
        for col in columns:
            records.append({"Column": col, "Metric": "RMSE", "Value": acc.rmse[col]})
            records.append({"Column": col, "Metric": "MAE", "Value": acc.mae[col]})
            records.append(
                {
                    "Column": col,
                    "Metric": "RMSE (Duration)",
                    "Value": acc.rmse_duration[col],
                }
            )

        df = pd.DataFrame(records)

        fig = px.bar(
            df,
            x="Column",
            y="Value",
            color="Metric",
            barmode="group",
            title=title,
        )

        return fig

    def segment_durations(self, title: str = "Segment Durations") -> go.Figure:
        """Plot segment durations (if segmentation was used).

        Parameters
        ----------
        title : str, default "Segment Durations"
            Plot title.

        Returns
        -------
        go.Figure

        Raises
        ------
        ValueError
            If no segmentation was used.
        """
        if self._result.segment_durations is None:
            raise ValueError("No segmentation was used in this aggregation")

        # segment_durations is tuple[tuple[int, ...], ...] - one tuple per period
        # Average durations across all typical periods for the bar chart
        durations = self._result.segment_durations

        # Validate uniform structure across periods
        segment_counts = {len(period) for period in durations}
        if len(segment_counts) != 1:
            raise ValueError(
                f"Inconsistent segment counts across periods: {segment_counts}. "
                "Cannot compute average durations."
            )

        n_segments = len(durations[0])
        avg_durations = [
            sum(period[s] for period in durations) / len(durations)
            for s in range(n_segments)
        ]

        df = pd.DataFrame(
            {
                "Segment": [f"Segment {s}" for s in range(n_segments)],
                "Duration": avg_durations,
            }
        )

        fig = px.bar(
            df,
            x="Segment",
            y="Duration",
            title=title,
            text="Duration",
            color="Duration",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="auto")
        fig.update_layout(showlegend=False, yaxis_title="Duration (timesteps)")

        return fig

    def compare(
        self,
        columns: list[str] | None = None,
        mode: str = "overlay",
        title: str | None = None,
    ) -> go.Figure:
        """Compare original vs reconstructed time series.

        Parameters
        ----------
        columns : list[str], optional
            Columns to compare. If None, compares all columns.
        mode : str, default "overlay"
            Comparison mode:
            - "overlay": Both series on same axes
            - "side_by_side": Separate subplots
            - "duration_curve": Compare sorted values
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.compare()  # Compare all columns
        >>> result.plot.compare(columns=["Load"])  # Compare specific column
        >>> result.plot.compare(mode="duration_curve")
        """
        orig = self._result.original
        recon = self._result.reconstructed

        columns = _validate_columns(columns, list(orig.columns), "original data")

        if mode == "duration_curve":
            return _duration_curve_figure(
                {"Original": orig, "Reconstructed": recon},
                columns=columns,
                title=title,
            )

        elif mode in ("overlay", "side_by_side"):
            # Build long-form data with Source (Original/Reconstructed) and Column
            orig_df = orig[columns].copy()
            orig_df["Source"] = "Original"
            recon_df = recon[columns].copy()
            recon_df["Source"] = "Reconstructed"

            combined = pd.concat([orig_df, recon_df])
            combined.index.name = "Time"
            long_df = combined.reset_index().melt(
                id_vars=["Time", "Source"],
                var_name="Column",
                value_name="Value",
            )

            if mode == "overlay":
                # Color by Column, dash by Source (Original/Reconstructed)
                fig = px.line(
                    long_df,
                    x="Time",
                    y="Value",
                    color="Column",
                    line_dash="Source",
                    title=title or "Original vs Reconstructed",
                )
            else:  # side_by_side
                fig = px.line(
                    long_df,
                    x="Time",
                    y="Value",
                    color="Column",
                    facet_row="Source",
                    title=title or "Original vs Reconstructed",
                )
                fig.update_layout(height=600)

            return fig

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'overlay', 'side_by_side', or 'duration_curve'."
            )

    def residuals(
        self,
        columns: list[str] | None = None,
        mode: str = "time_series",
        title: str | None = None,
    ) -> go.Figure:
        """Plot residuals (original - reconstructed).

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot. If None, plots all.
        mode : str, default "time_series"
            Display mode:
            - "time_series": Residuals over time
            - "histogram": Distribution of residuals
            - "by_period": Mean absolute error per period (bar chart)
            - "by_timestep": Mean absolute error by timestep within period
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.residuals()  # Time series of residuals
        >>> result.plot.residuals(mode="histogram")  # Error distribution
        >>> result.plot.residuals(mode="by_period")  # Which periods have highest error
        >>> result.plot.residuals(mode="by_timestep")  # Error pattern within day
        """
        resid = self._result.residuals
        columns = _validate_columns(columns, list(resid.columns), "residuals")

        if mode == "time_series":
            df_plot = resid[columns].copy()
            df_plot.index.name = "Time"
            long_df = df_plot.reset_index().melt(
                id_vars=["Time"],
                var_name="Column",
                value_name="Residual",
            )
            fig = px.line(
                long_df,
                x="Time",
                y="Residual",
                color="Column",
                title=title or "Residuals Over Time",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            return fig

        elif mode == "histogram":
            long_df = resid[columns].melt(var_name="Column", value_name="Residual")
            fig = px.histogram(
                long_df,
                x="Residual",
                color="Column",
                barmode="overlay",
                opacity=0.7,
                title=title or "Residual Distribution",
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            return fig

        elif mode == "by_period":
            n_timesteps = self._result.n_timesteps_per_period
            abs_resid = resid[columns].abs().copy()
            abs_resid["Period"] = np.arange(len(abs_resid)) // n_timesteps

            df = abs_resid.groupby("Period")[columns].mean().reset_index()
            long_df = df.melt(id_vars="Period", var_name="Column", value_name="MAE")

            fig = px.bar(
                long_df,
                x="Period",
                y="MAE",
                color="Column",
                barmode="group",
                title=title or "Mean Absolute Error by Period",
            )
            return fig

        elif mode == "by_timestep":
            n_timesteps = self._result.n_timesteps_per_period
            abs_resid = resid[columns].abs().copy()
            abs_resid["Timestep"] = np.arange(len(abs_resid)) % n_timesteps

            df = abs_resid.groupby("Timestep")[columns].mean().reset_index()
            long_df = df.melt(id_vars="Timestep", var_name="Column", value_name="MAE")

            fig = px.line(
                long_df,
                x="Timestep",
                y="MAE",
                color="Column",
                title=title or "Mean Absolute Error by Timestep",
            )
            return fig

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'time_series', 'histogram', 'by_period', or 'by_timestep'."
            )
