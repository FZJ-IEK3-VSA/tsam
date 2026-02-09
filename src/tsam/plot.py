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
from typing import TYPE_CHECKING

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
        title: str | None = None,
    ) -> go.Figure:
        """Plot all original periods grouped by cluster with representative highlighted.

        Shows individual member periods as faint lines and the cluster
        representative as a bold line. An animation slider lets you flip
        through clusters.

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot. If None, plots all columns.
        title : str, optional
            Plot title. Defaults to "Cluster Members".

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.cluster_members(columns=["Load"])
        >>> result.plot.cluster_members()  # all columns
        """
        from tsam.api import unstack_to_periods

        result = self._result
        orig = result.original
        available_columns = list(orig.columns)
        columns = _validate_columns(columns, available_columns, "original data")

        # Unstack original data into periods: MultiIndex columns (col, timestep)
        unstacked = unstack_to_periods(orig, result.n_timesteps_per_period)

        assignments = result.cluster_assignments
        representatives = result.cluster_representatives
        weights = result.cluster_weights
        n_timesteps = result.n_timesteps_per_period

        # Group period indices by cluster
        cluster_ids = sorted(set(assignments))
        members_per_cluster = {
            cid: np.where(assignments == cid)[0] for cid in cluster_ids
        }
        max_members = max(len(m) for m in members_per_cluster.values())

        # Build long-form DataFrame with consistent line_group slots
        # across all clusters so animation frames have equal trace counts.
        # Pad smaller clusters with NaN so those traces draw nothing.
        dfs: list[pd.DataFrame] = []
        timesteps = np.arange(n_timesteps)

        for cluster_id in cluster_ids:
            cluster_label = f"Cluster {cluster_id} (n={weights.get(cluster_id, 0)})"
            member_indices = members_per_cluster[cluster_id]

            for slot in range(max_members):
                if slot < len(member_indices):
                    period_idx = member_indices[slot]
                    for col in columns:
                        values = unstacked.loc[period_idx, col].values
                        dfs.append(
                            pd.DataFrame(
                                {
                                    "Timestep": timesteps,
                                    "Value": values,
                                    "Column": col,
                                    "Cluster": cluster_label,
                                    "Period": f"M{slot}_{col}",
                                    "Role": "Member",
                                }
                            )
                        )
                else:
                    # Pad with NaN so every frame has the same trace count
                    for col in columns:
                        dfs.append(
                            pd.DataFrame(
                                {
                                    "Timestep": timesteps,
                                    "Value": np.nan,
                                    "Column": col,
                                    "Cluster": cluster_label,
                                    "Period": f"M{slot}_{col}",
                                    "Role": "Member",
                                }
                            )
                        )

            # Representative period — expand segments to full timesteps
            rep_data = representatives.loc[cluster_id]
            if result.n_segments is not None:
                # Segmented: index is (Segment Step, Segment Duration)
                durations = rep_data.index.get_level_values("Segment Duration")
                for col in columns:
                    values = np.repeat(rep_data[col].values, durations)
                    dfs.append(
                        pd.DataFrame(
                            {
                                "Timestep": timesteps,
                                "Value": values,
                                "Column": col,
                                "Cluster": cluster_label,
                                "Period": f"Rep_{col}",
                                "Role": "Representative",
                            }
                        )
                    )
            else:
                for col in columns:
                    dfs.append(
                        pd.DataFrame(
                            {
                                "Timestep": timesteps,
                                "Value": rep_data[col].values,
                                "Column": col,
                                "Cluster": cluster_label,
                                "Period": f"Rep_{col}",
                                "Role": "Representative",
                            }
                        )
                    )

        long_df = pd.concat(dfs, ignore_index=True)

        fig = px.line(
            long_df,
            x="Timestep",
            y="Value",
            color="Role",
            line_group="Period",
            facet_col="Column" if len(columns) > 1 else None,
            animation_frame="Cluster",
            title=title or "Cluster Members",
        )

        # Style: members faint, representative bold
        fig.update_traces(selector={"name": "Member"}, opacity=0.3)
        fig.update_traces(selector={"name": "Representative"}, line={"width": 3})
        for frame in fig.frames:
            for trace in frame.data:
                if trace.name == "Member":
                    trace.opacity = 0.3
                elif trace.name == "Representative":
                    trace.line = {"width": 3}

        # Independent y-axes per column, fixed to global min/max across clusters
        if len(columns) > 1:
            fig.update_yaxes(matches=None, showticklabels=True)
            for i, col in enumerate(columns):
                col_data = long_df.loc[long_df["Column"] == col, "Value"]
                ymin, ymax = col_data.min(), col_data.max()
                margin = (ymax - ymin) * 0.05
                axis_key = "yaxis" if i == 0 else f"yaxis{i + 1}"
                fig.layout[axis_key].range = [ymin - margin, ymax + margin]
        else:
            ymin, ymax = long_df["Value"].min(), long_df["Value"].max()
            margin = (ymax - ymin) * 0.05
            fig.update_yaxes(range=[ymin - margin, ymax + margin])

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
