"""Plotting utilities for tsam using Plotly Express.

This module provides interactive visualizations for time series aggregation results.
Uses Plotly Express for clean, declarative plotting with automatic faceting and colors.

Two usage patterns are supported:

1. Accessor pattern (recommended):
   >>> result = tsam.aggregate(df, n_periods=8)
   >>> result.plot.heatmap(column="Load")
   >>> result.plot.duration_curve()
   >>> result.plot.typical_periods()

2. Standalone functions:
   >>> tsam.plot_heatmap(df, column="Load")
   >>> tsam.plot_duration_curve(df, columns=["Load", "GHI"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from tsam.result import AggregationResult


def plot_heatmap(
    data: pd.DataFrame,
    column: str | None = None,
    period_hours: int = 24,
    title: str | None = None,
    color_continuous_scale: str = "Viridis",
) -> go.Figure:
    """Create a heatmap of time series data organized by periods.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data to plot.
    column : str, optional
        Column to plot. If None, uses the first column.
    period_hours : int, default 24
        Number of hours per period.
    title : str, optional
        Plot title.
    color_continuous_scale : str, default "Viridis"
        Plotly color scale name.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    from tsam.timeseriesaggregation import unstackToPeriods

    if column is None:
        column = data.columns[0]

    stacked, _ = unstackToPeriods(data[[column]].copy(), period_hours)

    fig = px.imshow(
        stacked[column].values.T,
        labels={"x": "Period (Day)", "y": "Timestep (Hour)", "color": column},
        title=title or f"{column} Heatmap",
        color_continuous_scale=color_continuous_scale,
        aspect="auto",
    )

    return fig


def plot_duration_curve(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Duration Curve",
) -> go.Figure:
    """Plot duration curves (sorted descending values).

    Parameters
    ----------
    data : pd.DataFrame
        Time series data to plot.
    columns : list[str], optional
        Columns to plot. If None, plots all.
    title : str, default "Duration Curve"
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if columns is None:
        columns = list(data.columns)

    # Build long-form data with sorted values
    records = []
    for col in columns:
        sorted_vals = data[col].sort_values(ascending=False).reset_index(drop=True)
        for hour, val in enumerate(sorted_vals):
            records.append({"Hour": hour, "Value": val, "Column": col})

    long_df = pd.DataFrame(records)

    fig = px.line(
        long_df,
        x="Hour",
        y="Value",
        color="Column",
        title=title,
    )

    return fig


def compare_duration_curves(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Duration Curve Comparison",
) -> go.Figure:
    """Compare duration curves between original and reconstructed data.

    Parameters
    ----------
    original : pd.DataFrame
        Original time series data.
    reconstructed : pd.DataFrame
        Reconstructed time series from aggregation.
    columns : list[str], optional
        Columns to compare. If None, compares all.
    title : str, default "Duration Curve Comparison"
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure with faceted comparison.
    """
    if columns is None:
        columns = list(original.columns)

    records = []
    for col in columns:
        # Original
        sorted_orig = original[col].sort_values(ascending=False).reset_index(drop=True)
        for hour, val in enumerate(sorted_orig):
            records.append(
                {"Hour": hour, "Value": val, "Column": col, "Series": "Original"}
            )

        # Reconstructed
        sorted_recon = (
            reconstructed[col].sort_values(ascending=False).reset_index(drop=True)
        )
        for hour, val in enumerate(sorted_recon):
            records.append(
                {"Hour": hour, "Value": val, "Column": col, "Series": "Reconstructed"}
            )

    long_df = pd.DataFrame(records)

    fig = px.line(
        long_df,
        x="Hour",
        y="Value",
        color="Series",
        facet_col="Column",
        title=title,
    )

    return fig


def plot_time_slice(
    data: pd.DataFrame,
    start: str,
    end: str,
    columns: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot a time slice of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index.
    start : str
        Start date/time string.
    end : str
        End date/time string.
    columns : list[str], optional
        Columns to plot. If None, plots all.
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    sliced = data.loc[start:end]  # type: ignore[misc]

    if columns is None:
        columns = list(sliced.columns)

    sliced_subset = sliced[columns].copy()
    sliced_subset = sliced_subset.reset_index()
    sliced_subset.columns = pd.Index(["Time", *columns])

    long_df = sliced_subset.melt(
        id_vars=["Time"], var_name="Column", value_name="Value"
    )

    fig = px.line(
        long_df,
        x="Time",
        y="Value",
        color="Column",
        title=title or f"Time Series: {start} to {end}",
    )

    return fig


def compare_time_slices(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    start: str,
    end: str,
    columns: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Compare original and reconstructed data for a time slice.

    Parameters
    ----------
    original : pd.DataFrame
        Original time series data.
    reconstructed : pd.DataFrame
        Reconstructed time series from aggregation.
    start : str
        Start date/time string.
    end : str
        End date/time string.
    columns : list[str], optional
        Columns to compare. If None, compares first column.
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure with faceted comparison.
    """
    orig_slice = original.loc[start:end]  # type: ignore[misc]
    recon_slice = reconstructed.loc[start:end]  # type: ignore[misc]

    if columns is None:
        columns = [original.columns[0]]

    records = []
    for col in columns:
        for time, val in orig_slice[col].items():
            records.append(
                {"Time": time, "Value": val, "Column": col, "Series": "Original"}
            )
        for time, val in recon_slice[col].items():
            records.append(
                {"Time": time, "Value": val, "Column": col, "Series": "Reconstructed"}
            )

    long_df = pd.DataFrame(records)

    fig = px.line(
        long_df,
        x="Time",
        y="Value",
        color="Series",
        facet_row="Column" if len(columns) > 1 else None,
        title=title or f"Comparison: {start} to {end}",
    )

    return fig


class ResultPlotAccessor:
    """Plotting accessor for AggregationResult.

    Provides convenient plotting methods directly on the result object.

    Examples
    --------
    >>> result = tsam.aggregate(df, n_periods=8)
    >>> result.plot.heatmap(column="Load")
    >>> result.plot.duration_curve()
    >>> result.plot.typical_periods()
    >>> result.plot.cluster_weights()
    """

    def __init__(
        self, result: AggregationResult, original_data: pd.DataFrame | None = None
    ):
        self._result = result
        self._original = original_data

    def heatmap(
        self,
        column: str | None = None,
        use_original: bool = False,
        title: str | None = None,
        color_continuous_scale: str = "Viridis",
    ) -> go.Figure:
        """Plot heatmap of reconstructed (or original) data.

        Parameters
        ----------
        column : str, optional
            Column to plot.
        use_original : bool, default False
            If True and original data available, plot original instead.
        title : str, optional
            Plot title.
        color_continuous_scale : str, default "Viridis"
            Color scale.

        Returns
        -------
        go.Figure
        """
        if use_original and self._original is not None:
            data = self._original
        else:
            data = self._result.reconstruct()

        return plot_heatmap(
            data,
            column=column,
            period_hours=self._result.n_timesteps_per_period,
            title=title,
            color_continuous_scale=color_continuous_scale,
        )

    def duration_curve(
        self,
        columns: list[str] | None = None,
        compare_original: bool = False,
        title: str | None = None,
    ) -> go.Figure:
        """Plot duration curves.

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot.
        compare_original : bool, default False
            If True and original data available, show comparison.
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure
        """
        reconstructed = self._result.reconstruct()

        if compare_original and self._original is not None:
            return compare_duration_curves(
                self._original,
                reconstructed,
                columns=columns,
                title=title or "Duration Curve Comparison",
            )
        else:
            return plot_duration_curve(
                reconstructed,
                columns=columns,
                title=title or "Duration Curve",
            )

    def typical_periods(
        self,
        columns: list[str] | None = None,
        title: str = "Typical Periods",
    ) -> go.Figure:
        """Plot all typical periods.

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot.
        title : str, default "Typical Periods"
            Plot title.

        Returns
        -------
        go.Figure
        """
        typ = self._result.typical_periods
        weights = self._result.cluster_weights

        # Get column names (excluding index levels if they're columns)
        all_columns = [c for c in typ.columns if c not in ["period", "timestep"]]
        if columns is None:
            columns = all_columns
        else:
            columns = [c for c in columns if c in all_columns]

        # Build long-form data
        records = []

        if isinstance(typ.index, pd.MultiIndex):
            periods = typ.index.get_level_values(0).unique()
            for period in periods:
                period_data = typ.loc[period]
                weight = weights.get(period, 1)
                for timestep, row in period_data.iterrows():
                    for col in columns:
                        records.append(
                            {
                                "Timestep": timestep,
                                "Value": row[col],
                                "Column": col,
                                "Period": f"Period {period} (n={weight})",
                            }
                        )
        else:
            for _, row in typ.iterrows():
                period = row.get("period", 0)
                timestep = row.get("timestep", 0)
                weight = weights.get(period, 1)
                for col in columns:
                    records.append(
                        {
                            "Timestep": timestep,
                            "Value": row[col],
                            "Column": col,
                            "Period": f"Period {period} (n={weight})",
                        }
                    )

        long_df = pd.DataFrame(records)

        fig = px.line(
            long_df,
            x="Timestep",
            y="Value",
            color="Period",
            facet_col="Column" if len(columns) > 1 else None,
            title=title,
        )

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

        durations = self._result.segment_durations
        df = pd.DataFrame(
            {
                "Segment": [f"Segment {s}" for s in durations],
                "Duration": list(durations.values()),
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
        fig.update_traces(texttemplate="%{text:.1f}h", textposition="auto")
        fig.update_layout(showlegend=False, yaxis_title="Duration (hours)")

        return fig

    def time_slice(
        self,
        start: str,
        end: str,
        columns: list[str] | None = None,
        compare_original: bool = False,
        title: str | None = None,
    ) -> go.Figure:
        """Plot a time slice comparison.

        Parameters
        ----------
        start : str
            Start date/time.
        end : str
            End date/time.
        columns : list[str], optional
            Columns to plot.
        compare_original : bool, default False
            If True and original available, show comparison.
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure
        """
        reconstructed = self._result.reconstruct()

        if compare_original and self._original is not None:
            return compare_time_slices(
                self._original,
                reconstructed,
                start,
                end,
                columns=columns,
                title=title,
            )
        else:
            return plot_time_slice(
                reconstructed,
                start,
                end,
                columns=columns,
                title=title,
            )
