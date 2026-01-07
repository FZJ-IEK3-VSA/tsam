"""Plotting utilities for tsam using Plotly Express.

This module provides interactive visualizations for time series aggregation results.
Uses Plotly Express for clean, declarative plotting with automatic faceting and colors.

Two usage patterns are supported:

1. Module-level functions:
   >>> import tsam
   >>> tsam.plot.heatmap(df, column="Load")
   >>> tsam.plot.duration_curve(df)
   >>> tsam.plot.compare({"Original": df, "Aggregated": result.reconstruct()}, column="Load")

2. Accessor pattern on results:
   >>> result = tsam.aggregate(df, n_periods=8)
   >>> result.plot.heatmap(column="Load")
   >>> result.plot.duration_curve()

Note: This module requires the 'plotly' optional dependency.
Install with: pip install tsam[plot]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise ImportError(
        "The tsam.plot module requires plotly. Install it with: pip install tsam[plot]"
    ) from e

if TYPE_CHECKING:
    from tsam.result import AggregationResult


def heatmap(
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

    Examples
    --------
    >>> import tsam
    >>> tsam.plot.heatmap(df, column="Temperature", period_hours=24)
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


def heatmaps(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    period_hours: int = 24,
    title: str | None = None,
    color_continuous_scale: str = "Viridis",
    reference_data: pd.DataFrame | None = None,
) -> go.Figure:
    """Create stacked heatmaps for multiple columns.

    Creates a subplot with one heatmap per column, all sharing the same
    x-axis (days). Useful for visualizing how multiple time series are
    represented across periods.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data to plot.
    columns : list[str], optional
        Columns to plot. If None, plots all columns.
    period_hours : int, default 24
        Number of hours per period.
    title : str, optional
        Overall figure title.
    color_continuous_scale : str, default "Viridis"
        Plotly color scale name.
    reference_data : pd.DataFrame, optional
        Reference data for consistent color scaling (e.g., original data
        when plotting reconstructed data). Uses min/max from reference
        for color scale bounds.

    Returns
    -------
    go.Figure
        Plotly figure with stacked heatmaps.

    Examples
    --------
    >>> import tsam
    >>> result = tsam.aggregate(df, n_periods=8)
    >>> # Plot all columns from reconstructed data, scaled to original
    >>> tsam.plot.heatmaps(result.reconstruct(), reference_data=df)
    """
    from tsam.timeseriesaggregation import unstackToPeriods

    if columns is None:
        columns = list(data.columns)

    n_cols = len(columns)
    ref = reference_data if reference_data is not None else data

    fig = make_subplots(
        rows=n_cols,
        cols=1,
        subplot_titles=columns,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    for i, col in enumerate(columns, 1):
        stacked, _ = unstackToPeriods(data[[col]].copy(), period_hours)

        fig.add_trace(
            go.Heatmap(
                z=stacked[col].values.T,
                colorscale=color_continuous_scale,
                zmin=ref[col].min(),
                zmax=ref[col].max(),
                colorbar={
                    "title": col,
                    "y": 1 - (i - 0.5) / n_cols,
                    "len": 0.9 / n_cols,
                },
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text="Hour", row=i, col=1)

    fig.update_xaxes(title_text="Day", row=n_cols, col=1)
    fig.update_layout(
        height=200 * n_cols,
        title=title or "Time Series Heatmaps",
    )

    return fig


def duration_curve(
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

    Examples
    --------
    >>> import tsam
    >>> tsam.plot.duration_curve(df, columns=["Load", "GHI"])
    """
    if columns is None:
        columns = list(data.columns)

    # Build long-form data with sorted values using vectorized operations
    frames = []
    for col in columns:
        sorted_vals = data[col].sort_values(ascending=False).reset_index(drop=True)
        df_col = pd.DataFrame(
            {
                "Hour": range(len(sorted_vals)),
                "Value": sorted_vals.values,
                "Column": col,
            }
        )
        frames.append(df_col)
    long_df = pd.concat(frames, ignore_index=True)

    fig = px.line(
        long_df,
        x="Hour",
        y="Value",
        color="Column",
        title=title,
    )

    return fig


def time_slice(
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

    Examples
    --------
    >>> import tsam
    >>> tsam.plot.time_slice(df, start="20100210", end="20100218", columns=["Load"])
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


def compare(
    results: dict[str, pd.DataFrame],
    column: str,
    plot_type: str = "duration_curve",
    start: str | None = None,
    end: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Compare multiple DataFrames (e.g., from different aggregation methods).

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Dictionary mapping names to DataFrames.
        Example: {"Original": raw, "K-means": result1.reconstruct()}
    column : str
        Column to compare.
    plot_type : str, default "duration_curve"
        Type of plot: "duration_curve" or "time_slice".
    start : str, optional
        Start time (required for time_slice).
    end : str, optional
        End time (required for time_slice).
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import tsam
    >>> result1 = tsam.aggregate(df, n_periods=8, cluster=ClusterConfig(method="kmeans"))
    >>> result2 = tsam.aggregate(df, n_periods=8, cluster=ClusterConfig(method="hierarchical"))
    >>> fig = tsam.plot.compare(
    ...     {"Original": df, "K-means": result1.reconstruct(), "Hierarchical": result2.reconstruct()},
    ...     column="Load",
    ...     plot_type="duration_curve"
    ... )
    """
    records = []

    if plot_type == "duration_curve":
        for name, data in results.items():
            sorted_vals = (
                data[column].sort_values(ascending=False).reset_index(drop=True)
            )
            for hour, val in enumerate(sorted_vals):
                records.append({"Hour": hour, "Value": val, "Method": name})

        long_df = pd.DataFrame(records)
        fig = px.line(
            long_df,
            x="Hour",
            y="Value",
            color="Method",
            title=title or f"Duration Curve Comparison - {column}",
        )

    elif plot_type == "time_slice":
        if start is None or end is None:
            raise ValueError("start and end are required for time_slice plot")

        for name, data in results.items():
            sliced = data.loc[start:end]  # type: ignore[misc]
            for time, val in sliced[column].items():
                records.append({"Time": time, "Value": val, "Method": name})

        long_df = pd.DataFrame(records)
        fig = px.line(
            long_df,
            x="Time",
            y="Value",
            color="Method",
            title=title or f"Time Slice Comparison - {column}",
        )

    else:
        raise ValueError(
            f"Unknown plot_type: {plot_type}. Use 'duration_curve' or 'time_slice'."
        )

    return fig


# Aliases for backward compatibility and convenience
compare_results = compare


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

        return heatmap(
            data,
            column=column,
            period_hours=self._result.n_timesteps_per_period,
            title=title,
            color_continuous_scale=color_continuous_scale,
        )

    def heatmaps(
        self,
        columns: list[str] | None = None,
        use_original: bool = False,
        title: str | None = None,
        color_continuous_scale: str = "Viridis",
    ) -> go.Figure:
        """Plot heatmaps for all columns.

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot. If None, plots all.
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
        ref: pd.DataFrame | None
        if use_original and self._original is not None:
            data = self._original
            ref = self._original
        else:
            data = self._result.reconstruct()
            ref = self._original

        return heatmaps(
            data,
            columns=columns,
            period_hours=self._result.n_timesteps_per_period,
            title=title,
            color_continuous_scale=color_continuous_scale,
            reference_data=ref,
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
            # Use compare function for side-by-side comparison
            if columns is not None:
                col = columns[0]
            elif len(reconstructed.columns) > 0:
                col = reconstructed.columns[0]
            else:
                raise ValueError("No columns available to plot")
            return compare(
                {"Original": self._original, "Reconstructed": reconstructed},
                column=col,
                plot_type="duration_curve",
                title=title or "Duration Curve Comparison",
            )
        else:
            return duration_curve(
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
            if columns is not None:
                col = columns[0]
            elif len(reconstructed.columns) > 0:
                col = reconstructed.columns[0]
            else:
                raise ValueError("No columns available to plot")
            return compare(
                {"Original": self._original, "Reconstructed": reconstructed},
                column=col,
                plot_type="time_slice",
                start=start,
                end=end,
                title=title,
            )
        else:
            return time_slice(
                reconstructed,
                start,
                end,
                columns=columns,
                title=title,
            )
