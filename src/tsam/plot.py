"""Plotting accessor for tsam aggregation results.

Provides convenient plotting methods directly on the result object for
validation and visualization of aggregation quality.

Usage:
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> result.plot.compare()  # Compare original vs reconstructed
    >>> result.plot.residuals()  # View reconstruction errors
    >>> result.plot.cluster_representatives()
    >>> result.plot.cluster_members()  # All periods per cluster
    >>> result.plot.cluster_counts()
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
from typing import TYPE_CHECKING, Literal, cast

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

    if invalid and valid:
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
    color: str = "column",
) -> go.Figure:
    """Create duration curve comparison figure (internal helper).

    ``color`` selects which dimension drives the line colour: ``"column"``
    colours by column and dashes by series (e.g. original/reconstructed);
    ``"source"`` swaps them.
    """
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
    color_by, dash_by = (
        ("Column", "Method") if color == "column" else ("Method", "Column")
    )
    return px.line(
        long_df,
        x="Hour",
        y="Value",
        color=color_by,
        line_dash=dash_by,
        title=title or "Duration Curve Comparison",
    )


def _cluster_color_map(
    cluster_ids: list[int] | np.ndarray,
    palette: list[str] | None = None,
) -> dict[int, str]:
    """Map each cluster id to a stable colour from a qualitative palette.

    The ids are sorted before colours are assigned, so a given cluster gets the
    same colour in every plot (timeline, members, representatives,
    reconstruction). That lets a cluster be traced from one figure to the next.
    """
    if palette is None:
        palette = px.colors.qualitative.Plotly
    ids = sorted({int(c) for c in cluster_ids})
    return {cid: palette[i % len(palette)] for i, cid in enumerate(ids)}


def _to_rgba(color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` hex colour to an ``rgba(...)`` string."""
    h = color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# Distinguishable at small sizes and without relying on colour alone.
_PATH_SYMBOLS = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "x",
    "star",
    "hexagon",
    "cross",
)


class AttributeSpace:
    """A plane spanned by two attributes, in which each period is drawn as a path.

    A period is not a *point* in attribute space: it is a sequence of timesteps,
    so in a plane spanned by two attributes it traces a **path** from its first
    timestep to its last. Drawing periods this way makes the difference between
    representation rules legible — a ``medoid`` re-traces one member's path
    exactly, while a ``mean`` cuts a new path through the middle of the group
    that no member ever followed.

    Paths are added one at a time so that members and representatives can be
    styled differently in the same figure.

    Parameters
    ----------
    x_attr, y_attr : str
        Names of the two attributes spanning the plane. Used for axis titles.
    units : dict[str, str], optional
        Maps attribute name to a unit string, e.g. ``{"solar": "W/m²"}``.
        Attributes missing from the mapping are labelled without a unit.
    title : str, optional
        Figure title.
    legend_title : str, default "period"
        Heading for the legend.

    Examples
    --------
    >>> space = AttributeSpace("solar", "load", units={"solar": "W/m²", "load": "MW"})
    >>> space.add_path([0.1, 0.8, 0.6], [3.0, 4.5, 4.0], name="day 0")
    >>> space.add_path([0.2, 0.4, 0.3], [3.2, 3.9, 3.6], name="mean", dash="dot")
    >>> fig = space.figure
    """

    def __init__(
        self,
        x_attr: str,
        y_attr: str,
        *,
        units: dict[str, str] | None = None,
        title: str | None = None,
        legend_title: str = "period",
    ) -> None:
        self.x_attr = x_attr
        self.y_attr = y_attr
        self.units = units or {}
        self._n_paths = 0
        self._fig = go.Figure()
        self._fig.update_layout(
            title=title,
            xaxis_title=self._axis_label(x_attr),
            yaxis_title=self._axis_label(y_attr),
            legend_title=legend_title,
        )

    def _axis_label(self, attr: str) -> str:
        unit = self.units.get(attr)
        return f"{attr} [{unit}]" if unit else attr

    def add_path(
        self,
        x,
        y,
        name: str,
        *,
        color: str | None = None,
        symbol: str | None = None,
        dash: str | None = None,
        width: float = 2,
        arrows: bool = True,
        label_steps: bool = True,
        step_labels: list[str] | None = None,
        legendgroup: str | None = None,
    ) -> AttributeSpace:
        """Add one period to the plane as a path through its timesteps.

        Parameters
        ----------
        x, y : array-like
            The period's values for the two attributes, one entry per timestep,
            in chronological order.
        name : str
            Legend entry for this path.
        color : str, optional
            Any plotly colour. Defaults to the next colour of the qualitative
            palette, so successive paths are distinguishable without being
            named.
        symbol : str, optional
            Marker symbol (``"circle"``, ``"square"``, ``"diamond"``, ``"x"``,
            …). Defaults to the next symbol of an eight-symbol cycle, so paths
            stay distinguishable in greyscale and for colour-blind readers.
            Pass ``"circle"`` explicitly to opt out of the cycle.
        dash : str, optional
            Line dash style, e.g. ``"dot"``. Useful to separate derived profiles
            (representatives) from real ones (members).
        width : float, default 2
            Line width.
        arrows : bool, default True
            Draw an arrowhead on each segment, pointing from one timestep to the
            next, so the direction of travel through the period is unambiguous.
        label_steps : bool, default True
            Annotate each marker with its timestep label.
        step_labels : list[str], optional
            Text for each timestep. Defaults to ``t0, t1, …``.
        legendgroup : str, optional
            Group paths in the legend, so clicking toggles them together.

        Returns
        -------
        AttributeSpace
            ``self``, so calls can be chained.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}."
            )

        if color is None:
            palette = px.colors.qualitative.Plotly
            color = palette[self._n_paths % len(palette)]
        if symbol is None:
            symbol = _PATH_SYMBOLS[self._n_paths % len(_PATH_SYMBOLS)]
        if step_labels is None:
            step_labels = [f"t{i}" for i in range(len(x))]

        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                legendgroup=legendgroup or name,
                mode="lines+markers+text" if label_steps else "lines+markers",
                text=step_labels if label_steps else None,
                textposition="top center",
                textfont={"size": 9, "color": color},
                marker={"size": 9, "color": color, "symbol": symbol},
                line={"color": color, "width": width, "dash": dash},
                hovertemplate=(
                    f"{name}<br>%{{text}}<br>"
                    f"{self.x_attr} = %{{x:.3f}}<br>{self.y_attr} = %{{y:.3f}}"
                    "<extra></extra>"
                ),
                customdata=step_labels,
            )
        )

        if arrows and len(x) > 1:
            # A second, legend-less trace carrying only arrowheads. `angleref`
            # rotates each marker to the incoming segment, so the first point —
            # which has no incoming segment — is given size 0 instead.
            self._fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker={
                        "symbol": "arrow",
                        "size": [0] + [11] * (len(x) - 1),
                        "angleref": "previous",
                        "color": color,
                        "standoff": 5,
                    },
                    legendgroup=legendgroup or name,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        self._n_paths += 1
        return self

    @property
    def figure(self) -> go.Figure:
        """The assembled plotly figure."""
        return self._fig

    def show(self, **kwargs) -> None:
        """Display the figure."""
        self._fig.show(**kwargs)


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
    >>> result.plot.cluster_counts()
    """

    def __init__(self, result: AggregationResult):
        self._result = result

    def cluster_representatives(
        self,
        columns: list[str] | None = None,
        units: dict[str, str] | None = None,
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
        counts = self._result.cluster_counts

        available_columns = [c for c in typ.columns if c not in ["cluster", "timestep"]]
        columns = _validate_columns(
            columns, available_columns, "cluster_representatives"
        )

        # Reset index to get period/timestep as columns
        df = typ[columns].reset_index()
        df.columns = pd.Index(["Period", "Timestep", *columns])

        # Map period IDs to labels with their occurrence counts
        period_label = {p: f"Period {p} (n={counts.get(p, 1)})" for p in counts}
        df["Period"] = df["Period"].map(lambda p: period_label.get(p, f"Period {p}"))

        long_df = df.melt(
            id_vars=["Period", "Timestep"],
            var_name="Column",
            value_name="Value",
        )

        # Shared colour map so cluster colours match the other cluster plots.
        cmap = _cluster_color_map(list(counts.keys()))
        color_discrete_map = {
            period_label[p]: color for p, color in cmap.items() if p in period_label
        }

        fig = px.line(
            long_df,
            x="Timestep",
            y="Value",
            color="Period",
            color_discrete_map=color_discrete_map,
            facet_col="Column" if len(columns) > 1 else None,
            title=title,
        )

        fig.update_xaxes(title_text="timestep")
        if units and len(columns) == 1 and columns[0] in units:
            fig.update_yaxes(title_text=f"{columns[0]} [{units[columns[0]]}]")

        return fig

    def cluster_members(
        self,
        columns: list[str] | None = None,
        clusters: list[int] | None = None,
        slider: Literal["cluster", "column"] = "cluster",
        units: dict[str, str] | None = None,
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
        slider : ``"cluster"`` or ``"column"``, default ``"cluster"``
            Which dimension to put on the slider.
            The other dimension becomes ``facet_col``.

            - ``"cluster"``: slider flips through clusters, columns are facets.
            - ``"column"``: slider flips through columns, clusters are facets.
        title : str, optional
            Plot title. Defaults to "Cluster Members".

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.cluster_members(columns=["Load"])
        >>> result.plot.cluster_members(clusters=[0, 3])  # specific clusters
        >>> result.plot.cluster_members(slider="column")  # flip through columns
        """
        from plotly.subplots import make_subplots

        from tsam.api import unstack_to_periods

        _slider = slider.lower()
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
        counts = result.cluster_counts
        timesteps = np.arange(n_ts)

        all_cluster_ids = sorted(set(assignments))
        if clusters is not None:
            invalid = [c for c in clusters if c not in all_cluster_ids]
            cluster_ids = [c for c in clusters if c in all_cluster_ids]
            if invalid and cluster_ids:
                warnings.warn(
                    f"Cluster indices not found and will be ignored: {invalid}. "
                    f"Available clusters: {all_cluster_ids}",
                    UserWarning,
                    stacklevel=2,
                )
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

        if _slider not in ("cluster", "column"):
            raise ValueError(f"slider must be 'cluster' or 'column', got {slider!r}")

        # Pre-extract member data as numpy arrays for fast access.
        # member_arrays[cid][col] = 2D array (n_members, n_ts)
        member_arrays: dict[int, dict[str, np.ndarray]] = {}
        for cid in cluster_ids:
            members = members_by_cluster[cid]
            member_arrays[cid] = {
                col: np.asarray(unstacked[col].iloc[members].values) for col in columns
            }

        cluster_labels = {
            cid: f"Cluster {cid} (n={counts.get(cid, 1)})" for cid in cluster_ids
        }

        # Determine which dimension is animated vs faceted.
        anim_keys: list[int | str]
        if _slider == "cluster":
            anim_keys = list(cluster_ids)
            anim_labels = [cluster_labels[c] for c in cluster_ids]
            facet_labels = columns
        else:
            anim_keys = list(columns)
            anim_labels = list(columns)
            facet_labels = [cluster_labels[c] for c in cluster_ids]

        n_facets = len(facet_labels)
        traces_per_facet = 2  # one bundled member trace + one representative
        # Per-cluster colours, shared with the other cluster plots so a cluster
        # keeps the same colour across figures.
        cmap = _cluster_color_map(cluster_ids)

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
                if _slider == "cluster":
                    cid, col = cast("int", anim_key), columns[facet_idx]
                else:
                    cid, col = cluster_ids[facet_idx], cast("str", anim_key)

                color = cmap[int(cid)]
                n_m = len(members_by_cluster[cid])
                out.append(
                    go.Scatter(
                        x=_member_x[n_m],
                        y=_member_y(cid, col),
                        mode="lines",
                        # Members: the cluster colour, faded and thin, so they
                        # read as "all belong to this cluster" without competing
                        # with the representative.
                        line={"color": _to_rgba(color, 0.25), "width": 1},
                        name="Members",
                        legendgroup="Members",
                        showlegend=first_member,
                    )
                )
                first_member = False

                out.append(
                    go.Scatter(
                        x=timesteps,
                        y=_rep_values(cid, col),
                        mode="lines",
                        # Representative: the same cluster colour, solid and bold,
                        # highlighted as the one profile that stands in for them all.
                        line={"color": color, "width": 4},
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
        if _slider == "cluster":
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

        # Axis titles (x is always the timestep within a period; y is the
        # column, with units when provided).
        def _ylabel(c: str) -> str:
            return f"{c} [{units[c]}]" if units and c in units else c

        fig.update_xaxes(title_text="timestep")
        if n_facets > 1:
            if _slider == "cluster":
                for i, c in enumerate(columns):
                    fig.update_yaxes(title_text=_ylabel(c), row=1, col=i + 1)
            else:
                fig.update_yaxes(title_text="value")
        else:
            fig.update_yaxes(
                title_text=_ylabel(columns[0]) if _slider == "cluster" else "value"
            )

        return fig

    def clusters_over_time(
        self,
        columns: list[str] | None = None,
        *,
        reconstructed: bool = False,
        overlay_original: bool = False,
        mark_periods: bool = True,
        units: dict[str, str] | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Plot the series over time with each period shaded by its cluster.

        Each period (e.g. day) on the time axis is shaded with the colour of the
        cluster it was assigned to, so you can see which typical period stands in
        for each stretch of time — and whether a cluster spans several
        consecutive periods. Cluster colours match :meth:`cluster_members` and
        :meth:`cluster_representatives`, so a cluster can be traced across plots.

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot, one stacked subplot each. Defaults to all columns.
        reconstructed : bool, default False
            Plot the reconstructed series instead of the original.
        overlay_original : bool, default False
            Draw the original series as a dotted line on top. Combined with
            ``reconstructed=True`` this shows where the reconstruction differs.
        mark_periods : bool, default True
            Outline each period and add period boundary ticks to the x axis, so
            period boundaries and runs of consecutive same-cluster periods are
            easy to see.
        units : dict[str, str], optional
            Map of column name to unit (e.g. ``{"Load": "MW"}``), used to label
            the y axes as ``column [unit]``.
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.clusters_over_time(columns=["Load"])
        >>> result.plot.clusters_over_time(
        ...     columns=["Load"], reconstructed=True, overlay_original=True
        ... )
        """
        from plotly.subplots import make_subplots

        result = self._result
        columns = _validate_columns(
            columns, list(result.original.columns), "original data"
        )
        frame = result.reconstructed if reconstructed else result.original
        original = result.original
        assignments = result.cluster_assignments
        cmap = _cluster_color_map(assignments)

        times = frame.index
        n_periods = len(assignments)
        steps_per_period = result.n_timesteps_per_period
        step = (times[1] - times[0]) if len(times) > 1 else 1
        period_bounds = [times[p * steps_per_period] for p in range(n_periods)]
        period_bounds.append(times[-1] + step)

        n = len(columns)
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            subplot_titles=columns if n > 1 else None,
        )

        shapes: list[dict] = []
        for row, col in enumerate(columns, start=1):
            if overlay_original:
                fig.add_trace(
                    go.Scatter(
                        x=original.index,
                        y=original[col],
                        mode="lines",
                        line={"color": "#222", "width": 1.0, "dash": "dash"},
                        name="original",
                        legendgroup="original",
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
            series_name = "reconstructed" if reconstructed else col
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=frame[col],
                    mode="lines",
                    line={"color": "#444", "width": 1.2},
                    name=series_name,
                    legendgroup=series_name,
                    showlegend=(row == 1) if reconstructed else True,
                ),
                row=row,
                col=1,
            )
            # Collected rather than added one at a time: every `add_vrect` call
            # re-validates the shapes already on the figure, so shading a year of
            # days column by column is quadratic (minutes, not milliseconds).
            # These are exactly the shapes add_vrect would emit for this subplot.
            axis = "" if row == 1 else str(row)
            shapes.extend(
                {
                    "type": "rect",
                    "xref": f"x{axis}",
                    "yref": f"y{axis} domain",
                    "x0": period_bounds[period],
                    "x1": period_bounds[period + 1],
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": cmap[int(cluster)],
                    "opacity": 0.18,
                    "layer": "below",
                    "line": {
                        "width": 0.5 if mark_periods else 0,
                        "color": "rgba(0, 0, 0, 0.18)",
                    },
                }
                for period, cluster in enumerate(assignments)
            )

        fig.update_layout(shapes=shapes)

        # One legend entry per cluster (colours shared across all cluster plots).
        for cid, color in cmap.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={"color": color, "size": 10, "symbol": "square"},
                    name=f"cluster {cid}",
                ),
                row=1,
                col=1,
            )

        if mark_periods:
            fig.update_xaxes(
                minor={"tickvals": period_bounds, "ticklen": 6, "tickcolor": "grey"}
            )
        fig.update_xaxes(title_text="time", row=n, col=1)
        for row, col in enumerate(columns, start=1):
            label = f"{col} [{units[col]}]" if units and col in units else col
            fig.update_yaxes(title_text=label, row=row, col=1)
        fig.update_layout(
            title=title
            or (
                "Reconstructed series by cluster"
                if reconstructed
                else "Series by cluster"
            ),
            legend_title="cluster",
        )
        return fig

    def cluster_counts(self, title: str = "Cluster Counts") -> go.Figure:
        """Plot how many original periods each cluster represents.

        Parameters
        ----------
        title : str, default "Cluster Counts"
            Plot title.

        Returns
        -------
        go.Figure
        """
        counts = self._result.cluster_counts
        df = pd.DataFrame(
            {
                "Period": [f"Period {p}" for p in counts],
                "Count": list(counts.values()),
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

    def cluster_weights(self, title: str = "Cluster Weights") -> go.Figure:
        """Deprecated alias for :meth:`cluster_counts`."""
        warnings.warn(
            "plot.cluster_weights() is deprecated, use plot.cluster_counts().",
            FutureWarning,
            stacklevel=2,
        )
        return self.cluster_counts(title)

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
        time_slice: slice | None = None,
        color: str = "column",
        units: dict[str, str] | None = None,
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
        time_slice : slice, optional
            Restrict the comparison to a window of the time axis, e.g.
            ``slice("2010-01-11", "2010-01-17")``. Useful for zooming into a few
            periods so fine detail (such as segment step functions) is visible.
            Applies to all modes; for ``"duration_curve"`` the curve is computed
            over the sliced window.
        color : ``"column"`` or ``"source"``, default ``"column"``
            Which dimension drives the line colour:

            - ``"column"``: colour by column; the source (original vs.
              reconstructed) is the secondary encoding — dash in ``"overlay"``
              and ``"duration_curve"``, facet row in ``"side_by_side"``.
            - ``"source"``: colour by source, with the column as the secondary
              encoding instead. Clearer when comparing a single column, where the
              original/reconstructed split is the thing you want to see.

            Applies to all modes.

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.compare()  # Compare all columns
        >>> result.plot.compare(columns=["Load"])  # Compare specific column
        >>> result.plot.compare(mode="duration_curve")
        >>> result.plot.compare(columns=["Load"], time_slice=slice("2010-01-11", "2010-01-17"))
        >>> result.plot.compare(columns=["Load"], color="source")
        """
        if color not in ("column", "source"):
            raise ValueError(f"color must be 'column' or 'source', got {color!r}")

        orig = self._result.original
        recon = self._result.reconstructed

        if time_slice is not None:
            orig = orig.loc[time_slice]
            recon = recon.loc[time_slice]

        columns = _validate_columns(columns, list(orig.columns), "original data")

        if mode == "duration_curve":
            fig = _duration_curve_figure(
                {"Original": orig, "Reconstructed": recon},
                columns=columns,
                title=title,
                color=color,
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

            # color="column": colour by Column, with Source as the secondary
            # encoding. color="source": swap them.
            color_by, other = (
                ("Column", "Source") if color == "column" else ("Source", "Column")
            )
            if mode == "overlay":
                fig = px.line(
                    long_df,
                    x="Time",
                    y="Value",
                    color=color_by,
                    line_dash=other,
                    title=title or "Original vs Reconstructed",
                )
            else:  # side_by_side — facet by the non-colour dimension
                fig = px.line(
                    long_df,
                    x="Time",
                    y="Value",
                    color=color_by,
                    facet_row=other,
                    title=title or "Original vs Reconstructed",
                )
                fig.update_layout(height=600)

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'overlay', 'side_by_side', or 'duration_curve'."
            )

        if units and len(columns) == 1 and columns[0] in units:
            fig.update_yaxes(title_text=f"{columns[0]} [{units[columns[0]]}]")
        return fig

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
