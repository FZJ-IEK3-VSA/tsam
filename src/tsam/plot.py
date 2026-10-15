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
    >>> result.plot.feature_space()  # The space the clustering works in

To ask whether two configurations grouped the periods the same way, compare
their partitions side by side:
    >>> tsam.plot.compare_partitions({"kmeans": a, "kmedoids": b})

To compare chronological paths with consistent colours and shared axes:
    >>> tsam.plot.path_panels(days, "solar", "load", background=days)

To compare full series or their duration curves across several runs:
    >>> tsam.plot.compare_series(series, mode="duration_curve", reference="original")

To explain a duration representation: how member days are fitted and ordered,
where a reconstruction's error lies in time and in the duration curve, how each
cluster's duration curve is matched, and which reference each ordering follows:
    >>> tsam.plot.duration_fit(days, "solar", representative=profile)
    >>> tsam.plot.error_areas(original, {"mean": reconstructed}, "solar")
    >>> tsam.plot.cluster_duration_curves({"sunny": {"members": days}}, "solar")
    >>> tsam.plot.profile_panels(profiles, ["solar", "load"], background=days)

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

    Args:
        requested: Columns requested by user. If None, returns all available.
        available: Columns available in the data.
        context: Description for error messages (e.g., "original data").

    Returns:
        Valid columns to use.

    Raises:
        ValueError: If no valid columns remain after filtering.
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


# Opacity for the cluster-period shading in `clusters_over_time`, reused by
# the legend swatches so the two can never drift out of sync.
_PERIOD_SHADE_OPACITY = 0.35


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


# Multiplication sign for the count badges on merged markers ("sunny 5" reads as
# a name, "sunny <times>5" as a count). Written as an escape so it is not
# confused with an ASCII "x" here or by a linter.
_TIMES = "\u00d7"


def _canonical_labels(labels) -> np.ndarray:
    """Relabel clusters by order of first appearance.

    Cluster ids are arbitrary: two methods can produce the *same* grouping and
    still number it differently. Renumbering by first appearance makes
    identical groupings look identical, which is what lets partitions be
    compared across methods at a glance.
    """
    remap: dict[int, int] = {}
    out = []
    for value in labels:
        remap.setdefault(int(value), len(remap))
        out.append(remap[int(value)])
    return np.array(out, dtype=int)


def _period_matrix(result: AggregationResult) -> np.ndarray:
    """One row per period: the normalised vector the clustering actually saw.

    Clustering does not work on the raw series. Each period is flattened into a
    single ``n_timesteps x n_attributes`` vector of normalised values, and it is
    those vectors that get grouped. Rebuilding the matrix from the result —
    rather than re-deriving it in user code — keeps the plot faithful to what
    the algorithm was given.
    """
    norm = result._norm_values
    if norm is None:
        raise ValueError(
            "This result carries no normalised values, so its feature space "
            "cannot be reconstructed. Results restored from JSON do not keep "
            "them; re-run aggregate() on the data to plot the feature space."
        )
    values = np.asarray(norm, dtype=float)
    n_timesteps = result.n_timesteps_per_period
    n_periods = values.shape[0] // n_timesteps
    trimmed = values[: n_periods * n_timesteps]
    return trimmed.reshape(n_periods, n_timesteps * values.shape[1])


def _project_2d(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Project period vectors onto their first two principal components.

    Returns the 2-D coordinates and the share of variance those two components
    account for — the honesty check on the picture, since a low share means
    the plotted distances are not the distances the algorithm used.
    """
    centered = matrix - matrix.mean(axis=0)
    _u, singular, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:2].T
    if coords.shape[1] == 1:  # only one non-degenerate direction exists
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    total = float((singular**2).sum())
    explained = float((singular[:2] ** 2).sum() / total) if total > 0 else 1.0
    return coords, explained


def _collapse_nearby(
    coords: np.ndarray,
    keys: list[tuple],
    tolerance: float,
) -> dict[tuple, list[int]]:
    """Group periods that share a key and land within ``tolerance`` of each other.

    Repeated profiles — a designed example, or a real series with many alike
    days — put several periods on the same pixel. Drawing them as one marker
    carrying a count is the only way their labels stay legible. Merging is
    greedy and only ever joins periods with an identical key, so a marker
    never spans two clusters.
    """
    buckets: dict[tuple, list[list[int]]] = {}
    for i in range(len(coords)):
        candidates = buckets.setdefault(tuple(keys[i]), [])
        for bucket in candidates:
            center = coords[bucket].mean(axis=0)
            if float(np.hypot(*(coords[i] - center))) <= tolerance:
                bucket.append(i)
                break
        else:
            candidates.append([i])

    groups: dict[tuple, list[int]] = {}
    for key, candidates in buckets.items():
        for bucket in candidates:
            center = coords[bucket].mean(axis=0)
            groups[(float(center[0]), float(center[1]), *key)] = bucket
    return groups


def _fan_out(offsets: int, total: int, radius: float) -> tuple[float, float]:
    """Spread markers that would otherwise sit exactly on top of each other."""
    if total <= 1:
        return 0.0, 0.0
    angle = 2 * np.pi * offsets / total
    return radius * float(np.cos(angle)), radius * float(np.sin(angle))


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


def path_panels(
    paths: dict[str, pd.DataFrame],
    x_attr: str,
    y_attr: str,
    *,
    background: dict[str, pd.DataFrame] | None = None,
    units: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    path_color: str | None = None,
    path_label: str = "highlighted path",
    dash: str = "solid",
    label_steps: bool = True,
    n_cols: int = 2,
    title: str | None = None,
) -> go.Figure:
    """Compare chronological paths in separate panels with shared axis limits.

    Each panel highlights one path, retaining the reference paths as faint
    coloured lines and open markers. Arrows and timestep labels preserve the
    order of each highlighted path. The legend toggles a named path in every
    panel, including its arrows. No data points are displaced.

    Args:
        paths: Named data frames, one panel each in insertion order. Rows must
            be in chronological order; paths may have different lengths.
        x_attr: Column plotted on the horizontal axis.
        y_attr: Column plotted on the vertical axis.
        background: Named reference paths repeated in every panel.
        units: Optional units for axis and hover labels, keyed by attribute.
        colors: Colours keyed by path name, used for both references and
            highlights. Reuse this mapping across figures to keep identities
            consistent. Unspecified names use the Plotly qualitative palette.
        path_color: Override the colour of every highlighted path. Useful when
            colour identifies reference days and highlights represent methods.
        path_label: Shared legend label for highlights when ``path_color`` is
            set. Panel headings and hover labels still identify each path.
        dash: Line style for highlighted paths, such as ``"solid"`` or ``"dash"``.
        label_steps: Print timestep labels beside highlighted markers. Disable
            for longer paths; timestep labels remain available on hover.
        n_cols: Maximum number of panels per row.
        title: Optional figure title.

    Returns:
        A Plotly figure with matching axes and one legend entry per identity.

    Raises:
        ValueError: If paths is empty, n_cols is not positive, or a path is
            empty, lacks either attribute, or contains non-finite values.

    Examples:
        >>> days = {"day1": pd.DataFrame({"solar": [0, 3, 0], "load": [4, 5, 6]})}
        >>> fig = path_panels(days, "solar", "load", background=days)
        >>> fig = path_panels(
        ...     {"mean": days["day1"]}, "solar", "load", background=days,
        ...     colors={"day1": "#0072B2"}, path_color="#222222",
        ...     path_label="representative", dash="dash",
        ... )
    """
    from plotly.subplots import make_subplots

    if not paths:
        raise ValueError("paths is empty — pass at least one path.")
    if n_cols < 1:
        raise ValueError("n_cols must be positive.")
    background = background or {}
    units = units or {}
    frames = []
    for name, frame in [*background.items(), *paths.items()]:
        if frame.empty or x_attr not in frame or y_attr not in frame:
            raise ValueError(
                f"Path {name!r} must have rows and columns {x_attr!r}, {y_attr!r}."
            )
        values = frame[[x_attr, y_attr]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Path {name!r} contains non-finite values.")
        frames.append(values)

    names = list(dict.fromkeys([*background, *paths]))
    palette = px.colors.qualitative.Plotly
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    color_map.update(colors or {})
    n_cols = min(n_cols, len(paths))
    n_rows = (len(paths) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[
            [{} if row * n_cols + col < len(paths) else None for col in range(n_cols)]
            for row in range(n_rows)
        ],
        shared_xaxes="all",
        shared_yaxes="all",
        subplot_titles=list(paths),
        horizontal_spacing=0.25 / n_cols,
        vertical_spacing=0.4 / n_rows,
    )
    for panel, (name, profile) in enumerate(paths.items()):
        row, col = panel // n_cols + 1, panel % n_cols + 1
        space = AttributeSpace(x_attr, y_attr, units=units)
        for member, block in background.items():
            space.add_path(
                block[x_attr],
                block[y_attr],
                name=member,
                color=color_map[member],
                symbol="circle-open",
                width=2,
                label_steps=False,
                arrows=False,
            )
        space.figure.update_traces(marker_size=12, opacity=0.3)
        space.add_path(
            profile[x_attr],
            profile[y_attr],
            name=name,
            color=path_color or color_map[name],
            symbol="circle",
            width=3,
            dash=dash,
            label_steps=label_steps,
            legendgroup="highlight" if path_color else name,
        )
        # The arrow trace is absent for a single-timestep path.
        highlight = space.figure.data[len(background)]
        highlight.update(textfont_size=12, textposition="top right")
        if len(profile) > 1:
            space.figure.data[-1].update(
                marker_size=[0] + [13] * (len(profile) - 1),
                marker_standoff=6,
            )
        for trace in space.figure.data:
            if trace.hoverinfo != "skip":
                trace.hovertemplate = (
                    f"{trace.name}<br>%{{customdata}}<br>"
                    f"{space._axis_label(x_attr)} = %{{x:.3g}}<br>"
                    f"{space._axis_label(y_attr)} = %{{y:.3g}}<extra></extra>"
                )
            trace.showlegend = False
            fig.add_trace(trace, row=row, col=col)

    # Full-opacity legend samples identify even the faint reference paths.
    for name in names if path_color is None else background:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=name,
                legendgroup=name,
                mode="lines+markers",
                hoverinfo="skip",
                line={
                    "color": color_map[name],
                    "dash": dash if name in paths else "solid",
                },
                marker={"color": color_map[name], "symbol": "circle-open", "size": 10},
            )
        )
    if path_color is not None:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=path_label,
                legendgroup="highlight",
                mode="lines+markers",
                hoverinfo="skip",
                line={"color": path_color, "dash": dash, "width": 3},
                marker={"color": path_color, "size": 9},
            )
        )

    all_values = np.concatenate(frames)
    for index, (attr, update_axis) in enumerate(
        [(x_attr, fig.update_xaxes), (y_attr, fig.update_yaxes)]
    ):
        lower, upper = all_values[:, index].min(), all_values[:, index].max()
        padding = 0.15 * (upper - lower or max(abs(lower), 1.0))
        label = f"{attr} [{units[attr]}]" if attr in units else attr
        update_axis(
            range=[lower - padding, upper + padding],
            title_text=label,
            showticklabels=True,
            zeroline=False,
            nticks=5,
        )
    fig.update_layout(
        title={"text": title, "font": {"size": 17}},
        template="plotly_white",
        height=300 * n_rows + 130,
        legend={"orientation": "h", "y": -0.14, "groupclick": "togglegroup"},
        margin={"l": 55, "r": 35, "t": 95, "b": 125},
    )
    return fig


def compare_series(
    series: dict[str, pd.DataFrame],
    *,
    mode: Literal["time_series", "duration_curve"] = "time_series",
    columns: list[str] | None = None,
    units: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    reference: str | None = None,
    duration_range: tuple[float, float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Compare named series or profiles, one subplot per attribute.

    Args:
        series: Named data frames. In time-series mode their existing indices
            define the horizontal positions. In duration-curve mode each row
            has equal duration and the index is ignored. Pass reconstructed
            series to account for unequal typical-period occurrence counts.
        mode: ``"time_series"`` preserves chronology; ``"duration_curve"``
            sorts each attribute independently, descending, and draws exact
            steps over 0–100% of time. This permits different-length inputs.
        columns: Attributes to plot, taken from the first frame if omitted.
        units: Optional attribute units for axes and hover labels.
        colors: Stable colours keyed by series name.
        reference: Name of an input series to draw as a thicker grey line.
            An explicit entry in ``colors`` overrides its colour.
        duration_range: Optional percentage interval, such as ``(0, 5)`` for
            the highest 5% of values. Only valid in duration-curve mode. Both
            axes focus on this part of the already fitted curves; values are
            neither filtered before sorting nor fitted again.
        title: Optional figure title.

    Returns:
        A Plotly figure with one shared legend and separate physical scales
        for the attributes. Legend entries toggle a series in every subplot.

    Raises:
        ValueError: If mode or reference is invalid, or inputs are empty,
            lack a requested attribute, or contain non-finite values.

    Examples:
        >>> fig = compare_series(
        ...     {"original": data, "mean": result.reconstructed},
        ...     mode="duration_curve", reference="original",
        ... )
    """
    from plotly.subplots import make_subplots

    from tsam.metrics import _comparison_columns

    if mode not in ("time_series", "duration_curve"):
        raise ValueError("mode must be 'time_series' or 'duration_curve'.")
    if reference is not None and reference not in series:
        raise ValueError(f"Reference {reference!r} is not in series.")
    if duration_range is not None and (
        mode != "duration_curve"
        or not 0 <= duration_range[0] < duration_range[1] <= 100
    ):
        raise ValueError(
            "duration_range requires duration-curve mode and 0 <= start < end <= 100."
        )
    columns = _comparison_columns(series, columns)
    units = units or {}
    palette = px.colors.qualitative.Plotly
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(series)}
    if reference is not None:
        color_map[reference] = "#7b8490"
    color_map.update(colors or {})
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True)
    for row, column in enumerate(columns, start=1):
        label = f"{column} [{units[column]}]" if column in units else column
        visible_values = []
        for name, frame in series.items():
            values = frame[column].to_numpy(dtype=float)
            x: np.ndarray | pd.Index
            if mode == "duration_curve":
                values = np.sort(values)[::-1]
                x = np.linspace(0, 100, len(values) + 1)
                values = np.r_[values, values[-1]]
                x_hover = "sorted time = %{x:.2f}%"
                start, end = duration_range or (0, 100)
                visible_values.append(values[:-1][(x[:-1] <= end) & (x[1:] >= start)])
            else:
                x = frame.index
                x_hover = "%{x}"
                visible_values.append(values)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=values,
                    name=name,
                    legendgroup=name,
                    showlegend=row == 1,
                    mode="lines",
                    line={
                        "color": color_map[name],
                        "width": 3 if name == reference else 2,
                        "shape": "hv" if mode == "duration_curve" else "linear",
                    },
                    hovertemplate=f"{name}<br>{x_hover}<br>{label} = %{{y:.4g}}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        values = np.concatenate(visible_values)
        lower, upper = values.min(), values.max()
        padding = 0.08 * (upper - lower or max(abs(lower), 1.0))
        fig.update_yaxes(
            title_text=label, range=[lower - padding, upper + padding], row=row, col=1
        )
    if mode == "duration_curve":
        fig.update_xaxes(range=list(duration_range or (0, 100)), nticks=6)
        x_title = "Share of time [%], sorted descending"
    else:
        x_title = "time / timestep"
    fig.update_xaxes(title_text=x_title, row=len(columns), col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=270 * len(columns) + 120,
        legend={"orientation": "h", "y": -0.16, "groupclick": "togglegroup"},
        margin={"l": 65, "r": 35, "t": 65, "b": 115},
    )
    return fig


_TIMESTEP_SYMBOLS = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "cross",
    "x",
    "star",
    "pentagon",
    "hexagon",
    "star-square",
    "hourglass",
)


def _rank_numbers(values: np.ndarray) -> np.ndarray:
    """Rank 1 for the highest value, ties resolved in index order.

    This is the tie-breaking the distribution representation uses when it
    places its levels by the ranks of a profile.
    """
    ascending = np.argsort(np.round(values, 10), kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[ascending] = np.arange(len(values), 0, -1)
    return ranks


def _validate_days(days: dict[str, pd.DataFrame], attr: str) -> int:
    """Check named days of equal length for ``attr`` and return that length."""
    if not days:
        raise ValueError("days is empty — pass at least one named member day.")
    lengths = {len(frame) for frame in days.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("All member days must have the same, non-zero length.")
    for name, frame in days.items():
        if attr not in frame:
            raise ValueError(f"Day {name!r} has no column {attr!r}.")
        if not np.isfinite(frame[attr].to_numpy(dtype=float)).all():
            raise ValueError(f"Day {name!r} contains non-finite values.")
    return lengths.pop()


def _duration_steps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending duration curve of ``values`` as steps over the share of time."""
    sorted_values = np.sort(values)[::-1]
    return (
        np.linspace(0, 100, len(values) + 1),
        np.asarray(np.r_[sorted_values, sorted_values[-1]]),
    )


_SYMBOL_GLYPHS = {
    "circle": "●",
    "square": "■",
    "diamond": "◆",
    "triangle-up": "▲",
    "triangle-down": "▼",
    "cross": "✚",
    "x": "✖",
    "star": "★",
    "pentagon": "⬟",
    "hexagon": "⬢",
    "star-square": "❖",
    "hourglass": "⧗",
}


def _label_trace(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    offset: float,
    color: str,
    legendgroup: str,
) -> go.Scatter:
    """Text labels drawn a fixed distance above (or below) their markers."""
    return go.Scatter(
        x=x,
        y=np.asarray(y, dtype=float) + offset,
        legendgroup=legendgroup,
        showlegend=False,
        mode="text",
        text=labels,
        textposition="middle center",
        textfont={"size": 11, "color": color},
        hoverinfo="skip",
    )


def duration_fit(
    days: dict[str, pd.DataFrame],
    attr: str,
    *,
    representative: pd.DataFrame | None = None,
    ordering_profile: pd.DataFrame | None = None,
    ordering_label: str = "mean profile",
    n_levels: int | None = None,
    units: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Show how member days are fitted and ordered into a duration representation.

    Three panels. The upper left draws each day's values of ``attr`` over its
    timesteps, with one marker symbol per timestep, listed in the legend and
    in a key below the panel title, and the ordering profile on top with the
    rank of every timestep printed below its marker: rank 1 is the timestep
    where the profile is highest, ties are resolved in timestep order. The
    upper right pools all values, sorts them in descending order and draws one
    marker per value in the colour of its day and the symbol of its timestep,
    so that every sorted value can be traced back; the sorted values are
    divided into ``n_levels`` equal slices, and a numbered segment marks the
    average of each slice, the level the representation uses. The lower panel
    draws the duration curves of the pooled days, of the ordering profile and
    of the representative; each step of the two profiles is labelled with the
    timestep it comes from and its rank. The representative, if given, is
    drawn dashed in the upper left panel with the ranks printed above its
    markers: the timestep of rank *k* holds the level of slice *k*.

    Args:
        days: Named member days of equal length, rows in chronological order.
        attr: Column to show.
        representative: The cluster's representative profile with a column
            ``attr``.
        ordering_profile: The profile whose ranks place the levels. Defaults to
            the mean of ``days``, which is what the default ordering uses.
        ordering_label: Legend label of the ordering profile.
        n_levels: Number of slices. Defaults to the number of timesteps per day,
            which is what the cluster representation uses.
        units: Optional unit for the value axis, keyed by attribute.
        colors: Colours keyed by day name or by ``ordering_label``. Unspecified
            names use the Plotly qualitative palette.
        title: Optional figure title.

    Returns:
        A Plotly figure with three panels, one legend entry per day and one
        per timestep symbol.

    Raises:
        ValueError: If ``days`` is empty, the days differ in length, ``attr``
            is missing, values are non-finite, the pooled values cannot be
            divided into ``n_levels`` equal slices, or a profile does not
            match the days.
    """
    from plotly.subplots import make_subplots

    n_steps = _validate_days(days, attr)
    n_levels = n_steps if n_levels is None else n_levels
    n_pooled = n_steps * len(days)
    if n_levels < 1 or n_pooled % n_levels:
        raise ValueError(
            f"{n_pooled} pooled values cannot be divided into {n_levels} equal slices."
        )
    slice_size = n_pooled // n_levels
    for name, profile in {
        "ordering_profile": ordering_profile,
        "representative": representative,
    }.items():
        if profile is not None and (attr not in profile or len(profile) != n_steps):
            raise ValueError(f"{name} needs a column {attr!r} with {n_steps} rows.")

    palette = px.colors.qualitative.Plotly
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(days)}
    color_map[ordering_label] = "#0072B2"
    color_map.update(colors or {})
    ordering_color = color_map[ordering_label]
    label = f"{attr} [{units[attr]}]" if units and attr in units else attr
    steps = np.arange(n_steps)
    symbols = [_TIMESTEP_SYMBOLS[t % len(_TIMESTEP_SYMBOLS)] for t in steps]
    values_by_day = {
        name: frame[attr].to_numpy(dtype=float) for name, frame in days.items()
    }
    pooled_values = np.concatenate(list(values_by_day.values()))
    if ordering_profile is None:
        profile_values = np.mean(list(values_by_day.values()), axis=0)
    else:
        profile_values = ordering_profile[attr].to_numpy(dtype=float)
    ranks = _rank_numbers(profile_values)
    representative_values = (
        None if representative is None else representative[attr].to_numpy(dtype=float)
    )
    drawn = [pooled_values, profile_values]
    if representative_values is not None:
        drawn.append(representative_values)
    lo, hi = float(np.min(np.concatenate(drawn))), float(np.max(np.concatenate(drawn)))
    span = hi - lo or max(abs(hi), 1.0)
    offset = 0.07 * span

    pooled = pd.DataFrame(
        {
            "value": pooled_values,
            "day": np.repeat(list(days), n_steps),
            "timestep": np.tile(steps, len(days)),
        }
    )
    pooled = pooled.sort_values("value", ascending=False, kind="stable")
    pooled["rank"] = np.arange(1, n_pooled + 1)
    levels = pooled["value"].to_numpy().reshape(n_levels, slice_size).mean(axis=1)

    key = "  ".join(f"t{t} {_SYMBOL_GLYPHS[s]}" for t, s in zip(steps, symbols))
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.14,
        subplot_titles=(
            f"1 · in time: colour = day, symbol = timestep<br><sup>{key}</sup>",
            "2 · pooled and sorted, averaged in numbered slices",
            "3 · duration curves · step labels: timestep (rank)",
        ),
    )
    for name, values in values_by_day.items():
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                name=name,
                legendgroup=name,
                showlegend=True,
                mode="lines+markers",
                line={"color": color_map[name], "width": 1.5},
                marker={"symbol": symbols, "size": 9},
                hovertemplate=f"{name}<br>t%{{x}}<br>{label} = %{{y:.4g}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        own = pooled[pooled["day"] == name]
        fig.add_trace(
            go.Scatter(
                x=own["rank"],
                y=own["value"],
                name=name,
                legendgroup=name,
                showlegend=False,
                mode="markers",
                marker={
                    "color": color_map[name],
                    "size": 9,
                    "symbol": [symbols[t] for t in own["timestep"]],
                },
                customdata=own["timestep"],
                hovertemplate=(
                    f"{name} at t%{{customdata}}<br>sorted position %{{x}}<br>"
                    f"{label} = %{{y:.4g}}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    for t, symbol in zip(steps, symbols):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=f"t{t}",
                legendgroup="timestep",
                legendgrouptitle={"text": "timestep"} if t == 0 else None,
                showlegend=True,
                mode="markers",
                marker={"symbol": symbol, "color": "#68717c", "size": 9},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=profile_values,
            name=ordering_label,
            legendgroup="ordering",
            showlegend=True,
            mode="lines+markers",
            text=[str(rank) for rank in ranks],
            line={"color": ordering_color, "width": 2.5},
            marker={"size": 6},
            hovertemplate=(
                f"{ordering_label}<br>t%{{x}} has rank %{{text}}<br>"
                f"{label} = %{{y:.4g}}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _label_trace(
            steps,
            profile_values,
            [str(rank) for rank in ranks],
            -offset,
            ordering_color,
            "ordering",
        ),
        row=1,
        col=1,
    )
    if representative_values is not None:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=representative_values,
                name="representative",
                legendgroup="representative",
                showlegend=True,
                mode="lines+markers",
                text=[str(rank) for rank in ranks],
                line={"color": "#222222", "width": 3, "dash": "dash"},
                marker={"symbol": symbols, "size": 9},
                hovertemplate=(
                    f"representative<br>t%{{x}} has rank %{{text}} and holds "
                    f"slice %{{text}}<br>{label} = %{{y:.4g}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            _label_trace(
                steps,
                representative_values,
                [str(rank) for rank in ranks],
                offset,
                "#222222",
                "representative",
            ),
            row=1,
            col=1,
        )
    for k, level in enumerate(levels):
        start, end = k * slice_size + 0.5, (k + 1) * slice_size + 0.5
        if k:
            fig.add_vline(
                x=start, line={"color": "#b0b6bd", "dash": "dot"}, row=1, col=2
            )
        fig.add_trace(
            go.Scatter(
                x=[start, (start + end) / 2, end],
                y=[level, level, level],
                name="level (slice average)",
                legendgroup="level",
                showlegend=k == 0,
                mode="lines+text",
                text=["", str(k + 1), ""],
                textposition="top center",
                textfont={"size": 11, "color": "#222222"},
                line={"color": "#222222", "width": 3},
                hovertemplate=f"slice {k + 1}<br>{label} = {level:.4g}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    x, y = _duration_steps(pooled_values)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name="member days, pooled",
            legendgroup="pooled",
            showlegend=True,
            mode="lines",
            line={"color": "#7b8490", "width": 3, "shape": "hv"},
            hovertemplate=(
                f"member days, pooled<br>sorted time = %{{x:.1f}}%<br>"
                f"{label} = %{{y:.4g}}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    curves = [(ordering_label, profile_values, ordering_color, "solid", -offset)]
    if representative_values is not None:
        curves.append(
            ("representative", representative_values, "#222222", "dash", offset)
        )
    for name, values, color, dash, label_offset in curves:
        x, y = _duration_steps(values)
        group = "ordering" if name == ordering_label else name
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                legendgroup=group,
                showlegend=False,
                mode="lines",
                line={"color": color, "width": 2.5, "shape": "hv", "dash": dash},
                hovertemplate=(
                    f"{name}<br>sorted time = %{{x:.1f}}%<br>"
                    f"{label} = %{{y:.4g}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        order = np.argsort(ranks, kind="stable")
        fig.add_trace(
            _label_trace(
                (np.arange(n_steps) + 0.5) / n_steps * 100,
                values[order],
                [f"t{t} ({ranks[t]})" for t in order],
                label_offset,
                color,
                group,
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="timestep", row=1, col=1)
    fig.update_xaxes(title_text="sorted position", row=1, col=2)
    fig.update_xaxes(title_text="share of time [%]", range=[0, 100], row=2, col=1)
    for row, col in ((1, 1), (1, 2), (2, 1)):
        fig.update_yaxes(
            title_text=label,
            range=[lo - 0.25 * span, hi + 0.25 * span],
            row=row,
            col=col,
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1000,
        legend={"orientation": "h", "y": -0.08},
        margin={"l": 65, "r": 35, "t": 110, "b": 130},
    )
    return fig


def error_areas(
    original: pd.DataFrame,
    reconstructions: dict[str, pd.DataFrame],
    attr: str,
    *,
    units: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Shade the error between an original series and its reconstructions.

    One row per reconstruction and two columns: the series in time and their
    duration curves, i.e. the same values sorted in descending order over the
    share of time. The area between the original and the reconstruction is
    filled, and each panel's title states the root mean square error of
    ``attr`` in physical units, so that the timing error and the
    duration-curve error of a representation can be compared at a glance.

    Args:
        original: The original series with a column ``attr``.
        reconstructions: Named reconstructed series of the same length.
        attr: Column to show.
        units: Optional unit for the value axis, keyed by attribute.
        colors: Colours keyed by reconstruction name. Unspecified names use
            the Plotly qualitative palette.
        title: Optional figure title.

    Returns:
        A Plotly figure with two panels per reconstruction.

    Raises:
        ValueError: If ``reconstructions`` is empty, a frame lacks ``attr``,
            lengths differ, or values are non-finite.
    """
    from plotly.subplots import make_subplots

    if not reconstructions:
        raise ValueError("reconstructions is empty — pass at least one series.")
    frames = {"original": original, **reconstructions}
    for name, frame in frames.items():
        if attr not in frame:
            raise ValueError(f"Series {name!r} has no column {attr!r}.")
        if len(frame) != len(original):
            raise ValueError(f"Series {name!r} differs in length from the original.")
        if not np.isfinite(frame[attr].to_numpy(dtype=float)).all():
            raise ValueError(f"Series {name!r} contains non-finite values.")

    palette = px.colors.qualitative.Plotly
    color_map = {
        name: palette[i % len(palette)] for i, name in enumerate(reconstructions)
    }
    color_map.update(colors or {})
    unit = units.get(attr, "") if units else ""
    label = f"{attr} [{unit}]" if unit else attr
    original_values = original[attr].to_numpy(dtype=float)

    titles = []
    for name, frame in reconstructions.items():
        values = frame[attr].to_numpy(dtype=float)
        rmse = np.sqrt(np.mean((original_values - values) ** 2))
        rmse_sorted = np.sqrt(
            np.mean((np.sort(original_values) - np.sort(values)) ** 2)
        )
        titles += [
            f"{name} in time · RMSE {rmse:.3g} {unit}".rstrip(),
            f"{name} sorted · RMSE {rmse_sorted:.3g} {unit}".rstrip(),
        ]
    fig = make_subplots(rows=len(reconstructions), cols=2, subplot_titles=titles)
    for row, (name, frame) in enumerate(reconstructions.items(), start=1):
        values = frame[attr].to_numpy(dtype=float)
        share, sorted_original = _duration_steps(original_values)
        _, sorted_values = _duration_steps(values)
        for col, (x, y_original, y_values, shape) in enumerate(
            [
                (original.index, original_values, values, "linear"),
                (share, sorted_original, sorted_values, "hv"),
            ],
            start=1,
        ):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_original,
                    name="original",
                    legendgroup="original",
                    showlegend=row == 1 and col == 1,
                    mode="lines",
                    line={"color": "#7b8490", "width": 3, "shape": shape},
                    hovertemplate=f"original<br>{label} = %{{y:.4g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_values,
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    mode="lines",
                    line={"color": color_map[name], "width": 2, "shape": shape},
                    fill="tonexty",
                    fillcolor=_to_rgba(color_map[name], 0.3),
                    hovertemplate=f"{name}<br>{label} = %{{y:.4g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_yaxes(title_text=label, row=row, col=1)
    fig.update_xaxes(title_text="time", row=len(reconstructions), col=1)
    fig.update_xaxes(
        title_text="Share of time [%], sorted descending",
        row=len(reconstructions),
        col=2,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=260 * len(reconstructions) + 150,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 65, "r": 35, "t": 80, "b": 110},
    )
    return fig


def _duration_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate the descending duration curve of ``values`` at shares of time.

    ``grid`` holds shares of time in percent, ascending from 0 to 100. The
    curve is a step function: the share interval ``[k/n, (k+1)/n)`` carries
    the ``k``-th largest value. The last grid point repeats the last value so
    that the result can be drawn with ``line.shape="hv"``.
    """
    sorted_values = np.sort(values)[::-1]
    index = np.minimum(np.floor(grid / 100 * len(values)).astype(int), len(values) - 1)
    return np.asarray(sorted_values[index])


def cluster_duration_curves(
    clusters: dict[str, dict[str, pd.DataFrame]],
    attr: str,
    *,
    reference: str | None = None,
    fill: bool = False,
    fill_alpha: float = 0.3,
    split: bool = False,
    units: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    dashes: dict[str, str] | None = None,
    widths: dict[str, float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Compare duration curves cluster by cluster, one row per cluster.

    Each panel sorts the values of ``attr`` of every named series in
    descending order and draws them as steps over the share of time, so that
    series of different lengths, such as a cluster's pooled member days and
    its representative, are comparable. A series name keeps its colour in
    every panel and has one legend entry. With a reference and ``fill``, the
    area between the reference and every other series is shaded and each
    panel's title states the root mean square error between the curves in
    physical units. With ``split``, every series other than the reference
    gets its own column, so that the shaded areas do not overlap.

    Args:
        clusters: Named rows, each a mapping of series name to data frame.
        attr: Column to show.
        reference: Series name drawn as a thicker grey line in every panel.
        fill: Shade the error against the reference and state its RMSE.
        fill_alpha: Opacity of the shaded areas.
        split: One column per series other than the reference.
        units: Optional unit for the value axis, keyed by attribute.
        colors: Colours keyed by series name. Unspecified names use the Plotly
            qualitative palette.
        dashes: Line styles keyed by series name, such as ``"dash"`` or
            ``"dot"``; unspecified names are solid.
        widths: Line widths keyed by series name.
        title: Optional figure title.

    Returns:
        A Plotly figure with one row per cluster.

    Raises:
        ValueError: If ``clusters`` or a row is empty, a frame lacks ``attr``,
            values are non-finite, or ``fill`` or ``split`` is requested
            without a reference present in every row.
    """
    from plotly.subplots import make_subplots

    if not clusters or any(not series for series in clusters.values()):
        raise ValueError("clusters must map each panel to at least one series.")
    names: list[str] = []
    for panel, series in clusters.items():
        for name, frame in series.items():
            if attr not in frame:
                raise ValueError(
                    f"Series {name!r} in {panel!r} has no column {attr!r}."
                )
            if frame.empty or not np.isfinite(frame[attr].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Series {name!r} in {panel!r} is empty or non-finite."
                )
            if name not in names:
                names.append(name)
        if (fill or split) and reference not in series:
            raise ValueError(
                f"fill and split need the reference {reference!r} in {panel!r}."
            )
    ref: str = reference or ""
    palette = px.colors.qualitative.Plotly
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    if reference is not None:
        color_map[ref] = "#7b8490"
    color_map.update(colors or {})
    dashes = dashes or {}
    widths = widths or {}
    unit = units.get(attr, "") if units else ""
    label = f"{attr} [{unit}]" if unit else attr
    others = [name for name in names if name != ref]
    columns: list[list[str]] = [[name] for name in others] if split else [others]

    def rmse_sorted(values: np.ndarray, reference_values: np.ndarray) -> float:
        grid = np.unique(
            np.r_[
                np.linspace(0, 100, len(values) + 1),
                np.linspace(0, 100, len(reference_values) + 1),
            ]
        )
        gap = _duration_on_grid(values, grid[:-1]) - _duration_on_grid(
            reference_values, grid[:-1]
        )
        return float(np.sqrt(np.sum(np.diff(grid) / 100 * gap**2)))

    titles = []
    for panel, series in clusters.items():
        for column in columns:
            present = [name for name in column if name in series]
            heading = f"{panel} · {present[0]}" if split and present else panel
            if not fill or not present:
                titles.append(heading)
                continue
            reference_values = series[ref][attr].to_numpy(dtype=float)
            errors = {
                name: rmse_sorted(
                    series[name][attr].to_numpy(dtype=float), reference_values
                )
                for name in present
            }
            if split:
                titles.append(f"{heading} · RMSE {errors[present[0]]:.3g} {unit}")
            else:
                parts = ", ".join(f"{n} {e:.3g}" for n, e in errors.items())
                titles.append(f"{heading} · RMSE sorted: {parts} {unit}")
    titles = [t.rstrip() for t in titles]

    fig = make_subplots(
        rows=len(clusters),
        cols=len(columns),
        shared_xaxes=True,
        shared_yaxes=split,
        subplot_titles=titles,
    )
    shown: set[str] = set()
    for row, series in enumerate(clusters.values(), start=1):
        for col, column in enumerate(columns, start=1):
            if not fill and not split and ref in series:
                column = [*column, ref]
            for name in column:
                if name not in series:
                    continue
                frame = series[name]
                x, y = _duration_steps(frame[attr].to_numpy(dtype=float))
                if fill and name != ref:
                    reference_x, _ = _duration_steps(
                        series[ref][attr].to_numpy(dtype=float)
                    )
                    x = np.unique(np.r_[x, reference_x])
                    y = _duration_on_grid(frame[attr].to_numpy(dtype=float), x)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=_duration_on_grid(
                                series[ref][attr].to_numpy(dtype=float), x
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                            mode="lines",
                            line={"color": "rgba(0,0,0,0)", "shape": "hv"},
                        ),
                        row=row,
                        col=col,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=name,
                        legendgroup=name,
                        showlegend=name not in shown,
                        mode="lines",
                        line={
                            "color": color_map[name],
                            "width": widths.get(name, 3 if name == ref else 2),
                            "shape": "hv",
                            "dash": dashes.get(name, "solid"),
                        },
                        fill="tonexty" if fill and name != ref else None,
                        fillcolor=_to_rgba(color_map[name], fill_alpha)
                        if fill and name != ref
                        else None,
                        hovertemplate=(
                            f"{name}<br>sorted time = %{{x:.2f}}%<br>"
                            f"{label} = %{{y:.4g}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                shown.add(name)
            if (fill or split) and ref in series:
                x, y = _duration_steps(series[ref][attr].to_numpy(dtype=float))
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=ref,
                        legendgroup=ref,
                        showlegend=ref not in shown,
                        mode="lines",
                        line={"color": color_map[ref], "width": 3, "shape": "hv"},
                        hovertemplate=(
                            f"{ref}<br>sorted time = %{{x:.2f}}%<br>"
                            f"{label} = %{{y:.4g}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                shown.add(ref)
            if col == 1:
                fig.update_yaxes(title_text=label, row=row, col=col)
            if row == len(clusters):
                fig.update_xaxes(
                    title_text="share of time [%], sorted descending",
                    row=row,
                    col=col,
                )
    fig.update_xaxes(range=[0, 100], nticks=6)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=250 * len(clusters) + 140,
        legend={"orientation": "h", "y": -0.16, "groupclick": "togglegroup"},
        margin={"l": 65, "r": 35, "t": 80, "b": 110},
    )
    return fig


def profile_panels(
    profiles: dict[str, pd.DataFrame],
    attrs: list[str],
    *,
    background: dict[str, pd.DataFrame] | None = None,
    references: dict[str, pd.DataFrame] | None = None,
    rank_sources: dict[str, dict[str, tuple[str, str] | str]] | None = None,
    units: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Draw profiles in time, one row per profile and one column per attribute.

    Every panel shows the background days faded and the row's own profile in
    black, with the rank of each value among the profile's values of that
    attribute printed above its marker, 1 for the highest. Profiles that hold
    the same values in a different order therefore show the same numbers
    permuted. ``rank_sources`` names, per profile and attribute, the
    reference profile and the attribute whose ranks placed the values. The
    panel title states this source, and when the source attribute is the
    panel's own, the reference is drawn in colour with its ranks printed
    below its markers, so that the ranks of the reference and of the profile
    can be compared marker by marker. A plain string describes a source that
    is not a drawable profile, such as a shared order.

    Args:
        profiles: Named profiles, one row each, with the columns ``attrs``.
        attrs: Columns, one panel column each.
        background: Named days drawn faded in every panel, such as the
            cluster's member days.
        references: Named reference profiles, such as a mean profile or a
            medoid day, that ``rank_sources`` can refer to.
        rank_sources: Per profile name and attribute, either a tuple of
            reference name and source attribute, or a description string.
        units: Optional units for the value axes, keyed by attribute.
        colors: Colours keyed by reference name. Unspecified names use the
            Plotly qualitative palette.
        title: Optional figure title.

    Returns:
        A Plotly figure with ``len(profiles)`` rows and ``len(attrs)`` columns.

    Raises:
        ValueError: If ``profiles`` or ``attrs`` is empty, a frame lacks one
            of ``attrs`` or contains non-finite values, or a rank source
            names an unknown profile, reference or attribute.
    """
    from plotly.subplots import make_subplots

    if not profiles or not attrs:
        raise ValueError("profiles and attrs must not be empty.")
    background = background or {}
    references = references or {}
    rank_sources = rank_sources or {}
    for name, frame in {**profiles, **background, **references}.items():
        missing = [attr for attr in attrs if attr not in frame]
        if missing:
            raise ValueError(f"Profile {name!r} lacks columns {missing}.")
        if not np.isfinite(frame[attrs].to_numpy(dtype=float)).all():
            raise ValueError(f"Profile {name!r} contains non-finite values.")
    for name, sources in rank_sources.items():
        if name not in profiles:
            raise ValueError(f"rank_sources names an unknown profile {name!r}.")
        for attr, source in sources.items():
            if attr not in attrs:
                raise ValueError(f"rank_sources names an unknown attribute {attr!r}.")
            if isinstance(source, tuple) and (
                source[0] not in references or source[1] not in attrs
            ):
                raise ValueError(f"rank source {source!r} is not available.")
    palette = px.colors.qualitative.Plotly
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(references)}
    color_map.update(colors or {})

    def source_of(name: str, attr: str) -> tuple[str, str] | str | None:
        return rank_sources.get(name, {}).get(attr)

    def drawn_reference(name: str, attr: str) -> str | None:
        source = source_of(name, attr)
        if isinstance(source, tuple) and source[1] == attr:
            return source[0]
        return None

    titles = []
    for name in profiles:
        for attr in attrs:
            rank_source = source_of(name, attr)
            if rank_source is None:
                titles.append(f"{name} · {attr}")
            elif isinstance(rank_source, tuple):
                titles.append(
                    f"{name} · {attr}<br><sup>ranks of {rank_source[0]} "
                    f"({rank_source[1]})</sup>"
                )
            else:
                titles.append(f"{name} · {attr}<br><sup>ranks of {rank_source}</sup>")
    spans = {}
    for attr in attrs:
        stacked = np.concatenate(
            [frame[attr].to_numpy(dtype=float) for frame in profiles.values()]
            + [frame[attr].to_numpy(dtype=float) for frame in background.values()]
            + [frame[attr].to_numpy(dtype=float) for frame in references.values()]
        )
        lo, hi = float(stacked.min()), float(stacked.max())
        spans[attr] = (lo, hi, hi - lo or max(abs(hi), 1.0))

    fig = make_subplots(
        rows=len(profiles),
        cols=len(attrs),
        shared_xaxes=True,
        vertical_spacing=0.14 / max(len(profiles) - 1, 1) + 0.04,
        subplot_titles=titles,
    )
    shown: set[str] = set()
    for row, (name, profile) in enumerate(profiles.items(), start=1):
        for col, attr in enumerate(attrs, start=1):
            label = f"{attr} [{units[attr]}]" if units and attr in units else attr
            lo, hi, span = spans[attr]
            offset = 0.09 * span
            for day, frame in background.items():
                fig.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=frame[attr].to_numpy(dtype=float),
                        name="member days",
                        legendgroup="background",
                        showlegend="background" not in shown,
                        mode="lines",
                        line={"color": "#c3c8ce", "width": 1.5},
                        hovertemplate=f"{day}<br>{label} = %{{y:.4g}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                shown.add("background")
            ref = drawn_reference(name, attr)
            if ref is not None:
                frame = references[ref]
                values = frame[attr].to_numpy(dtype=float)
                ranks = [str(rank) for rank in _rank_numbers(values)]
                fig.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=values,
                        name=ref,
                        legendgroup=ref,
                        showlegend=ref not in shown,
                        mode="lines+markers",
                        text=ranks,
                        line={"color": color_map[ref], "width": 2},
                        marker={"size": 6},
                        hovertemplate=(
                            f"{ref}<br>rank %{{text}}<br>{label} = %{{y:.4g}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    _label_trace(
                        np.asarray(frame.index),
                        values,
                        ranks,
                        -offset,
                        color_map[ref],
                        ref,
                    ),
                    row=row,
                    col=col,
                )
                shown.add(ref)
            values = profile[attr].to_numpy(dtype=float)
            ranks = [str(rank) for rank in _rank_numbers(values)]
            fig.add_trace(
                go.Scatter(
                    x=profile.index,
                    y=values,
                    name="representative",
                    legendgroup="representative",
                    showlegend="representative" not in shown,
                    mode="lines+markers",
                    text=ranks,
                    line={"color": "#222222", "width": 3},
                    marker={"size": 6},
                    hovertemplate=(
                        f"{name}<br>level %{{text}}<br>{label} = %{{y:.4g}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                _label_trace(
                    np.asarray(profile.index),
                    values,
                    ranks,
                    offset,
                    "#222222",
                    "representative",
                ),
                row=row,
                col=col,
            )
            shown.add("representative")
            fig.update_yaxes(range=[lo - 0.3 * span, hi + 0.3 * span], row=row, col=col)
            if row == 1:
                fig.update_yaxes(title_text=label, row=row, col=col)
            if row == len(profiles):
                fig.update_xaxes(title_text="timestep", row=row, col=col)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=260 * len(profiles) + 180,
        legend={"orientation": "h", "y": -0.1 - 0.03 * (6 - min(len(profiles), 6))},
        margin={"l": 65, "r": 35, "t": 90, "b": 100},
    )
    return fig


class ResultPlotAccessor:
    """Plotting accessor for AggregationResult.

    Provides convenient plotting methods directly on the result object.

    Examples:
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> result.plot.compare()  # Compare original vs reconstructed
        >>> result.plot.residuals()  # View reconstruction errors
        >>> result.plot.cluster_representatives()
        >>> result.plot.cluster_members()
        >>> result.plot.cluster_counts()
        >>> result.plot.feature_space()
    """

    def __init__(self, result: AggregationResult) -> None:
        self._result = result

    def cluster_representatives(
        self,
        columns: list[str] | None = None,
        units: dict[str, str] | None = None,
        title: str = "Cluster Representatives",
    ) -> go.Figure:
        """Plot all cluster representatives (typical periods).

        Args:
            columns: Columns to plot. If None, plots all available columns.
            title: Plot title.

        Returns:
            A Plotly figure of the cluster representatives.
        """
        typ = self._result.cluster_representatives
        counts = self._result.cluster_counts

        available_columns = [c for c in typ.columns if c not in ["cluster", "timestep"]]
        columns = _validate_columns(
            columns, available_columns, "cluster_representatives"
        )

        # Reset index to get period/timestep as columns. The representatives
        # index has 2 levels normally (period, timestep) but 3 with
        # segmentation (period, segment step, segment duration). Name the
        # first two levels and drop any extra segment-metadata levels so the
        # column assignment matches the actual number of columns.
        df = typ[columns].reset_index()
        n_index_levels = typ.index.nlevels
        index_names = [
            "Period",
            "Timestep",
            *[f"_extra_{i}" for i in range(n_index_levels - 2)],
        ]
        df.columns = pd.Index([*index_names, *columns])

        # Map period IDs to labels with their occurrence counts
        period_label = {p: f"Period {p} (n={counts.get(p, 1)})" for p in counts}
        df["Period"] = df["Period"].map(lambda p: period_label.get(p, f"Period {p}"))

        long_df = df.melt(
            id_vars=["Period", "Timestep"],
            value_vars=columns,
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

        Args:
            columns: Columns to plot. If None, plots all columns.
            clusters: Cluster indices to include. If None, includes all clusters.
            slider: Which dimension to put on the slider; the other dimension
                becomes ``facet_col``.

                - ``"cluster"``: slider flips through clusters, columns are facets.
                - ``"column"``: slider flips through columns, clusters are facets.
            title: Plot title. Defaults to "Cluster Members".

        Returns:
            A Plotly figure with member periods and highlighted representatives.

        Examples:
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
                return np.repeat(np.asarray(rep[col].values), durations)
            return np.asarray(rep[col].values)

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
                    "opacity": _PERIOD_SHADE_OPACITY,
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
        # The swatch fill is blended to the same opacity as the shaded bands
        # (over a white background), so what's shown in the legend is what the
        # band actually looks like on the chart. A solid outline in the full
        # saturated colour keeps the swatch identifiable despite the fade —
        # without it, hues like blue (#636EFA) and purple (#AB63FA) become
        # nearly indistinguishable pastels at 18% opacity, which is what made
        # the legend and shading look mismatched even though the underlying
        # cluster->colour mapping (cmap) was always identical in both places.

        for cid, color in cmap.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={
                        "color": _to_rgba(color, _PERIOD_SHADE_OPACITY),
                        "size": 14,
                        "symbol": "square",
                    },
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

    def feature_space(
        self,
        *,
        labels: list[str] | None = None,
        show_centers: bool = True,
        show_assignments: bool = True,
        merge_tolerance: float = 0.04,
        title: str | None = None,
    ) -> go.Figure:
        """Plot the space the clustering works in — one point per period.

        A clustering method never sees day-shapes. It sees each period as a
        single point in an ``n_timesteps x n_attributes`` space and applies a
        rule for covering those points with ``k`` centers. This plot is that
        space, and it is where the difference between methods becomes
        geometric rather than anecdotal.

        That space has too many dimensions to draw, so it is flattened to two
        with **principal component analysis (PCA)**: PCA finds the two
        directions along which the periods differ most and plots each period's
        position along them. Those two directions are the axes — the *first*
        and *second principal component*, ``PC 1`` and ``PC 2``. They are
        mixtures of all timesteps and attributes, so they carry no physical
        unit and no single meaning; only the *relative positions* of the
        periods do. The default title reports how much of the original
        variation the flattening kept — when that share is low, the distances
        on screen are not the distances the algorithm used.

        Two details keep it readable. Periods landing on the same point are
        drawn as **one marker carrying a count** — repeated profiles otherwise
        stack their labels into an unreadable smudge. And the axes are locked
        to **equal aspect**, because a plot whose subject is distance must not
        distort it.

        Parameters
        ----------
        labels : list[str], optional
            One descriptive name per period, e.g. an archetype such as
            ``"sunny"``. Sets the marker fill colour and the printed label, so
            you can see how a known grouping relates to the one found. Without
            it markers are filled by cluster.
        show_centers : bool, default True
            Draw each cluster's center. A ringed marker means the center is a
            **real period** (k-medoids, k-maxoids, medoid-based hierarchical);
            a star means it is a **computed mean** that no period matches.
        show_assignments : bool, default True
            Draw a line from every period to the center it was assigned to.
        merge_tolerance : float, default 0.04
            How close two periods must be, as a fraction of the plot's widest
            axis, before they are drawn as one counted marker. Only periods in
            the same cluster (and with the same label) are ever merged. Set to
            ``0`` to merge only periods that coincide *exactly* — identical
            profiles still share one marker, since two markers cannot occupy
            one pixel legibly.
        title : str, optional
            Figure title. Defaults to a title naming the explained variance.

        Returns
        -------
        go.Figure

        Raises
        ------
        ValueError
            If ``labels`` has a different length than the number of periods,
            or the result carries no normalised values (e.g. restored
            from JSON).

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=3, period_duration="1D")
        >>> result.plot.feature_space()  # doctest: +SKIP
        >>> archetypes = ["sunny", "cloudy", "storm", ...]
        >>> result.plot.feature_space(labels=archetypes)  # doctest: +SKIP
        """
        matrix = _period_matrix(self._result)
        coords, explained = _project_2d(matrix)
        assignments = self._result.cluster_assignments
        n_periods = len(coords)

        if labels is not None and len(labels) != n_periods:
            raise ValueError(
                f"labels has {len(labels)} entries but there are {n_periods} "
                f"periods. Pass one label per period."
            )

        cluster_ids = sorted({int(c) for c in assignments})
        qualitative = px.colors.qualitative.Plotly
        cluster_color = {
            cid: qualitative[i % len(qualitative)] for i, cid in enumerate(cluster_ids)
        }
        if labels is not None:
            categories = list(dict.fromkeys(labels))
            bold = px.colors.qualitative.Bold
            fill_color = {
                name: bold[i % len(bold)] for i, name in enumerate(categories)
            }
        else:
            categories = [f"cluster {cid}" for cid in cluster_ids]
            fill_color = {f"cluster {cid}": cluster_color[cid] for cid in cluster_ids}

        def category_of(period: int) -> str:
            if labels is not None:
                return labels[period]
            return f"cluster {int(assignments[period])}"

        keys = [(int(assignments[i]), category_of(i)) for i in range(n_periods)]
        span = float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1e-9))
        groups = _collapse_nearby(coords, keys, merge_tolerance * span)

        # Markers that would still overlap after merging (same location,
        # different cluster) get fanned onto a small circle so both stay visible.
        radius = 0.025 * span
        by_location: dict[tuple[float, float], list[tuple]] = {}
        for key in groups:
            by_location.setdefault((key[0], key[1]), []).append(key)

        drawn: dict[tuple, tuple[float, float, list[int]]] = {}
        for location, keys_here in by_location.items():
            for offset, key in enumerate(keys_here):
                dx, dy = _fan_out(offset, len(keys_here), radius)
                drawn[key] = (location[0] + dx, location[1] + dy, groups[key])

        fig = go.Figure()

        # Centers first, so period markers sit on top of the assignment lines.
        center_indices = self._result.clustering.cluster_centers
        center_xy: dict[int, tuple[float, float]] = {}
        real_center: dict[int, bool] = {}
        for cid in cluster_ids:
            members = np.flatnonzero(assignments == cid)
            point = (
                coords[members].mean(axis=0) if len(members) else coords.mean(axis=0)
            )
            center_xy[cid] = (float(point[0]), float(point[1]))
            real_center[cid] = False
        if center_indices is not None:
            for period_idx in center_indices:
                idx = int(period_idx)
                if 0 <= idx < n_periods:
                    cid = int(assignments[idx])
                    center_xy[cid] = (float(coords[idx, 0]), float(coords[idx, 1]))
                    real_center[cid] = True

        if show_assignments:
            for key, (x, y, _periods) in drawn.items():
                cid = int(key[2])
                cx, cy = center_xy[cid]
                fig.add_trace(
                    go.Scatter(
                        x=[x, cx],
                        y=[y, cy],
                        mode="lines",
                        line={"color": cluster_color[cid], "width": 1.5},
                        opacity=0.75,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        for category in categories:
            xs, ys, texts, hovers, rings, sizes = [], [], [], [], [], []
            for key, (x, y, periods) in drawn.items():
                if key[3] != category:
                    continue
                cid = int(key[2])
                count = len(periods)
                xs.append(x)
                ys.append(y)
                texts.append(f"{category} {_TIMES}{count}" if count > 1 else category)
                hovers.append(
                    f"{category}<br>cluster {cid}<br>"
                    f"period{'s' if count > 1 else ''} "
                    f"{', '.join(str(m) for m in periods)}"
                )
                rings.append(cluster_color[cid])
                sizes.append(min(13 + 4 * (count - 1), 34))
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=texts,
                    textposition="top center",
                    hovertext=hovers,
                    hoverinfo="text",
                    marker={
                        "size": sizes,
                        "color": fill_color[category],
                        "line": {"color": rings, "width": 3},
                    },
                    name=category,
                    legendgroup=category,
                )
            )

        if show_centers:
            for cid in cluster_ids:
                cx, cy = center_xy[cid]
                is_real = real_center[cid]
                fig.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode="markers",
                        marker={
                            "symbol": "circle-open" if is_real else "star",
                            "size": 24 if is_real else 17,
                            "color": cluster_color[cid],
                            "line": {"color": cluster_color[cid], "width": 3},
                        },
                        name=f"center {cid}"
                        + (" (real period)" if is_real else " (mean)"),
                        hoverinfo="name",
                        legendgroup="centers",
                        legendgrouptitle_text="cluster centers",
                    )
                )

        fig.update_layout(
            title=title
            or (
                "Periods in feature space "
                f"(flattened to 2-D, keeping {explained:.0%} of the variation)"
            ),
            xaxis_title="main direction in which periods differ (PC 1)",
            yaxis_title="second direction in which periods differ (PC 2)",
            legend_title_text="period" if labels is not None else "cluster",
            height=520,
        )
        # Distance is the subject of this plot, so it must not be distorted.
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    def cluster_counts(self, title: str = "Cluster Counts") -> go.Figure:
        """Plot how many original periods each cluster represents.

        Args:
            title: Plot title.

        Returns:
            A bar chart of cluster counts.
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

        Args:
            title: Plot title.

        Returns:
            A grouped bar chart of accuracy metrics per column.
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

        Args:
            title: Plot title.

        Returns:
            A bar chart of average segment durations.

        Raises:
            ValueError: If no segmentation was used.
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

        Args:
            columns: Columns to compare. If None, compares all columns.
            mode: Comparison mode.

                - "overlay": Both series on same axes
                - "side_by_side": Separate subplots
                - "duration_curve": Compare sorted values
            title: Plot title.
            time_slice: Restrict the comparison to a window of the time axis, e.g.
                ``slice("2010-01-11", "2010-01-17")``. Useful for zooming into a
                few periods so fine detail (such as segment step functions) is
                visible. Applies to all modes; for ``"duration_curve"`` the curve
                is computed over the sliced window.
            color: Which dimension drives the line colour, ``"column"`` or
                ``"source"`` (default ``"column"``).

                - ``"column"``: colour by column; the source (original vs.
                  reconstructed) is the secondary encoding — dash in ``"overlay"``
                  and ``"duration_curve"``, facet row in ``"side_by_side"``.
                - ``"source"``: colour by source, with the column as the secondary
                  encoding instead. Clearer when comparing a single column, where
                  the original/reconstructed split is the thing you want to see.

                Applies to all modes.

        Returns:
            A Plotly figure comparing original and reconstructed series.

        Raises:
            ValueError: If an unknown mode is given.

        Examples:
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

        Args:
            columns: Columns to plot. If None, plots all.
            mode: Display mode.

                - "time_series": Residuals over time
                - "histogram": Distribution of residuals
                - "by_period": Mean absolute error per period (bar chart)
                - "by_timestep": Mean absolute error by timestep within period
            title: Plot title.

        Returns:
            A Plotly figure of the residuals.

        Raises:
            ValueError: If an unknown mode is given.

        Examples:
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


def compare_partitions(
    partitions: dict,
    *,
    labels: list[str] | None = None,
    canonical: bool = True,
    title: str | None = None,
    show_ids: bool = True,
) -> go.Figure:
    """Compare which periods several clusterings group together.

    Answers one question exactly: *do these configurations put the same
    periods in the same group?* Each row is a clustering, each column an
    original period, and the colour is the group that period landed in. Rows
    that look alike agree; rows that look different disagree, and you can read
    off precisely which periods moved.

    Unlike a feature-space scatter this involves no projection, so nothing is
    approximated — but it shows membership only, not distance.

    Parameters
    ----------
    partitions : dict
        Maps a name to either an :class:`~tsam.result.AggregationResult` or a
        raw sequence of cluster ids, one per period. Rows appear in insertion
        order.
    labels : list[str], optional
        One descriptive name per period, e.g. an archetype. Printed under the
        period index so you can see what each column *is*.
    canonical : bool, default True
        Renumber each row's clusters by order of first appearance. Cluster ids
        are arbitrary, so without this two identical groupings can look
        different purely because they numbered their groups differently.
        Turn it off to see the ids a result actually carries.
    title : str, optional
        Figure title.
    show_ids : bool, default True
        Print the cluster id inside each cell.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``partitions`` is empty, the rows differ in length, or ``labels``
        does not match the number of periods.

    Examples
    --------
    >>> runs = {m: tsam.aggregate(df, n_clusters=3, cluster=ClusterConfig(method=m))
    ...         for m in ["kmeans", "kmedoids", "averaging"]}
    >>> tsam.plot.compare_partitions(runs)  # doctest: +SKIP
    """
    if not partitions:
        raise ValueError("partitions is empty — pass at least one clustering.")

    names = list(partitions)
    rows = []
    for name in names:
        entry = partitions[name]
        assignments = getattr(entry, "cluster_assignments", entry)
        values = np.asarray(list(assignments), dtype=int)
        rows.append(_canonical_labels(values) if canonical else values)

    widths = {len(row) for row in rows}
    if len(widths) > 1:
        raise ValueError(
            f"All partitions must cover the same number of periods, got {sorted(widths)}."
        )
    n_periods = widths.pop()

    if labels is not None and len(labels) != n_periods:
        raise ValueError(
            f"labels has {len(labels)} entries but there are {n_periods} periods."
        )

    matrix = np.vstack(rows)
    n_groups = int(matrix.max()) + 1

    # A discrete scale: cluster ids are categories, so neighbouring ids must not
    # read as "nearly the same" the way a continuous ramp would imply.
    qualitative = px.colors.qualitative.Plotly
    colorscale = []
    for i in range(n_groups):
        color = qualitative[i % len(qualitative)]
        colorscale.append([i / n_groups, color])
        colorscale.append([(i + 1) / n_groups, color])

    if labels is not None:
        tick_text = [f"{i}<br>{labels[i]}" for i in range(n_periods)]
    else:
        tick_text = [str(i) for i in range(n_periods)]

    hover = [
        [
            f"{names[r]}<br>period {c}"
            + (f" ({labels[c]})" if labels is not None else "")
            + f"<br>cluster {matrix[r, c]}"
            for c in range(n_periods)
        ]
        for r in range(len(names))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=list(range(n_periods)),
            y=names,
            colorscale=colorscale,
            zmin=-0.5,
            zmax=n_groups - 0.5,
            showscale=False,
            xgap=2,
            ygap=4,
            hoverinfo="text",
            text=hover,
            texttemplate="%{z}" if show_ids else None,
            textfont={"size": 11, "color": "white"},
        )
    )
    fig.update_layout(
        title=title or "Which periods each method groups together",
        xaxis_title="original period",
        yaxis_title="",
        height=110 + 42 * len(names),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_periods)),
        ticktext=tick_text,
        showgrid=False,
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig
