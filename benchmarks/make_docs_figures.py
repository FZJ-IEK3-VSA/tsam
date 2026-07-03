"""Render the docs performance figures from a saved ``pytest-benchmark`` run.

Loads the run into a tidy DataFrame via ``pytest_benchmem.load_long_df`` and writes
self-contained plotly HTML into ``docs/assets/benchmarks/``: two line figures
(method / representation scaling) and two log-coloured runtime heatmaps. Wall-clock
time only, in seconds.

Usage::

    python benchmarks/make_docs_figures.py [path/to/run.json]

With no argument, the most recent ``*_headline.json`` under ``.benchmarks/`` is
used.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytest_benchmem as pb
from _data import METHODS
from bench_scaling import REPRESENTATIONS as _REPS
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "assets" / "benchmarks"
PERIOD = {24: "daily (365 periods)", 168: "weekly (52 periods)"}
PERIOD_ORDER = list(PERIOD.values())
RES_TICKS = {
    "tickvals": [2190, 8760, 35040, 105120],
    "ticktext": ["4h", "1h", "15min", "5min"],
}
REPRESENTATIONS = [k for k, _ in _REPS]


def _load(path: str | None) -> pd.DataFrame:
    if path is None:
        hits = sorted(glob.glob(str(ROOT / ".benchmarks" / "*" / "*_headline.json")))
        if not hits:
            raise SystemExit(
                "No *_headline.json found; run with --benchmark-save=headline"
            )
        path = hits[-1]
    df, _ = pb.load_long_df(path, metric="time", stat="median")
    df["runtime_s"] = df["value"]  # load_long_df returns seconds
    df["period"] = df["period_hours"].map(PERIOD)
    return df


def _render(
    s: pd.DataFrame, color: str, order: list[str], title: str, out: str
) -> None:
    """Write one faceted method/representation scaling figure to ``out``."""
    present = [v for v in order if v in set(s[color])]
    fig = px.line(
        s.sort_values([color, "period_hours", "timesteps"]),
        x="timesteps",
        y="runtime_s",
        color=color,
        facet_col="period",
        markers=True,
        log_x=True,
        log_y=True,
        category_orders={color: present, "period": PERIOD_ORDER},
        labels={
            "timesteps": "resolution of the 1-year input",
            "runtime_s": "runtime (s)",
        },
        title=title,
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(**RES_TICKS)
    fig.update_traces(marker={"size": 7}, line={"width": 2})
    fig.update_layout(
        template="plotly_white",
        height=440,
        margin={"l": 70, "r": 20, "t": 70, "b": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text=color,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUT_DIR / out,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )
    print(f"wrote {OUT_DIR / out}")


_RES_ORDER = [240, 60, 15, 5]
_RES_LABEL = {240: "4h", 60: "1h", 15: "15min", 5: "5min"}
_HEAT_PERIODS = [(24, "daily (365 periods)"), (168, "weekly (52 periods)")]
# Log-scale colourbar ticks: 0.01 s .. 10 s.
_CBAR = {"tickvals": [-2, -1, 0, 1], "ticktext": ["0.01", "0.1", "1", "10"]}


def _heatmap(s: pd.DataFrame, key: str, order: list[str], title: str, out: str) -> None:
    """Write the underlying runtimes as a log-coloured heatmap (cell text = seconds).

    Rows are the methods/representations, columns the four resolutions, faceted into
    daily vs weekly. Colour is ``log10(seconds)`` so the orders-of-magnitude spread
    stays legible; the printed number is the actual runtime. A skipped case is a
    blank cell.
    """
    names = [v for v in order if v in set(s[key])]
    res_labels = [_RES_LABEL[r] for r in _RES_ORDER]
    logs, texts = {}, {}
    for period_hours, _ in _HEAT_PERIODS:
        z = np.full((len(names), len(_RES_ORDER)), np.nan)
        text = [["—"] * len(_RES_ORDER) for _ in names]
        for i, name in enumerate(names):
            rows = s[(s[key] == name) & (s["period_hours"] == period_hours)]
            by_res = dict(zip(rows["resolution_min"], rows["value"]))
            for j, r in enumerate(_RES_ORDER):
                if r in by_res:
                    z[i, j] = np.log10(by_res[r])
                    text[i][j] = f"{by_res[r]:.2f}"
        logs[period_hours], texts[period_hours] = z, text
    zmin = float(np.nanmin(list(logs.values())))
    zmax = float(np.nanmax(list(logs.values())))

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=[label for _, label in _HEAT_PERIODS],
    )
    for col, (period_hours, _) in enumerate(_HEAT_PERIODS, start=1):
        fig.add_trace(
            go.Heatmap(
                z=logs[period_hours],
                x=res_labels,
                y=names,
                text=texts[period_hours],
                texttemplate="%{text}",
                textfont={"size": 12},
                zmin=zmin,
                zmax=zmax,
                colorscale="YlOrRd",
                xgap=2,
                ygap=2,
                showscale=(col == 2),
                colorbar={"title": "runtime (s)", "len": 0.9, **_CBAR},
                hovertemplate="%{y} · %{x}: %{text}s<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_yaxes(autorange="reversed")  # first method on top
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(260, 96 + 46 * len(names)),
        margin={"l": 120, "r": 20, "t": 70, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUT_DIR / out,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )
    print(f"wrote {OUT_DIR / out}")


def main() -> None:
    df = _load(sys.argv[1] if len(sys.argv) > 1 else None)
    scale = df[
        (df["node.func"] == "test_scale")
        & (df["columns"] == 4)
        & (df["clusters"] == 12)
    ]
    _render(
        scale.copy(),
        color="method",
        order=METHODS,
        title="How clustering methods scale — 1 year, 4 columns, 12 clusters",
        out="method_scaling.html",
    )
    _heatmap(
        scale,
        key="method",
        order=METHODS,
        title="Method runtimes (seconds) — 1 year, 4 columns, 12 clusters",
        out="method_runtime.html",
    )

    rep = df[df["node.func"] == "test_representation"].copy()
    _render(
        rep,
        color="representation",
        order=REPRESENTATIONS,
        title="Representation cost — hierarchical, 1 year, 4 columns, 12 clusters",
        out="representation_scaling.html",
    )
    _heatmap(
        rep,
        key="representation",
        order=REPRESENTATIONS,
        title="Representation runtimes (seconds) — hierarchical, 1 year, 4 columns",
        out="representation_runtime.html",
    )


if __name__ == "__main__":
    main()
