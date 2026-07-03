"""Render the docs performance figures from a saved ``pytest-benchmark`` run.

Loads the run into a tidy DataFrame via ``pytest_benchmem.load_long_df`` and writes
two self-contained plotly HTML figures into ``docs/assets/benchmarks/``. Wall-clock
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

import plotly.express as px
import pytest_benchmem as pb
from _data import METHODS
from bench_scaling import REPRESENTATIONS as _REPS

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
    _render(
        df[df["node.func"] == "test_representation"].copy(),
        color="representation",
        order=REPRESENTATIONS,
        title="Representation cost — hierarchical, 1 year, 4 columns, 12 clusters",
        out="representation_scaling.html",
    )


if __name__ == "__main__":
    main()
