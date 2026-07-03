"""Render the method-scaling figure for the performance docs page.

Loads a saved ``pytest-benchmark`` run into a tidy DataFrame via
``pytest_benchmem.load_long_df`` and writes one self-contained plotly HTML figure
into ``docs/assets/benchmarks/``. Wall-clock time only, in seconds.

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

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "assets" / "benchmarks"
TEMPLATE = "plotly_white"
_PERIOD = {24: "daily (365 periods)", 168: "weekly (52 periods)"}
_PERIOD_ORDER = ["daily (365 periods)", "weekly (52 periods)"]
_RES_TICKS = {"tickvals": [8760, 35040, 105120], "ticktext": ["1h", "15min", "5min"]}
REPRESENTATIONS = [
    "mean",
    "medoid",
    "distribution",
    "distribution_minmax",
    "concurrency_consensus",
    "concurrency_assignment",
    "minmax_mean",
]


def _load(path: str | None) -> pd.DataFrame:
    if path is None:
        hits = sorted(glob.glob(str(ROOT / ".benchmarks" / "*" / "*_headline.json")))
        if not hits:
            raise SystemExit(
                "No *_headline.json found; run with --benchmark-save=headline"
            )
        path = hits[-1]
    df, _ = pb.load_long_df(path, metric="time", stat="median")
    df["s"] = df["value"]  # load_long_df returns seconds
    return df


def fig_method_scaling(df: pd.DataFrame) -> None:
    """How each method scales, and how that depends on the period length.

    x = timesteps (resolution, log), y = runtime (log), colour = method, facets =
    period (daily vs weekly). Tells the whole story: (a) the scalable methods stay
    fast even at fine resolution, (b) methods differ by orders of magnitude, and
    (c) the difference *depends on the setting* — the exact ``kmedoids`` / heuristic
    ``kmaxoids`` are infeasible on 365 daily periods but usable on 52 weekly ones.
    """
    s = df[
        (df["node.func"] == "test_scale")
        & (df["columns"] == 4)
        & (df["clusters"] == 12)
    ].copy()
    s["runtime_s"] = s["s"]
    s["period"] = s["period_hours"].map(
        {24: "daily (365 periods)", 168: "weekly (52 periods)"}
    )
    methods_present = [m for m in METHODS if m in set(s["method"])]
    fig = px.line(
        s.sort_values(["method", "period_hours", "timesteps"]),
        x="timesteps",
        y="runtime_s",
        color="method",
        facet_col="period",
        markers=True,
        log_x=True,
        log_y=True,
        category_orders={
            "method": methods_present,
            "period": ["daily (365 periods)", "weekly (52 periods)"],
        },
        labels={
            "timesteps": "resolution of the 1-year input",
            "runtime_s": "runtime (s)",
        },
        title="How clustering methods scale — 1 year, 4 columns, 12 clusters",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(
        tickvals=[8760, 35040, 105120],
        ticktext=["1h<br>8 760", "15min<br>35 040", "5min<br>105 120"],
    )
    fig.update_traces(marker={"size": 7}, line={"width": 2})
    fig.update_layout(
        template=TEMPLATE,
        height=440,
        margin={"l": 70, "r": 20, "t": 70, "b": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="method",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUT_DIR / "method_scaling.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )
    print(f"wrote {OUT_DIR / 'method_scaling.html'}")


def fig_representation_scaling(df: pd.DataFrame) -> None:
    """How each representation method scales, all clustered with hierarchical."""
    s = df[df["node.func"] == "test_representation"].copy()
    s["runtime_s"] = s["s"]
    s["period"] = s["period_hours"].map(_PERIOD)
    reps_present = [r for r in REPRESENTATIONS if r in set(s["representation"])]
    fig = px.line(
        s.sort_values(["representation", "period_hours", "timesteps"]),
        x="timesteps",
        y="runtime_s",
        color="representation",
        facet_col="period",
        markers=True,
        log_x=True,
        log_y=True,
        category_orders={"representation": reps_present, "period": _PERIOD_ORDER},
        labels={
            "timesteps": "resolution of the 1-year input",
            "runtime_s": "runtime (s)",
        },
        title="Representation cost — hierarchical, 1 year, 4 columns, 12 clusters",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(**_RES_TICKS)
    fig.update_traces(marker={"size": 7}, line={"width": 2})
    fig.update_layout(
        template=TEMPLATE,
        height=440,
        margin={"l": 70, "r": 20, "t": 70, "b": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="representation",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUT_DIR / "representation_scaling.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )
    print(f"wrote {OUT_DIR / 'representation_scaling.html'}")


def main() -> None:
    df = _load(sys.argv[1] if len(sys.argv) > 1 else None)
    fig_method_scaling(df)
    fig_representation_scaling(df)


if __name__ == "__main__":
    main()
