#!/usr/bin/env python
"""Build a standalone HTML performance report from pytest-benchmark runs.

Reads two (or more) benchmark JSON files, pairs them by benchmark id, and
writes one self-contained page: scaling curves on log-log axes plus a table
attributing the change per benchmark. The output has plotly inlined, so it
opens from disk with no network.

The dims the suite stamps (`n_columns`, `n_clusters`, `method`, ...) are what
make this possible: `pytest_benchmem.load_long_df` turns each run into a tidy
frame with one column per dim, so a chart is a groupby rather than a parser.

Usage::

    python benchmarks/plot_report.py before.json after.json -o report.html
    python benchmarks/plot_report.py before.json after.json --label develop --label stack
"""

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytest_benchmem import load_long_df

#: Validated categorical pair (CVD ΔE 24.7): orange = baseline, blue = candidate.
COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"]

#: Which scaling curves to draw: (test function, dim on x, human title).
#:
#: Ordered by the four axes that drive tsam's cost, so the report reads as one
#: sweep rather than a pile of curves: input width, input length, typical-period
#: count, and segments per period. The paths a change is expected to move follow.
CURVES = [
    ("test_scale_columns", "n_columns", "Baseline · width"),
    ("test_scale_timesteps", "n_timesteps", "Baseline · length"),
    ("test_scale_clusters", "n_clusters", "Baseline · typical periods"),
    ("test_scale_segments", "n_segments", "Baseline · segments per period"),
    ("test_segmentation_scale", "n_clusters", "Segmentation · typical periods"),
    ("test_accuracy", "n_columns", "Accuracy metrics · width"),
    ("test_representation_global_scale", "n_columns", "Global duration repr. · width"),
    ("test_disaggregate", "n_columns", "Disaggregate · width"),
]


def _frame(runs: list[Path], labels: list[str]):
    """Tidy long frame of medians, one row per (run, benchmark)."""
    df, unit = load_long_df(runs, metric="time", stat="median", labels=labels)
    df["func"] = df["id"].str.split("::").str[-1].str.split("[").str[0]
    df["ms"] = df["value"] * (1000 if unit == "s" else 1)
    return df


def _scaling_figure(df, labels: list[str]) -> go.Figure:
    """Small multiples: one panel per curve, log-log, one line per run."""
    present = [
        (f, x, t) for f, x, t in CURVES if f in set(df["func"]) and x in df.columns
    ]
    if not present:
        raise SystemExit("no scaling curves found — do the runs carry shape dims?")

    cols = 2
    rows = -(-len(present) // cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{t}<br><sub>vs {x}</sub>" for _, x, t in present],
        vertical_spacing=0.13,
        horizontal_spacing=0.09,
    )

    for i, (func, xdim, _title) in enumerate(present):
        r, c = divmod(i, cols)
        sub = df[(df["func"] == func) & df[xdim].notna()]
        for j, label in enumerate(labels):
            arm = (
                sub[sub["snapshot"] == label]
                .groupby(xdim, as_index=False)["ms"]
                .median()
            )
            if arm.empty:
                continue
            arm = arm.sort_values(xdim)
            fig.add_trace(
                go.Scatter(
                    x=arm[xdim],
                    y=arm["ms"],
                    name=label,
                    legendgroup=label,
                    showlegend=(i == 0),
                    mode="lines+markers",
                    line={"color": COLORS[j % len(COLORS)], "width": 2},
                    marker={"size": 8, "line": {"width": 2, "color": "white"}},
                    hovertemplate=f"{label}<br>{xdim} %{{x}}<br>%{{y:.3g}} ms<extra></extra>",
                ),
                row=r + 1,
                col=c + 1,
            )
        fig.update_xaxes(type="log", title_text=xdim, row=r + 1, col=c + 1)
        fig.update_yaxes(type="log", title_text="ms", row=r + 1, col=c + 1)

    fig.update_layout(
        height=300 * rows,
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        legend={"orientation": "h", "y": 1.06, "x": 0},
        hovermode="closest",
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 12},
    )
    fig.update_annotations(font_size=13)
    return fig


def _attribution_rows(df, labels: list[str]) -> list[tuple[str, float, float, float]]:
    """(benchmark, first-run ms, last-run ms, % change), worst change first."""
    wide = df.pivot_table(index="id", columns="snapshot", values="ms", aggfunc="median")
    first, last = labels[0], labels[-1]
    if first not in wide or last not in wide:
        return []
    wide = wide.dropna(subset=[first, last])
    rows = [
        (
            str(bench).split("::")[-1],
            float(row[first]),
            float(row[last]),
            100.0 * (row[last] - row[first]) / row[first],
        )
        for bench, row in wide.iterrows()
    ]
    return sorted(rows, key=lambda r: r[3])


def _table_html(rows, labels: list[str]) -> str:
    head = (
        "<tr><th>benchmark</th>"
        f"<th>{labels[0]}</th><th>{labels[-1]}</th><th>change</th></tr>"
    )
    body = "".join(
        f"<tr><td><code>{name}</code></td><td>{before:.3g} ms</td>"
        f"<td>{after:.3g} ms</td>"
        f'<td class="{"gain" if pct < -5 else "flat"}">'
        f"{f'{pct:.1f}%' if pct < -5 else 'no change'}</td></tr>"
        for name, before, after, pct in rows
    )
    return f"<table>{head}{body}</table>"


CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
     padding:28px 20px 56px;background:#f9f9f7;color:#0b0b0b;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.05rem;margin:32px 0 6px}
p{color:#52514e;font-size:.9rem;margin:0 0 10px}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin-top:8px;
      background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:8px}
th,td{padding:6px 10px;border-bottom:1px solid rgba(11,11,11,.08);
      text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
th{color:#52514e;font-weight:600}
.gain{color:#2a78d6;font-weight:600}.flat{color:#898781}
.scroll{overflow-x:auto}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", type=Path, help="pytest-benchmark JSON files, oldest first"
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("report.html"))
    parser.add_argument("--label", action="append", dest="labels", help="one per run")
    parser.add_argument("--title", default="tsam performance report")
    args = parser.parse_args()

    labels = args.labels or [p.stem for p in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit(f"{len(labels)} labels for {len(args.runs)} runs")

    df = _frame(args.runs, labels)
    fig = _scaling_figure(df, labels)
    rows = _attribution_rows(df, labels)

    chart = fig.to_html(
        full_html=False, include_plotlyjs="inline", config={"displaylogo": False}
    )
    moved = sum(1 for _n, _b, _a, pct in rows if pct < -5)

    args.out.write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{args.title}</title><style>{CSS}</style></head><body><div class='wrap'>"
        f"<h1>{args.title}</h1>"
        f"<p>{' → '.join(labels)} · {len(rows)} benchmarks compared, "
        f"{moved} moved by more than 5%. Log-log axes: a straight line is a power law "
        f"and its slope is the complexity exponent.</p>"
        f"<h2>Scaling</h2>{chart}"
        f"<h2>Attribution</h2>"
        f"<p>Rows that did not move are kept deliberately — they are the evidence "
        f"that nothing regressed.</p>"
        f"<div class='scroll'>{_table_html(rows, labels)}</div>"
        "</div></body></html>\n"
    )
    print(
        f"wrote {args.out} ({args.out.stat().st_size // 1024} KiB, {len(rows)} benchmarks)"
    )


if __name__ == "__main__":
    main()
