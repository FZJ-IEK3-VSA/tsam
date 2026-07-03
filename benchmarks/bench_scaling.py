"""Realistic-usage scaling benchmark for :func:`tsam.aggregate`.

One parametrized test over a fixed **1-year** horizon, with the levers a user
actually dials as native pytest dimensions:

* ``resolution``  — 1h / 15min / 5min (drives timesteps *and* W together)
* ``period``      — daily (24h) or weekly (168h) typical periods
* ``columns``     — number of attribute columns (up to large multi-region models)
* ``clusters``    — ``n_clusters`` (typical periods)
* ``method``      — clustering method

Derived per case: ``timesteps = horizon / resolution``,
``timesteps_per_period = period / resolution``, ``periods = horizon / period``
(≈365 daily, ≈52 weekly). So finer resolution grows the input and the feature
width while keeping the period count — and hence the clustering problem —
realistic.

Every launched case is bounded up front, so the suite stays fast without any
runtime timeout machinery: a memory budget drops the biggest corners, and the
explosion-prone ``kmedoids`` / ``kmaxoids`` are only run where they stay tractable
(see ``_slow_feasible``). Each case is measured exactly once — we want the scaling
shape, not statistical rigor.

Select any slice from the CLI with ``-k``; the full cross-product is the default.

Examples::

    pytest benchmarks/bench_scaling.py --benchmark-only --benchmark-save=headline
    pytest benchmarks/bench_scaling.py -k "hierarchical and period_24h and clusters_12"
"""

from __future__ import annotations

import numpy as np
import pytest
from _data import (
    CLUSTERS,
    COLUMNS,
    MAX_INPUT_CELLS,
    METHODS,
    PERIOD_HOURS,
    RESOLUTIONS_MIN,
    SLOW_METHODS,
    derive,
    make_data,
)

import tsam
from tsam import ClusterConfig, Distribution, MinMaxMean

# One measured call per case — we want the scaling shape, not statistical rigor.
ROUNDS = 1

# Representation methods to compare (all clustered with hierarchical). The base
# columns are GHI, T, Wind, Load, so the min/max references below always exist.
REP_COLUMNS = 4
REP_CLUSTERS = 12
REPRESENTATIONS = [
    ("mean", "mean"),
    ("medoid", "medoid"),
    ("distribution", Distribution()),
    ("distribution_minmax", Distribution(preserve_minmax=True)),
    ("concurrency_consensus", Distribution(concurrency="consensus")),
    ("concurrency_assignment", Distribution(concurrency="assignment")),
    ("minmax_mean", MinMaxMean(max_columns=["Load"], min_columns=["T"])),
]


def _slow_feasible(
    method: str, columns: int, resolution_min: int, period_hours: int
) -> bool:
    """Whether an expensive method is cheap enough to benchmark at this point.

    ``kmedoids`` (exact MILP) and ``kmaxoids`` can run for minutes, so they are
    measured only at the base column width (the method figure uses 4 columns
    anyway) and — for ``kmedoids`` — only where the period count stays small: every
    weekly point, but daily only at hourly resolution. Everything else is skipped
    rather than launched, so no single case can blow up the suite.
    """
    if method not in SLOW_METHODS:
        return True
    if columns != REP_COLUMNS:
        return False
    return not (method == "kmedoids" and period_hours < 168 and resolution_min < 60)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("clusters", CLUSTERS, ids=lambda v: f"clusters_{v}")
@pytest.mark.parametrize("columns", COLUMNS, ids=lambda v: f"columns_{v}")
@pytest.mark.parametrize("period_hours", PERIOD_HOURS, ids=lambda v: f"period_{v}h")
@pytest.mark.parametrize("resolution_min", RESOLUTIONS_MIN, ids=lambda v: f"res_{v}min")
def test_scale(benchmark, resolution_min, period_hours, columns, clusters, method):
    resolution_hours, timesteps, steps_per_period, periods = derive(
        resolution_min, period_hours
    )
    if periods < clusters:
        pytest.skip(f"clusters={clusters} exceeds periods={periods}")
    if timesteps * columns > MAX_INPUT_CELLS:
        pytest.skip(f"input {timesteps}x{columns} exceeds memory budget")
    if not _slow_feasible(method, columns, resolution_min, period_hours):
        pytest.skip(f"{method} not benchmarked at this scale (would run for minutes)")

    data = make_data(timesteps, columns, resolution_hours=resolution_hours)
    benchmark.extra_info.update(
        {
            "method": method,
            "resolution_min": resolution_min,
            "period_hours": period_hours,
            "columns": columns,
            "clusters": clusters,
            "timesteps": timesteps,
            "timesteps_per_period": steps_per_period,
            "periods": periods,
            "feature_width": steps_per_period * columns,
        }
    )

    def run():
        np.random.seed(42)  # determinism for kmeans / kmaxoids
        result = tsam.aggregate(
            data,
            n_clusters=clusters,
            period_duration=period_hours,
            temporal_resolution=resolution_hours,
            cluster=ClusterConfig(method=method),
        )
        _ = result.reconstructed  # force reconstruction (computed lazily)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


@pytest.mark.parametrize("period_hours", PERIOD_HOURS, ids=lambda v: f"period_{v}h")
@pytest.mark.parametrize("resolution_min", RESOLUTIONS_MIN, ids=lambda v: f"res_{v}min")
@pytest.mark.parametrize(
    ("representation_key", "representation"),
    REPRESENTATIONS,
    ids=[k for k, _ in REPRESENTATIONS],
)
def test_representation(
    benchmark, representation_key, representation, resolution_min, period_hours
):
    """How the representation method costs scale, all clustered with hierarchical."""
    resolution_hours, timesteps, steps_per_period, periods = derive(
        resolution_min, period_hours
    )
    if periods < REP_CLUSTERS:
        pytest.skip(f"clusters={REP_CLUSTERS} exceeds periods={periods}")

    data = make_data(timesteps, REP_COLUMNS, resolution_hours=resolution_hours)
    benchmark.extra_info.update(
        {
            "representation": representation_key,
            "resolution_min": resolution_min,
            "period_hours": period_hours,
            "columns": REP_COLUMNS,
            "clusters": REP_CLUSTERS,
            "timesteps": timesteps,
            "periods": periods,
        }
    )

    def run():
        np.random.seed(42)
        result = tsam.aggregate(
            data,
            n_clusters=REP_CLUSTERS,
            period_duration=period_hours,
            temporal_resolution=resolution_hours,
            cluster=ClusterConfig(method="hierarchical", representation=representation),
        )
        _ = result.reconstructed

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
