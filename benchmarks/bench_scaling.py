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

Select any slice from the CLI with ``-k``; the full cross-product is the default.
To keep the suite fast, any case slower than ``CASE_TIMEOUT_S`` is dropped: cheap
methods are measured once and, if over budget, their wider-column variants are
skipped; the hang-prone ``kmedoids`` / ``kmaxoids`` are probed in a subprocess and
hard-killed. Cases over the memory budget or with ``clusters > periods`` are also
skipped.

Examples::

    pytest benchmarks/bench_scaling.py --benchmark-only --benchmark-save=headline
    pytest benchmarks/bench_scaling.py -k "hierarchical and period_24h and clusters_12"
"""

from __future__ import annotations

import time

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
from _timeout import completes_within

import tsam
from tsam import ClusterConfig, Distribution, MinMaxMean

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

# Cases slower than this are dropped so the suite stays fast and bounded. Cheap
# methods are measured once and, if over budget, their wider-column variants are
# skipped. The hang-prone MILP/heuristic methods are probed in a subprocess and
# hard-killed — with a longer budget, since they are only usable on small period
# counts and we still want a few data points to show how they scale.
CASE_TIMEOUT_S = 3.0
SLOW_TIMEOUT_S = 15.0
# One measured call per case — we want the scaling shape, not statistical rigor.
ROUNDS = 1
# (method, resolution_min, period_hours) -> smallest columns count that timed out
_timed_out: dict[tuple[str, int, int], int] = {}


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

    key = (method, resolution_min, period_hours)
    if key in _timed_out and columns >= _timed_out[key]:
        pytest.skip(f"{method}: wider inputs skipped after an earlier timeout")

    # Hang-prone methods: hard-kill in a subprocess before we ever run them here.
    if method in SLOW_METHODS and not completes_within(
        SLOW_TIMEOUT_S,
        timesteps,
        columns,
        method,
        period_hours,
        resolution_hours,
        clusters,
    ):
        _timed_out[key] = min(columns, _timed_out.get(key, columns))
        pytest.skip(f"{method} exceeded {SLOW_TIMEOUT_S:.0f}s at columns={columns}")

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

    elapsed = 0.0

    def run():
        nonlocal elapsed
        np.random.seed(42)  # determinism for kmeans / kmaxoids
        start = time.perf_counter()
        result = tsam.aggregate(
            data,
            n_clusters=clusters,
            period_duration=period_hours,
            temporal_resolution=resolution_hours,
            cluster=ClusterConfig(method=method),
        )
        _ = result.reconstructed  # force reconstruction (computed lazily)
        elapsed = time.perf_counter() - start

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)

    # Cheap methods aren't probed up-front; if this one blew the budget, skip its
    # wider-column variants (which are strictly heavier).
    if method not in SLOW_METHODS and elapsed > CASE_TIMEOUT_S:
        _timed_out[key] = min(columns, _timed_out.get(key, columns))


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
