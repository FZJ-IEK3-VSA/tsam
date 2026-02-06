"""Benchmarks for tsam using pytest-benchmark.

Self-contained: only imports ``tsam.timeseriesaggregation`` so it works with
both the current dev version and older releases (e.g. tsam==2.3.9).

Usage::

    # Benchmark an old version
    uv pip install tsam==2.3.9
    pytest benchmarks/bench.py --benchmark-save=v2.3.9

    # Switch back to dev and compare
    uv pip install -e .
    pytest benchmarks/bench.py --benchmark-compare='*v2.3.9'

    # Compare two saved snapshots
    pytest-benchmark compare '*v2.3.9' '*v3.0.0' --group-by=name
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add test/ to sys.path so we can import _configs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import tsam.timeseriesaggregation as tsam
from _configs import build_old_cases, case_ids, get_data

# ---------------------------------------------------------------------------
# Build parametrized cases from shared configs
# ---------------------------------------------------------------------------

OLD_CASES = build_old_cases()
_IDS = case_ids(OLD_CASES)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(case) -> None:
    data = get_data(case.dataset)
    if case.seed is not None:
        np.random.seed(case.seed)
    agg = tsam.TimeSeriesAggregation(timeSeries=data, **case.old_kwargs)
    agg.createTypicalPeriods()
    agg.predictOriginalData()


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", OLD_CASES, ids=_IDS)
def test_bench(case, benchmark):
    benchmark(lambda: _run(case))
