"""Benchmarks for tsam using pytest-benchmark.

Benchmarks the public :func:`tsam.aggregate` API across the shared golden
regression cases.

Usage::

    # Benchmark the current dev version
    uv pip install -e .
    pytest benchmarks/bench.py --benchmark-save=dev

    # Compare two saved snapshots
    pytest-benchmark compare '*v3.0.0' '*dev' --group-by=name
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add test/ to sys.path so we can import the shared case definitions.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

from _golden_cases import CASES, case_ids, get_data, run_new

_IDS = case_ids(CASES)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(case) -> None:
    data = get_data(case.dataset, max_timesteps=case.max_timesteps)
    result = run_new(data, case)
    _ = result.reconstructed  # force reconstruction


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_bench(case, benchmark):
    benchmark(lambda: _run(case))
