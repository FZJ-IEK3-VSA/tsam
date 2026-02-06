#!/usr/bin/env python
"""Generate golden regression files using the OLD tsam API.

This script is meant to be run against a specific tsam version (e.g. v2.3.9)
to produce baseline reconstructed time series for regression testing.

Usage::

    # Run with tsam v2.3.9 in an ephemeral environment
    uv run --with tsam==2.3.9 --no-project test/generate_golden.py

    # Or with any installed tsam version
    python test/generate_golden.py
"""

from __future__ import annotations

import numpy as np

import tsam.timeseriesaggregation as tsam
from _configs import CONFIGS, GOLDEN_DIR, build_old_cases, get_data

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def main() -> None:
    print(
        f"tsam version: {tsam.__version__ if hasattr(tsam, '__version__') else 'unknown'}"
    )
    print(f"Output dir:   {GOLDEN_DIR}")
    print()

    cases = build_old_cases(CONFIGS)
    total = 0

    for case in cases:
        data = get_data(case.dataset)

        if case.seed is not None:
            np.random.seed(case.seed)

        agg = tsam.TimeSeriesAggregation(timeSeries=data, **case.old_kwargs)
        agg.createTypicalPeriods()
        reconstructed = agg.predictOriginalData()

        # Save
        path = GOLDEN_DIR / f"{case.id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        reconstructed.to_csv(path)

        total += 1
        print(f"  {case.id}  ({reconstructed.shape})")

    print(f"\nGenerated {total} golden files.")


if __name__ == "__main__":
    main()
