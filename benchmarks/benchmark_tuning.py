#!/usr/bin/env python
"""Benchmark script for tuning parallelization.

Compares sequential vs parallel (file-based) execution.
The file-based approach saves data to a temp file and workers load from disk,
avoiding any DataFrame pickling - safe for sensitive/corporate data.

Usage:
    uv run python benchmark_tuning.py
    uv run python benchmark_tuning.py --workers 4       # Specific worker count
    uv run python benchmark_tuning.py --reduction 0.05  # Different reduction target
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import pandas as pd

from tsam.tuning import find_optimal_combination


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tuning parallelization")
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of workers (-1 for all CPUs)",
    )
    parser.add_argument(
        "--reduction",
        type=float,
        default=0.02,
        help="Data reduction target (default: 0.02 = 2%%)",
    )
    args = parser.parse_args()

    # Load test data
    data_path = (
        Path(__file__).parent.parent / "docs/source/examples_notebooks/testdata.csv"
    )
    if not data_path.exists():
        print(f"Error: Test data not found at {data_path}")
        print("Please ensure the examples_notebooks directory exists.")
        return

    print("Loading test data...")
    raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  Shape: {raw.shape}")
    print(f"  Columns: {list(raw.columns)}")
    print()

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # Benchmark sequential
    print("=" * 60)
    print("Benchmark: Sequential (n_jobs=1)")
    print("=" * 60)
    start = time.perf_counter()
    result_seq = find_optimal_combination(
        raw,
        data_reduction=args.reduction,
        n_jobs=1,
        show_progress=True,
    )
    time_sequential = time.perf_counter() - start
    print(f"  Time: {time_sequential:.2f}s")
    print(
        f"  Optimal: {result_seq.n_clusters} periods, {result_seq.n_segments} segments"
    )
    print(f"  RMSE: {result_seq.rmse:.4f}")
    print(f"  Configs tested: {len(result_seq.history)}")
    print()

    # Benchmark parallel (file-based)
    print("=" * 60)
    print(f"Benchmark: Parallel file-based (n_jobs={n_workers})")
    print("  Data saved to temp file, workers load from disk")
    print("  No DataFrame pickling - safe for sensitive data")
    print("=" * 60)
    start = time.perf_counter()
    result_par = find_optimal_combination(
        raw,
        data_reduction=args.reduction,
        n_jobs=args.workers,
        show_progress=True,
    )
    time_parallel = time.perf_counter() - start
    print(f"  Time: {time_parallel:.2f}s")
    print(f"  Speedup vs sequential: {time_sequential / time_parallel:.2f}x")
    print()

    # Validation
    assert math.isclose(result_par.rmse, result_seq.rmse, rel_tol=1e-6), (
        "Parallel and sequential results differ (RMSE mismatch)"
    )

    assert result_par.n_clusters == result_seq.n_clusters, (
        "Parallel and sequential results differ (n_clusters mismatch)"
    )

    assert result_par.n_segments == result_seq.n_segments, (
        "Parallel and sequential results differ (n_segments mismatch)"
    )

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"  Optimal: {result_seq.n_clusters} periods, "
        f"{result_seq.n_segments} segments, "
        f"RMSE: {result_seq.rmse:.4f}"
    )
    print()
    print(f"  Sequential:     {time_sequential:6.2f}s  (baseline)")
    print(
        f"  Parallel:       {time_parallel:6.2f}s  "
        f"({time_sequential / time_parallel:.2f}x speedup)"
    )
    print()
    print("  Security: No DataFrame pickling - only file paths passed to workers")


if __name__ == "__main__":
    main()
