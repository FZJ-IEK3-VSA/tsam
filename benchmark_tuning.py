#!/usr/bin/env python
"""Benchmark script for tuning parallelization.

Compares:
- Sequential execution
- ProcessPoolExecutor with initializer (current implementation - pickle once per worker)
- ProcessPoolExecutor with partial (old approach - pickle per task)

Usage:
    uv run python benchmark_tuning.py
    uv run python benchmark_tuning.py --compare-old     # Also test old pickling approach
    uv run python benchmark_tuning.py --workers 4       # Specific worker count
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from tsam.api import aggregate
from tsam.config import ClusterConfig, SegmentConfig
from tsam.tuning import find_optimal_combination

# Globals for old-style benchmark (pickle per task)
_OLD_DATA: pd.DataFrame | None = None
_OLD_CLUSTER: ClusterConfig | None = None


def _test_single_config_old_style(
    args: tuple[int, int],
    data: pd.DataFrame,
    period_hours: int,
    resolution: float,
    cluster: ClusterConfig,
) -> tuple[int, int, float]:
    """Test function that receives data as argument (pickled per task)."""
    n_periods, n_segments = args
    try:
        result = aggregate(
            data,
            n_periods=n_periods,
            period_hours=period_hours,
            resolution=resolution,
            cluster=cluster,
            segments=SegmentConfig(n_segments=n_segments),
        )
        rmse = float(result.accuracy.rmse.mean())
        return (n_periods, n_segments, rmse)
    except Exception:
        return (n_periods, n_segments, float("inf"))


def benchmark_old_pickle_per_task(
    data: pd.DataFrame,
    configs: list[tuple[int, int]],
    n_workers: int,
) -> float:
    """Benchmark OLD approach: pickle data for every task."""
    cluster = ClusterConfig()
    resolution = 1.0
    period_hours = 24

    test_func = partial(
        _test_single_config_old_style,
        data=data,
        period_hours=period_hours,
        resolution=resolution,
        cluster=cluster,
    )

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(test_func, configs))
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tuning parallelization")
    parser.add_argument(
        "--compare-old",
        action="store_true",
        help="Also benchmark old pickle-per-task approach",
    )
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
    data_path = Path(__file__).parent / "docs/source/examples_notebooks/testdata.csv"
    if not data_path.exists():
        print(f"Error: Test data not found at {data_path}")
        print("Please ensure the examples_notebooks directory exists.")
        return

    print("Loading test data...")
    raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  Shape: {raw.shape}")
    print(f"  Columns: {list(raw.columns)}")
    print()

    import os

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
        f"  Optimal: {result_seq.optimal_n_periods} periods, "
        f"{result_seq.optimal_n_segments} segments"
    )
    print(f"  RMSE: {result_seq.optimal_rmse:.4f}")
    print(f"  Configs tested: {len(result_seq.history)}")
    print()

    # Benchmark parallel with initializer (current implementation)
    print("=" * 60)
    print(f"Benchmark: ProcessPoolExecutor + initializer (n_jobs={n_workers})")
    print("  (Current implementation - pickle once per worker)")
    print("=" * 60)
    start = time.perf_counter()
    find_optimal_combination(
        raw,
        data_reduction=args.reduction,
        n_jobs=args.workers,
        show_progress=True,
    )
    time_parallel = time.perf_counter() - start
    print(f"  Time: {time_parallel:.2f}s")
    print(f"  Speedup vs sequential: {time_sequential / time_parallel:.2f}x")
    print()

    # Optionally benchmark old pickle-per-task approach
    if args.compare_old:
        # Get the configs that were tested
        configs = [(h["n_periods"], h["n_segments"]) for h in result_seq.history]

        print("=" * 60)
        print(f"Benchmark: ProcessPoolExecutor + partial (n_jobs={n_workers})")
        print("  (Old approach - pickles DataFrame for EACH task)")
        print("=" * 60)
        time_old = benchmark_old_pickle_per_task(raw, configs, n_workers)
        print(f"  Time: {time_old:.2f}s")
        print(f"  Speedup vs sequential: {time_sequential / time_old:.2f}x")
        print()

        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(
            f"  Optimal: {result_seq.optimal_n_periods} periods, "
            f"{result_seq.optimal_n_segments} segments, "
            f"RMSE: {result_seq.optimal_rmse:.4f}"
        )
        print()
        print(f"  Sequential:              {time_sequential:6.2f}s  (baseline)")
        print(
            f"  Parallel + initializer:  {time_parallel:6.2f}s  "
            f"({time_sequential / time_parallel:.2f}x speedup)"
        )
        print(
            f"  Parallel + partial:      {time_old:6.2f}s  "
            f"({time_sequential / time_old:.2f}x speedup)"
        )
        print()
        if time_parallel < time_old:
            improvement = (time_old - time_parallel) / time_old * 100
            print(
                f"  Initializer approach is {improvement:.1f}% faster than "
                "pickle-per-task!"
            )
        else:
            print("  Old approach was faster (unexpected)")
    else:
        # Summary without old comparison
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(
            f"  Optimal: {result_seq.optimal_n_periods} periods, "
            f"{result_seq.optimal_n_segments} segments, "
            f"RMSE: {result_seq.optimal_rmse:.4f}"
        )
        print()
        print(f"  Sequential:              {time_sequential:6.2f}s  (baseline)")
        print(
            f"  Parallel + initializer:  {time_parallel:6.2f}s  "
            f"({time_sequential / time_parallel:.2f}x speedup)"
        )
        print()
        print(
            "Run with --compare-old to also benchmark the old pickle-per-task approach"
        )


if __name__ == "__main__":
    main()
