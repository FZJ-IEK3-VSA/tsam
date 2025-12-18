"""Hyperparameter tuning for tsam aggregation.

This module provides functions for finding optimal aggregation parameters.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tqdm

from tsam.api import aggregate
from tsam.config import ClusterConfig, SegmentConfig

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tsam.result import AggregationResult

logger = logging.getLogger(__name__)


def _test_single_config_file(
    args: tuple[int, int, int, float, str, dict],
) -> tuple[int, int, float, AggregationResult | None]:
    """Test a single configuration for parallel execution.

    Loads data from file - no DataFrame pickling.
    Args contains (n_periods, n_segments, period_hours, resolution, data_path, cluster_dict).

    Returns (n_periods, n_segments, rmse, result).
    """
    n_periods, n_segments, period_hours, resolution, data_path, cluster_dict = args
    try:
        # Load data fresh from file - no pickling
        data = pd.read_csv(
            data_path, index_col=0, parse_dates=True, sep=",", decimal="."
        )
        cluster = ClusterConfig(**cluster_dict)

        result = aggregate(
            data,
            n_periods=n_periods,
            period_hours=period_hours,
            resolution=resolution,
            cluster=cluster,
            segments=SegmentConfig(n_segments=n_segments),
        )
        rmse = float(result.accuracy.rmse.mean())
        return (n_periods, n_segments, rmse, result)
    except Exception as e:
        logger.debug("Config (%d, %d) failed: %s", n_periods, n_segments, e)
        return (n_periods, n_segments, float("inf"), None)


def _infer_resolution(data: pd.DataFrame) -> float:
    """Infer time resolution in hours from DataFrame datetime index."""
    if len(data) < 2:
        return 1.0  # Default to hourly
    try:
        timedelta = data.index[1] - data.index[0]
        return float(timedelta.total_seconds()) / 3600
    except (AttributeError, TypeError):
        # Try converting to datetime
        try:
            index = pd.to_datetime(data.index)
            timedelta = index[1] - index[0]
            return float(timedelta.total_seconds()) / 3600
        except Exception:
            # Default to hourly if can't infer
            return 1.0


@contextmanager
def _parallel_context(
    data: pd.DataFrame,
    cluster: ClusterConfig,
    prefix: str = "tsam_",
) -> Iterator[tuple[str, dict]]:
    """Context manager for parallel execution setup.

    Saves data to temp file and yields (data_path, cluster_dict).
    Cleans up temp files on exit.
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    data_path = str(Path(temp_dir) / "data.csv")
    data.to_csv(data_path, sep=",", decimal=".")
    cluster_dict = asdict(cluster)
    try:
        yield data_path, cluster_dict
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _test_configs(
    configs: list[tuple[int, int]],
    data: pd.DataFrame,
    period_hours: int,
    resolution: float,
    cluster: ClusterConfig,
    n_workers: int,
    show_progress: bool = False,
    progress_desc: str = "Testing configurations",
) -> list[tuple[int, int, float, AggregationResult | None]]:
    """Test a batch of configurations, either sequentially or in parallel.

    Args:
        configs: List of (n_periods, n_segments) tuples to test.
        data: Input time series data.
        period_hours: Hours per period.
        resolution: Time resolution in hours.
        cluster: Clustering configuration.
        n_workers: Number of parallel workers (1 for sequential).
        show_progress: Whether to show progress bar.
        progress_desc: Description for progress bar.

    Returns:
        List of (n_periods, n_segments, rmse, result) tuples.
    """
    if not configs:
        return []

    results: list[tuple[int, int, float, AggregationResult | None]] = []

    if n_workers > 1:
        with _parallel_context(data, cluster) as (data_path, cluster_dict):
            full_configs = [
                (n_per, n_seg, period_hours, resolution, data_path, cluster_dict)
                for n_per, n_seg in configs
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                if show_progress:
                    results_iter = tqdm.tqdm(
                        executor.map(_test_single_config_file, full_configs),
                        total=len(full_configs),
                        desc=f"{progress_desc} ({n_workers} workers)",
                    )
                else:
                    results_iter = executor.map(_test_single_config_file, full_configs)
                results = list(results_iter)
    else:
        iterator: list[tuple[int, int]] | tqdm.tqdm[tuple[int, int]] = configs
        if show_progress:
            iterator = tqdm.tqdm(configs, desc=progress_desc)

        for n_per, n_seg in iterator:
            try:
                result = aggregate(
                    data,
                    n_periods=n_per,
                    period_hours=period_hours,
                    resolution=resolution,
                    cluster=cluster,
                    segments=SegmentConfig(n_segments=n_seg),
                )
                rmse = float(result.accuracy.rmse.mean())
                results.append((n_per, n_seg, rmse, result))
            except Exception as e:
                logger.debug("Config (%d, %d) failed: %s", n_per, n_seg, e)
                results.append((n_per, n_seg, float("inf"), None))

    return results


def _make_tuning_result(
    n_periods: int,
    n_segments: int,
    rmse: float,
    result: AggregationResult,
) -> TuningResult:
    """Create a TuningResult from aggregation output."""
    return TuningResult(
        optimal_n_periods=n_periods,
        optimal_n_segments=n_segments,
        optimal_rmse=rmse,
        history=[{"n_periods": n_periods, "n_segments": n_segments, "rmse": rmse}],
        best_result=result,
    )


def _get_n_workers(n_jobs: int | None) -> int:
    """Convert n_jobs parameter to actual worker count.

    Follows joblib convention for negative values:
    - n_jobs=None or 1: single worker (no parallelization)
    - n_jobs=-1: all CPUs
    - n_jobs=-2: all CPUs minus 1
    - n_jobs=-N: all CPUs minus (N-1)
    - n_jobs>1: exactly that many workers
    """
    if n_jobs is None or n_jobs == 1:
        return 1
    elif n_jobs < 0:
        # Negative values: all CPUs + n_jobs + 1 (e.g., -1 = all, -2 = all-1)
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count + n_jobs + 1)
    else:
        return max(1, n_jobs)


@dataclass
class TuningResult:
    """Result of hyperparameter tuning.

    Attributes
    ----------
    optimal_n_periods : int
        Optimal number of typical periods.
    optimal_n_segments : int
        Optimal number of segments per period.
    optimal_rmse : float
        RMSE of the optimal configuration.
    history : list[dict]
        History of all tested configurations with their RMSE values.
    best_result : AggregationResult
        The AggregationResult for the optimal configuration.
    all_results : list[AggregationResult]
        All AggregationResults from tuning (only populated if save_all_results=True).
    """

    optimal_n_periods: int
    optimal_n_segments: int
    optimal_rmse: float
    history: list[dict]
    best_result: AggregationResult
    all_results: list[AggregationResult] = field(default_factory=list)


def periods_for_reduction(
    n_timesteps: int,
    n_segments: int,
    data_reduction: float,
) -> int:
    """Calculate max periods for a target data reduction.

    Parameters
    ----------
    n_timesteps : int
        Number of original timesteps.
    n_segments : int
        Number of segments per period.
    data_reduction : float
        Target reduction factor (e.g., 0.1 for 10% of original size).

    Returns
    -------
    int
        Maximum number of periods that achieves the reduction.

    Examples
    --------
    >>> periods_for_reduction(8760, 24, 0.01)  # 1% of hourly year
    3
    """
    return int(np.floor(data_reduction * float(n_timesteps) / n_segments))


def segments_for_reduction(
    n_timesteps: int,
    n_periods: int,
    data_reduction: float,
) -> int:
    """Calculate max segments for a target data reduction.

    Parameters
    ----------
    n_timesteps : int
        Number of original timesteps.
    n_periods : int
        Number of typical periods.
    data_reduction : float
        Target reduction factor (e.g., 0.1 for 10% of original size).

    Returns
    -------
    int
        Maximum number of segments that achieves the reduction.

    Examples
    --------
    >>> segments_for_reduction(8760, 8, 0.01)  # 1% with 8 periods
    10
    """
    return int(np.floor(data_reduction * float(n_timesteps) / n_periods))


def find_optimal_combination(
    data: pd.DataFrame,
    data_reduction: float,
    *,
    period_hours: int = 24,
    resolution: float | None = None,
    cluster: ClusterConfig | None = None,
    show_progress: bool = True,
    save_all_results: bool = False,
    n_jobs: int | None = None,
) -> TuningResult:
    """Find optimal period/segment combination for a target data reduction.

    Searches the Pareto-optimal frontier of period/segment combinations
    that achieve the specified data reduction, returning the one with
    minimum RMSE.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data.
    data_reduction : float
        Target reduction factor (e.g., 0.01 for 1% of original size).
    period_hours : int, default 24
        Hours per period.
    resolution : float, optional
        Time resolution of input data in hours.
        If not provided, inferred from the datetime index.
        Examples: 1.0 (hourly), 0.25 (15-minute), 0.5 (30-minute)
    cluster : ClusterConfig, optional
        Clustering configuration.
    show_progress : bool, default True
        Show progress bar during search.
    save_all_results : bool, default False
        If True, save all AggregationResults in all_results attribute.
        Useful for detailed analysis but increases memory usage.
    n_jobs : int, optional
        Number of parallel jobs. If None or 1, runs sequentially.
        Use -1 for all available CPUs, or a positive integer for
        a specific number of workers. Parallel execution uses a file-based
        approach where data is saved to a temp file and workers load from
        disk - no DataFrame pickling, safe for sensitive data.

    Returns
    -------
    TuningResult
        Result containing optimal parameters and history.

    Examples
    --------
    >>> result = find_optimal_combination(df, data_reduction=0.01)
    >>> print(f"Optimal: {result.optimal_n_periods} periods, "
    ...       f"{result.optimal_n_segments} segments")

    >>> # Use all CPUs for faster search (file-based, no DataFrame pickling)
    >>> result = find_optimal_combination(df, data_reduction=0.01, n_jobs=-1)
    """
    if cluster is None:
        cluster = ClusterConfig()

    if resolution is None:
        resolution = _infer_resolution(data)

    n_timesteps = len(data)
    timesteps_per_period = int(period_hours / resolution)

    max_periods = n_timesteps // timesteps_per_period
    max_segments = timesteps_per_period

    # Find valid combinations on the Pareto frontier
    possible_segments = np.arange(1, max_segments + 1)
    possible_periods = np.arange(1, max_periods + 1)

    combined_timesteps = np.outer(possible_segments, possible_periods)
    valid_mask = combined_timesteps <= n_timesteps * data_reduction
    valid_timesteps = combined_timesteps * valid_mask

    optimal_periods_idx = np.zeros_like(valid_timesteps, dtype=bool)
    optimal_periods_idx[
        np.arange(valid_timesteps.shape[0]),
        valid_timesteps.argmax(axis=1),
    ] = True

    optimal_segments_idx = np.zeros_like(valid_timesteps, dtype=bool)
    optimal_segments_idx[
        valid_timesteps.argmax(axis=0),
        np.arange(valid_timesteps.shape[1]),
    ] = True

    pareto_mask = optimal_periods_idx & optimal_segments_idx
    pareto_points = np.nonzero(pareto_mask)

    configs_to_test = [
        (int(possible_periods[per_idx]), int(possible_segments[seg_idx]))
        for seg_idx, per_idx in zip(pareto_points[0], pareto_points[1])
    ]

    n_workers = _get_n_workers(n_jobs)
    results = _test_configs(
        configs_to_test,
        data,
        period_hours,
        resolution,
        cluster,
        n_workers,
        show_progress=show_progress,
        progress_desc="Searching configurations",
    )

    history: list[dict] = []
    all_results: list[AggregationResult] = []
    best_rmse = float("inf")
    best_result = None
    best_periods = 1
    best_segments = 1

    for n_periods, n_segments, rmse, result in results:
        if result is not None:
            history.append(
                {"n_periods": n_periods, "n_segments": n_segments, "rmse": rmse}
            )
            if save_all_results:
                all_results.append(result)
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = result
                best_periods = n_periods
                best_segments = n_segments

    if best_result is None:
        raise ValueError("No valid configuration found")

    return TuningResult(
        optimal_n_periods=best_periods,
        optimal_n_segments=best_segments,
        optimal_rmse=best_rmse,
        history=history,
        best_result=best_result,
        all_results=all_results,
    )


def find_pareto_front(
    data: pd.DataFrame,
    *,
    period_hours: int = 24,
    resolution: float | None = None,
    max_timesteps: int | None = None,
    timesteps: Sequence[int] | None = None,
    cluster: ClusterConfig | None = None,
    show_progress: bool = True,
    n_jobs: int | None = None,
) -> list[TuningResult]:
    """Find all Pareto-optimal aggregations from 1 period to full resolution.

    Uses a steepest-descent approach to efficiently explore the
    period/segment space, finding configurations that are optimal
    for their complexity level.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data.
    period_hours : int, default 24
        Hours per period.
    resolution : float, optional
        Time resolution of input data in hours.
        If not provided, inferred from the datetime index.
        Examples: 1.0 (hourly), 0.25 (15-minute), 0.5 (30-minute)
    max_timesteps : int, optional
        Stop when reaching this many timesteps. If None, explores
        up to full resolution. Ignored if `timesteps` is provided.
    timesteps : Sequence[int], optional
        Specific timestep counts to explore. If provided, only evaluates
        configurations that produce approximately these timestep counts.
        Useful for faster exploration with large steps or specific ranges.
        Examples: range(10, 500, 10), [10, 50, 100, 200, 500]
    cluster : ClusterConfig, optional
        Clustering configuration.
    show_progress : bool, default True
        Show progress bar.
    n_jobs : int, optional
        Number of parallel jobs for testing configurations.
        If None or 1, runs sequentially. Use -1 for all available CPUs.
        During steepest-descent phase, tests both directions in parallel.

    Returns
    -------
    list[TuningResult]
        List of Pareto-optimal configurations, ordered by increasing
        complexity (timesteps).

    Examples
    --------
    >>> pareto = find_pareto_front(df, max_timesteps=500)
    >>> for result in pareto:
    ...     print(f"{result.optimal_n_periods}x{result.optimal_n_segments}: "
    ...           f"RMSE={result.optimal_rmse:.4f}")

    >>> # Use parallel execution for faster search
    >>> pareto = find_pareto_front(df, max_timesteps=500, n_jobs=-1)

    >>> # Explore only specific timestep counts (faster)
    >>> pareto = find_pareto_front(df, timesteps=range(10, 500, 50))

    >>> # Explore a specific list of timestep targets
    >>> pareto = find_pareto_front(df, timesteps=[10, 50, 100, 200, 500])
    """
    if cluster is None:
        cluster = ClusterConfig()

    if resolution is None:
        resolution = _infer_resolution(data)

    n_timesteps = len(data)
    timesteps_per_period = int(period_hours / resolution)

    max_periods = n_timesteps // timesteps_per_period
    max_segments = timesteps_per_period

    if max_timesteps is None:
        max_timesteps = n_timesteps

    n_workers = _get_n_workers(n_jobs)

    # If specific timesteps are provided, use targeted exploration
    if timesteps is not None:
        return _find_pareto_front_targeted(
            data=data,
            timesteps=timesteps,
            period_hours=period_hours,
            resolution=resolution,
            max_periods=max_periods,
            max_segments=max_segments,
            cluster=cluster,
            show_progress=show_progress,
            n_workers=n_workers,
        )

    # Steepest descent exploration
    return _find_pareto_front_steepest(
        data=data,
        period_hours=period_hours,
        resolution=resolution,
        max_periods=max_periods,
        max_segments=max_segments,
        max_timesteps=max_timesteps,
        cluster=cluster,
        show_progress=show_progress,
        n_workers=n_workers,
    )


def _find_pareto_front_targeted(
    data: pd.DataFrame,
    timesteps: Sequence[int],
    period_hours: int,
    resolution: float,
    max_periods: int,
    max_segments: int,
    cluster: ClusterConfig,
    show_progress: bool,
    n_workers: int,
) -> list[TuningResult]:
    """Find Pareto front for specific target timestep counts."""
    # Build all configurations to test
    configs_with_target: list[tuple[int, int, int]] = []  # (target, n_per, n_seg)

    for target in sorted(set(timesteps)):
        if target < 1:
            continue
        for n_seg in range(1, min(target, max_segments) + 1):
            if target % n_seg == 0:
                n_per = target // n_seg
                if 1 <= n_per <= max_periods:
                    configs_with_target.append((target, n_per, n_seg))

    if not configs_with_target:
        return []

    # Test all configurations
    configs = [(n_per, n_seg) for _, n_per, n_seg in configs_with_target]
    results = _test_configs(
        configs,
        data,
        period_hours,
        resolution,
        cluster,
        n_workers,
        show_progress=show_progress,
        progress_desc="Testing configurations",
    )

    # Group results by target timestep
    results_by_target: dict[
        int, list[tuple[int, int, float, AggregationResult | None]]
    ] = {}
    for (target, _, _), result in zip(configs_with_target, results):
        if target not in results_by_target:
            results_by_target[target] = []
        results_by_target[target].append(result)

    # For each target, pick the best configuration (lowest RMSE)
    pareto_results: list[TuningResult] = []
    for target in sorted(results_by_target.keys()):
        best_rmse = float("inf")
        best_result: AggregationResult | None = None
        best_n_per = 0
        best_n_seg = 0

        for n_per, n_seg, rmse, agg_result in results_by_target[target]:
            if agg_result is not None and rmse < best_rmse:
                best_rmse = rmse
                best_result = agg_result
                best_n_per = n_per
                best_n_seg = n_seg

        if best_result is not None:
            pareto_results.append(
                _make_tuning_result(best_n_per, best_n_seg, best_rmse, best_result)
            )

    return pareto_results


def _find_pareto_front_steepest(
    data: pd.DataFrame,
    period_hours: int,
    resolution: float,
    max_periods: int,
    max_segments: int,
    max_timesteps: int,
    cluster: ClusterConfig,
    show_progress: bool,
    n_workers: int,
) -> list[TuningResult]:
    """Find Pareto front using steepest descent exploration."""
    pareto_results: list[TuningResult] = []
    n_periods = 1
    n_segments = 1

    pbar = None
    if show_progress:
        pbar = tqdm.tqdm(total=max_timesteps, desc="Building Pareto front")

    def update_progress() -> None:
        if pbar is not None:
            pbar.update(n_segments * n_periods - pbar.n)

    # Start with (1, 1)
    results = _test_configs(
        [(n_periods, n_segments)],
        data,
        period_hours,
        resolution,
        cluster,
        n_workers=1,  # Single config, no parallelization needed
    )
    if results:
        _, _, rmse, agg_result = results[0]
        if agg_result is not None:
            pareto_results.append(
                _make_tuning_result(n_periods, n_segments, rmse, agg_result)
            )

    # Steepest descent phase
    while (
        n_periods < max_periods
        and n_segments < max_segments
        and (n_segments + 1) * n_periods <= max_timesteps
        and n_segments * (n_periods + 1) <= max_timesteps
    ):
        # Test both directions
        candidates = [
            (n_periods, n_segments + 1),
            (n_periods + 1, n_segments),
        ]
        results = _test_configs(
            candidates,
            data,
            period_hours,
            resolution,
            cluster,
            n_workers=min(n_workers, 2),
        )
        _, _, rmse_seg, result_seg = results[0]
        _, _, rmse_per, result_per = results[1]

        # Calculate gradients
        current_rmse = (
            pareto_results[-1].optimal_rmse if pareto_results else float("inf")
        )
        gradient_seg = (
            (current_rmse - rmse_seg) / n_periods if rmse_seg < float("inf") else 0
        )
        gradient_per = (
            (current_rmse - rmse_per) / n_segments if rmse_per < float("inf") else 0
        )

        # Follow steeper gradient
        if gradient_per > gradient_seg and result_per:
            n_periods += 1
            pareto_results.append(
                _make_tuning_result(n_periods, n_segments, rmse_per, result_per)
            )
        elif result_seg:
            n_segments += 1
            pareto_results.append(
                _make_tuning_result(n_periods, n_segments, rmse_seg, result_seg)
            )
        else:
            break

        update_progress()

    # Continue with periods only
    remaining_periods = []
    while n_periods < max_periods and n_segments * (n_periods + 1) <= max_timesteps:
        n_periods += 1
        remaining_periods.append((n_periods, n_segments))

    if remaining_periods:
        results = _test_configs(
            remaining_periods,
            data,
            period_hours,
            resolution,
            cluster,
            n_workers,
        )
        for n_per, n_seg, rmse, result in results:
            if result is not None:
                pareto_results.append(_make_tuning_result(n_per, n_seg, rmse, result))
            if pbar is not None:
                pbar.update(n_seg * n_per - pbar.n)

    # Continue with segments only
    remaining_segments = []
    while n_segments < max_segments and (n_segments + 1) * n_periods <= max_timesteps:
        n_segments += 1
        remaining_segments.append((n_periods, n_segments))

    if remaining_segments:
        results = _test_configs(
            remaining_segments,
            data,
            period_hours,
            resolution,
            cluster,
            n_workers,
        )
        for n_per, n_seg, rmse, result in results:
            if result is not None:
                pareto_results.append(_make_tuning_result(n_per, n_seg, rmse, result))
            if pbar is not None:
                pbar.update(n_seg * n_per - pbar.n)

    if pbar is not None:
        pbar.close()

    return pareto_results
