"""Hyperparameter tuning for tsam aggregation.

This module provides functions for finding optimal aggregation parameters.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tqdm

from tsam.api import aggregate
from tsam.config import ClusterConfig, SegmentConfig

if TYPE_CHECKING:
    from tsam.result import AggregationResult


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
    except Exception:
        return (n_periods, n_segments, float("inf"), None)


def _infer_resolution(data: pd.DataFrame) -> float:
    """Infer time resolution in hours from DataFrame datetime index."""
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

    # Infer resolution if not provided
    if resolution is None:
        resolution = _infer_resolution(data)

    n_timesteps = len(data)
    timesteps_per_period = int(period_hours / resolution)

    max_periods = n_timesteps // timesteps_per_period
    max_segments = timesteps_per_period

    # Find valid combinations on the Pareto frontier
    possible_segments = np.arange(1, max_segments + 1)
    possible_periods = np.arange(1, max_periods + 1)

    # Number of timesteps for all combinations
    combined_timesteps = np.outer(possible_segments, possible_periods)

    # Filter to valid combinations for target reduction
    valid_mask = combined_timesteps <= n_timesteps * data_reduction
    valid_timesteps = combined_timesteps * valid_mask

    # Find Pareto-optimal points (max timesteps for each segment count)
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

    # Build list of configurations to test
    configs_to_test = [
        (int(possible_periods[per_idx]), int(possible_segments[seg_idx]))
        for seg_idx, per_idx in zip(pareto_points[0], pareto_points[1])
    ]

    # Determine number of workers
    if n_jobs is None or n_jobs == 1:
        n_workers = 1
    elif n_jobs == -1:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = max(1, n_jobs)

    history: list[dict] = []
    all_results: list[AggregationResult] = []
    best_rmse = float("inf")
    best_result = None
    best_periods = 1
    best_segments = 1

    if n_workers == 1:
        # Sequential execution (original behavior)
        iterator = configs_to_test
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="Searching configurations")

        for n_periods, n_segments in iterator:
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
                history.append(
                    {
                        "n_periods": n_periods,
                        "n_segments": n_segments,
                        "rmse": rmse,
                    }
                )

                if save_all_results:
                    all_results.append(result)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_result = result
                    best_periods = n_periods
                    best_segments = n_segments

            except Exception:
                continue
    else:
        # Parallel execution: file-based approach
        # Data saved to temp file, workers load from disk - no DataFrame pickling
        import shutil

        temp_dir = tempfile.mkdtemp(prefix="tsam_tuning_")
        data_path = Path(temp_dir) / "data.csv"
        data.to_csv(data_path, sep=",", decimal=".")

        # Convert cluster to dict (only primitive types passed to workers)
        cluster_dict = asdict(cluster)

        full_configs = [
            (
                n_periods,
                n_segments,
                period_hours,
                resolution,
                str(data_path),
                cluster_dict,
            )
            for n_periods, n_segments in configs_to_test
        ]

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                if show_progress:
                    results_iter = tqdm.tqdm(
                        executor.map(_test_single_config_file, full_configs),
                        total=len(full_configs),
                        desc=f"Searching configurations ({n_workers} workers)",
                    )
                else:
                    results_iter = executor.map(_test_single_config_file, full_configs)

                for n_periods, n_segments, rmse, result in results_iter:
                    if result is not None:
                        history.append(
                            {
                                "n_periods": n_periods,
                                "n_segments": n_segments,
                                "rmse": rmse,
                            }
                        )

                        if save_all_results:
                            all_results.append(result)

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_result = result
                            best_periods = n_periods
                            best_segments = n_segments
        finally:
            # Clean up temp file
            shutil.rmtree(temp_dir, ignore_errors=True)

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
        up to full resolution.
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
    """
    if cluster is None:
        cluster = ClusterConfig()

    # Infer resolution if not provided
    if resolution is None:
        resolution = _infer_resolution(data)

    n_timesteps = len(data)
    timesteps_per_period = int(period_hours / resolution)

    max_periods = n_timesteps // timesteps_per_period
    max_segments = timesteps_per_period

    if max_timesteps is None:
        max_timesteps = n_timesteps

    # Determine number of workers
    if n_jobs is None or n_jobs == 1:
        n_workers = 1
    elif n_jobs == -1:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = max(1, n_jobs)

    pareto_results: list[TuningResult] = []
    n_periods = 1
    n_segments = 1

    # Set up file-based parallel execution if needed (no DataFrame pickling)
    temp_dir: str | None = None
    data_path: str | None = None
    cluster_dict: dict | None = None
    if n_workers > 1:
        import shutil

        temp_dir = tempfile.mkdtemp(prefix="tsam_pareto_")
        data_path = str(Path(temp_dir) / "data.csv")
        data.to_csv(data_path, sep=",", decimal=".")
        cluster_dict = asdict(cluster)

    if show_progress:
        pbar = tqdm.tqdm(total=max_timesteps, desc="Building Pareto front")

    def test_config(n_per: int, n_seg: int) -> tuple[float, AggregationResult | None]:
        try:
            result = aggregate(
                data,
                n_periods=n_per,
                period_hours=period_hours,
                resolution=resolution,
                cluster=cluster,
                segments=SegmentConfig(n_segments=n_seg),
            )
            return float(result.accuracy.rmse.mean()), result
        except Exception:
            return float("inf"), None

    def test_configs_parallel(
        configs: list[tuple[int, int]],
        data_path: str,
        cluster_dict: dict,
    ) -> list[tuple[int, int, float, AggregationResult | None]]:
        """Test multiple configurations in parallel (file-based, no pickling)."""
        full_configs = [
            (n_per, n_seg, period_hours, resolution, data_path, cluster_dict)
            for n_per, n_seg in configs
        ]
        with ProcessPoolExecutor(max_workers=min(n_workers, len(configs))) as executor:
            return list(executor.map(_test_single_config_file, full_configs))

    # Start with (1, 1)
    rmse, result = test_config(n_periods, n_segments)
    if result:
        pareto_results.append(
            TuningResult(
                optimal_n_periods=n_periods,
                optimal_n_segments=n_segments,
                optimal_rmse=rmse,
                history=[
                    {"n_periods": n_periods, "n_segments": n_segments, "rmse": rmse}
                ],
                best_result=result,
            )
        )

    # Steepest descent exploration
    while (
        n_periods < max_periods
        and n_segments < max_segments
        and (n_segments + 1) * n_periods <= max_timesteps
        and n_segments * (n_periods + 1) <= max_timesteps
    ):
        if n_workers > 1:
            # Test both directions in parallel
            assert data_path is not None and cluster_dict is not None
            configs = [
                (n_periods, n_segments + 1),  # Add segment
                (n_periods + 1, n_segments),  # Add period
            ]
            results = test_configs_parallel(configs, data_path, cluster_dict)
            _, _, rmse_seg, result_seg = results[0]
            _, _, rmse_per, result_per = results[1]
        else:
            # Sequential testing
            rmse_seg, result_seg = test_config(n_periods, n_segments + 1)
            rmse_per, result_per = test_config(n_periods + 1, n_segments)

        # Calculate gradients (RMSE improvement per timestep added)
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
                TuningResult(
                    optimal_n_periods=n_periods,
                    optimal_n_segments=n_segments,
                    optimal_rmse=rmse_per,
                    history=[
                        {
                            "n_periods": n_periods,
                            "n_segments": n_segments,
                            "rmse": rmse_per,
                        }
                    ],
                    best_result=result_per,
                )
            )
        elif result_seg:
            n_segments += 1
            pareto_results.append(
                TuningResult(
                    optimal_n_periods=n_periods,
                    optimal_n_segments=n_segments,
                    optimal_rmse=rmse_seg,
                    history=[
                        {
                            "n_periods": n_periods,
                            "n_segments": n_segments,
                            "rmse": rmse_seg,
                        }
                    ],
                    best_result=result_seg,
                )
            )
        else:
            break

        if show_progress:
            pbar.update(n_segments * n_periods - pbar.n)

    # Continue with periods only - can batch these for parallel execution
    remaining_periods = []
    while n_periods < max_periods and n_segments * (n_periods + 1) <= max_timesteps:
        n_periods += 1
        remaining_periods.append((n_periods, n_segments))

    if remaining_periods:
        if n_workers > 1 and len(remaining_periods) > 1:
            # Batch test remaining period configurations
            assert data_path is not None and cluster_dict is not None
            results = test_configs_parallel(remaining_periods, data_path, cluster_dict)
            for n_per, n_seg, rmse, result in results:
                if result:
                    pareto_results.append(
                        TuningResult(
                            optimal_n_periods=n_per,
                            optimal_n_segments=n_seg,
                            optimal_rmse=rmse,
                            history=[
                                {"n_periods": n_per, "n_segments": n_seg, "rmse": rmse}
                            ],
                            best_result=result,
                        )
                    )
                if show_progress:
                    pbar.update(n_seg * n_per - pbar.n)
        else:
            for n_per, n_seg in remaining_periods:
                rmse, result = test_config(n_per, n_seg)
                if result:
                    pareto_results.append(
                        TuningResult(
                            optimal_n_periods=n_per,
                            optimal_n_segments=n_seg,
                            optimal_rmse=rmse,
                            history=[
                                {"n_periods": n_per, "n_segments": n_seg, "rmse": rmse}
                            ],
                            best_result=result,
                        )
                    )
                if show_progress:
                    pbar.update(n_seg * n_per - pbar.n)

    # Continue with segments only - can batch these for parallel execution
    remaining_segments = []
    while n_segments < max_segments and (n_segments + 1) * n_periods <= max_timesteps:
        n_segments += 1
        remaining_segments.append((n_periods, n_segments))

    if remaining_segments:
        if n_workers > 1 and len(remaining_segments) > 1:
            # Batch test remaining segment configurations
            assert data_path is not None and cluster_dict is not None
            results = test_configs_parallel(remaining_segments, data_path, cluster_dict)
            for n_per, n_seg, rmse, result in results:
                if result:
                    pareto_results.append(
                        TuningResult(
                            optimal_n_periods=n_per,
                            optimal_n_segments=n_seg,
                            optimal_rmse=rmse,
                            history=[
                                {"n_periods": n_per, "n_segments": n_seg, "rmse": rmse}
                            ],
                            best_result=result,
                        )
                    )
                if show_progress:
                    pbar.update(n_seg * n_per - pbar.n)
        else:
            for n_per, n_seg in remaining_segments:
                rmse, result = test_config(n_per, n_seg)
                if result:
                    pareto_results.append(
                        TuningResult(
                            optimal_n_periods=n_per,
                            optimal_n_segments=n_seg,
                            optimal_rmse=rmse,
                            history=[
                                {"n_periods": n_per, "n_segments": n_seg, "rmse": rmse}
                            ],
                            best_result=result,
                        )
                    )
                if show_progress:
                    pbar.update(n_seg * n_per - pbar.n)

    if show_progress:
        pbar.close()

    # Clean up temp file if used
    if temp_dir is not None:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    return pareto_results
