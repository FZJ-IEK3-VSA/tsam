"""Hyperparameter tuning for tsam aggregation.

This module provides functions for finding optimal aggregation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tqdm

from tsam.api import aggregate
from tsam.config import ClusterConfig, SegmentConfig

if TYPE_CHECKING:
    from tsam.result import AggregationResult


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

    Returns
    -------
    TuningResult
        Result containing optimal parameters and history.

    Examples
    --------
    >>> result = find_optimal_combination(df, data_reduction=0.01)
    >>> print(f"Optimal: {result.optimal_n_periods} periods, "
    ...       f"{result.optimal_n_segments} segments")
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

    history: list[dict] = []
    all_results: list[AggregationResult] = []
    best_rmse = float("inf")
    best_result = None
    best_periods = 1
    best_segments = 1

    iterator = zip(pareto_points[0], pareto_points[1])
    if show_progress:
        iterator = tqdm.tqdm(list(iterator), desc="Searching configurations")

    for seg_idx, per_idx in iterator:
        n_segments = int(possible_segments[seg_idx])
        n_periods = int(possible_periods[per_idx])

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

    pareto_results = []
    n_periods = 1
    n_segments = 1

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
        # Test adding a segment
        rmse_seg, result_seg = test_config(n_periods, n_segments + 1)
        # Test adding a period
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

    # Continue with periods only
    while n_periods < max_periods and n_segments * (n_periods + 1) <= max_timesteps:
        n_periods += 1
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
        if show_progress:
            pbar.update(n_segments * n_periods - pbar.n)

    # Continue with segments only
    while n_segments < max_segments and (n_segments + 1) * n_periods <= max_timesteps:
        n_segments += 1
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
        if show_progress:
            pbar.update(n_segments * n_periods - pbar.n)

    if show_progress:
        pbar.close()

    return pareto_results
