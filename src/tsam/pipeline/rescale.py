"""Rescaling of cluster periods to match original means."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tsam.pipeline.types import NormalizedData

MAX_ITERATOR = 20
TOLERANCE = 1e-6


def rescale_representatives(
    cluster_periods: list | np.ndarray,
    cluster_period_no_occur: dict[int, float],
    extreme_cluster_idx: list[int],
    profiles_df: pd.DataFrame,
    norm_data: NormalizedData,
    n_timesteps_per_period: int,
    exclude_columns: list[str],
) -> tuple[np.ndarray, dict]:
    """Rescale cluster periods so weighted mean matches original.

    Replicates _rescale_cluster_periods (monolith lines 930-1027).
    Must produce identical floating-point results.

    Returns (rescaled_periods, deviations_dict).
    """
    original_data = norm_data.original_data
    columns = list(original_data.columns)
    normalize_column_means = norm_data.normalize_column_means
    weights = norm_data.weights

    rescale_deviations: dict = {}

    weighting_vec = pd.Series(cluster_period_no_occur).values
    n_clusters = len(cluster_periods)
    n_cols = len(columns)
    n_timesteps = n_timesteps_per_period

    # Convert to 3D numpy array: (n_clusters, n_cols, n_timesteps)
    arr = np.array(cluster_periods).reshape(n_clusters, n_cols, n_timesteps)

    # Indices for non-extreme clusters
    idx_wo_peak = np.delete(np.arange(n_clusters), extreme_cluster_idx)
    extreme_cluster_idx_arr = np.array(extreme_cluster_idx, dtype=int)

    for ci, column in enumerate(columns):
        if column in exclude_columns:
            continue

        col_data = arr[:, ci, :]  # (n_clusters, n_timesteps)
        sum_raw = profiles_df[column].sum().sum()

        # Sum of extreme periods (weighted)
        if len(extreme_cluster_idx_arr) > 0:
            sum_peak = np.sum(
                weighting_vec[extreme_cluster_idx_arr]
                * col_data[extreme_cluster_idx_arr, :].sum(axis=1)
            )
        else:
            sum_peak = 0.0

        sum_clu_wo_peak = np.sum(
            weighting_vec[idx_wo_peak] * col_data[idx_wo_peak, :].sum(axis=1)
        )

        # Define the upper scale dependent on the weighting of the series
        scale_ub = 1.0
        if normalize_column_means:
            scale_ub = (
                scale_ub * original_data[column].max() / original_data[column].mean()
            )
        if weights and column in weights:
            scale_ub = scale_ub * weights[column]

        # Difference between predicted and original sum
        diff = abs(sum_raw - (sum_clu_wo_peak + sum_peak))

        a = 0
        while diff > sum_raw * TOLERANCE and a < MAX_ITERATOR:
            # Rescale values (only non-extreme clusters)
            arr[idx_wo_peak, ci, :] *= (sum_raw - sum_peak) / sum_clu_wo_peak

            # Reset values higher than the upper scale or less than zero
            arr[:, ci, :] = np.clip(arr[:, ci, :], 0, scale_ub)

            # Handle NaN (replace with 0)
            np.nan_to_num(arr[:, ci, :], copy=False, nan=0.0)

            # Calc new sum and new diff to orig data
            col_data = arr[:, ci, :]
            sum_clu_wo_peak = np.sum(
                weighting_vec[idx_wo_peak] * col_data[idx_wo_peak, :].sum(axis=1)
            )
            diff = abs(sum_raw - (sum_clu_wo_peak + sum_peak))
            a += 1

        # Calculate and store final deviation
        deviation_pct = (diff / sum_raw) * 100 if sum_raw != 0 else 0.0
        converged = a < MAX_ITERATOR
        rescale_deviations[column] = {
            "deviation_pct": deviation_pct,
            "converged": converged,
            "iterations": a,
        }

        if not converged and deviation_pct > 0.01:
            warnings.warn(
                f'Max iteration number reached for "{column}" while rescaling '
                f"the cluster periods. The integral of the aggregated time series "
                f"deviates by: {round(deviation_pct, 2)}%"
            )

    # Reshape back to 2D: (n_clusters, n_cols * n_timesteps)
    return arr.reshape(n_clusters, -1), rescale_deviations
