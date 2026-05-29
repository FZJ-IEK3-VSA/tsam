"""Rescaling of cluster periods to match original means."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tsam.options import options


def rescale_representatives(
    cluster_periods: list | np.ndarray,
    cluster_period_no_occur: dict[int, float],
    extreme_cluster_idx: list[int],
    profiles_df: pd.DataFrame,
    original_data: pd.DataFrame,
    normalize_column_means: bool,
    n_timesteps_per_period: int,
    exclude_columns: list[str],
) -> tuple[np.ndarray, dict]:
    """Correct cluster representatives so their weighted mean matches the input.

    Optional stage, enabled by ``preserve_column_means`` (the
    ``rescale_cluster_periods`` config flag).

    **Problem.** Clustering can shift column means. Aggregating 365 daily load
    profiles into 8 typical days may leave the weighted average of the 8
    representatives not matching the original annual average.

    **Solution.** Iteratively scale each column of each non-extreme cluster
    center until its occurrence-weighted sum matches the original total within
    tolerance. Values are clipped to ``[0, scale_ub]``, where ``scale_ub``
    depends on ``normalize_column_means`` (ratio of max to mean). Because the
    data is unweighted at this point, no weight compensation is needed for the
    clip bound.

    Extreme clusters (from `add_extreme_periods`) are excluded so their extreme
    values are preserved. Columns in ``exclude_columns`` are also skipped —
    useful for binary 0/1 columns that should not be scaled.

    Parameters
    ----------
    cluster_periods
        Unweighted cluster representatives to rescale.
    cluster_period_no_occur
        Occurrence count per cluster (the rescaling weights).
    extreme_cluster_idx
        Indices of extreme clusters to leave untouched.
    profiles_df
        Normalized period profiles, source of the target column means.
    original_data
        Original input, defining the column order and totals to match.
    normalize_column_means
        Whether column-mean normalization was applied (sets the clip bound).
    n_timesteps_per_period
        Timesteps per period, used to reshape the representatives.
    exclude_columns
        Columns to skip during rescaling.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(rescaled_periods, deviations_dict)`` — the corrected representatives
        and per-column residual deviations after rescaling.

    See Also
    --------
    cluster_periods : Produces the representatives rescaled here.
    add_extreme_periods : Its extreme clusters are excluded from rescaling.
    """
    columns = list(original_data.columns)

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

        # Difference between predicted and original sum
        diff = abs(sum_raw - (sum_clu_wo_peak + sum_peak))

        iteration = 0
        while (
            diff > sum_raw * options.rescale_tolerance
            and iteration < options.rescale_max_iterations
        ):
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
            iteration += 1

        # Calculate and store final deviation
        deviation_pct = (diff / sum_raw) * 100 if sum_raw != 0 else 0.0
        converged = iteration < options.rescale_max_iterations
        rescale_deviations[column] = {
            "deviation_pct": deviation_pct,
            "converged": converged,
            "iterations": iteration,
        }

        if not converged and deviation_pct > 0.01:
            warnings.warn(
                f'Max iteration number reached for "{column}" while rescaling '
                f"the cluster periods. The integral of the aggregated time series "
                f"deviates by: {round(deviation_pct, 2)}%"
            )

    # Reshape back to 2D: (n_clusters, n_cols * n_timesteps)
    return arr.reshape(n_clusters, -1), rescale_deviations
