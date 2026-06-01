"""Period unstacking and feature augmentation."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from tsam.pipeline.types import PeriodProfiles


def unstack_to_periods(
    normalized_ts: pd.DataFrame,
    n_timesteps_per_period: int,
) -> PeriodProfiles:
    """Reshape the flat time series into a (period × timestep-feature) matrix.

    Clustering groups whole periods, so the flat series must first become a
    matrix where each row is one period and each column is an
    ``(attribute, timestep)`` pair.

    **Example.** 365 days of hourly data for 3 columns is an ``(8760, 3)``
    DataFrame. Unstacking with ``n_timesteps_per_period=24`` yields a
    ``(365, 72)`` matrix — each row is a 72-dimensional point
    (3 columns × 24 hours).

    If the series length is not an integer multiple of the period length, the
    last period is padded by repeating the first rows so the reshape succeeds;
    the padded period's weight is corrected later during post-processing.

    Parameters
    ----------
    normalized_ts
        Normalized flat time series (output of `normalize`).
    n_timesteps_per_period
        Timesteps in one period, e.g. ``24`` for daily periods of hourly data.

    Returns
    -------
    PeriodProfiles
        The candidate matrix plus the column ``MultiIndex`` and original time
        index needed to reshape and reconstruct later.

    Raises
    ------
    ValueError
        If the reshaped data contains NaN (indicates malformed input).

    See Also
    --------
    add_period_sum_features : Optionally append per-period column sums as
        extra clustering features.
    """
    unstacked = normalized_ts.copy()

    # Extend to integer multiple of period length
    if len(normalized_ts) % n_timesteps_per_period == 0:
        pass
    else:
        attached_timesteps = (
            n_timesteps_per_period - len(normalized_ts) % n_timesteps_per_period
        )
        rep_data = unstacked.head(attached_timesteps)
        unstacked = pd.concat([unstacked, rep_data])

    # Create period and step index
    period_index = []
    step_index = []
    for ii in range(len(unstacked)):
        period_index.append(int(ii / n_timesteps_per_period))
        step_index.append(
            ii - int(ii / n_timesteps_per_period) * n_timesteps_per_period
        )

    # Save old index
    time_index = copy.deepcopy(unstacked.index)

    # Create new double index and unstack
    unstacked.index = pd.MultiIndex.from_arrays(
        [step_index, period_index], names=["TimeStep", "PeriodNum"]
    )
    unstacked = unstacked.unstack(level="TimeStep")  # type: ignore[assignment]

    # Check for NaN
    if unstacked.isnull().values.any():
        raise ValueError(
            "Pre processed data includes NaN. Please check the time_series input data."
        )

    n_periods = len(unstacked)
    n_columns = len(normalized_ts.columns)

    return PeriodProfiles(
        column_index=unstacked.columns,  # type: ignore[arg-type]
        time_index=time_index,
        profiles_dataframe=unstacked,
        n_timesteps_per_period=n_timesteps_per_period,
        n_columns=n_columns,
        n_periods=n_periods,
    )


def add_period_sum_features(
    profiles_df: pd.DataFrame,
    candidates: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Append each period's per-column sum as extra clustering features.

    Optional stage, enabled by ``ClusterConfig.include_period_sums``. The
    per-column sum of each period is appended as extra columns so that periods
    with similar totals are pulled together, not just periods with similar
    shapes.

    These extra columns influence **only** which periods get grouped — they are
    stripped from the cluster centers during post-processing (the trim step) so
    they never reach the representation logic, which expects the original
    columns. When per-column weights are active they are already baked into
    ``candidates``, so the sums are appended to the weighted candidates.

    Parameters
    ----------
    profiles_df
        The unstacked period profiles (used to compute per-period sums).
    candidates
        Current candidate matrix (possibly already weighted) to augment.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(augmented_candidates, n_extra_features)`` — the second value is the
        number of appended columns, kept so the trim step can remove them.

    See Also
    --------
    cluster_periods : Consumes the (possibly augmented) candidate matrix.
    """
    evaluation_values = (
        profiles_df.stack(future_stack=True, level=0).sum(axis=1).unstack(level=1)  # type: ignore[arg-type]
    )
    n_extra = len(evaluation_values.columns)
    augmented = np.concatenate(
        (candidates, evaluation_values.values),
        axis=1,
    )
    return augmented, n_extra
