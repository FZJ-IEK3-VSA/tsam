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
    """Extend to integer multiple of period length, reshape to period matrix.

    Replicates unstack_to_periods (monolith lines 27-85) and the call at lines 656-660.
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
    """Append per-column sums as extra features.

    Replicates monolith lines 1137-1151.

    Returns (augmented_candidates, n_extra_features_to_trim).
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
