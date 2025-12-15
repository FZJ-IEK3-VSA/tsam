"""Utility functions for tsam."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reshape_to_periods(
    data: pd.DataFrame | pd.Series,
    period_length: int,
) -> pd.DataFrame:
    """Reshape time series into periods x timesteps format.

    Transforms a flat time series into a 2D array where each row is a period
    (e.g., a day) and each column is a timestep within that period (e.g., an hour).

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Time series data with shape (n_timesteps,) or (n_timesteps, n_columns).
    period_length : int
        Number of timesteps per period (e.g., 24 for hourly data with daily periods).

    Returns
    -------
    pd.DataFrame
        Reshaped data with shape (n_periods, period_length) for Series input,
        or (n_periods, period_length * n_columns) for DataFrame input.
        Columns are MultiIndex (column, timestep) for DataFrame input.

    Examples
    --------
    >>> import pandas as pd
    >>> import tsam
    >>> data = pd.DataFrame({"temp": range(8760), "load": range(8760)})
    >>> reshaped = tsam.reshape_to_periods(data, period_length=24)
    >>> reshaped.shape
    (365, 48)
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    n_timesteps = len(data)
    n_periods = n_timesteps // period_length

    data = data.iloc[: n_periods * period_length]

    result = {}
    for col in data.columns:
        arr = np.asarray(data[col]).reshape(n_periods, period_length)
        for t in range(period_length):
            result[(col, t)] = arr[:, t]

    return pd.DataFrame(result)


def reshape_to_periods_array(
    data: pd.DataFrame | pd.Series,
    period_length: int,
) -> np.ndarray:
    """Reshape time series into a 3D numpy array.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Time series data.
    period_length : int
        Number of timesteps per period.

    Returns
    -------
    np.ndarray
        Array with shape (n_columns, period_length, n_periods).
        For Series input, shape is (1, period_length, n_periods).
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    n_timesteps = len(data)
    n_periods = n_timesteps // period_length
    n_cols = len(data.columns)

    arr = np.asarray(data.iloc[: n_periods * period_length])
    reshaped: np.ndarray = arr.reshape(n_periods, period_length, n_cols).transpose(
        2, 1, 0
    )
    return reshaped
