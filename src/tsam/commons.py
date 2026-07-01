"""Generic, dependency-light helpers shared across tsam modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def infer_resolution(data: pd.DataFrame) -> float:
    """Infer temporal resolution (hours per step) from a data index."""
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        return (data.index[1] - data.index[0]).total_seconds() / 3600
    return 1.0


def time_index_to_dict(idx: pd.DatetimeIndex) -> dict[str, Any] | list[str]:
    """Serialize a DatetimeIndex compactly when possible.

    Regular indices are stored as ``{start, periods, freq}`` (~3 values).
    Irregular indices fall back to a full ISO string list.
    """
    freq = pd.infer_freq(idx)
    if freq is not None:
        return {"start": idx[0].isoformat(), "periods": len(idx), "freq": freq}
    return [t.isoformat() for t in idx]


def time_index_from_dict(
    raw: dict[str, Any] | list[str],
) -> pd.DatetimeIndex:
    """Deserialize a DatetimeIndex from either compact or list format."""
    if isinstance(raw, dict):
        return pd.date_range(raw["start"], periods=raw["periods"], freq=raw["freq"])
    return pd.DatetimeIndex(raw)


def parse_duration_hours(value: int | float | str, param_name: str) -> float:
    """Parse a duration value to hours.

    Accepts:
    - int/float: interpreted as hours (e.g., 24 -> 24.0 hours)
    - str: pandas Timedelta string (e.g., '24h', '1d', '15min')

    Returns duration in hours as float.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            td = pd.Timedelta(value)
            return td.total_seconds() / 3600
        except ValueError as e:
            raise ValueError(
                f"{param_name}: invalid duration string '{value}': {e}"
            ) from e
    raise TypeError(
        f"{param_name} must be int, float, or string, got {type(value).__name__}"
    )


def weighted_mean(
    per_column: pd.Series,
    weights: dict[str, float] | None,
) -> float:
    """Weighted arithmetic mean of per-column values.

    Parameters
    ----------
    per_column : pd.Series
        One value per column (e.g. per-column MAE).
    weights : dict or None
        Column name → weight. Missing columns default to 1.
        ``None`` is equivalent to uniform weights.

    Returns
    -------
    float
        ``sum(value_i * w_i) / sum(w_i)``
    """
    if weights:
        w = pd.Series(weights).reindex(per_column.index, fill_value=1.0)
        return float((per_column * w).sum() / w.sum())
    return float(per_column.mean())


def weighted_rms(
    per_column: pd.Series,
    weights: dict[str, float] | None,
) -> float:
    """Weighted root-mean-square of per-column values.

    Appropriate for aggregating RMSE across columns: the result equals
    the RMSE you would obtain by pooling all (weighted) residuals into
    a single series.

    Parameters
    ----------
    per_column : pd.Series
        One RMSE value per column.
    weights : dict or None
        Column name → weight. Missing columns default to 1.
        ``None`` is equivalent to uniform weights.

    Returns
    -------
    float
        ``sqrt(sum(value_i² * w_i) / sum(w_i))``
    """
    squared = per_column**2
    if weights:
        w = pd.Series(weights).reindex(squared.index, fill_value=1.0)
        return float(((squared * w).sum() / w.sum()) ** 0.5)
    return float(squared.mean() ** 0.5)


def bounded_water_fill(
    values: np.ndarray,
    weights: np.ndarray,
    lower: float,
    upper: float,
    target_weighted_sum: float,
    *,
    rel_tolerance: float,
    max_passes: int = 100,
) -> tuple[np.ndarray, bool, int]:
    """Adjust ``values`` within ``[lower, upper]`` to hit a weighted-sum target.

    Solves a bounded water-fill: each pass moves mass proportional to every
    element's remaining headroom (``upper - value``) when the weighted sum is
    too low, or its depth above the floor (``value - lower``) when it is too
    high. The weighted sum changes by exactly the residual (capped at the
    feasible amount), so it moves monotonically toward the target and every
    element stays inside ``[lower, upper]``. This preserves the integral without
    flattening the distribution against either bound.

    Parameters
    ----------
    values : np.ndarray
        Starting values (any shape). Not mutated; a clipped copy is returned.
    weights : np.ndarray
        Per-element weights, broadcastable against ``values`` (e.g. one weight
        per period, shape ``(n_periods, 1)``).
    lower, upper : float
        Inclusive bounds enforced on every element.
    target_weighted_sum : float
        Desired value of ``sum(weights * values)``.
    rel_tolerance : float
        Relative convergence tolerance; the absolute tolerance on the
        weighted-sum residual is ``max(abs(target_weighted_sum), 1) *
        rel_tolerance``.
    max_passes : int
        Safety cap on redistribution passes.

    Returns
    -------
    tuple[np.ndarray, bool, int]
        The adjusted array, whether the target was reached within tolerance
        (``False`` if the passes were exhausted or no feasible room remained),
        and the number of redistribution passes performed.
    """
    tolerance = max(abs(target_weighted_sum), 1.0) * rel_tolerance
    adjusted = np.clip(np.array(values, dtype=float), lower, upper)
    passes = 0
    while passes < max_passes:
        residual = target_weighted_sum - float(np.sum(weights * adjusted))
        if abs(residual) <= tolerance:
            return adjusted, True, passes
        # Room to grow toward the cap, or depth to shrink toward the floor.
        room = (upper - adjusted) if residual > 0 else (adjusted - lower)
        capacity = float(np.sum(weights * room))
        if capacity <= tolerance:
            break  # no feasible room left inside [lower, upper]
        step = min(abs(residual), capacity)
        adjusted = adjusted + np.sign(residual) * step * room / capacity
        np.clip(adjusted, lower, upper, out=adjusted)
        passes += 1
    return adjusted, False, passes
