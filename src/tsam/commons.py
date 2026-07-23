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

    Accepts an int/float, interpreted as hours (e.g. 24 -> 24.0 hours), or a
    string in pandas Timedelta format (e.g. '24h', '1d', '15min').

    Args:
        value: Duration as a number of hours or a pandas Timedelta string.
        param_name: Name of the parameter, used in error messages.

    Returns:
        The duration in hours.

    Raises:
        ValueError: If value is a string that cannot be parsed as a duration.
        TypeError: If value is not an int, float, or string.
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

    Args:
        per_column: One value per column (e.g. per-column MAE).
        weights: Column name to weight mapping. Missing columns default to 1.
            None is equivalent to uniform weights.

    Returns:
        ``sum(value_i * w_i) / sum(w_i)``.
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

    Args:
        per_column: One RMSE value per column.
        weights: Column name to weight mapping. Missing columns default to 1.
            None is equivalent to uniform weights.

    Returns:
        ``sqrt(sum(value_i² * w_i) / sum(w_i))``.
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

    Args:
        values: Starting values (any shape). Not mutated; a clipped copy is
            returned.
        weights: Per-element weights, broadcastable against ``values`` (e.g. one
            weight per period, shape ``(n_periods, 1)``).
        lower: Inclusive lower bound enforced on every element.
        upper: Inclusive upper bound enforced on every element.
        target_weighted_sum: Desired value of ``sum(weights * values)``.
        rel_tolerance: Relative convergence tolerance; the absolute tolerance on
            the weighted-sum residual is
            ``max(abs(target_weighted_sum), 1) * rel_tolerance``.
        max_passes: Safety cap on redistribution passes.

    Returns:
        A tuple of the adjusted array, whether the target was reached within
        tolerance (False if the passes were exhausted or no feasible room
        remained), and the number of redistribution passes performed.
    """
    tolerance = max(abs(target_weighted_sum), 1.0) * rel_tolerance
    adjusted = np.clip(np.array(values, dtype=float), lower, upper)
    passes = 0
    while passes < max_passes:
        residual = target_weighted_sum - float(np.sum(weights * adjusted))
        if abs(residual) <= tolerance:
            return adjusted, True, passes
        room = (upper - adjusted) if residual > 0 else (adjusted - lower)
        capacity = float(np.sum(weights * room))
        if capacity <= tolerance:
            break
        step = min(abs(residual), capacity)
        adjusted = adjusted + np.sign(residual) * step * room / capacity
        np.clip(adjusted, lower, upper, out=adjusted)
        passes += 1
    return adjusted, False, passes
