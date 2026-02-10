"""Unified weight validation for tsam."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

MIN_WEIGHT = 1e-6


def validate_weights(
    columns: pd.Index,
    weights: dict[str, float] | None,
) -> dict[str, float] | None:
    """Validate and normalize per-column clustering weights.

    Consolidates:
    - Column existence check (raises ValueError for unknown columns)
    - MIN_WEIGHT clamping (warns and clamps near-zero weights)
    - Returns None if all weights are effectively 1.0

    Parameters
    ----------
    columns : pd.Index
        Column names from the input data.
    weights : dict or None
        Per-column weights. Columns not listed default to 1.0.

    Returns
    -------
    dict or None
        Validated weights dict, or None if no weighting is needed
        (all weights are 1.0 or input is None/empty).

    Raises
    ------
    ValueError
        If any weight key is not present in *columns*.
    """
    if not weights:
        return None

    missing = set(weights.keys()) - set(columns)
    if missing:
        raise ValueError(f"Weight columns not found in data: {missing}")

    any_non_unit = False
    cleaned: dict[str, float] = {}
    for col, w in weights.items():
        if w < MIN_WEIGHT:
            warnings.warn(
                f'weight of "{col}" set to the minimal tolerable weighting',
                stacklevel=2,
            )
            w = MIN_WEIGHT
        if w != 1.0:
            any_non_unit = True
        cleaned[col] = w

    return cleaned if any_non_unit else None
