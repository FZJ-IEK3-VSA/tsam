"""Small shared helpers used across tsam modules.

Generic, dependency-free utilities (temporal-resolution inference and
DatetimeIndex serialization) that several modules need. Kept here, rather than
in ``config`` or ``result``, so neither module's public API reference is
cluttered by them.
"""

from __future__ import annotations

from typing import Any

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
