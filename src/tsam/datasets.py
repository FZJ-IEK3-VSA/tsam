"""Designed example datasets for the documentation.

Each function returns a small time series whose properties are chosen so that
the differences between aggregation options are visible and checkable. None of
the data is measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_HOURS = np.arange(0, 24, 3)

# One tuple per day: solar peak [W/m²], its hour, load base [MW], evening load
# peak [MW], its hour. Days 0-5 have a high solar peak, days 6-9 a low one.
_REPRESENTATION_DAYS = (
    (800, 12, 50, 30, 18),
    (900, 15, 55, 40, 21),
    (850, 12, 45, 25, 18),
    (950, 15, 50, 45, 21),
    (750, 12, 60, 35, 18),
    (880, 15, 52, 50, 21),
    (250, 12, 70, 40, 18),
    (300, 15, 75, 55, 21),
    (200, 12, 80, 45, 18),
    (350, 15, 72, 60, 21),
)


def _day(
    solar_peak: float,
    solar_hour: float,
    load_base: float,
    evening_peak: float,
    evening_hour: float,
) -> np.ndarray:
    """One day: a solar bell around ``solar_hour``; a load base with a morning bump and an evening peak."""
    solar = solar_peak * np.exp(-(((_HOURS - solar_hour) / 4) ** 2))
    solar[solar < 0.05 * solar_peak] = 0.0
    load = (
        load_base
        + 15 * np.exp(-(((_HOURS - 7) / 2) ** 2))
        + evening_peak * np.exp(-(((_HOURS - evening_hour) / 2.5) ** 2))
    )
    return np.column_stack([solar, load])


def representation_days() -> pd.DataFrame:
    """Ten designed days for comparing duration representations.

    The series has eight timesteps of three hours per day and the attributes
    ``solar`` (W/m²) and ``load`` (MW). Days 0 to 5 have a high solar peak of
    750 to 950 W/m², at 12:00 on days 0, 2 and 4 and at 15:00 on days 1, 3
    and 5, and an evening load peak of 70 to 102 MW three hours later. Days 6
    to 9 have a low solar peak of 200 to 350 W/m², as high as the other days'
    morning and evening values, and a higher load of 70 to 132 MW that
    overlaps the other days' evening peaks. Hierarchical clustering into two
    clusters separates the high-solar days from the low-solar days.

    Returns:
        A data frame with 80 rows at a three-hour resolution, starting on
        2024-06-01, and the columns ``solar`` and ``load``.
    """
    values = np.vstack([_day(*spec) for spec in _REPRESENTATION_DAYS]).round(1)
    index = pd.date_range(
        "2024-06-01", periods=len(_REPRESENTATION_DAYS) * len(_HOURS), freq="3h"
    )
    return pd.DataFrame(values, columns=["solar", "load"], index=index)
