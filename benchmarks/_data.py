"""Deterministic synthetic data + the benchmark dimension levels.

The generator tiles/replicates the real ``testdata.csv`` so benchmark inputs
scale to any ``(n_rows, n_cols)`` while keeping realistic time-series structure.
Everything is seeded, so a given ``(n_rows, n_cols)`` always yields byte-identical
data — a hard requirement for comparable, regression-grade benchmark numbers.

``bench_scaling.py`` parametrizes over the level lists below; the figure scripts
import them for axis ordering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_TESTDATA_CSV = (
    Path(__file__).resolve().parent.parent / "docs" / "notebooks" / "testdata.csv"
)
_BASE_COLS = ["GHI", "T", "Wind", "Load"]

_base_cache: np.ndarray | None = None


def _base_values() -> np.ndarray:
    """Return the real testdata as an ``(8760, 4)`` float array (cached)."""
    global _base_cache
    if _base_cache is None:
        df = pd.read_csv(_TESTDATA_CSV, index_col=0, parse_dates=True)
        _base_cache = df[_BASE_COLS].to_numpy(dtype=float)
    return _base_cache


def make_data(
    n_rows: int,
    n_cols: int,
    resolution_hours: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a deterministic ``n_rows x n_cols`` time series from real testdata.

    Rows are produced by tiling the base year and applying small seeded jitter so
    tiled periods are not byte-identical. Columns beyond the base four are seeded
    scaled/noised replicas of the base attributes, preserving realistic
    correlation structure while widening the clustering feature matrix.

    Args:
        n_rows: Number of time steps (rows) to generate.
        n_cols: Number of attribute columns to generate.
        resolution_hours: Spacing of the DatetimeIndex, in hours.
        seed: RNG seed controlling jitter and column replicas.

    Returns:
        A DataFrame with a regular DatetimeIndex. The first ``min(n_cols, 4)``
        columns keep the real attribute names (``GHI``, ``T``, ``Wind``, ``Load``);
        extras are named ``<attr>_<replica>``.
    """
    rng = np.random.RandomState(seed)
    base = _base_values()
    n_base = base.shape[0]

    reps = int(np.ceil(n_rows / n_base))
    tiled = np.tile(base, (reps, 1))[:n_rows]
    tiled = tiled * (1.0 + 0.02 * rng.standard_normal(tiled.shape))

    columns = []
    names = []
    for i in range(n_cols):
        attr = i % 4
        replica = i // 4
        scale = 1.0 if replica == 0 else 0.5 + rng.rand()
        noise = rng.standard_normal(n_rows) * tiled[:, attr].std() * 0.01
        columns.append(tiled[:, attr] * scale + noise)
        names.append(
            _BASE_COLS[attr] if replica == 0 else f"{_BASE_COLS[attr]}_{replica}"
        )

    index = pd.date_range(
        "2010-01-01", periods=n_rows, freq=pd.Timedelta(hours=resolution_hours)
    )
    return pd.DataFrame(np.column_stack(columns), index=index, columns=names)


# ---------------------------------------------------------------------------
# Benchmark dimension levels (see bench_scaling.test_scale)
# ---------------------------------------------------------------------------

# Realistic energy-modelling usage: a fixed 1-year horizon, refined by
# resolution (which drives both the total timesteps and the timesteps-per-period)
# and split into daily or weekly periods. Columns range up to large multi-region
# models; clusters up to ~50 typical periods.
HORIZON_HOURS = 8760  # one year
RESOLUTIONS_MIN = [60, 15, 5]  # 1h, 15min, 5min
PERIOD_HOURS = [24, 168]  # daily, weekly
COLUMNS = [4, 256, 1000]  # single region -> large multi-region model
CLUSTERS = [12]  # n_clusters (typical periods); cost is nearly flat in this axis
METHODS = [
    "averaging",
    "kmeans",
    "hierarchical",
    "contiguous",
    "kmaxoids",
    "kmedoids",
]

# Methods whose worst case can explode; these get the subprocess timeout gate.
SLOW_METHODS = {"kmedoids", "kmaxoids"}

# Skip cases whose input array (timesteps x columns) exceeds this, to avoid OOM
# on the fine-resolution x many-columns corner (~0.5 GB of float64).
MAX_INPUT_CELLS = 60_000_000


def derive(resolution_min: int, period_hours: int) -> tuple[float, int, int, int]:
    """Return ``(resolution_hours, timesteps, timesteps_per_period, periods)``.

    A whole number of periods is used, so ``timesteps == periods * W`` is always
    divisible by the period length.
    """
    resolution_hours = resolution_min / 60
    steps_per_period = round(period_hours / resolution_hours)
    periods = round(HORIZON_HOURS / period_hours)
    timesteps = periods * steps_per_period
    return resolution_hours, timesteps, steps_per_period, periods
