"""Shared golden-regression cases — new-API only.

Self-contained helper for ``test_golden_regression.py`` (and benchmarks).
Defines the datasets, the algorithm configurations expressed purely with the
new :func:`tsam.aggregate` API, and the cross-product of config x dataset that
makes up the golden-file regression suite.

This module intentionally contains **no** reference to the legacy
``tsam.timeseriesaggregation`` API so that it keeps working once the legacy
wrapper is removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tsam import (
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    SegmentConfig,
    aggregate,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"
GOLDEN_DIR = TEST_DATA_DIR / "golden"

EXAMPLES_DIR = TEST_DIR.parent / "docs" / "notebooks"
TESTDATA_CSV = EXAMPLES_DIR / "testdata.csv"
WIDE_CSV = TEST_DATA_DIR / "wide.csv"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _make_constant() -> pd.DataFrame:
    """Two constant columns, 10 days of hourly data."""
    idx = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
    return pd.DataFrame({"A": 42.0, "B": 7.0}, index=idx)


def _make_with_zero_column() -> pd.DataFrame:
    """testdata with an extra all-zero column."""
    df = pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)
    df["Zero"] = 0.0
    return df


DATASETS = {
    "testdata": lambda: pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True),
    "wide": lambda: pd.read_csv(WIDE_CSV, index_col=0, parse_dates=True),
    "constant": _make_constant,
    "with_zero_column": _make_with_zero_column,
}

_DATA_CACHE: dict[str, pd.DataFrame] = {}


def get_data(name: str, max_timesteps: int | None = None) -> pd.DataFrame:
    """Return dataset by name, optionally truncated. Full data is cached."""
    if name not in _DATA_CACHE:
        _DATA_CACHE[name] = DATASETS[name]()
    df = _DATA_CACHE[name]
    if max_timesteps is not None:
        return df.iloc[:max_timesteps]
    return df


# Smaller datasets cannot support the default cluster count.
SMALL_DATASET_CLUSTERS: dict[str, int] = {"constant": 3}


# ---------------------------------------------------------------------------
# Algorithm configurations (new-API kwargs for ``aggregate``)
# ---------------------------------------------------------------------------

_NEW_KWARGS: dict[str, dict] = {
    "hierarchical_default": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
    },
    "hierarchical_mean": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical", representation="mean"),
    },
    "kmeans": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmeans"),
    },
    "hierarchical_distribution": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical", representation="distribution"),
    },
    "hierarchical_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "segments": SegmentConfig(n_segments=8),
    },
    "hierarchical_no_rescale": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "preserve_column_means": False,
    },
    "contiguous": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="contiguous"),
    },
    "averaging": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="averaging"),
    },
    "kmaxoids": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmaxoids"),
    },
    "kmedoids": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmedoids"),
    },
    "hierarchical_maxoid": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical", representation="maxoid"),
    },
    "hierarchical_distribution_minmax": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical", representation="distribution_minmax"
        ),
    },
    "distribution_global": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            representation=Distribution(scope="global"),
        ),
        "preserve_column_means": False,
    },
    "distribution_minmax_global": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            representation=Distribution(scope="global", preserve_minmax=True),
        ),
        "preserve_column_means": False,
    },
    "minmaxmean": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            representation=MinMaxMean(max_columns=["GHI"], min_columns=["T", "Load"]),
        ),
        "preserve_column_means": False,
    },
    "hierarchical_weighted": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "weights": {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
    },
    "segmentation_samemean": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical", normalize_column_means=True),
        "segments": SegmentConfig(n_segments=4),
    },
    "segmentation_distribution_global": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            representation=Distribution(scope="global"),
        ),
        "segments": SegmentConfig(
            n_segments=4, representation=Distribution(scope="global")
        ),
        "preserve_column_means": False,
    },
    "extremes_append": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="append", max_value=["Load"]),
    },
    "extremes_replace": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="replace", max_value=["Load"]),
    },
    "extremes_new_cluster": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="new_cluster", max_value=["Load"]),
    },
    "extremes_min_value": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="append", min_value=["T"]),
    },
    "extremes_max_period": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="append", max_period=["GHI"]),
    },
    "extremes_min_period": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="append", min_period=["Wind"]),
    },
    "extremes_multi": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(
            method="append",
            max_value=["Load"],
            min_value=["T"],
            max_period=["GHI"],
        ),
    },
    "extremes_with_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="append", max_value=["Load"]),
        "segments": SegmentConfig(n_segments=6),
    },
    "extremes_wide_multi": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(
            method="append",
            max_value=["DE_Load"],
            min_value=["FR_T"],
        ),
    },
    "extremes_constant": {
        "n_clusters": 3,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="append", max_value=["A"]),
    },
    "extremes_zero_column": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="append", max_value=["Zero"]),
    },
    # --- Clustering method x segmentation ---
    "kmeans_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmeans"),
        "segments": SegmentConfig(n_segments=8),
    },
    "kmedoids_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmedoids"),
        "segments": SegmentConfig(n_segments=8),
    },
    "kmaxoids_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmaxoids"),
        "segments": SegmentConfig(n_segments=8),
    },
    "averaging_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="averaging"),
        "segments": SegmentConfig(n_segments=8),
    },
    "contiguous_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="contiguous"),
        "segments": SegmentConfig(n_segments=8),
    },
    # --- Untested boolean clustering options ---
    "hierarchical_duration_curves": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical", use_duration_curves=True),
    },
    "hierarchical_eval_sum_periods": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical", include_period_sums=True),
    },
    "kmeans_duration_curves": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmeans", use_duration_curves=True),
    },
    # --- rescale_exclude_columns ---
    "hierarchical_rescale_exclude": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "rescale_exclude_columns": ["GHI"],
    },
    # --- round_decimals ---
    "hierarchical_round": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "round_decimals": 2,
    },
    # --- Clustering method x extremes ---
    "kmeans_extremes_append": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmeans"),
        "extremes": ExtremeConfig(method="append", max_value=["Load"]),
    },
    "contiguous_extremes_append": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="contiguous"),
        "extremes": ExtremeConfig(method="append", max_value=["Load"]),
    },
    # --- Weight x feature interactions ---
    "kmeans_weighted": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmeans"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
    },
    "kmedoids_weighted": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmedoids"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
    },
    "kmaxoids_weighted": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmaxoids"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
    },
    "hierarchical_weighted_duration_curves": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            use_duration_curves=True,
        ),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
    },
    "hierarchical_weighted_extremes": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "extremes": ExtremeConfig(method="append", max_value=["Load"]),
    },
    "hierarchical_weighted_samemean": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            normalize_column_means=True,
        ),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
    },
    "hierarchical_weighted_rescale_exclude": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "rescale_exclude_columns": ["GHI"],
    },
    "kmeans_weighted_distribution": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="kmeans",
            representation="distribution",
        ),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
    },
    # --- Cross-feature interactions ---
    "hierarchical_weighted_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "weights": {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "segments": SegmentConfig(n_segments=8),
    },
    "kmeans_distribution": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmeans", representation="distribution"),
    },
    "extremes_replace_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "extremes": ExtremeConfig(method="replace", max_value=["Load"]),
        "segments": SegmentConfig(n_segments=6),
    },
    # --- Weight x segmentation x feature (three-way) ---
    "hierarchical_weighted_segmentation_extremes": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "segments": SegmentConfig(n_segments=6),
        "extremes": ExtremeConfig(method="append", max_value=["Load"]),
    },
    "hierarchical_weighted_segmentation_samemean": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            normalize_column_means=True,
        ),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "segments": SegmentConfig(n_segments=4),
    },
    "hierarchical_weighted_segmentation_distribution": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            representation="distribution",
        ),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "segments": SegmentConfig(n_segments=4),
    },
    "kmeans_weighted_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="kmeans"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "segments": SegmentConfig(n_segments=8),
    },
    # --- Weight x no rescale ---
    "hierarchical_weighted_no_rescale": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(method="hierarchical"),
        "weights": {"Load": 5.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        "preserve_column_means": False,
    },
}

# Per-config metadata: seed, dataset restriction, data truncation, tolerance.
_META: dict[str, dict] = {
    "hierarchical_default": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_mean": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans": {"seed": 42, "only_datasets": None, "max_timesteps": None, "rtol": 1e-05},
    "hierarchical_distribution": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_segmentation": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_no_rescale": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "contiguous": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "averaging": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmaxoids": {
        "seed": 42,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "kmedoids": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": 2016,
        "rtol": 1e-10,
    },
    "hierarchical_maxoid": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_distribution_minmax": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "distribution_global": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "distribution_minmax_global": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "minmaxmean": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_weighted": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "segmentation_samemean": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "segmentation_distribution_global": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_append": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_replace": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_new_cluster": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_min_value": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_max_period": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_min_period": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_multi": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_with_segmentation": {
        "seed": None,
        "only_datasets": ["testdata", "with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_wide_multi": {
        "seed": None,
        "only_datasets": ["wide"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_constant": {
        "seed": None,
        "only_datasets": ["constant"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "extremes_zero_column": {
        "seed": None,
        "only_datasets": ["with_zero_column"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans_segmentation": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "kmedoids_segmentation": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": 2016,
        "rtol": 1e-10,
    },
    "kmaxoids_segmentation": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "averaging_segmentation": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "contiguous_segmentation": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_duration_curves": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_eval_sum_periods": {
        "seed": None,
        "only_datasets": None,
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans_duration_curves": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "hierarchical_rescale_exclude": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_round": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans_extremes_append": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "contiguous_extremes_append": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans_weighted": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "kmedoids_weighted": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": 2016,
        "rtol": 1e-10,
    },
    "kmaxoids_weighted": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "hierarchical_weighted_duration_curves": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_weighted_extremes": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_weighted_samemean": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_weighted_rescale_exclude": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans_weighted_distribution": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "hierarchical_weighted_segmentation": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans_distribution": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "extremes_replace_segmentation": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_weighted_segmentation_extremes": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_weighted_segmentation_samemean": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "hierarchical_weighted_segmentation_distribution": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
    "kmeans_weighted_segmentation": {
        "seed": 42,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-05,
    },
    "hierarchical_weighted_no_rescale": {
        "seed": None,
        "only_datasets": ["testdata"],
        "max_timesteps": None,
        "rtol": 1e-10,
    },
}


# ---------------------------------------------------------------------------
# Build cross-product: config x dataset → GoldenCase
# ---------------------------------------------------------------------------


@dataclass
class GoldenCase:
    """Fully resolved test case (config + dataset) using new-API kwargs."""

    id: str
    dataset: str
    new_kwargs: dict
    seed: int | None = None
    rtol: float = 1e-10
    max_timesteps: int | None = None
    only_datasets: set[str] = field(default_factory=set)


def _build_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for config_id, new_kw_base in _NEW_KWARGS.items():
        meta = _META[config_id]
        eligible = (
            set(meta["only_datasets"]) if meta["only_datasets"] else set(DATASETS)
        )
        for ds in sorted(eligible):
            n_override = SMALL_DATASET_CLUSTERS.get(ds)
            new_kw = dict(new_kw_base)
            if n_override is not None:
                new_kw["n_clusters"] = n_override
            cases.append(
                GoldenCase(
                    id=f"{config_id}/{ds}",
                    dataset=ds,
                    new_kwargs=new_kw,
                    seed=meta["seed"],
                    rtol=meta["rtol"],
                    max_timesteps=meta["max_timesteps"],
                )
            )
    return cases


CASES = _build_cases()


def case_ids(cases: list) -> list[str]:
    """Extract test IDs from a list of cases (any object with an ``id`` attr)."""
    return [c.id for c in cases]


def run_new(data: pd.DataFrame, case: GoldenCase):
    """Run the new API for a case and return the ``AggregationResult``."""
    if case.seed is not None:
        np.random.seed(case.seed)
    return aggregate(data, **case.new_kwargs)
