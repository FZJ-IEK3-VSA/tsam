#!/usr/bin/env python
"""Generate golden regression files using the OLD tsam API.

This script is meant to be run against a specific tsam version (e.g. v2.3.9)
to produce baseline reconstructed time series for regression testing.

Usage::

    # Run with tsam v2.3.9 in an ephemeral environment
    uv run --with tsam==2.3.9 --no-project test/generate_golden.py

    # Or with any installed tsam version
    python test/generate_golden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
TESTDATA_CSV = (
    SCRIPT_DIR.parent / "docs" / "source" / "examples_notebooks" / "testdata.csv"
)
WIDE_CSV = SCRIPT_DIR / "data" / "wide.csv"
GOLDEN_DIR = SCRIPT_DIR / "data" / "golden"


def _make_constant() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
    return pd.DataFrame({"A": 42.0, "B": 7.0}, index=idx)


def _make_with_zero_column() -> pd.DataFrame:
    df = pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)
    df["Zero"] = 0.0
    return df


DATASETS: dict[str, callable] = {
    "testdata": lambda: pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True),
    "wide": lambda: pd.read_csv(WIDE_CSV, index_col=0, parse_dates=True),
    "constant": _make_constant,
    "with_zero_column": _make_with_zero_column,
}

# ---------------------------------------------------------------------------
# Configs — old_kwargs only (no new API references)
# ---------------------------------------------------------------------------

CONFIGS: list[dict] = [
    {
        "id": "hierarchical_default",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
        },
    },
    {
        "id": "hierarchical_mean",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "meanRepresentation",
        },
    },
    {
        "id": "kmeans",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_means",
        },
        "seed": 42,
    },
    {
        "id": "hierarchical_distribution",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "durationRepresentation",
        },
    },
    {
        "id": "hierarchical_segmentation",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "segmentation": True,
            "noSegments": 8,
        },
    },
    {
        "id": "hierarchical_no_rescale",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "rescaleClusterPeriods": False,
        },
    },
    {
        "id": "contiguous",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "adjacent_periods",
        },
    },
    {
        "id": "averaging",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "averaging",
        },
    },
    {
        "id": "kmaxoids",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_maxoids",
        },
        "seed": 42,
    },
    {
        "id": "kmedoids",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_medoids",
        },
        "seed": 42,
        "only_datasets": {"testdata"},
    },
    {
        "id": "hierarchical_maxoid",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "maxoidRepresentation",
        },
    },
    {
        "id": "hierarchical_distribution_minmax",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
        },
    },
    {
        "id": "distribution_global",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
    },
    {
        "id": "distribution_minmax_global",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
    },
    {
        "id": "minmaxmean",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "minmaxmeanRepresentation",
            "representationDict": {
                "GHI": "max",
                "T": "min",
                "Wind": "mean",
                "Load": "min",
            },
            "rescaleClusterPeriods": False,
        },
        "only_datasets": {"testdata"},
    },
    {
        "id": "hierarchical_weighted",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "weightDict": {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        },
        "only_datasets": {"testdata"},
    },
    {
        "id": "extremes_append",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_replace",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "replace_cluster_center",
            "addPeakMax": ["Load"],
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_new_cluster",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "new_cluster_center",
            "addPeakMax": ["Load"],
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_min_value",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMin": ["T"],
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_max_period",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMax": ["GHI"],
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_min_period",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMin": ["Wind"],
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_multi",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
            "addPeakMin": ["T"],
            "addMeanMax": ["GHI"],
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_with_segmentation",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
            "segmentation": True,
            "noSegments": 6,
        },
        "only_datasets": {"testdata", "with_zero_column"},
    },
    {
        "id": "extremes_wide_multi",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["DE_Load"],
            "addPeakMin": ["FR_T"],
        },
        "only_datasets": {"wide"},
    },
    {
        "id": "extremes_constant",
        "old_kwargs": {
            "noTypicalPeriods": 3,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["A"],
        },
        "only_datasets": {"constant"},
    },
    {
        "id": "extremes_zero_column",
        "old_kwargs": {
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Zero"],
        },
        "only_datasets": {"with_zero_column"},
    },
]

# Default n_clusters overrides for small datasets.
_SMALL_DATASET_CLUSTERS = {"constant": 3}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def main() -> None:
    print(
        f"tsam version: {tsam.__version__ if hasattr(tsam, '__version__') else 'unknown'}"
    )
    print(f"Output dir:   {GOLDEN_DIR}")
    print()

    total = 0
    for cfg in CONFIGS:
        cfg_id = cfg["id"]
        old_kwargs = cfg["old_kwargs"]
        seed = cfg.get("seed")
        only_datasets = cfg.get("only_datasets", set())

        eligible = only_datasets or set(DATASETS)
        for ds_name in sorted(eligible):
            if ds_name not in DATASETS:
                print(f"  SKIP {cfg_id}/{ds_name} — dataset not available")
                continue

            # Apply n_clusters override for small datasets
            kw = dict(old_kwargs)
            n_override = _SMALL_DATASET_CLUSTERS.get(ds_name)
            if n_override is not None and "noTypicalPeriods" in kw:
                kw["noTypicalPeriods"] = n_override

            data = DATASETS[ds_name]()

            if seed is not None:
                np.random.seed(seed)

            agg = tsam.TimeSeriesAggregation(timeSeries=data, **kw)
            agg.createTypicalPeriods()
            reconstructed = agg.predictOriginalData()

            # Save
            case_id = f"{cfg_id}/{ds_name}"
            path = GOLDEN_DIR / f"{case_id}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            reconstructed.to_csv(path)

            total += 1
            print(f"  {case_id}  ({reconstructed.shape})")

    print(f"\nGenerated {total} golden files.")


if __name__ == "__main__":
    main()
