"""Benchmarks for tsam using pytest-benchmark.

Self-contained: only imports ``tsam.timeseriesaggregation`` so it works with
both the current dev version and older releases (e.g. tsam==2.3.9).

Usage::

    # Benchmark an old version
    uv pip install tsam==2.3.9
    pytest test/test_benchmarks.py --benchmark-save=v2.3.9

    # Switch back to dev and compare
    uv pip install -e .
    pytest test/test_benchmarks.py --benchmark-compare='*v2.3.9'

    # Compare two saved snapshots
    pytest-benchmark compare '*v2.3.9' '*v3.0.0' --group-by=name
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as tsam

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEST_DIR = Path(__file__).parent
_EXAMPLES_DIR = _TEST_DIR.parent / "docs" / "source" / "examples_notebooks"
_TESTDATA_CSV = _EXAMPLES_DIR / "testdata.csv"
_OPSD_CSV = _TEST_DIR / "data" / "opsd_germany_2019.csv"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_DATASETS: dict[str, callable] = {
    "testdata": lambda: pd.read_csv(_TESTDATA_CSV, index_col=0, parse_dates=True),
    "opsd": lambda: pd.read_csv(_OPSD_CSV, index_col=0, parse_dates=True),
    "constant": lambda: pd.DataFrame(
        {"A": 42.0, "B": 7.0},
        index=pd.date_range("2020-01-01", periods=10 * 24, freq="h"),
    ),
    "with_zero_column": lambda: pd.read_csv(
        _TESTDATA_CSV, index_col=0, parse_dates=True
    ).assign(Zero=0.0),
}

_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _get_data(name: str) -> pd.DataFrame:
    if name not in _DATA_CACHE:
        _DATA_CACHE[name] = _DATASETS[name]()
    return _DATA_CACHE[name]


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    id: str
    kwargs: dict = field(default_factory=dict)
    seed: int | None = None
    only_datasets: set[str] = field(default_factory=set)


_SMALL_DATASET_CLUSTERS = {"constant": 3}

CONFIGS: list[Case] = [
    Case(
        id="hierarchical_default",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
        },
    ),
    Case(
        id="hierarchical_mean",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "meanRepresentation",
        },
    ),
    Case(
        id="kmeans",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_means",
        },
        seed=42,
    ),
    Case(
        id="hierarchical_distribution",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "durationRepresentation",
        },
    ),
    Case(
        id="hierarchical_segmentation",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "segmentation": True,
            "noSegments": 8,
        },
    ),
    Case(
        id="hierarchical_no_rescale",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "rescaleClusterPeriods": False,
        },
    ),
    Case(
        id="contiguous",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "adjacent_periods",
        },
    ),
    Case(
        id="averaging",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "averaging",
        },
    ),
    Case(
        id="kmaxoids",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_maxoids",
        },
        seed=42,
    ),
    Case(
        id="kmedoids",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_medoids",
        },
        seed=42,
        only_datasets={"testdata"},
    ),
    Case(
        id="hierarchical_maxoid",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "maxoidRepresentation",
        },
    ),
    Case(
        id="hierarchical_distribution_minmax",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
        },
    ),
    Case(
        id="distribution_global",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
    ),
    Case(
        id="distribution_minmax_global",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
    ),
    Case(
        id="minmaxmean",
        kwargs={
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
        only_datasets={"testdata"},
    ),
    Case(
        id="hierarchical_weighted",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "weightDict": {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        },
        only_datasets={"testdata"},
    ),
    Case(
        id="extremes_append",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_replace",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "replace_cluster_center",
            "addPeakMax": ["Load"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_new_cluster",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "new_cluster_center",
            "addPeakMax": ["Load"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_min_value",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMin": ["T"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_max_period",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMax": ["GHI"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_min_period",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMin": ["Wind"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_multi",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
            "addPeakMin": ["T"],
            "addMeanMax": ["GHI"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_with_segmentation",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
            "segmentation": True,
            "noSegments": 6,
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Case(
        id="extremes_opsd_multi",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["DE_Load"],
            "addPeakMin": ["DE_Price"],
        },
        only_datasets={"opsd"},
    ),
    Case(
        id="extremes_constant",
        kwargs={
            "noTypicalPeriods": 3,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["A"],
        },
        only_datasets={"constant"},
    ),
    Case(
        id="extremes_zero_column",
        kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Zero"],
        },
        only_datasets={"with_zero_column"},
    ),
]


# ---------------------------------------------------------------------------
# Build parametrized list: config x dataset
# ---------------------------------------------------------------------------


@dataclass
class BenchCase:
    case_id: str
    dataset: str
    kwargs: dict
    seed: int | None = None


def _build_bench_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []
    for cfg in CONFIGS:
        eligible = cfg.only_datasets or set(_DATASETS)
        for ds in eligible:
            kw = dict(cfg.kwargs)
            n_override = _SMALL_DATASET_CLUSTERS.get(ds)
            if n_override is not None:
                kw["noTypicalPeriods"] = n_override
            cases.append(
                BenchCase(
                    case_id=f"{cfg.id}/{ds}",
                    dataset=ds,
                    kwargs=kw,
                    seed=cfg.seed,
                )
            )
    return cases


BENCH_CASES = _build_bench_cases()
_IDS = [c.case_id for c in BENCH_CASES]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(data: pd.DataFrame, case: BenchCase) -> None:
    if case.seed is not None:
        np.random.seed(case.seed)
    agg = tsam.TimeSeriesAggregation(timeSeries=data, **case.kwargs)
    agg.createTypicalPeriods()
    agg.predictOriginalData()


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", BENCH_CASES, ids=_IDS)
def test_bench(case: BenchCase, benchmark):
    data = _get_data(case.dataset)
    benchmark(lambda: _run(data, case))
