"""Shared test configuration — no new-API imports.

Importable under any tsam version (including v2.3.9) because it only
references old-API parameter names.  Used by:

- ``test/test_old_new_equivalence.py`` (adds new_kwargs on top)
- ``test/test_golden_regression.py`` (via test_old_new_equivalence)
- ``test/generate_golden.py``
- ``benchmarks/bench.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"
GOLDEN_DIR = TEST_DATA_DIR / "golden"

EXAMPLES_DIR = TEST_DIR.parent / "docs" / "source" / "examples_notebooks"
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


def get_data(name: str) -> pd.DataFrame:
    """Return dataset by name, cached across calls."""
    if name not in _DATA_CACHE:
        _DATA_CACHE[name] = DATASETS[name]()
    return _DATA_CACHE[name]


# ---------------------------------------------------------------------------
# Small-dataset overrides
# ---------------------------------------------------------------------------

SMALL_DATASET_CLUSTERS: dict[str, int] = {"constant": 3}


# ---------------------------------------------------------------------------
# Base config (old-API kwargs only)
# ---------------------------------------------------------------------------


@dataclass
class BaseConfig:
    """One algorithm configuration using old-API parameter names only."""

    id: str
    old_kwargs: dict = field(default_factory=dict)
    seed: int | None = None
    only_datasets: set[str] = field(default_factory=set)


CONFIGS: list[BaseConfig] = [
    BaseConfig(
        id="hierarchical_default",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
        },
    ),
    BaseConfig(
        id="hierarchical_mean",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "meanRepresentation",
        },
    ),
    BaseConfig(
        id="kmeans",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_means",
        },
        seed=42,
    ),
    BaseConfig(
        id="hierarchical_distribution",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "durationRepresentation",
        },
    ),
    BaseConfig(
        id="hierarchical_segmentation",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "segmentation": True,
            "noSegments": 8,
        },
    ),
    BaseConfig(
        id="hierarchical_no_rescale",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "rescaleClusterPeriods": False,
        },
    ),
    BaseConfig(
        id="contiguous",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "adjacent_periods",
        },
    ),
    BaseConfig(
        id="averaging",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "averaging",
        },
    ),
    BaseConfig(
        id="kmaxoids",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_maxoids",
        },
        seed=42,
    ),
    BaseConfig(
        id="kmedoids",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_medoids",
        },
        seed=42,
        only_datasets={"testdata"},
    ),
    BaseConfig(
        id="hierarchical_maxoid",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "maxoidRepresentation",
        },
    ),
    BaseConfig(
        id="hierarchical_distribution_minmax",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
        },
    ),
    BaseConfig(
        id="distribution_global",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
    ),
    BaseConfig(
        id="distribution_minmax_global",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
    ),
    BaseConfig(
        id="minmaxmean",
        old_kwargs={
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
    BaseConfig(
        id="hierarchical_weighted",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "weightDict": {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        },
        only_datasets={"testdata"},
    ),
    BaseConfig(
        id="segmentation_samemean",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "sameMean": True,
            "segmentation": True,
            "noSegments": 4,
        },
        only_datasets={"testdata"},
    ),
    BaseConfig(
        id="segmentation_distribution_global",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionRepresentation",
            "distributionPeriodWise": False,
            "segmentation": True,
            "noSegments": 4,
            "rescaleClusterPeriods": False,
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    BaseConfig(
        id="extremes_append",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    BaseConfig(
        id="extremes_replace",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "replace_cluster_center",
            "addPeakMax": ["Load"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    BaseConfig(
        id="extremes_new_cluster",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "new_cluster_center",
            "addPeakMax": ["Load"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    BaseConfig(
        id="extremes_min_value",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMin": ["T"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    BaseConfig(
        id="extremes_max_period",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMax": ["GHI"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    BaseConfig(
        id="extremes_min_period",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMin": ["Wind"],
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    BaseConfig(
        id="extremes_multi",
        old_kwargs={
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
    BaseConfig(
        id="extremes_with_segmentation",
        old_kwargs={
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
    BaseConfig(
        id="extremes_wide_multi",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["DE_Load"],
            "addPeakMin": ["FR_T"],
        },
        only_datasets={"wide"},
    ),
    BaseConfig(
        id="extremes_constant",
        old_kwargs={
            "noTypicalPeriods": 3,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["A"],
        },
        only_datasets={"constant"},
    ),
    BaseConfig(
        id="extremes_zero_column",
        old_kwargs={
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
# Resolved old-API case (config + dataset)
# ---------------------------------------------------------------------------


@dataclass
class OldCase:
    """Fully resolved old-API test case."""

    id: str
    dataset: str
    old_kwargs: dict
    seed: int | None = None


def build_old_cases(configs: list[BaseConfig] | None = None) -> list[OldCase]:
    """Build cross-product of configs x datasets for old-API testing."""
    if configs is None:
        configs = CONFIGS
    cases: list[OldCase] = []
    for cfg in configs:
        eligible = cfg.only_datasets or set(DATASETS)
        for ds in sorted(eligible):
            if ds not in DATASETS:
                continue
            kw = dict(cfg.old_kwargs)
            n_override = SMALL_DATASET_CLUSTERS.get(ds)
            if n_override is not None:
                kw["noTypicalPeriods"] = n_override
            cases.append(
                OldCase(
                    id=f"{cfg.id}/{ds}",
                    dataset=ds,
                    old_kwargs=kw,
                    seed=cfg.seed,
                )
            )
    return cases


def case_ids(cases: list) -> list[str]:
    """Extract test IDs from a list of cases (any object with an ``id`` attr)."""
    return [c.id for c in cases]
