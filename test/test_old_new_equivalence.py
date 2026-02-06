"""Parametrized equivalence tests: old TimeSeriesAggregation vs new tsam.aggregate().

Every ``Config`` is tested against every dataset (cross-product).
To extend coverage add a Config to ``CONFIGS``, a dataset to ``DATASETS``,
or a comparison method to ``TestOldNewEquivalence``.

Configs that reference specific column names list the datasets they are
compatible with in ``only_datasets``.  Generic configs leave it empty and
run everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as old_tsam
from conftest import TEST_DATA_DIR, TESTDATA_CSV
from tsam import (
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    SegmentConfig,
    aggregate,
)

# ---------------------------------------------------------------------------
# Datasets — add an entry here to introduce a new dataset
# ---------------------------------------------------------------------------

WIDE_CSV = TEST_DATA_DIR / "wide.csv"

DATASETS: dict[str, callable] = {
    "testdata": lambda: pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True),
    "wide": lambda: pd.read_csv(WIDE_CSV, index_col=0, parse_dates=True),
    "constant": lambda: _make_constant(),
    "with_zero_column": lambda: _make_with_zero_column(),
}


def _make_constant() -> pd.DataFrame:
    """Two constant columns, 10 days of hourly data."""
    idx = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
    return pd.DataFrame({"A": 42.0, "B": 7.0}, index=idx)


def _make_with_zero_column() -> pd.DataFrame:
    """testdata with an extra all-zero column."""
    df = pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)
    df["Zero"] = 0.0
    return df


# Cache so each dataset is loaded at most once per process.
_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _get_data(name: str) -> pd.DataFrame:
    if name not in _DATA_CACHE:
        _DATA_CACHE[name] = DATASETS[name]()
    return _DATA_CACHE[name]


# ---------------------------------------------------------------------------
# Config definition
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """One algorithm configuration to test across datasets.

    Parameters
    ----------
    id : str
        Short label (combined with dataset name for the pytest ID).
    old_kwargs : dict
        ``TimeSeriesAggregation`` constructor kwargs (excluding ``timeSeries``).
    new_kwargs : dict
        ``aggregate()`` kwargs (excluding ``data``).
    seed : int | None
        Seed for stochastic methods.
    rtol : float
        Relative tolerance for accuracy comparisons.
    only_datasets : set[str]
        If non-empty, only run with these datasets (for column-specific configs).
    n_clusters_override : dict[str, int]
        Per-dataset override for ``noTypicalPeriods`` / ``n_clusters``
        (e.g. ``{"constant": 3}`` because it only has 10 periods).
    """

    id: str
    old_kwargs: dict = field(default_factory=dict)
    new_kwargs: dict = field(default_factory=dict)
    seed: int | None = None
    rtol: float = 1e-10
    only_datasets: set[str] = field(default_factory=set)
    n_clusters_override: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configs — append new entries to extend coverage
# ---------------------------------------------------------------------------

CONFIGS: list[Config] = [
    Config(
        id="hierarchical_default",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
        },
    ),
    Config(
        id="hierarchical_mean",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "meanRepresentation",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical", representation="mean"),
        },
    ),
    Config(
        id="kmeans",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_means",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="kmeans"),
        },
        seed=42,
        rtol=1e-5,
    ),
    Config(
        id="hierarchical_distribution",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "durationRepresentation",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(
                method="hierarchical", representation="distribution"
            ),
        },
    ),
    Config(
        id="hierarchical_segmentation",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "segmentation": True,
            "noSegments": 8,
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "segments": SegmentConfig(n_segments=8),
        },
    ),
    Config(
        id="hierarchical_no_rescale",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "rescaleClusterPeriods": False,
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "preserve_column_means": False,
        },
    ),
    # --- More cluster methods ---
    Config(
        id="contiguous",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "adjacent_periods",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="contiguous"),
        },
    ),
    Config(
        id="averaging",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "averaging",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="averaging"),
        },
    ),
    Config(
        id="kmaxoids",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_maxoids",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="kmaxoids"),
        },
        seed=42,
        rtol=1e-5,
    ),
    Config(
        id="kmedoids",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "k_medoids",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="kmedoids"),
        },
        seed=42,
        # kmedoids is slow (~30s), only run on testdata
        only_datasets={"testdata"},
    ),
    # --- More representations ---
    Config(
        id="hierarchical_maxoid",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "maxoidRepresentation",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical", representation="maxoid"),
        },
    ),
    Config(
        id="hierarchical_distribution_minmax",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(
                method="hierarchical", representation="distribution_minmax"
            ),
        },
    ),
    Config(
        id="distribution_global",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(
                method="hierarchical",
                representation=Distribution(scope="global"),
            ),
            "preserve_column_means": False,
        },
    ),
    Config(
        id="distribution_minmax_global",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "representationMethod": "distributionAndMinMaxRepresentation",
            "distributionPeriodWise": False,
            "rescaleClusterPeriods": False,
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(
                method="hierarchical",
                representation=Distribution(scope="global", preserve_minmax=True),
            ),
            "preserve_column_means": False,
        },
    ),
    # --- Column-specific configs (only compatible datasets) ---
    Config(
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
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(
                method="hierarchical",
                representation=MinMaxMean(
                    max_columns=["GHI"], min_columns=["T", "Load"]
                ),
            ),
            "preserve_column_means": False,
        },
        only_datasets={"testdata"},
    ),
    Config(
        id="hierarchical_weighted",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "weightDict": {"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(
                method="hierarchical",
                weights={"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
            ),
        },
        only_datasets={"testdata"},
    ),
    # --- Extreme period configs ---
    Config(
        id="extremes_append",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Load"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append", max_value=["Load"], preserve_n_clusters=False
            ),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Config(
        id="extremes_replace",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "replace_cluster_center",
            "addPeakMax": ["Load"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="replace", max_value=["Load"], preserve_n_clusters=False
            ),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Config(
        id="extremes_new_cluster",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "new_cluster_center",
            "addPeakMax": ["Load"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="new_cluster", max_value=["Load"], preserve_n_clusters=False
            ),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Config(
        id="extremes_min_value",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMin": ["T"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append", min_value=["T"], preserve_n_clusters=False
            ),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Config(
        id="extremes_max_period",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMax": ["GHI"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append", max_period=["GHI"], preserve_n_clusters=False
            ),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Config(
        id="extremes_min_period",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addMeanMin": ["Wind"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append", min_period=["Wind"], preserve_n_clusters=False
            ),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Config(
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
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append",
                max_value=["Load"],
                min_value=["T"],
                max_period=["GHI"],
                preserve_n_clusters=False,
            ),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    Config(
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
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append", max_value=["Load"], preserve_n_clusters=False
            ),
            "segments": SegmentConfig(n_segments=6),
        },
        only_datasets={"testdata", "with_zero_column"},
    ),
    # --- Extremes on wide dataset (different column names) ---
    Config(
        id="extremes_wide_multi",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["DE_Load"],
            "addPeakMin": ["FR_T"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append",
                max_value=["DE_Load"],
                min_value=["FR_T"],
                preserve_n_clusters=False,
            ),
        },
        only_datasets={"wide"},
    ),
    # --- Extremes on constant data (all periods identical) ---
    Config(
        id="extremes_constant",
        old_kwargs={
            "noTypicalPeriods": 3,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["A"],
        },
        new_kwargs={
            "n_clusters": 3,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append", max_value=["A"], preserve_n_clusters=False
            ),
        },
        only_datasets={"constant"},
    ),
    # --- Extremes on zero column ---
    Config(
        id="extremes_zero_column",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "extremePeriodMethod": "append",
            "addPeakMax": ["Zero"],
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "extremes": ExtremeConfig(
                method="append", max_value=["Zero"], preserve_n_clusters=False
            ),
        },
        only_datasets={"with_zero_column"},
    ),
]

# Default n_clusters overrides for small datasets.
_SMALL_DATASET_CLUSTERS = {"constant": 3}

# ---------------------------------------------------------------------------
# Build cross-product: Config x Dataset → EquivalenceCase
# ---------------------------------------------------------------------------


@dataclass
class EquivalenceCase:
    """Fully resolved test case (config + dataset)."""

    id: str
    dataset: str
    old_kwargs: dict
    new_kwargs: dict
    seed: int | None = None
    rtol: float = 1e-10


def _build_cases() -> list[EquivalenceCase]:
    cases: list[EquivalenceCase] = []
    for cfg in CONFIGS:
        eligible = cfg.only_datasets or set(DATASETS)
        for ds in eligible:
            # Apply n_clusters override for small datasets
            n_override = cfg.n_clusters_override.get(
                ds, _SMALL_DATASET_CLUSTERS.get(ds)
            )

            old_kw = dict(cfg.old_kwargs)
            new_kw = dict(cfg.new_kwargs)

            if n_override is not None:
                old_kw["noTypicalPeriods"] = n_override
                new_kw = {**new_kw, "n_clusters": n_override}

            cases.append(
                EquivalenceCase(
                    id=f"{cfg.id}/{ds}",
                    dataset=ds,
                    old_kwargs=old_kw,
                    new_kwargs=new_kw,
                    seed=cfg.seed,
                    rtol=cfg.rtol,
                )
            )
    return cases


CASES = _build_cases()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_old(data: pd.DataFrame, case: EquivalenceCase):
    """Run old API and return (result_df, agg_object)."""
    if case.seed is not None:
        np.random.seed(case.seed)
    agg = old_tsam.TimeSeriesAggregation(timeSeries=data, **case.old_kwargs)
    result = agg.createTypicalPeriods()
    return result, agg


def _run_new(data: pd.DataFrame, case: EquivalenceCase):
    """Run new API and return AggregationResult."""
    if case.seed is not None:
        np.random.seed(case.seed)
    return aggregate(data, **case.new_kwargs)


def _case_ids(cases):
    return [c.id for c in cases]


# ---------------------------------------------------------------------------
# Parametrized test class
# ---------------------------------------------------------------------------


class TestOldNewEquivalence:
    """Parametrized comparison of old and new API across all configs x datasets."""

    @pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
    def test_cluster_representatives(self, case: EquivalenceCase):
        """Typical-period DataFrames must be equal."""
        data = _get_data(case.dataset)
        old_result, _ = _run_old(data, case)
        new_result = _run_new(data, case)

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    @pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
    def test_cluster_assignments(self, case: EquivalenceCase):
        """Cluster order arrays must match."""
        data = _get_data(case.dataset)
        _, old_agg = _run_old(data, case)
        new_result = _run_new(data, case)

        np.testing.assert_array_equal(
            old_agg.clusterOrder,
            new_result.cluster_assignments,
        )

    @pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
    def test_accuracy(self, case: EquivalenceCase):
        """RMSE and MAE must match within tolerance."""
        data = _get_data(case.dataset)
        _, old_agg = _run_old(data, case)
        new_result = _run_new(data, case)

        old_acc = old_agg.accuracyIndicators()

        np.testing.assert_allclose(
            old_acc["RMSE"].values,
            new_result.accuracy.rmse.values,
            rtol=case.rtol,
        )
        np.testing.assert_allclose(
            old_acc["MAE"].values,
            new_result.accuracy.mae.values,
            rtol=case.rtol,
        )

    @pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
    def test_reconstruction(self, case: EquivalenceCase):
        """Reconstructed time series must match."""
        data = _get_data(case.dataset)
        _, old_agg = _run_old(data, case)
        new_result = _run_new(data, case)

        old_reconstructed = old_agg.predictOriginalData()

        pd.testing.assert_frame_equal(
            old_reconstructed,
            new_result.reconstructed,
            check_names=False,
        )
