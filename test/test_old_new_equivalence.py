"""Parametrized equivalence tests: old TimeSeriesAggregation vs new tsam.aggregate().

Each test case is an EquivalenceCase dataclass — to add a new scenario, append
to the CASES list.  To add a new dataset, add an entry to DATA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as old_tsam
from conftest import TEST_DATA_DIR, TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, SegmentConfig, aggregate

# ---------------------------------------------------------------------------
# Data loaders — add an entry here to introduce a new dataset
# ---------------------------------------------------------------------------

OPSD_CSV = TEST_DATA_DIR / "opsd_germany_2019.csv"

DATA: dict[str, callable] = {
    "testdata": lambda: pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True),
    "opsd": lambda: pd.read_csv(OPSD_CSV, index_col=0, parse_dates=True),
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


# ---------------------------------------------------------------------------
# Test-case definition
# ---------------------------------------------------------------------------


@dataclass
class EquivalenceCase:
    """One old-vs-new equivalence scenario.

    Parameters
    ----------
    id : str
        Pytest ID shown in ``-v`` output.
    old_kwargs : dict
        ``TimeSeriesAggregation`` constructor kwargs (excluding ``timeSeries``).
    new_kwargs : dict
        ``aggregate()`` kwargs (excluding ``data``).
    seed : int | None
        If set, ``np.random.seed(seed)`` is called before *each* API call
        so stochastic methods (kmeans, kmaxoids) are reproducible.
    dataset : str
        Key into ``DATA``; defaults to ``"testdata"``.
    rtol : float
        Relative tolerance for accuracy comparisons (RMSE/MAE).
    """

    id: str
    old_kwargs: dict = field(default_factory=dict)
    new_kwargs: dict = field(default_factory=dict)
    seed: int | None = None
    dataset: str = "testdata"
    rtol: float = 1e-10


# ---------------------------------------------------------------------------
# Cases — append new entries to extend coverage
# ---------------------------------------------------------------------------

CASES: list[EquivalenceCase] = [
    EquivalenceCase(
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
    EquivalenceCase(
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
    EquivalenceCase(
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
    EquivalenceCase(
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
    EquivalenceCase(
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
    EquivalenceCase(
        id="hierarchical_extremes_append",
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
    ),
    EquivalenceCase(
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
    EquivalenceCase(
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
    ),
    # --- OPSD dataset (12 columns, 91 days) ---
    EquivalenceCase(
        id="opsd_hierarchical",
        dataset="opsd",
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
    EquivalenceCase(
        id="opsd_distribution",
        dataset="opsd",
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
    EquivalenceCase(
        id="opsd_segmentation",
        dataset="opsd",
        old_kwargs={
            "noTypicalPeriods": 8,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "segmentation": True,
            "noSegments": 6,
        },
        new_kwargs={
            "n_clusters": 8,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "segments": SegmentConfig(n_segments=6),
        },
    ),
    # --- Constant data (edge case) ---
    EquivalenceCase(
        id="constant_no_rescale",
        dataset="constant",
        old_kwargs={
            "noTypicalPeriods": 3,
            "hoursPerPeriod": 24,
            "clusterMethod": "hierarchical",
            "rescaleClusterPeriods": False,
        },
        new_kwargs={
            "n_clusters": 3,
            "period_duration": 24,
            "cluster": ClusterConfig(method="hierarchical"),
            "preserve_column_means": False,
        },
    ),
    # --- Data with a zero column (edge case) ---
    EquivalenceCase(
        id="zero_column_no_rescale",
        dataset="with_zero_column",
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
]


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


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Cache loaded datasets so each is read at most once per process.
_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _get_data(name: str) -> pd.DataFrame:
    if name not in _DATA_CACHE:
        _DATA_CACHE[name] = DATA[name]()
    return _DATA_CACHE[name]


def _case_ids(cases):
    return [c.id for c in cases]


# ---------------------------------------------------------------------------
# Parametrized test class
# ---------------------------------------------------------------------------


class TestOldNewEquivalence:
    """Parametrized comparison of old and new API across all CASES."""

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
