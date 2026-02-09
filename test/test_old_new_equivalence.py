"""Parametrized equivalence tests: old TimeSeriesAggregation vs new tsam.aggregate().

Every config is tested against every compatible dataset (cross-product).
To extend coverage add a ``BaseConfig`` to ``_configs.CONFIGS``, a dataset to
``_configs.DATASETS``, or a comparison method to ``TestOldNewEquivalence``.

Old-API kwargs live in ``_configs.py`` (shared with generate_golden.py and
benchmarks/bench.py).  New-API kwargs are defined here in ``_NEW_KWARGS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as old_tsam
from _configs import (
    CONFIGS as BASE_CONFIGS,
)
from _configs import (
    DATASETS,
    SMALL_DATASET_CLUSTERS,
    case_ids,
    get_data,
)
from tsam import (
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    SegmentConfig,
    aggregate,
)

# ---------------------------------------------------------------------------
# New-API kwargs for each config ID
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
        "cluster": ClusterConfig(
            method="hierarchical",
            weights={"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        ),
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
    # --- Cross-feature interactions ---
    "hierarchical_weighted_segmentation": {
        "n_clusters": 8,
        "period_duration": 24,
        "cluster": ClusterConfig(
            method="hierarchical",
            weights={"Load": 2.0, "GHI": 1.0, "T": 1.0, "Wind": 1.0},
        ),
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
}

_RTOL: dict[str, float] = {
    "kmeans": 1e-5,
    "kmaxoids": 1e-5,
    "kmeans_segmentation": 1e-5,
    "kmaxoids_segmentation": 1e-5,
    "kmeans_duration_curves": 1e-5,
    "kmeans_extremes_append": 1e-5,
    "kmeans_distribution": 1e-5,
}


# ---------------------------------------------------------------------------
# Build cross-product: BaseConfig x Dataset → EquivalenceCase
# ---------------------------------------------------------------------------


@dataclass
class EquivalenceCase:
    """Fully resolved test case (config + dataset) with both old and new kwargs."""

    id: str
    dataset: str
    old_kwargs: dict
    new_kwargs: dict
    seed: int | None = None
    rtol: float = 1e-10


def _build_cases() -> list[EquivalenceCase]:
    cases: list[EquivalenceCase] = []
    for base in BASE_CONFIGS:
        new_kw_base = _NEW_KWARGS[base.id]
        eligible = base.only_datasets or set(DATASETS)
        for ds in sorted(eligible):
            n_override = SMALL_DATASET_CLUSTERS.get(ds)

            old_kw = dict(base.old_kwargs)
            new_kw = dict(new_kw_base)

            if n_override is not None:
                old_kw["noTypicalPeriods"] = n_override
                new_kw["n_clusters"] = n_override

            cases.append(
                EquivalenceCase(
                    id=f"{base.id}/{ds}",
                    dataset=ds,
                    old_kwargs=old_kw,
                    new_kwargs=new_kw,
                    seed=base.seed,
                    rtol=_RTOL.get(base.id, 1e-10),
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


# ---------------------------------------------------------------------------
# Parametrized test class
# ---------------------------------------------------------------------------


class TestOldNewEquivalence:
    """Parametrized comparison of old and new API across all configs x datasets."""

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_cluster_representatives(self, case: EquivalenceCase):
        """Typical-period DataFrames must be equal."""
        data = get_data(case.dataset)
        old_result, _ = _run_old(data, case)
        new_result = _run_new(data, case)

        pd.testing.assert_frame_equal(
            old_result,
            new_result.cluster_representatives,
            check_names=False,
        )

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_cluster_assignments(self, case: EquivalenceCase):
        """Cluster order arrays must match."""
        data = get_data(case.dataset)
        _, old_agg = _run_old(data, case)
        new_result = _run_new(data, case)

        np.testing.assert_array_equal(
            old_agg.clusterOrder,
            new_result.cluster_assignments,
        )

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_accuracy(self, case: EquivalenceCase):
        """RMSE and MAE must match within tolerance."""
        data = get_data(case.dataset)
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

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_reconstruction(self, case: EquivalenceCase):
        """Reconstructed time series must match."""
        data = get_data(case.dataset)
        _, old_agg = _run_old(data, case)
        new_result = _run_new(data, case)

        old_reconstructed = old_agg.predictOriginalData()

        pd.testing.assert_frame_equal(
            old_reconstructed,
            new_result.reconstructed,
            check_names=False,
        )
