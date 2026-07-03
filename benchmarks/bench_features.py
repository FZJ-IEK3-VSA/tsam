"""Feature/parameter-cost benchmarks for :func:`tsam.aggregate`.

Holds the data size fixed (a medium ``8760 x 12`` dataset, daily periods,
``k=8``) and varies one feature axis at a time — clustering method,
representation / concurrency strategy, extreme handling, segmentation, and the
various toggles — so the numbers show the *relative* cost of each feature rather
than data scaling (which lives in ``bench_scaling.py``).

Non-deterministic methods (``kmeans``, ``kmaxoids``) are seeded per run. The
exact-MILP ``kmedoids`` method is capped to a smaller series so it stays
tractable.

Run::

    pytest benchmarks/bench_features.py --benchmark-only --benchmark-memory \\
        --benchmark-group-by=name --benchmark-save=$(git rev-parse --short HEAD)
"""

from __future__ import annotations

import numpy as np
import pytest
from _data import make_data

import tsam
from tsam import (
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    SegmentConfig,
)

FEATURE_T = 8760
FEATURE_C = 12
FEATURE_W = 24
FEATURE_K = 8
KMEDOIDS_CAP = 2016  # exact MILP: keep the candidate count tractable


def _cases() -> list[tuple[str, dict]]:
    """Return ``(id, aggregate_kwargs)`` pairs, one per feature under test.

    An optional ``_cap`` key (popped before the call) shrinks the series for a
    case; everything else is forwarded to :func:`tsam.aggregate`.
    """
    cases: list[tuple[str, dict]] = []

    def add(case_id: str, **kwargs) -> None:
        cases.append((case_id, kwargs))

    # Clustering methods (representation kept at each method's natural default).
    add(
        "method=averaging",
        cluster=ClusterConfig(method="averaging", representation="mean"),
    )
    add("method=kmeans", cluster=ClusterConfig(method="kmeans", representation="mean"))
    add(
        "method=hierarchical",
        cluster=ClusterConfig(method="hierarchical", representation="medoid"),
    )
    add(
        "method=contiguous",
        cluster=ClusterConfig(method="contiguous", representation="medoid"),
    )
    add(
        "method=kmaxoids",
        cluster=ClusterConfig(method="kmaxoids", representation="maxoid"),
    )
    add(
        "method=kmedoids",
        cluster=ClusterConfig(
            method="kmedoids", representation="medoid", solver="highs"
        ),
        _cap=KMEDOIDS_CAP,
    )

    # Representation / concurrency strategies (hierarchical clustering fixed).
    add("rep=mean", cluster=ClusterConfig(representation="mean"))
    add("rep=medoid", cluster=ClusterConfig(representation="medoid"))
    add("rep=distribution", cluster=ClusterConfig(representation=Distribution()))
    add(
        "rep=distribution_global",
        cluster=ClusterConfig(representation=Distribution(scope="global")),
    )
    add(
        "rep=distribution_minmax",
        cluster=ClusterConfig(representation=Distribution(preserve_minmax=True)),
    )
    add(
        "rep=concurrency_consensus",
        cluster=ClusterConfig(representation=Distribution(concurrency="consensus")),
    )
    add(
        "rep=concurrency_assignment",
        cluster=ClusterConfig(representation=Distribution(concurrency="assignment")),
    )
    add(
        "rep=minmax_mean",
        cluster=ClusterConfig(
            representation=MinMaxMean(max_columns=["Load"], min_columns=["T"])
        ),
    )

    # Extreme-period handling.
    for method in ("append", "replace", "new_cluster"):
        add(
            f"extreme={method}",
            extremes=ExtremeConfig(method=method, max_value=["Load"], min_value=["T"]),
        )

    # Segmentation.
    add("segments=12", segments=SegmentConfig(n_segments=12))
    add("segments=6", segments=SegmentConfig(n_segments=6))

    # Toggles.
    add("duration_curves", cluster=ClusterConfig(use_duration_curves=True))
    add("period_sums", cluster=ClusterConfig(include_period_sums=True))
    add("weights", weights={"Load": 5.0, "GHI": 0.5})
    add("no_rescale", preserve_column_means=False)

    return cases


FEATURE_CASES = _cases()


@pytest.mark.parametrize("case", FEATURE_CASES, ids=[c[0] for c in FEATURE_CASES])
def test_features(benchmark, case):
    _, kwargs = case
    kwargs = dict(kwargs)
    n_rows = kwargs.pop("_cap", FEATURE_T)
    data = make_data(n_rows, FEATURE_C)

    benchmark.extra_info.update(
        {
            "n_rows": n_rows,
            "columns": FEATURE_C,
            "timesteps_per_period": FEATURE_W,
            "clusters": FEATURE_K,
        }
    )

    def run():
        np.random.seed(42)  # determinism for kmeans / kmaxoids
        result = tsam.aggregate(
            data, n_clusters=FEATURE_K, period_duration=FEATURE_W, **kwargs
        )
        _ = result.reconstructed

    benchmark(run)
