"""End-to-end workflows, written the way a caller would write them.

`test_bench.py` isolates one dimension per benchmark, which is what makes it
plottable but also means no single number there answers "how long does my job
take". These benchmarks do the opposite: each one is a whole task, spelled out
in full — loading its own data, building its own config, and reading back the
fields that task uses. Nothing is shared between them on purpose. A workflow
that quietly reuses a fixture is no longer a description of what a caller does.

Each workflow runs over the configurations callers actually reach for, not one
arbitrary pick, because they exercise different code: ``medoid`` selects a real
period, ``mean`` averages one, ``distribution`` fits duration curves, and
``kmeans`` spends its time inside scikit-learn instead of tsam. A change that
helps one can easily miss the others.

Two things these catch that the dimensional suite cannot:

- **Lazy work.** ``accuracy`` is a cached property and ``disaggregate()`` is
  called by the caller, so an `aggregate()`-only benchmark charges for
  neither. A workflow that reports metrics or expands an optimization result
  pays for both.
- **Per-call overhead.** Tuning sweeps and scenario batches call
  ``aggregate()`` tens of times on small inputs, where fixed overhead
  dominates and a win on one large call may not show up at all.

Usage::

    pytest benchmarks/test_workflows.py --benchmark-only
    pytest benchmarks/test_workflows.py --benchmark-only --large
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsam import (
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    SegmentConfig,
    aggregate,
)

ROOT = Path(__file__).resolve().parent.parent
FINE_CSV = ROOT / "benchmarks" / "data" / "fine_multiregional.csv.gz"
WIDE_CSV = ROOT / "test" / "data" / "wide.csv"
TESTDATA_CSV = ROOT / "docs" / "data" / "testdata.csv"

HEAVY = {"rounds": 3, "iterations": 1, "warmup_rounds": 1}
LIGHT = {"rounds": 10, "iterations": 1, "warmup_rounds": 1}

#: The four setups that cover what callers configure in practice: the library
#: default, the classic k-means run, FINE's duration-curve-with-segments, and
#: a peak-preserving model run. Used as a scalar parametrize value so
#: ``benchmem plot --x setup`` slices across them.
SETUPS = ["hierarchical_medoid", "kmeans_mean", "duration_segments", "mean_extremes"]


def _setup_kwargs(setup: str, peak_columns: list[str]) -> dict:
    """Spell out one named setup as ``aggregate()`` keyword arguments."""
    if setup == "hierarchical_medoid":
        return {
            "cluster": ClusterConfig(method="hierarchical", representation="medoid")
        }
    if setup == "kmeans_mean":
        return {"cluster": ClusterConfig(method="kmeans", representation="mean")}
    if setup == "duration_segments":
        return {
            "cluster": ClusterConfig(
                method="hierarchical", representation="distribution"
            ),
            "segments": SegmentConfig(n_segments=12),
            "preserve_column_means": False,
        }
    return {
        "cluster": ClusterConfig(method="hierarchical", representation="mean"),
        "extremes": ExtremeConfig(method="append", max_value=peak_columns),
    }


@pytest.mark.benchmark(group="workflow")
@pytest.mark.parametrize("setup", SETUPS)
def test_workflow_energy_system_model(benchmark, setup):
    """Aggregate one region's year for an energy-system model, and read it back.

    ``duration_segments`` is FINE's ``aggregateTemporally()`` on its own
    multi-regional example: 40 typical days, hierarchical clustering with
    duration representation, 12 segments per day, no rescaling. The model then
    wants the typical-period values, the segment durations, and which day maps
    to which typical day — and nothing else. It never asks for the accuracy
    indicators, and never reads the reconstructed series back.
    """
    demand = pd.read_csv(FINE_CSV, index_col=0, parse_dates=True)
    peak_columns = [c for c in demand.columns if c.startswith("ElecDemand")]
    kwargs = _setup_kwargs(setup, peak_columns)

    def run():
        result = aggregate(demand, 40, **kwargs)
        typical_periods = result.cluster_representatives
        return typical_periods.to_dict(), list(result.cluster_assignments)

    benchmark.pedantic(run, **HEAVY)


@pytest.mark.large
@pytest.mark.benchmark(group="workflow")
def test_workflow_energy_system_model_production(benchmark):
    """The same model at production scale: eight regions, two years, 400 profiles.

    A multi-region run also pins each region's peak-demand day as its own
    typical period, and switches the duration representation to global scope
    so the fitted curve is shared across periods.
    """
    one_region = pd.read_csv(FINE_CSV, index_col=0, parse_dates=True)
    all_regions = pd.concat(
        [one_region.add_suffix(f"_{region}") for region in range(10)], axis=1
    )
    demand = pd.concat([all_regions] * 2, ignore_index=True)
    demand.index = pd.date_range("2020-01-01", periods=len(demand), freq="h")
    peak_columns = [c for c in demand.columns if c.startswith("ElecDemand")]

    benchmark.extra_info["n_timesteps"] = len(demand)
    benchmark.extra_info["n_columns"] = demand.shape[1]

    def run():
        result = aggregate(
            demand,
            40,
            cluster=ClusterConfig(
                method="hierarchical", representation=Distribution(scope="global")
            ),
            segments=SegmentConfig(n_segments=12),
            extremes=ExtremeConfig(method="append", max_value=peak_columns),
            preserve_column_means=False,
        )
        typical_periods = result.cluster_representatives
        segment_durations = typical_periods.index.get_level_values("Segment Duration")
        return (
            typical_periods.to_dict(),
            list(segment_durations),
            list(result.cluster_assignments),
        )

    benchmark.pedantic(run, **HEAVY)


@pytest.mark.benchmark(group="workflow")
@pytest.mark.parametrize("resolution", ["typical_periods", "segments"])
def test_workflow_optimization_roundtrip(benchmark, resolution):
    """Cluster once, solve on the typical periods, expand every variable back.

    The shape of an optimization run: the solver returns one value per
    (typical period, timestep) for each of its variables, and every one of
    them has to be mapped back onto the original time index before the result
    means anything. Only the expansion repeats per variable. Segmented results
    expand through a different path than unsegmented ones, so both run here.
    """
    profiles = pd.read_csv(WIDE_CSV, index_col=0, parse_dates=True)
    profiles = pd.concat([profiles.add_suffix(f"_{i}") for i in range(5)], axis=1).iloc[
        :, :20
    ]
    segments = SegmentConfig(n_segments=12) if resolution == "segments" else None
    n_variables = 50
    benchmark.extra_info["n_variables"] = n_variables

    def run():
        result = aggregate(profiles, 36, segments=segments)
        solved_variables = result.cluster_representatives
        return [result.disaggregate(solved_variables) for _ in range(n_variables)]

    benchmark.pedantic(run, **HEAVY)


@pytest.mark.benchmark(group="workflow")
@pytest.mark.parametrize("setup", ["hierarchical_medoid", "kmeans_mean"])
def test_workflow_accuracy_report(benchmark, setup):
    """Aggregate a wide dataset and report how much accuracy it cost.

    What someone does when choosing between configurations: run one, look at
    the per-attribute error. ``accuracy`` is lazy, so this is the only
    workflow that pays for it on a single call.
    """
    profiles = pd.read_csv(WIDE_CSV, index_col=0, parse_dates=True)
    profiles = pd.concat(
        [profiles.add_suffix(f"_{i}") for i in range(12)], axis=1
    ).iloc[:, :48]
    kwargs = _setup_kwargs(setup, [])

    def run():
        result = aggregate(profiles, 8, **kwargs)
        accuracy = result.accuracy
        return accuracy.rmse, accuracy.mae, accuracy.rmse_duration

    benchmark.pedantic(run, **LIGHT)


@pytest.mark.benchmark(group="workflow")
@pytest.mark.parametrize("setup", ["hierarchical_medoid", "kmeans_mean"])
def test_workflow_scenario_batch(benchmark, setup):
    """Aggregate eight scenario years in one job, scoring each.

    Batch jobs multiply per-call overhead by the number of scenarios instead
    of amortizing it, so a fixed cost shows up here that a single large
    aggregation would hide.
    """
    profiles = pd.read_csv(WIDE_CSV, index_col=0, parse_dates=True)
    wide = pd.concat([profiles.add_suffix(f"_{i}") for i in range(16)], axis=1)
    scenarios = [wide.iloc[:, i * 8 : (i + 1) * 8] for i in range(8)]
    kwargs = _setup_kwargs(setup, [])
    benchmark.extra_info["n_scenarios"] = len(scenarios)

    def run():
        return [
            aggregate(scenario, 8, **kwargs).accuracy.rmse.mean()
            for scenario in scenarios
        ]

    benchmark.pedantic(run, **HEAVY)


@pytest.mark.benchmark(group="workflow")
@pytest.mark.parametrize("setup", ["hierarchical_medoid", "duration_segments"])
def test_workflow_tuning_sweep(benchmark, setup):
    """Search for the smallest typical-period count that still fits the data.

    Twelve candidate cluster counts, each scored by its duration-curve error —
    the loop ``tsam.tuning`` runs internally, written out so it behaves
    identically on every version being compared.
    """
    measurements = pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)
    candidates = [4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 60]
    kwargs = _setup_kwargs(setup, [])
    benchmark.extra_info["n_configs"] = len(candidates)

    def run():
        scored = []
        for n_clusters in candidates:
            result = aggregate(measurements, n_clusters, **kwargs)
            scored.append((n_clusters, float(result.accuracy.rmse_duration.mean())))
        return min(scored, key=lambda candidate: candidate[1])

    benchmark.pedantic(run, **HEAVY)
