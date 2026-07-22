"""Benchmark the concurrency-preserving duration representation strategies.

For each ordering strategy supported by the distribution representation, this
aggregates the shared example dataset and reports two competing error axes:

- distribution error: how well each attribute's marginal (duration curve) is
  preserved (``RMSE_duration``, lower is better);
- concurrency error: how well the joint structure across attributes is
  preserved (``corr_frobenius`` / ``spearman_frobenius``, lower is better).

A good concurrency-preserving method lowers the concurrency error relative to
``independent`` without materially worsening the distribution error.

Run with::

    python benchmark/concurrency_representations.py
"""

import warnings
from pathlib import Path

import pandas as pd

import tsam.timeseriesaggregation as tsam

DATA = Path(__file__).resolve().parent.parent / "docs" / "notebooks" / "testdata.csv"

# (label, kwargs) for each strategy. "reference" needs an explicit attribute.
STRATEGIES = [
    ("independent", {"representationConcurrencyMethod": "independent"}),
    (
        "reference(GHI)",
        {
            "representationConcurrencyMethod": "reference",
            "representationReferenceAttribute": "GHI",
        },
    ),
    ("medoid", {"representationConcurrencyMethod": "medoid"}),
    ("consensus", {"representationConcurrencyMethod": "consensus"}),
    ("assignment", {"representationConcurrencyMethod": "assignment"}),
]


def run(raw, extra):
    agg = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        representationMethod="distributionRepresentation",
        rescaleClusterPeriods=False,
        **extra,
    )
    agg.createTypicalPeriods()
    dist = agg.totalAccuracyIndicators()["RMSE_duration"]
    conc = agg.concurrencyIndicators()
    return dist, conc["corr_frobenius"], conc["spearman_frobenius"]


def main():
    raw = pd.read_csv(DATA, index_col=0)
    print(f"dataset: {DATA.name}  columns={list(raw.columns)}\n")
    header = f"{'strategy':16s} {'RMSE_duration':>14s} {'corr_frob':>12s} {'spearman_frob':>14s}"
    print(header)
    print("-" * len(header))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, extra in STRATEGIES:
            dist, corr_f, spear_f = run(raw, extra)
            print(f"{label:16s} {dist:14.6f} {corr_f:12.4f} {spear_f:14.4f}")
    print(
        "\nLower is better on every column. 'independent' is the distribution-"
        "optimal baseline with no concurrency preservation; the other rows trade "
        "a little distribution error for lower concurrency error."
    )


if __name__ == "__main__":
    main()
