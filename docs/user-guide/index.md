# User Guide

The [Quickstart](../notebooks/quickstart.ipynb) covers the core workflow —
shrinking a time series to a few typical periods and checking the result. Once
you know it, these three notebooks cover everything else, grouped by what you're
trying to do.

## Make it smaller

**[Segmentation](../notebooks/segmentation.ipynb)** — the second reduction lever:
keep fewer time steps *within* each period by merging adjacent steps into a few
variable-length segments.

**[Sizing & tuning](../notebooks/sizing_tuning.ipynb)** — the accuracy-vs-size
trade-off across both levers (periods and segments), and how to let tsam search
for the combination that hits a target.

## Control what it preserves

**[Representations, methods & extremes](../notebooks/representations_methods.ipynb)**
— pick the clustering method, choose how each typical period is built (mean,
medoid, value distribution, per-step min/max), and force extreme periods to be
kept exactly so peaks survive aggregation.

## Put it to work

**[Optimization workflow](../notebooks/optimization_workflow.ipynb)** — the handoff
to a downstream model: the representatives, counts, and assignments your model
needs; weighting columns by importance; mapping results back with disaggregation;
and reusing a clustering across datasets.

---

For the methods and maths behind the pipeline, see
[Background](../background/index.md). For function- and class-level detail, see the
[API Reference](../api/index.md).
