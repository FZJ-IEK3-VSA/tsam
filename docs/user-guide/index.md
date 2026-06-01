# User Guide

The [Quickstart](../notebooks/quickstart.ipynb) covers the core workflow —
shrinking a time series to a few typical periods and checking the result. These
notebooks cover everything else, grouped by what you're trying to do.

## Make it smaller

**[Segmentation](../notebooks/segmentation.ipynb)** — the second reduction lever:
keep fewer time steps *within* each period by merging adjacent steps into a few
variable-length segments.

**[How small can you go?](../notebooks/tuning.ipynb)** — the accuracy-vs-size
trade-off across both levers (periods and segments), and how to let tsam search
for the best combination at a target size.

## Control what it preserves

**[Clustering methods](../notebooks/clustering_methods.ipynb)** — how periods are
grouped: hierarchical, k-means, k-medoids, k-maxoids, averaging, contiguous, and
when to pick each.

**[Representations](../notebooks/representations.ipynb)** — how each cluster
becomes one profile: mean, medoid, the value distribution, or per-step min/max —
and why the choice changes which features survive.

**[Extreme periods](../notebooks/extreme_periods.ipynb)** — force the peak (or
trough) day to be kept exactly, so it isn't averaged away.

## Put it to work

**[Optimization workflow](../notebooks/optimization_workflow.ipynb)** — the handoff
to a downstream model: the representatives, counts, and assignments your model
needs; weighting columns by importance; mapping results back with disaggregation;
and reusing a clustering across datasets.

---

For the methods and maths behind the pipeline, see
[Background](../background/index.md). For function- and class-level detail, see the
[API Reference](../api/index.md).
