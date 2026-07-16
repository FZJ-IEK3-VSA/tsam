# How-to guides

Task-focused recipes for getting things done with tsam, assuming you have skimmed the
[tutorial](../tutorials/quickstart.ipynb). For *why* each method works, see the
[How aggregation works](../explanation/how-it-works/00_overview.ipynb) explanation series.

## Start here

**[How to aggregate a time series](how_to_aggregate.ipynb)** — the core
recipe: load your data, call `aggregate()` with the two parameters that matter, and read the
outputs a downstream model needs. Everything below is a variation on this.

## Make it smaller

**[Segmentation](segmentation.ipynb)** — the second reduction lever: keep
fewer time steps *within* each period by merging adjacent steps into a few variable-length
segments.

**[How small can you go?](tuning.ipynb)** — the accuracy-vs-size trade-off
across both levers (periods and segments), and how to let tsam search for the best
combination at a target size.

## Control what it preserves

**[Clustering methods](clustering_methods.ipynb)** — how periods are
grouped: hierarchical, k-means, k-medoids, k-maxoids, averaging, and contiguous, and when to
pick each.

**[Representations](representations.ipynb)** — how each cluster becomes one
profile: mean, medoid, the value distribution, or per-step min/max, and why the choice changes
which features survive.

**[Extreme periods](extreme_periods.ipynb)** — force the peak (or trough)
day to be kept exactly, so it is not averaged away.

## Put it to work

**[Working with typical periods](working_with_typical_periods.ipynb)** —
read the links between typical days and the original calendar, map model results back with
`disaggregate()`, and see how the ordered assignments feed inter-period storage formulations.

**[Optimization workflow](optimization_workflow.ipynb)** — the full handoff
to a downstream model: the representatives, counts, and assignments your model needs, weighting
columns by importance, mapping results back with disaggregation, and reusing a clustering
across datasets.

---

For the methods and maths behind the pipeline, see [Background](../explanation/background/index.md). For
function- and class-level detail, see the [API Reference](../reference/api/index.md).
