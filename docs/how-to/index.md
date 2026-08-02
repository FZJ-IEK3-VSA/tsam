# How-to guides

Task-focused recipes. They assume you already know what `aggregate()` does and which lever you
want — if you are still deciding, [Choosing a method](../tutorials/choosing_a_method.ipynb) walks
all four levers on one dataset. For *why* each method works, see the
[How aggregation works](../explanation/how-aggregation-works/00_overview.ipynb) explanation series.

## Pick a lever

`aggregate()` makes four decisions for you. Find the one you need to change:

| I want to… | Change | Page |
|---|---|---|
| just make it smaller, sensibly | nothing — the default is strong | [Aggregate a time series](how_to_aggregate.ipynb) |
| control **which days group together** | `ClusterConfig(method=…)` | [Clustering methods](clustering_methods.ipynb) |
| control **what each group becomes** | `ClusterConfig(representation=…)` | [Representations](representations.ipynb) |
| keep **one specific period** exactly | `extremes=ExtremeConfig(…)` | [Extreme periods](extreme_periods.ipynb) |
| cut detail **inside** each period | `segments=SegmentConfig(…)` | [Segmentation](segmentation.ipynb) |
| hit a **target size** | let tsam search both levers | [How small can you go?](tuning.ipynb) |
| keep **calendar order** | `method="contiguous"` | [Clustering methods](clustering_methods.ipynb) |
| preserve the **duration curve** | `representation="distribution"` | [Representations](representations.ipynb) |
| make it **finish faster** | `method=…`, `period_duration=…` | [How long will this take?](runtime.ipynb) |

## Start here

**[Aggregate a time series](how_to_aggregate.ipynb)** — the core recipe: load your data, call
`aggregate()` with the two parameters that matter, and read the outputs a downstream model needs.
Everything below is a variation on this.

## Make it smaller

**[Segmentation](segmentation.ipynb)** — keep fewer timesteps *within* each period by merging
adjacent steps into variable-length segments.

**[How small can you go?](tuning.ipynb)** — the accuracy-vs-size trade-off across both levers, and
how to let tsam search for the best combination at a target size.

## Control what it preserves

**[Clustering methods](clustering_methods.ipynb)** — how periods are grouped: hierarchical, k-means,
k-medoids, k-maxoids, averaging, and contiguous.

**[Representations](representations.ipynb)** — how each cluster becomes one profile: mean, medoid,
the value distribution, or per-step min/max.

**[Extreme periods](extreme_periods.ipynb)** — force the peak (or trough) day to be kept exactly.

## Make it fast

**[How long will this take?](runtime.ipynb)** — what each clustering method costs to run, how that
grows with dataset size, and the one parameter that moves it most.

## Put it to work

**[Working with typical periods](working_with_typical_periods.ipynb)** — read the links between
typical days and the original calendar, map model results back with `disaggregate()`, and feed
inter-period storage formulations.

**[Optimization workflow](optimization_workflow.ipynb)** — the full handoff to a downstream model:
representatives, counts, assignments, column weights, and reusing a clustering across datasets.

---

For the methods behind the pipeline, see [How aggregation works](../explanation/how-aggregation-works/00_overview.ipynb);
for the equations, [Notation and equations](../reference/notation.md); and for function- and
class-level detail, the [API Reference](../reference/api/index.md).
