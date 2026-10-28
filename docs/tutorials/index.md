# Tutorials

Tutorials are meant to teach you something about ETHOS.TSAM, by providing you with an example to try out TSAM functionality. Each example defines one and guides you to reach that goal. If you already know what
you need and want a recipe, go to the [how-to guides](../how-to/index.md).

## Start here

**[Your first aggregation](quickstart.ipynb)** — run an aggregation with `tsam.aggregate()`, read
its typical periods, their weights and its accuracy, reconstruct the original timeline to judge the
error, and trade size for accuracy.

**[Choosing a method](choosing_a_method.ipynb)** — name the four levers and what each one throws
away, choose a clustering method and a representation independently, decide between the three ways
to preserve a peak, and read a configuration off what you are modelling.

## Going deeper

**[Comparing clustering methods](comparing_clustering_methods.ipynb)** — predict which methods group
days by value and which by calendar position, explain why they split the same days differently, and
weigh what each grouping costs in accuracy and runtime.

**[Comparing representations](comparing_representations.ipynb)** — tell the rules that select a
whole observed day (`medoid`, `maxoid`) from those that calculate each attribute at each timestep
separately (`mean`, `minmax_mean`), and predict what each one keeps of the cluster's means,
extremes and paths.

**[Duration representations](duration_representations.ipynb)** — treat a distribution
representation as two separate choices (fit each attribute's duration curve, then order the fitted
values in time), and choose its scope, min/max preservation and concurrency ordering by measuring
what each one changes.

---

Once these make sense, the [how-to guides](../how-to/index.md) are the task-oriented recipes, and
[How aggregation works](../explanation/how-aggregation-works/00_overview.ipynb) traces the whole pipeline by
hand on a six-day dataset.
