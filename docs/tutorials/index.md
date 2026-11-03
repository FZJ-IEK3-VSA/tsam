# Tutorials

Tutorials teach tsam by doing. Each one sets a learning goal, works through one example step by
step, and ends with what you have learned. Run the code as you read: every result the text
describes is one you can reproduce. If you just want instructions how to do certain tasks, go to [how-to guides](../how-to/index.md).

`aggregate()` makes several decisions for you, and each section below teaches one of them. Start
with [Your first aggregation](quickstart.ipynb), then follow the decision your model depends on.
Within a section, the first tutorial compares the options and the ones after it take one question
at a time. Every tutorial also works on its own.

## Getting started: how many typical days?

**[Your first aggregation](quickstart.ipynb)** — run an aggregation with `tsam.aggregate()`, read
its typical periods, their weights and its accuracy, reconstruct the original timeline to judge the
error, and trade size for accuracy: how many typical days, and how many timesteps in each.

## Clustering: which days belong together? {#clustering}

Set with `ClusterConfig(method=…)`.

**[Comparing clustering methods](comparing_clustering_methods.ipynb)** — follow six methods on
twelve designed days: see three of them group by similarity and agree, `kmaxoids` give the storm a
typical day of its own, and `averaging` and `contiguous` group days by the calendar.

**[Method or representation](method_or_representation.ipynb)** — compare `hierarchical`, `kmeans`
and `kmedoids` with `medoid` and with `mean` on two weeks to a full year of hourly data, and see
the representation move the fit far more than the method.

Two tutorials then show what the grouping means for a model, on the hourly example data:

**[1. As similar as possible](clustering/01_similar_days.ipynb)** — compare how close
`hierarchical`, `kmeans` and `kmedoids` come and how long they take, and watch the runtime of
`kmedoids` grow much faster than the data.

**[2. Seasonal storage](clustering/02_seasonal_storage.ipynb)** — size a seasonal storage from
typical days, see `kmeans` make it too small and `averaging` get it almost exactly right, and find
out why `contiguous` needs `representation="mean"`.

## Representations: what does each group become? {#representations}

Set with `ClusterConfig(representation=…)`.

**[Comparing representations](comparing_representations.ipynb)** — tell the rules that copy a
whole observed day (`medoid`, `maxoid`) from those that calculate each attribute at each timestep
(`mean`, `minmax_mean`) and from `Distribution()`, which sorts and averages all values, and
compare what each one keeps: the energy, the peak, a real day or the value distribution.

### Duration curves

A series on one representation. `Distribution()` keeps each attribute's **duration curve**: how
many hours each value is reached or exceeded. It works in two steps: it **fits the values** of
each typical day, then **orders them in time**. Each tutorial changes one setting on the same
two weeks of hourly data.

#### Fitting the values

**[1. Fitting a duration curve](duration-curves/01_fitting.ipynb)** — read a duration curve,
see the mean flatten it and `Distribution()` keep it, rebuild the fitted values by hand, and
measure what keeping the curve costs in hour-by-hour accuracy.

**[2. Local or global](duration-curves/02_scope.ipynb)** — fit each typical day to the days it
represents or all typical days to the whole series, and see how `scope="global"` moves sun from
one typical day to another.

**[3. Keeping the extremes](duration-curves/03_extremes.ipynb)** — see `Distribution()` miss the
real highest and lowest load, keep them with `preserve_minmax=True`, and count how often a kept
extreme then occurs.

#### Ordering them in time

**[4. Which values occur together](duration-curves/04_ordering.ipynb)** — check that all five
orderings keep the same values, see the ordering change the residual load, and tell the
orderings that change which values pair up from those that only move the pairs to other hours.

**[5. Making peaks coincide on purpose](duration-curves/05_coincident_peaks.ipynb)** — size a
grid connection shared by two districts, see the default typical days underestimate their
combined peak, get a conservative estimate from a shared ordering with `preserve_minmax=True`,
and choose the hour of that peak with `reference`.

## Extreme periods: which days must survive? {#extreme-periods}

Set with `extremes=ExtremeConfig(…)`.

**[1. Keeping the peak day](extreme-periods/01_peak_day.ipynb)** — size a grid connection from
typical days, see them miss the peak load and more typical days fail to bring it back, keep it
exactly with `ExtremeConfig`, and see segments flatten it again.

---

Once these make sense, the [how-to guides](../how-to/index.md) are the task-oriented recipes, and
[How aggregation works](../explanation/how-aggregation-works/00_overview.ipynb) traces the whole
pipeline by hand on a six-day dataset.
