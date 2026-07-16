# Tutorials

Learning-oriented lessons. Each one is a guided walk you can follow start to finish, on data small
enough to check by eye. If you want to *understand by doing*, start here; if you already know what
you need and want a recipe, go to the [how-to guides](../how-to/index.md).

## Start here

**[Your first aggregation](quickstart.ipynb)** — load a series, shrink six weeks of hourly data to a
handful of typical days, read the results a downstream model needs, and measure what the reduction
cost you. Assumes nothing.

## Going deeper

**[Comparing clustering methods](comparing_clustering_methods.ipynb)** — give six clustering methods
the same twelve days and the same budget, and watch them disagree. Explains *why* each method carves
the data differently, and what that means for the typical periods your model sees.

**[Choosing a method](choosing_a_method.ipynb)** — the four levers `aggregate()` sets for you, what
each one throws away, and how to read a configuration off what you are modelling. Includes the three
different ways to preserve a peak and why they are not interchangeable.

---

Once these make sense, the [how-to guides](../how-to/index.md) are the task-oriented recipes, and
[How aggregation works](../explanation/how-it-works/00_overview.ipynb) traces the whole pipeline by
hand on a six-day dataset.
