# System Context

The diagram below shows ETHOS.TSAM in its environment: who uses it, what it touches, and what it produces.

![System context diagram](../../assets/architecture/context_diagram.svg)

ETHOS.TSAM is a Python library used directly by an **energy-system modeler or data scientist**. The user loads time series data into a `pandas.DataFrame`, calls `tsam.aggregate()`, and receives a set of typical periods in return. tsam does not read files, write files, or call any downstream framework — all of that is the user's responsibility.

The **downstream optimization framework** (e.g. [ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE), [flixopt](https://github.com/flixOpt/flixopt), [oemof](https://oemof.org/)) consumes the aggregated result, but tsam has no knowledge of it.

The only optional external dependency is a **MILP solver**, required when `ClusterConfig(method="kmedoids")` is used. All other clustering methods run without an external solver.
