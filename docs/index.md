<div class="landing-logos">
  <div class="tsam-logo-light">
    <img src="assets/tsam-logo-light.svg" alt="ETHOS.TSAM">
  </div>
  <div class="tsam-logo-dark">
    <img src="assets/tsam-logo-dark.svg" alt="ETHOS.TSAM">
  </div>
  <div class="jsa-logo">
    <a href="https://www.fz-juelich.de/en/iek/iek-3">
      <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header.svg" alt="Jülich System Analysis">
    </a>
  </div>
</div>

# ETHOS.TSAM

**Time series aggregation for large optimization models — and any other time series.**

ETHOS.TSAM compresses long, high-resolution time series into a small set of
**typical periods** (e.g. 8 representative days standing in for a whole year) and,
optionally, into **coarser time steps** within each period. The result keeps the
statistical character of the original data while drastically cutting the number of
time steps a downstream model has to solve.

It was built to shrink the temporal complexity of energy system optimization
models, but it works on any time series — weather, prices, load, behaviour, or all
of them at once.

```python
import pandas as pd
import tsam

data = pd.read_csv("timeseries.csv", index_col=0, parse_dates=True)

result = tsam.aggregate(data, n_clusters=8, period_duration="1D")

result.cluster_representatives   # the 8 typical days
result.accuracy.rmse             # how well they match the original
result.plot.cluster_representatives()   # ...and a plot of them
```

[Install ETHOS.TSAM](installation.md){ .md-button .md-button--primary }
[Quickstart](notebooks/quickstart.ipynb){ .md-button }
[User Guide](user-guide/index.md){ .md-button }

## Main features

- **One function, many methods.** [`aggregate`][tsam.aggregate] is the single
  entry point. Choose between **k-means, k-medoids, k-maxoids, hierarchical,
  contiguous, and averaging** clustering — backed by scikit-learn or solved
  exactly with an MILP solver.
- **Two aggregation dimensions, freely combined.** Reduce the number of *periods*
  (typical days) and/or the *temporal resolution* within them, via
  [how small you can go](notebooks/tuning.ipynb).
- **Representations that preserve what matters.** Beyond means and medoids, keep
  the [value distribution](notebooks/representations.ipynb) (duration
  curve), per-time-step min/max, or force **extreme periods** so peaks survive
  aggregation.
- **Built-in evaluation and plotting.** Every result carries accuracy metrics and
  an interactive `.plot` accessor (Plotly) — see the
  [Quickstart](notebooks/quickstart.ipynb).
- **Automatic hyperparameter tuning.** Let tsam
  [find the period/segment combination](notebooks/tuning.ipynb) that hits a
  target data reduction, or map the full Pareto front.
- **Built for downstream models.** Hand the representatives, counts, and
  assignments to an optimization, then map results back — see the
  [optimization workflow](notebooks/optimization_workflow.ipynb).

## Where to start

| If you want to… | Go to |
|------------------|-------|
| Install the package | [Installation](installation.md) |
| Run your first aggregation | [Quickstart](notebooks/quickstart.ipynb) |
| Learn the features one by one | [User Guide](user-guide/index.md) |
| See an end-to-end optimization workflow | [Optimization workflow](notebooks/optimization_workflow.ipynb) |
| Look up a function or class | [API Reference](api/index.md) |
| Understand the methods and maths | [Background](background/index.md) |

## About

ETHOS.TSAM is open source and developed on
[GitHub](https://github.com/FZJ-IEK3-VSA/tsam) — contributions, questions, and
issues are welcome. It is part of the
[Energy Transformation PatHway Optimization Suite (ETHOS)](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services)
at ICE-2 and is tightly integrated with
[ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE).

If you use ETHOS.TSAM in scientific work, please
[cite one of our publications](further-reading.md).
