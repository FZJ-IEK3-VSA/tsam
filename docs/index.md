<p class="landing-logos">
  <img src="assets/tsam-logo-light.svg#only-light" alt="ETHOS.TSAM" class="hero-logo hero-logo--tsam">
  <img src="assets/tsam-logo-dark.svg#only-dark" alt="ETHOS.TSAM" class="hero-logo hero-logo--tsam">
  <a href="https://www.fz-juelich.de/en/ice/ice-2" class="hero-logo-link">
    <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header.svg#only-light" alt="Jülich Systems Analysis" class="hero-logo hero-logo--jsa">
    <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header-dark.svg#only-dark" alt="Jülich Systems Analysis" class="hero-logo hero-logo--jsa">
  </a>
</p>

# ETHOS.TSAM

**Time series aggregation for large optimization models — and any other time series.**

ETHOS.TSAM compresses long, high-resolution time series into a small set of
**typical periods** (e.g. 8 typical days standing in for a whole year) and,
optionally, into **coarser timesteps** within each period. The result preserves key statistical characteristics and recurring patterns of the original data as closely as possible, while substantially reducing the number of time steps.

It is designed to shrink the temporal complexity of energy system optimization
models, but it works on any time series — weather, prices, load, behavior, or all
of them at once.

## How this documentation is organized

It follows the [Diátaxis framework](https://diataxis.fr/), which splits documentation by what
you came for:

| Section | Use it to |
|---------|-----------|
| **[Tutorials](tutorials/quickstart.ipynb)** | Learn tsam by working through a complete example. |
| **[How-to guides](how-to/index.md)** | Get one specific task done. |
| **[Explanation](explanation/how-aggregation-works/00_overview.ipynb)** | Understand the algorithms, the architecture, and the decisions behind them. |
| **[Reference](reference/api/index.md)** | Look up the API, the notation, Glossary and implementation details. |

## Main features

- **One function, many methods.** [`aggregate`][tsam.aggregate] is the single
  entry point. Choose between **k-means, k-medoids, k-maxoids, hierarchical,
  contiguous, and averaging** clustering — backed by scikit-learn or solved
  exactly with an MILP solver.
- **Two aggregation dimensions, freely combined.** Reduce the number of *periods*
  (typical days) and/or the *temporal resolution* within them, via
  [how small you can go](how-to/tuning.ipynb).
- **Representations that preserve what matters.** Beyond means and medoids, keep
  the [value distribution](how-to/representations.ipynb) (duration
  curve), per-timestep min/max, or force **extreme periods** so peaks survive
  aggregation.
- **Built-in evaluation and plotting.** Every result carries accuracy metrics and
  an interactive `.plot` accessor (Plotly) — see the
  [Quickstart](tutorials/quickstart.ipynb).
- **Automatic hyperparameter tuning.** Let tsam
  [find the period/segment combination](how-to/tuning.ipynb) that hits a
  target data reduction, or map the full Pareto front.
- **Built for downstream models.** Hand the representatives, counts, and
  assignments to an optimization, then map results back — see the
  [optimization workflow](how-to/optimization_workflow.ipynb).

## Where to start

| If you want to… | Go to |
|------------------|-------|
| Install the package | [Installation](installation.md) |
| Run your first aggregation | [Quickstart](tutorials/quickstart.ipynb) |
| Solve a specific task | [How-to guides](how-to/index.md) |
| See an end-to-end optimization workflow | [Optimization workflow](how-to/optimization_workflow.ipynb) |
| Understand how aggregation works | [How aggregation works](explanation/how-aggregation-works/00_overview.ipynb) |
| Look up an equation or symbol | [Notation and equations](reference/notation.md) |
| Look up a function or class | [API Reference](reference/api/index.md) |
| Upgrade from v2 or v3 | [Migration guide](migration-guide.md) |

## About

ETHOS.TSAM is open source and developed on
[GitHub](https://github.com/FZJ-IEK3-VSA/tsam) — contributions, questions, and
issues are welcome. It is part of the
[Energy Transformation PatHway Optimization Suite (ETHOS)](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services)
at ICE-2 and is tightly integrated with
[ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE).

If you use or reference ETHOS.TSAM in scientific work, please
[cite one of our publications](explanation/further-reading.md).
