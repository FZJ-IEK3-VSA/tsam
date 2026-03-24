[![Version](https://img.shields.io/pypi/v/tsam.svg)](https://pypi.python.org/pypi/tsam) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/tsam.svg)](https://anaconda.org/conda-forge/tsam) [![Documentation Status](https://readthedocs.org/projects/tsam/badge/?version=latest)](https://tsam.readthedocs.io/en/latest/) [![PyPI - License](https://img.shields.io/pypi/l/tsam)]((https://github.com/FZJ-IEK3-VSA/tsam/blob/master/LICENSE.txt)) [![codecov](https://codecov.io/gh/FZJ-IEK3-VSA/tsam/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/tsam)

<p align="left">
  <a href="https://tsam.readthedocs.io/en/latest/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tsam-logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/assets/tsam-logo-light.svg">
      <img src="docs/assets/tsam-logo-light.svg" alt="ETHOS.TSAM Logo" height="80px">
    </picture>
  </a>
  <a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="docs/assets/JSA-Header.svg" alt="Jülich System Analysis Header" height="80px"></a>
</p>

# ETHOS.TSAM - Time Series Aggregation Module
ETHOS.TSAM is a python package which uses different machine learning algorithms for the aggregation of time series. The data aggregation can be performed in two freely combinable dimensions: By representing the time series by a user-defined number of typical periods or by decreasing the temporal resolution.
ETHOS.TSAM was originally designed for reducing the computational load for large-scale energy system optimization models by aggregating their input data, but is applicable for all types of time series, e.g., weather data, load data, both simultaneously or other arbitrary groups of time series.

ETHOS.TSAM is part of the [Energy Transformation PatHway Optimization Suite (ETHOS) at ICE-2](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services). It is tightly integrated into [ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE) to reduce the temporal complexity of energy system models.

The documentation of the ETHOS.TSAM code can be found [**here**](https://tsam.readthedocs.io/).

## Features
* flexible handling of multidimensional time-series via the pandas module
* different aggregation methods implemented (averaging, k-means, exact k-medoids, hierarchical, k-maxoids, k-medoids with contiguity), which are based on scikit-learn, or self-programmed with pyomo
* hypertuning of aggregation parameters to find the optimal combination of the number of segments inside a period and the number of typical periods
* novel representation methods, keeping statistical attributes, such as the distribution
* flexible integration of extreme periods as own cluster centers
* weighting for the case of multidimensional time-series to represent their relevance

## Installation

To avoid dependency conflicts, it is recommended that you install ETHOS.TSAM in its own environment. You can use either [uv](https://docs.astral.sh/uv/)  or [conda/mamba](https://conda-forge.org/download/) to manage environments and installations. Before proceeding, you must install either UV or Conda/Mamba, or both.

**Quick Install with uv**

```bash
uv venv tsam_env
uv pip install tsam
```

Or from conda-forge:

```bash
conda create -n tsam_env -c conda-forge tsam
```

conda and mamba can be used interchangeably

### Development Installation

```bash
git clone https://github.com/FZJ-IEK3-VSA/tsam.git
cd tsam
```

# Using uv (recommended)
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[develop]"
```

# Using conda-forge

```bash
conda env create -n tsam_env --file=environment.yml
conda activate tsam_env
pip install -e . --no-deps
```

# Set up pre-commit hooks
```bash
pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

### MILP Solver for k-medoids

[HiGHS](https://github.com/ERGO-Code/HiGHS) is installed by default. For better performance on large problems, commercial solvers (Gurobi, CPLEX) are recommended if you have a license


## Examples

### Basic workflow

A small example how ETHOS.TSAM can be used is described as follows:
```python
import pandas as pd
import tsam
```

Read in the time series data set with pandas
```python
raw = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)
```

Run the aggregation - specify the number of typical periods and configure clustering/segmentation options:
```python
from tsam import aggregate, ClusterConfig, SegmentConfig

result = tsam.aggregate(
    raw,
    n_clusters=8,
    period_duration='24h',  # or 24, '1d'
    cluster=ClusterConfig(
        method='hierarchical',
        representation='distribution_minmax',
    ),
    segments=SegmentConfig(n_segments=8),
)
```

Access the results:
```python
# Get the typical periods DataFrame
cluster_representatives = result.cluster_representatives

# Check accuracy metrics
print(f"RMSE: {result.accuracy.rmse.mean():.4f}")

# Reconstruct the original time series from typical periods
reconstructed = result.reconstructed

# Save results
cluster_representatives.to_csv('cluster_representatives.csv')
```

### Legacy API

For backward compatibility, the class-based API of TSAM Version 2 is still available.
```python
import tsam.timeseriesaggregation as tsam_legacy

aggregation = tsam_legacy.TimeSeriesAggregation(
    raw,
    noTypicalPeriods=8,
    hoursPerPeriod=24,
    segmentation=True,
    noSegments=8,
    representationMethod="distributionAndMinMaxRepresentation",
    clusterMethod='hierarchical'
)
cluster_representatives = aggregation.createTypicalPeriods()
```

### Detailed examples
Detailed examples can be found at:/docs/notebooks/

A [**quickstart example**](/docs/notebooks/quickstart.ipynb) shows the capabilities of ETHOS.TSAM as a Jupyter notebook.

A [**second example**](/docs/notebooks/optimization_input.ipynb) shows in more detail how to access the relevant aggregation results required for parameterizing e.g. an optimization.

The example time series are based on a department [publication](https://www.mdpi.com/1996-1073/10/3/361) and the [test reference years of the DWD](https://www.dwd.de/DE/leistungen/testreferenzjahre/testreferenzjahre.html).

## License

[MIT License](LICENSE.txt)

Copyright (C) 2017-2025 Leander Kotzur (FZJ ICE-2), Maximilian Hoffmann (FZJ ICE-2), Peter Markewitz (FZJ ICE-2), Martin Robinius (FZJ ICE-2), Detlef Stolten (FZJ ICE-2)


## Citing and further reading

If you want to use ETHOS.TSAM in a published work, **please kindly cite** our latest journal articles:
* Hoffmann et al. (2022):\
[**The Pareto-Optimal Temporal Aggregation of Energy System Models**](https://www.sciencedirect.com/science/article/abs/pii/S0306261922004342)


If you are further interested in the impact of time series aggregation on the cost-optimal results on different energy system use cases, you can find a publication which validates the methods and describes their cababilites via the following [**link**](https://www.sciencedirect.com/science/article/pii/S0960148117309783). A second publication introduces a method how to model state variables (e.g. the state of charge of energy storage components) between the aggregated typical periods which can be found [**here**](https://www.sciencedirect.com/science/article/pii/S0306261918300242). Finally yet importantly the potential of time series aggregation to simplify mixed integer linear problems is investigated [**here**](https://www.mdpi.com/1996-1073/12/14/2825).

The publications about time series aggregation for energy system optimization models published alongside the development of ETHOS.TSAM are listed below:
* Hoffmann et al. (2021):\
[**The Pareto-Optimal Temporal Aggregation of Energy System Models**](https://www.sciencedirect.com/science/article/abs/pii/S0306261922004342)\
(open access manuscript to be found [**here**](https://arxiv.org/abs/1710.07593))
* Hoffmann et al. (2021):\
[**Typical periods or typical time steps? A multi-model analysis to determine the optimal temporal aggregation for energy system models**](https://www.sciencedirect.com/science/article/abs/pii/S0306261921011545)
* Hoffmann et al. (2020):\
[**A Review on Time Series Aggregation Methods for Energy System Models**](https://www.mdpi.com/1996-1073/13/3/641)
* Kannengießer et al. (2019):\
[**Reducing Computational Load for Mixed Integer Linear Programming: An Example for a District and an Island Energy System**](https://www.mdpi.com/1996-1073/12/14/2825)
* Kotzur et al. (2018):\
[**Time series aggregation for energy system design: Modeling seasonal storage**](https://www.sciencedirect.com/science/article/pii/S0306261918300242)\
(open access manuscript to be found [**here**](https://arxiv.org/abs/1710.07593))
* Kotzur et al. (2018):\
[**Impact of different time series aggregation methods on optimal energy system design**](https://www.sciencedirect.com/science/article/abs/pii/S0960148117309783)\
(open access manuscript to be found [**here**](https://arxiv.org/abs/1708.00420))


## About Us 

We are the <a href="https://www.fz-juelich.de/en/ice/ice-2">Institute of Climate and Energy Systems – Jülich Systems Analysis (ICE-2)</a> at the <a href="https://www.fz-juelich.de/en"> Forschungszentrum Jülich</a>.
Our work focuses on independent, interdisciplinary research in energy, bioeconomy, infrastructure, and sustainability. We support a just, greenhouse gas–neutral transformation through open models and policy-relevant science.


## Code of Conduct
Please respect our [code of conduct](https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/CODE_CONDUCT.md).


## Acknowledgement

This work is supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/) and the program ["Energy System Design"](https://www.esd.kit.edu/index.php) and within the [BMWi/BMWk](https://www.bmwk.de/Navigation/DE/Home/home.html) funded project **METIS**.

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
