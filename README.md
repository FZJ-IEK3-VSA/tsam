[![Build Status](https://travis-ci.com/FZJ-IEK3-VSA/tsam.svg?branch=master)](https://travis-ci.com/FZJ-IEK3-VSA/tsam) [![Version](https://img.shields.io/pypi/v/tsam.svg)](https://pypi.python.org/pypi/tsam) [![Documentation Status](https://readthedocs.org/projects/tsam/badge/?version=latest)](https://tsam.readthedocs.io/en/latest/) [![PyPI - License](https://img.shields.io/pypi/l/tsam)]((https://github.com/FZJ-IEK3-VSA/tsam/blob/master/LICENSE.txt)) [![codecov](https://codecov.io/gh/FZJ-IEK3-VSA/tsam/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/tsam)
[![badge](https://img.shields.io/badge/launch-binder-579aca.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/FZJ-IEK3-VSA/voila-tsam/HEAD?urlpath=voila/render/Time-Series-Aggregation-Module.ipynb)

<a href="https://www.fz-juelich.de/iek/iek-3"><img src="https://www.fz-juelich.de/SiteGlobals/StyleBundles/Bilder/NeuesLayout/logo.jpg?__blob=normal" alt="Forschungszentrum Juelich Logo"></a> 

# tsam - Time Series Aggregation Module
tsam is a python package which uses different machine learning algorithms for the aggregation of time series. The data aggregation can be performed in two freely combinable dimensions: By representing the time series by a user-defined number of typical periods or by decreasing the temporal resolution.
tsam was originally designed for reducing the computational load for large-scale energy system optimization models by aggregating their input data, but is applicable for all types of time series, e.g., weather data, load data, both simultaneously or other arbitrary groups of time series.

If you want to use tsam in a published work, **please kindly cite** one of our latest journal articles:
* Hoffmann et al. (2021):\
[**The Pareto-Optimal Temporal Aggregation of Energy System Models**](https://arxiv.org/abs/2111.12072)
* Hoffmann et al. (2021):\
[**Typical periods or typical time steps? A multi-model analysis to determine the optimal temporal aggregation for energy system models**](https://www.sciencedirect.com/science/article/abs/pii/S0306261921011545)

The documentation of the tsam code can be found [**here**](https://tsam.readthedocs.io/en/latest/index.html).

## Features
* flexible handling of multidimensional time-series via the pandas module
* different aggregation methods implemented (averaging, k-means, exact k-medoids, hierarchical, k-maxoids, k-medoids with contiguity), which are based on scikit-learn, or self-programmed with pyomo
* novel representation methods, keeping statistical attributes, such as the distribution 
* flexible integration of extreme periods as own cluster centers
* weighting for the case of multidimensional time-series to represent their relevance


## Installation
Directly install via pip as follows:

	pip install tsam

Alternatively, clone a local copy of the repository to your computer

	git clone https://github.com/FZJ-IEK3-VSA/tsam.git
	
Then install tsam via pip as follow
	
	cd tsam
	pip install . 
	
Or install directly via python as 

	python setup.py install
	
In order to use the k-medoids clustering, make sure that you have installed a MILP solver. As default coin-cbc is used. Nevertheless, in case you have access to a license we recommend commercial solvers (e.g. Gurobi or CPLEX) since they have a better performance.
	
	
## Examples

### Basic workflow

A small example how tsam can be used is decribed as follows
```python
	import pandas as pd
	import tsam.timeseriesaggregation as tsam
```


Read in the time series data set with pandas
```python
	raw = pd.read_csv('testdata.csv', index_col = 0)
```

Initialize an aggregation object and define the length of a single period, the number of typical periods, the number of segments in each period, the aggregation method and the representation method - here duration/distribution representation which 
```python
	aggregation = tsam.TimeSeriesAggregation(raw, 
						noTypicalPeriods = 8, 
						hoursPerPeriod = 24, 
						rescaleClusterPeriods = False,
						segmentation = True,
						representationMethod = "distributionRepresentation",
						distributionPeriodWise = False
						clusterMethod = 'hierarchical')
```

Run the aggregation to typical periods
```python
	typPeriods = aggregation.createTypicalPeriods()
```

Store the results as .csv file
	
```python
	typPeriods.to_csv('typperiods.csv')
```

### Detailed examples

A [**first example**](/examples/aggregation_example.ipynb) shows the capabilites of tsam as jupyter notebook. 

A [**second example**](/examples/aggregation_optiinput.ipynb) shows in more detail how to access the relevant aggregation results required for paramtrizing e.g. an optimization.

The example time series are based on a department [publication](https://www.mdpi.com/1996-1073/10/3/361) and the [test reference years of the DWD](https://www.dwd.de/DE/leistungen/testreferenzjahre/testreferenzjahre.html).

## License

MIT License

Copyright (C) 2016-2019 Leander Kotzur (FZJ IEK-3), Maximilian Hoffmann (FZJ IEK-3), Peter Markewitz (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT

The core developer team sits in the [Institute of Energy and Climate Research - Techno-Economic Energy Systems Analysis (IEK-3)](https://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the [Forschungszentrum Jülich](https://www.fz-juelich.de/).

## Further Reading

If you are further interested in the impact of time series aggregation on the cost-optimal results on different energy system use cases, you can find a publication which validates the methods and describes their cababilites via the following [**link**](https://www.sciencedirect.com/science/article/pii/S0960148117309783). A second publication introduces a method how to model state variables (e.g. the state of charge of energy storage components) between the aggregated typical periods which can be found [**here**](https://www.sciencedirect.com/science/article/pii/S0306261918300242). Finally yet importantly the potential of time series aggregation to simplify mixed integer linear problems is investigated [**here**](https://www.mdpi.com/1996-1073/12/14/2825).

The publications about time series aggregation for energy system optimization models published alongside the development of tsam are listed below:
* Hoffmann et al. (2021):\
[**The Pareto-Optimal Temporal Aggregation of Energy System Models**](https://arxiv.org/abs/2111.12072)
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



## Acknowledgement

This work is supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/) and the program ["Energy System Design"](https://www.esd.kit.edu/index.php) and within the [BMWi/BMWk](https://www.bmwk.de/Navigation/DE/Home/home.html) funded project [**METIS**](http://www.metis-platform.net/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>


