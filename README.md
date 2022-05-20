[![Build Status](https://travis-ci.com/FZJ-IEK3-VSA/tsam.svg?branch=master)](https://travis-ci.com/FZJ-IEK3-VSA/tsam) [![Version](https://img.shields.io/pypi/v/tsam.svg)](https://pypi.python.org/pypi/tsam) [![Documentation Status](https://readthedocs.org/projects/tsam/badge/?version=latest)](https://tsam.readthedocs.io/en/latest/) [![PyPI - License](https://img.shields.io/pypi/l/tsam)]((https://github.com/FZJ-IEK3-VSA/tsam/blob/master/LICENSE.txt)) [![codecov](https://codecov.io/gh/FZJ-IEK3-VSA/tsam/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/tsam)

<a href="https://www.fz-juelich.de/iek/iek-3"><img src="https://www.fz-juelich.de/SiteGlobals/StyleBundles/Bilder/NeuesLayout/logo.jpg?__blob=normal" alt="Forschungszentrum Juelich Logo"></a> 

# tsam - Time Series Aggregation Module
tsam is a python package which uses different machine learning algorithms for the aggregation of time series. The data aggregation can be performed in two freely combinable dimensions: By representing the time series by a user-defined number of typical periods or by decreasing the temporal resolution.
tsam was originally designed for reducing the computational load for large-scale energy system optimization models by aggregating their input data, but is applicable for all types of time series, e.g., weather data, load data, both simultaneously or other arbitrary groups of time series.

If you want to use tsam in a published work, **please kindly cite** our latest journal article "[**A Review on Time Series Aggregation Methods for Energy System Models**](https://www.mdpi.com/1996-1073/13/3/641)".

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
        				representationMethod = "durationRepresentation",
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
* Kotzur et al. (2018):\
[**Impact of different time series aggregation methods on optimal energy system design**](https://www.sciencedirect.com/science/article/abs/pii/S0960148117309783)\
(open access manuscript to be found [**here**](https://arxiv.org/abs/1708.00420))
* Kotzur et al. (2018):\
[**Time series aggregation for energy system design: Modeling seasonal storage**](https://www.sciencedirect.com/science/article/pii/S0306261918300242)\
(open access manuscript to be found [**here**](https://arxiv.org/abs/1710.07593))
* Kannengießer et al. (2019):\
[**Reducing Computational Load for Mixed Integer Linear Programming: An Example for a District and an Island Energy System**](https://www.mdpi.com/1996-1073/12/14/2825)
* Hoffmann et al. (2020):\
[**A Review on Time Series Aggregation Methods for Energy System Models**](https://www.mdpi.com/1996-1073/13/3/641)


## Acknowledgement

This work is supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/) and the program ["Energy System Design"](https://www.esd.kit.edu/index.php) and within the [BMWi/BMWk](https://www.bmwk.de/Navigation/DE/Home/home.html) funded project [**METIS**](http://www.metis-platform.net/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>


