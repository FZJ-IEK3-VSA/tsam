[![Build Status](https://travis-ci.com/FZJ-IEK3-VSA/tsam.svg?branch=master)](https://travis-ci.com/FZJ-IEK3-VSA/tsam)

<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

# tsam - Time Series Aggregation Module
tsam is a python package which uses different machine learning algorithms for the aggregation of typical periods. It is applicable for all type of time series, eather weather data, load data or both simultaneously. The module is able to significantly reduce input time series for energy system models, and therefore the model's complexity and computational time. 


If you want to use tsam in a published work, please [**kindly cite following publication**](https://www.sciencedirect.com/science/article/pii/S0960148117309783) which validates the methods and describes their cababilites. The manuscript is found [**here**](https://arxiv.org/abs/1708.00420). 

A [**second publication**](https://www.sciencedirect.com/science/article/pii/S0306261918300242) introduces a method how to model states (e.g. state of charge of storage) between the aggregated typical periods.

## Features
* flexible read-in and handling of multidimensional time-series via the pandas module
* different aggregation methods implemented (averaging, k-means, exact k-medoids, hierarchical), which are based on scikit-learn or pyomo
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
	
In order to use the k-medoids clustering, make sure that you have installed a MILP solver. As default solver GLPK is used. Nevertheless, in case you have access to a license we recommend commercial solvers (e.g. Gurobi or CPLEX) since they have a better performance.
	
	
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

Initialize an aggregation object and define the number of typical periods, the length of a single period and the aggregation method
```python
	aggregation = tsam.TimeSeriesAggregation(raw, 
						noTypicalPeriods = 8, 
						hoursPerPeriod = 24, 
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

A [**first example**](https://github.com/FZJ-IEK3-VSA/tsam/blob/master/example/aggregation_example.ipynb) shows the capabilites of tsam as jupyter notebook. 

A [**second example**](https://github.com/FZJ-IEK3-VSA/tsam/blob/master/example/aggregation_optiinput.ipynb) shows in more detail how to access the relevant aggregation results required for paramtrizing e.g. an optimization.

The example time series are based on a department [publication](http://www.mdpi.com/1996-1073/10/3/361) and the [test reference years of the DWD](http://www.dwd.de/DE/leistungen/testreferenzjahre/testreferenzjahre.html).

## License

MIT License

Copyright (C) 2016-2018 Leander Kotzur (FZJ IEK-3), Peter Markewitz (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT

## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="https://www.fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2019-02-04_459x244_2480x1317.jpg?__blob=normal" alt="Abteilung VSA"></a> 

We are the [Techno-Economic Energy Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the [Forschungszentrum Jülich](www.fz-juelich.de/). Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Contributions and Users

Within the BMWi funded project [**METIS**](http://www.metis-platform.net/) we extend the methodology together with the RWTH-Aachen ([**Prof. Aaron Praktiknjo**](http://www.wiwi.rwth-aachen.de/cms/Wirtschaftswissenschaften/Die-Fakultaet/Institute-und-Lehrstuehle/Professoren/~jgfr/Praktiknjo-Aaron/?allou=1&lidx=1)), the EDOM Team at FAU ([**PD Lars Schewe**](http://www.mso.math.fau.de/de/edom/team/schewe-lars/dr-lars-schewe/)) and the [**Jülich Supercomputing Centre**](http://www.fz-juelich.de/ias/jsc/DE/Home/home_node.html).

<a href="http://www.metis-platform.net/"><img src="http://www.metis-platform.net/metis-platform/DE/_Documents/Pictures/projectTeamAtKickOffMeeting_640x338.jpg?__blob=normal" alt="METIS Team" width="400px" style="float:center"></a> 


## Acknowledgement

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.597956.svg)](https://doi.org/10.5281/zenodo.597956)
