<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

# tsam - Time Series Aggregation Module
tsam is a python package which uses different machine learning algorithms for the aggregation of typical periods. It is applicable for all type of time series, eather weather data, load data or both simultaneously. The module is able to significantly reduce input time series for energy system models, and therefore the model's complexity and computational time. 

A publication will follow soon which validates the methods and describes their capabilites.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.579998.svg)](https://doi.org/10.5281/zenodo.579998)
 
## Features
* flexible read-in and handling of multidimensional time-series via the pandas module
* different aggregation methods implemented (averaging, k-mean, exact k-medoid, hierarchical), which are based on scikit-learn or pyomo
* flexible integration of extreme periods as own cluster centers
* weighting for the case of multidimensional time-series to represent their relevance


## Installation

First clone a local copy of the repository to your computer

	git clone https://github.com/FZJ-IEK3-VSA/tsam.git
	
Then install tsa via pip as follow
	
	cd tsam
	pip install . 
	
Or install directly via python as 

	python setup.py install
	
	
## Examples

A small example how tsa can be used is decribed as follows

	import pandas as pd
	import tsam.timeseriesaggregation as tsam

Read in the time series data set with pandas
	
	raw = pd.read_csv('testdata.csv', index_col = 0)

Initialize an aggregation object and define the number of typical periods, the length of a single period and the aggregation method
	
	aggregation = tsam.TimeSeriesAggregation(raw, 
			noTypicalPeriods = 8, 
			hoursPerPeriod = 24, 
			clusterMethod = 'hierarchical')

Run the aggregation to typical periods
	
	typPeriods = aggregation.createTypicalPeriods()

Store the results as .csv file
	
	typPeriods.to_csv('typperiods.csv')

A more detailed example showing the capabilites of tsa is found [`here`](https://134.94.251.61/l-kotzur/tsa/blob/opensource/example/aggregation_example.ipynb) as jupyter notebook. 

The example time series are based on a department [publication](http://www.mdpi.com/1996-1073/10/3/361) and the [test reference years of the DWD](http://www.dwd.de/DE/leistungen/testreferenzjahre/testreferenzjahre.html).

## License

Copyright (C) 2016-2017 Leander Kotzur (FZJ IEK-3), Peter Markewitz (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2017.jpg?__blob=normal" alt="Abteilung VSA"></a> 

We are the [Process and Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the Forschungszentrum Jülich. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

#### Selected publications:

* <a href="http://www.sciencedirect.com/science/article/pii/S0360319915001913">Power to gas: Technological overview, systems analysis and economic assessment for a case study in Germany</a>
* <a href="http://www.mdpi.com/1996-1073/10/4/451">Power-to-Steel: Reducing CO2 through the Integration of Renewable Energy and Hydrogen into the German Steel Industry</a>
* <a href="http://www.sciencedirect.com/science/article/pii/S0306261916309199">Early power to gas applications: Reducing wind farm forecast errors and providing secondary control reserve</a>
* <a href="http://www.mdpi.com/1996-1073/10/3/361">A Top-Down Spatially Resolved Electrical Load Model</a>
* <a href="http://www.sciencedirect.com/science/article/pii/S0360319917310054">Long-term power-to-gas potential from wind and solar power: A country analysis for Italy</a>
* <a href="http://pubs.rsc.org/en/Content/ArticleLanding/2015/EE/c5ee02591e">Closing the loop: captured CO2 as a feedstock in the chemical industry</a>


## Acknowledgement

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050 – A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>