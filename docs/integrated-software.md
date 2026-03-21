# Integrated Software

* [scikit-learn](https://scikit-learn.org/stable/)
* [cbc](https://www.gnu.org/software/cbc/)
* [gurobi](https://www.gurobi.com/)

**Installation of additional packages**

ETHOS.TSAM is intrinsically implemented in the energy system modelling framework ETHOS.FINE. If you would like to use time series
aggregation for effectively reducing the computational load of energy system optimization problems,
the Python packages [ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE) and [PYOMO](http://www.pyomo.org/) should be
installed by pip alongside ETHOS.TSAM. Some plots in FINE require the GeoPandas package to be installed (nice-to-have).
Installation instructions are given [here](http://geopandas.org/install.html). In some cases, the dependencies of
the GeoPandas package have to be installed manually before the package itself can be installed.

* [FINE](https://github.com/FZJ-IEK3-VSA/FINE)
* [PYOMO](http://www.pyomo.org/)
* [GeoPandas](http://geopandas.org/install.html)
