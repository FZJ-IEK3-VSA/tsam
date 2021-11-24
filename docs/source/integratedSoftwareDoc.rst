###################
Integrated Software
###################

* `scikit-learn <https://scikit-learn.org/stable/>`_
* `cbc <https://www.gnu.org/software/cbc/>`_
* `gurobi <https://www.gurobi.com/downloads/?campaignid=2027425870&adgroupid=77414946211&creative=355014679607&keyword=gurobi&matchtype=e&gclid=CjwKCAiAhc7yBRAdEiwAplGxXykdP_5vQi3wmH752LzSgmH-kBJ1g2fXLTA6tbbtmyAOB2KV8YFG3RoCxxkQAvD_BwE>`_

**Installation of additional packages**

tsam is intrinsically implemented in the energy system modelling framework FINE. If you would like to use time series
aggregation for effectively reducing the computational load of energy system optimization problems,
the Python packages `FINE <https://github.com/FZJ-IEK3-VSA/FINE>`_ and `PYOMO <http://www.pyomo.org/>`_ should be
installed by pip alongside tsam. Some plots in FINE require the GeoPandas package to be installed (nice-to-have).
Installation instructions are given `here <http://geopandas.org/install.html>`_. In some cases, the dependencies of
the GeoPandas package have to be installed manually before the package itself can be installed.

* `FINE <https://github.com/FZJ-IEK3-VSA/FINE>`_
* `PYOMO <http://www.pyomo.org/>`_
* `GeoPandas <http://geopandas.org/install.html>`_