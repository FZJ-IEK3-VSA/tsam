.. _installation:

###############
Installation
###############

It is recommended to install tsam within its own environment. If you are no familiar with python environments, plaese consider to read some `external documentation <https://realpython.com/python-virtual-environments-a-primer/>`_. In the following we assume you have a `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ or `conda <https://www.anaconda.com/>`_ installation.  All conda and mamba command are interchangeable.

**Direct Installations from Package Manager Repositories**

If you want to prevent any possible dependency conflicts create a new environment using the following command:

.. code-block:: bash

    mamba create -n tsam_env python pip

Activate an existing or the newly create environment afterward

.. code-block:: bash

    mamba activate tsam_env

Directly install via pip from pypi as follows:

.. code-block:: bash

    pip install tsam

or install from conda forge with the following command:

.. code-block:: bash

    conda install tsam -c conda-forge

**Local Installation for Development**

Alternatively, clone a local copy of the repository to your computer

.. code-block:: bash

    git clone https://github.com/FZJ-IEK3-VSA/tsam.git

Change the directory of your shell into the root folder of the repository

.. code-block:: bash

    cd tsam

For development, it is recommended to install tsam into its own environment using conda e.g.

.. code-block:: bash

    conda env create --file=requirement.yml

Afterward activate the environment
.. code-block:: bash

    conda activate tsam_dev

Then install tsam via pip as follows

.. code-block:: bash

    pip install -e .[dev]


**Installation of an optimization solver**

Some clustering algorithms implemented in tsam are based on Mixed-Integer Linear Programming. Accordingy, an appropriate solver for using these functionalities is required that can be accessed by `Pyomo <https://github.com/Pyomo/pyomo/>`_. As default `HiGHS <https://github.com/ERGO-Code/HiGHS>`_ is installed and used. Nevertheless, in case you have access to a license we recommend commercial solvers (e.g. Gurobi or CPLEX) since they have a better performance.

**Developer installation**

In order to setup a virtual environment in Linux, correct the python name in the Makefile and call

.. code-block:: bash

    make setup_venv


