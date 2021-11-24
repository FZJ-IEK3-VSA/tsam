.. _getting_started:

###############
Getting started
###############

In the following, instructions for installing and using the tsam package on Windows are given. The installation
instructions for installing and using tsam on Linux/macOS systems are however quite similar and can be, hopefully
easily, derived from the instructions below.

**tsam installation**

Install via pip by typing

.. code-block:: bash

    pip install tsam

into the command prompt. Alternatively, download or clone a local copy of the repository to your computer

.. code-block:: bash

    git clone https://github.com/FZJ-IEK3-VSA/tsam.git

and install tsam in the folder where the setup.py is located with

.. code-block:: bash

    pip install -e .

or install directly via python as

.. code-block:: bash

    python setup.py install

**Installation of an optimization solver**

Some clustering algorithms implemented in tsam are based on Mixed-Integer Linear Programming. Accordingy, an appropriate solver for using these functionalities is required that can be accessed by `Pyomo <https://github.com/Pyomo/pyomo/>`_. In theory many solvers can be used (e.g. `GUROBI <http://www.gurobi.com/>`_  or `cbc <https://sourceforge.net/projects/wincbc/files/latest/download>`_). For the installation of GUROBI, follow the instructions on the solver's website. For installation of cbc, move the downloaded folder to a desired location. Then, manually append the Environment Variable *Path* with the absolute path leading to the folder in which the glpsol.exe is located (c.f. w32/w64 folder, depending on operating system type).
