.. _installation:

###############
Installation
###############

In the following, instructions for installing and using the tsam package are given.

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


Some clustering algorithms implemented in tsam are based on Mixed-Integer Linear Programming. Accordingy, an appropriate solver for using these functionalities is required that can be accessed by `Pyomo <https://github.com/Pyomo/pyomo/>`_. The `HiGHS <https://github.com/ERGO-Code/HiGHS>`_ solver is set as default and installed via `highspy` but can be replaced by any other solver that is supported by Pyomo.
