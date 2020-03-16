###############
Getting started
###############

**************************
Purpose and Vision of tsam
**************************

tsam is a python package which uses different machine learning algorithms for the aggregation of typical periods. It is applicable
for all type of time series, either weather data, load data or both simultaneously. The module is able to significantly reduce input
time series for energy system models, and therefore the model's complexity and computational time.

The concept of tsam is that scientists, programmers and anyone who is interested all around the world can use tsam to aggregate their
time series whatever they might be used for. Therefore, tsam is open source available and completely for free.
The authors of both, the program and the publications, **kindly request you to cite their work** listed in the
:ref:`Further Reading Section <further_reading>` if tsam is used in a scientific context.

************
Installation
************

In the following, instructions for installing and using the tsam package on Windows are given. The installation
instructions for installing and using tsam on Linux/macOS systems are however quite similar and can be, hopefully
easily, derived from the instructions below.

**tsam installation**

Install via pip by typing

    $ pip install tsam

into the command prompt. Alternatively, download or clone a local copy of the repository to your computer

    $ git clone https://github.com/FZJ-IEK3-VSA/tsam.git

and install tsam in the folder where the setup.py is located with

    $ pip install -e .

or install directly via python as

    $ python setup.py install

**Installation of an optimization solver**

Some clustering algorithms implemented in tsam are based on Mixed-Integer Linear Programming. Accordingy,
an appropriate solver for using these functionalities is  required.
In theory many solvers can be used (e.g. `GUROBI <http://www.gurobi.com/>`_  or
`GLPK <https://sourceforge.net/projects/winglpk/files/latest/download>`_). For the installation of GUROBI, follow
the instructions on the solver's website. GUROBI has, if applicable, an academic license option. For installation
of GLPK, move the downloaded folder to a desired location. Then, manually append the Environment Variable *Path*
with the absolute path leading to the folder in which the glpsol.exe is located (c.f. w32/w64 folder, depending on
operating system type).

********
About Us
********

.. image:: https://www.fz-juelich.de/iek/iek-3/DE/_Documents/Pictures/IEK-3Team_2019-02-04.jpg?__blob=poster
    :target: https://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html
    :alt: Abteilung TSA
    :align: center

We are the `Institute of Energy and Climate Research: Techno-Economic Energy Systems Analysis (IEK-3)
<https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html>`_ belonging to the Forschungszentrum Jülich.
Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and
system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and
costs of energy systems. The results are used for performing comparative assessment studies between the various systems.
Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s
greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and
by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

**Contributions and Users**

Within the BMWi funded project `METIS <http://www.metis-platform.net/>`_ we develop together with the RWTH-Aachen
`(Prof. Aaron Praktiknjo) <http://www.wiwi.rwth-aachen.de/cms/Wirtschaftswissenschaften/Die-Fakultaet/Institute-und-Lehrstuehle/Professoren/~jgfr/Praktiknjo-Aaron/?allou=1&lidx=1>`_,
the EDOM Team at FAU `(PD Lars Schewe) <http://www.mso.math.fau.de/de/edom/team/schewe-lars/dr-lars-schewe>`_ and the
`Jülich Supercomputing Centre (JSC) <http://www.fz-juelich.de/ias/jsc/DE/Home/home_node.html>`_ new methods and models
within FINE.

.. image:: http://www.metis-platform.net/metis-platform/DE/_Documents/Pictures/projectTeamAtKickOffMeeting_640x338.jpg?__blob=normal
    :target: http://www.metis-platform.net
    :alt: METIS Team
    :align: center

Dr. Martin Robinius is teaching a `course <https://www.campus-elgouna.tu-berlin.de/energy/v_menu/msc_business_engineering_energy/modules_and_curricula/project_market_coupling/>`_
at TU Berlin in which he is introducing tsam to students.