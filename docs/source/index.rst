.. FINE documentation master file, created by
   sphinx-quickstart on Sat Nov 10 21:04:19 2018.


.. image:: http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster
    :target: http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html
    :width: 230px
    :alt: Forschungszentrum Juelich Logo
    :align: right

Welcome to FINE's documentation!
================================

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FINE - A Framework for Integrated Energy System Assessment
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The FINE python package provides a framework for modeling, optimizing and assessing energy systems. With the provided
framework, systems with multiple regions, commodities and time steps can be modeled. Target of the optimization is the
minimization of the total annual cost while considering technical and environmental constraints. Besides using the full
temporal resolution, an interconnected typical period storage formulation can be applied, that reduces the complexity
and computational time of the model.

**Features**

* representation of an energy system by multiple locations, commodities and time steps
* complexity reducing storage formulation based on typical periods


**Citing FINE**

If you want to use FINE in a published work, please kindly cite following publication:
https://www.sciencedirect.com/science/article/pii/S036054421830879X which gives
a description of the first stages of the framework. The python package which provides the time series aggregation module
and its corresponding literatur can be found `here <https://github.com/FZJ-IEK3-VSA/tsam>`_.

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    gettingStartedDoc
    usersGuideDoc
    newsDoc
    integratedSoftwareDoc
    legalNoticeDoc
    furtherReadingDoc

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



