.. _tutorial:

********
Tutorial
********

**Basic Workflow**

A small example how tsam can be used is decribed as follows

.. code-block:: python
  :linenos:
  :lineno-start: 1

   import pandas as pd
   import tsam.timeseriesaggregation as tsam

Read in the time series data set with pandas

.. code-block:: python
  :linenos:
  :lineno-start: 3

   raw = pd.read_csv('testdata.csv', index_col = 0)

Initialize an aggregation object and define the number of typical periods, the length of a single period and the aggregation method

.. code-block:: python
  :linenos:
  :lineno-start: 4

   aggregation = tsam.TimeSeriesAggregation(raw, 
					    noTypicalPeriods = 8, 
					    hoursPerPeriod = 24, 
					    clusterMethod = 'hierarchical')

Run the aggregation to typical periods

.. code-block:: python
  :linenos:
  :lineno-start: 8

   typPeriods = aggregation.createTypicalPeriods()

Store the results as .csv file

.. code-block:: python
  :linenos:
  :lineno-start: 9

   typPeriods.to_csv('typperiods.csv')

**Additional Examples**

More detailed examples can be found on the `GitHub page of tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_.