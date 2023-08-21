.. _start:

********
Getting started
********

**Basic Workflow**

A small example how tsam can be used is decribed as follows

.. code-block:: python

   import pandas as pd
   import tsam.timeseriesaggregation as tsam

Read in the time series data set with pandas

.. code-block:: python

   raw = pd.read_csv('testdata.csv', index_col = 0)

Initialize an aggregation object and define the number of typical periods, the length of a single period, the aggregation method

.. code-block:: python

  aggregation = tsam.TimeSeriesAggregation(raw, 
    noTypicalPeriods = 8, 
    hoursPerPeriod = 24, 
    segmentation = True,
    noSegments = 8,
    representationMethod = "distributionAndMinMaxRepresentation",
    distributionPeriodWise = False
    clusterMethod = 'hierarchical'
  )

Run the aggregation to typical periods

.. code-block:: python

   typPeriods = aggregation.createTypicalPeriods()

Store the results as .csv file

.. code-block:: python

   typPeriods.to_csv('typperiods.csv')


**Hypertuned aggregation**

In case you do not know, which number of segments or typical periods to choose, you should make first a run with the HyperTunedAggregations class. It will find the best combination of typical periods and segments for the aggregation by minimizing the error between the original and the aggregated time series. The following example shows how to use it.


Either create or provide a TimeSeriesAggregation object and use it as base case to get tuned by the HyperTunedAggregations class.

.. code-block:: python

	tune_aggregation = tune.HyperTunedAggregations(
    tsam.TimeSeriesAggregation(
      raw,
      hoursPerPeriod=24,
      clusterMethod="hierarchical",
      representationMethod="distributionRepresentation",
      rescaleClusterPeriods=False,
      segmentation=True,
    )
  )

Afterwards, define the aggregation level you want to reach. The following examples shows how to reduce the original time series to 5% of the original data set.

.. code-block:: python

  segments, periods, RMSE = tune_aggregation.identifyOptimalSegmentPeriodCombination(
    dataReduction=0.05
  )


Since, it is quite time consuming, I would recommend to just run it once at the beginning for your time series set, save the resulting segment and period number, and use it as fix values for the original TimeSeriesAggregation object in production.

The scientific documentation of the methodology can be found here: 
`The Pareto-Optimal Temporal Aggregation of Energy System Models <https://www.sciencedirect.com/science/article/abs/pii/S0306261922004342>`_

**Additional Examples**

More detailed examples can be found on the `GitHub page of tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_.