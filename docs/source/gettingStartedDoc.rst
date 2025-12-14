.. _start:

****************
Getting started
****************

**Basic Workflow**

A small example how tsam can be used is described as follows:

.. code-block:: python

   import pandas as pd
   import tsam

Read in the time series data set with pandas

.. code-block:: python

   raw = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)

Run the aggregation using the new function-based API. Specify the number of typical periods, period length, and optionally configure clustering and segmentation:

.. code-block:: python

   from tsam import aggregate, ClusterConfig, SegmentConfig

   result = tsam.aggregate(
       raw,
       n_periods=8,
       period_hours=24,
       cluster=ClusterConfig(
           method='hierarchical',
           representation='distribution_minmax',
       ),
       segments=SegmentConfig(n_segments=8),
   )

Access the results:

.. code-block:: python

   # Get the typical periods DataFrame
   typical_periods = result.typical_periods

   # Check accuracy metrics
   print(f"RMSE: {result.accuracy.rmse.mean():.4f}")

   # Reconstruct the original time series
   reconstructed = result.reconstruct()

Store the results as .csv file

.. code-block:: python

   typical_periods.to_csv('typical_periods.csv')


**Hypertuned aggregation**

In case you do not know which number of segments or typical periods to choose, you can use the tuning functions. They will find the best combination of typical periods and segments for a target data reduction by minimizing the error between the original and the aggregated time series.

.. code-block:: python

   from tsam import ClusterConfig
   from tsam.tuning import find_optimal_combination

   result = find_optimal_combination(
       raw,
       data_reduction=0.05,  # Reduce to 5% of original size
       period_hours=24,
       cluster=ClusterConfig(
           method='hierarchical',
           representation='distribution',
       ),
   )

   print(f"Optimal configuration: {result.optimal_n_periods} periods, "
         f"{result.optimal_n_segments} segments")
   print(f"RMSE: {result.optimal_rmse:.4f}")

   # Access the best aggregation result directly
   typical_periods = result.best_result.typical_periods

Since tuning can be time consuming, it is recommended to run it once at the beginning for your time series set, save the resulting segment and period numbers, and use them as fixed values in production.

For exploring the full Pareto front of period/segment combinations:

.. code-block:: python

   from tsam.tuning import find_pareto_front

   pareto = find_pareto_front(raw, max_timesteps=500)
   for p in pareto:
       print(f"{p.optimal_n_periods}x{p.optimal_n_segments}: RMSE={p.optimal_rmse:.4f}")

The scientific documentation of the methodology can be found here:
`The Pareto-Optimal Temporal Aggregation of Energy System Models <https://www.sciencedirect.com/science/article/abs/pii/S0306261922004342>`_


**Legacy API**

The class-based API is still available for backward compatibility:

.. code-block:: python

   import tsam.timeseriesaggregation as tsam_legacy

   aggregation = tsam_legacy.TimeSeriesAggregation(
       raw,
       noTypicalPeriods=8,
       hoursPerPeriod=24,
       segmentation=True,
       noSegments=8,
       representationMethod="distributionAndMinMaxRepresentation",
       clusterMethod='hierarchical'
   )
   typical_periods = aggregation.createTypicalPeriods()


**Additional Examples**

More detailed examples can be found on the `GitHub page of tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_.
