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

   from tsam import ClusterConfig, SegmentConfig

   result = tsam.aggregate(
       raw,
       n_clusters=8,
       period_duration='1D',
       cluster=ClusterConfig(
           method='hierarchical',
           representation='distribution_minmax',
       ),
       segments=SegmentConfig(n_segments=8),
   )

Access the results:

.. code-block:: python

   # Get the typical periods DataFrame
   cluster_representatives = result.cluster_representatives

   # Check accuracy metrics
   print(f"RMSE: {result.accuracy.rmse.mean():.4f}")

   # Reconstruct the original time series
   reconstructed = result.reconstructed

Store the results as .csv file

.. code-block:: python

   cluster_representatives.to_csv('cluster_representatives.csv')


**Hypertuned aggregation**

In case you do not know which number of segments or typical periods to choose, you can use the tuning functions. They will find the best combination of typical periods and segments for a target data reduction by minimizing the error between the original and the aggregated time series.

.. code-block:: python

   from tsam import ClusterConfig
   from tsam.tuning import find_optimal_combination

   result = find_optimal_combination(
       raw,
       data_reduction=0.05,  # Reduce to 5% of original size
       period_duration='1D',
       cluster=ClusterConfig(
           method='hierarchical',
           representation='distribution',
       ),
   )

   print(f"Optimal configuration: {result.n_clusters} clusters, "
         f"{result.n_segments} segments")
   print(f"RMSE: {result.rmse:.4f}")

   # Access the best aggregation result directly
   cluster_representatives = result.best_result.cluster_representatives

Since tuning can be time consuming, it is recommended to run it once at the beginning for your time series set, save the resulting segment and period numbers, and use them as fixed values in production.

For exploring the full Pareto front of period/segment combinations:

.. code-block:: python

   from tsam.tuning import find_pareto_front

   pareto = find_pareto_front(raw, max_timesteps=500)
   for row in pareto.summary.itertuples():
       print(f"{row.n_clusters}x{row.n_segments}: RMSE={row.rmse:.4f}")

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
   cluster_representatives = aggregation.createTypicalPeriods()


**Additional Examples**

More detailed examples can be found on the `GitHub page of tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_.


**Glossary**

Key concepts used in the tsam API:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Concept
     - Description
   * - **Period**
     - A fixed-length time window (e.g., 24 hours = 1 day). The original time series is divided into periods for clustering.
   * - **Typical Period**
     - A representative period selected or computed to represent a cluster of similar periods.
   * - **Cluster**
     - A group of similar original periods. Each cluster is represented by one typical period.
   * - **Segment**
     - A subdivision within a period. Consecutive timesteps are grouped into segments to reduce temporal resolution.
   * - **Timestep**
     - A single time point within a period (e.g., one hour in a 24-hour period).
   * - **Duration Curve**
     - A sorted representation of values within a period (highest to lowest). Used with ``use_duration_curves=True`` to cluster by value distribution rather than temporal pattern.
   * - ``n_clusters``
     - Number of clusters to create. Each cluster is represented by one typical period.
   * - ``n_segments``
     - Number of segments per period. If not specified, equals timesteps per period (no segmentation).
   * - ``period_duration``
     - Length of each period. Accepts int/float (hours) or pandas Timedelta strings (e.g., ``24``, ``'24h'``, ``'1d'``).
   * - ``timestep_duration``
     - Time resolution of input data. Accepts float (hours) or pandas Timedelta strings (e.g., ``1.0``, ``'1h'``, ``'15min'``). If not provided, inferred from the datetime index.
   * - ``cluster_assignments``
     - Array mapping each original period to its cluster index (0 to n_clusters-1).
   * - ``cluster_weights``
     - Dictionary mapping cluster index to occurrence count (how many original periods each cluster represents).
   * - ``segment_durations``
     - Nested tuple with duration (in timesteps) for each segment in each typical period.
   * - ``cluster_representatives``
     - MultiIndex DataFrame with aggregated data. Index levels are (cluster, timestep) or (cluster, segment) if segmented.
