###########################
API Reference
###########################

.. |br| raw:: html

   <br />

This page documents the tsam API including the new function-based API and the legacy class-based API.

*****************************
Function-based API (v3.0+)
*****************************

The new API provides a simpler, more intuitive interface for time series aggregation.

Main Function
=============

.. automodule:: tsam.api
   :members: aggregate
   :member-order: bysource

Configuration Classes
=====================

.. automodule:: tsam.config
   :members: ClusterConfig, SegmentConfig, ExtremeConfig
   :member-order: bysource

Result Classes
==============

.. automodule:: tsam.result
   :members: AggregationResult, AccuracyMetrics
   :member-order: bysource

Tuning Functions
================

.. automodule:: tsam.tuning
   :members: find_optimal_combination, find_pareto_front, periods_for_reduction, segments_for_reduction, TuningResult
   :member-order: bysource

*****************************
Legacy Class-based API
*****************************

The class-based API is still available for backward compatibility.

.. automodule:: tsam.timeseriesaggregation
   :members: unstackToPeriods
   :member-order: bysource

   .. autoclass:: TimeSeriesAggregation
      :members:
      :member-order: bysource

      .. automethod:: __init__
