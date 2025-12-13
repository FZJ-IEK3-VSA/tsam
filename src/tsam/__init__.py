"""tsam - Time Series Aggregation Module.

A Python package for aggregating time series data using clustering algorithms.
Designed for reducing computational load in energy system optimization models.

Quick Start
-----------
>>> import pandas as pd
>>> import tsam
>>>
>>> # Load your time series data
>>> df = pd.read_csv("data.csv", index_col=0, parse_dates=True)
>>>
>>> # Aggregate to 8 typical days
>>> result = tsam.aggregate(df, n_periods=8)
>>>
>>> # Access results
>>> typical_periods = result.typical_periods
>>> print(f"RMSE: {result.accuracy.rmse.mean():.4f}")

For more control, use configuration objects:

>>> from tsam import aggregate, ClusterConfig, SegmentConfig
>>>
>>> result = aggregate(
...     df,
...     n_periods=8,
...     cluster=ClusterConfig(method="hierarchical", representation="distribution"),
...     segments=SegmentConfig(n_segments=12),
... )

Legacy API
----------
The original class-based API is still available:

>>> from tsam.timeseriesaggregation import TimeSeriesAggregation
>>> agg = TimeSeriesAggregation(df, noTypicalPeriods=8)
>>> typical = agg.createTypicalPeriods()
"""

from tsam import plot
from tsam.api import aggregate
from tsam.config import ClusterConfig, ExtremeConfig, SegmentConfig
from tsam.result import AccuracyMetrics, AggregationResult

# Legacy imports for backward compatibility
from tsam.timeseriesaggregation import TimeSeriesAggregation, unstackToPeriods

__version__ = "3.0.0"

__all__ = [
    "AccuracyMetrics",
    "AggregationResult",
    "ClusterConfig",
    "ExtremeConfig",
    "SegmentConfig",
    "TimeSeriesAggregation",
    "aggregate",
    "plot",
    "unstackToPeriods",
]
