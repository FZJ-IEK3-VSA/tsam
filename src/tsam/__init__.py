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
>>> result = tsam.aggregate(df, n_clusters=8)
>>>
>>> # Access results
>>> typical_periods = result.cluster_representatives
>>> print(f"RMSE: {result.accuracy.rmse.mean():.4f}")

For more control, use configuration objects:

>>> from tsam import aggregate, ClusterConfig, SegmentConfig
>>>
>>> result = aggregate(
...     df,
...     n_clusters=8,
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

from tsam.api import aggregate

# Optional modules loaded on-demand to avoid importing heavy dependencies (e.g., plotly)
_LAZY_MODULES = ("plot", "tuning")


def __getattr__(name: str):
    """Lazy import handler for optional modules."""
    import importlib

    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from tsam.config import ClusterConfig, ExtremeConfig, PredefinedConfig, SegmentConfig
from tsam.result import AccuracyMetrics, AggregationResult

# Legacy imports for backward compatibility
from tsam.timeseriesaggregation import TimeSeriesAggregation, unstackToPeriods

__version__ = "3.0.0"

__all__ = [
    "AccuracyMetrics",
    "AggregationResult",
    "ClusterConfig",
    "ExtremeConfig",
    "PredefinedConfig",
    "SegmentConfig",
    "TimeSeriesAggregation",
    "aggregate",
    "plot",
    "tuning",
    "unstackToPeriods",
]
