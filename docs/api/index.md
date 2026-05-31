# API Reference

ETHOS.TSAM's public API. Most users only need [`aggregate`][tsam.aggregate] —
the single entry point — plus the configuration objects and the result it
returns. The pipeline internals are here for completeness.

| Topic | Contents |
|-------|----------|
| [Configuration](configuration.md) | `ClusterConfig`, `SegmentConfig`, `ExtremeConfig`, `Distribution`, `MinMaxMean` |
| [Results](results.md) | `AggregationResult`, `AccuracyMetrics`, `ClusteringResult` |
| [Pipeline internals](pipeline.md) | `run_pipeline`, the four phases, and the stage functions |
| [Tuning](tuning.md) | Hyperparameter sweeps over aggregation settings |
| [Utilities](utilities.md) | Options, weights, plotting, low-level aggregation helpers |

## Aggregation

::: tsam.api
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3
