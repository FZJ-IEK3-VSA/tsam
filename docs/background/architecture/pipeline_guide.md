# Pipeline Guide

When you call [`tsam.aggregate()`][tsam.aggregate] it builds a configuration and
hands off to `run_pipeline()`, which runs the aggregation in **four phases**:

1. **Prepare data** — normalize the input and reshape it into clustering candidates.
2. **Cluster & post-process** — group the periods, add extremes, count, rescale.
3. **Format & reconstruct** — shape the outputs, denormalize, rebuild the series.
4. **Assemble** — pack everything into the result object.

This page is the stable conceptual map of that flow. Each phase below names the
*stage functions* it runs and links to their full reference — the precise
signatures, parameters, and behavior live in the auto-generated
[API reference](../../api/tsam/), which tracks the code. The public input and
output types are linked from [Configuration reference](#configuration-reference)
and [Working with results](#working-with-results) at the end.

---

## Overview

The diagram below groups the eight pipeline steps into the four phases. Dashed
boxes are optional steps, each activated by the named config parameter.

![Pipeline data flow](../../assets/architecture/pipeline_diagram.svg)

**Legend — abbreviations:** T = input timesteps · C = columns (variables) · S = timesteps per period ·
P = T÷S periods · K = number of clusters · K' = extra extreme clusters · seg = segments per period ·
df = `pd.DataFrame` · arr = `np.ndarray`

**Dashed boxes** are optional steps, each activated by the named config parameter.
`2a` (weights) and `2b` (include_period_sums) **can both be active simultaneously** — when both are on,
column weights are applied first and period-sum features are then appended to the
already-weighted candidates. The double arrow between them in the diagram shows this sequential combination.

---

## Entry points

There are two ways into the pipeline. Both end up calling `run_pipeline()`.

### `tsam.aggregate()` — the primary API

```python
import tsam
from tsam import ClusterConfig, SegmentConfig, ExtremeConfig

result = tsam.aggregate(
    data,                     # DataFrame with DatetimeIndex
    n_clusters=8,             # how many typical periods
    period_duration=24,       # hours (or '1d', '24h')
    temporal_resolution=1.0,  # hours (or '1h', '15min'); auto-inferred if omitted
    cluster=ClusterConfig(),  # clustering options
    segments=SegmentConfig(n_segments=8),  # optional intra-period segmentation
    extremes=ExtremeConfig(max_value=["demand"]),  # optional extremes
    weights={"demand": 2.0},       # per-column clustering weights (optional)
    preserve_column_means=True,    # rescale so totals match
    rescale_exclude_columns=None,  # columns to skip during rescaling
    round_decimals=None,           # optional output rounding
    numerical_tolerance=1e-13,     # tolerance for bounds-check warnings
)
```

`aggregate()` validates inputs, computes `n_timesteps_per_period` from
`period_duration / temporal_resolution`, and calls `run_pipeline()`.

### `ClusteringResult.apply()` — transfer to new data

```python
# Cluster one dataset, apply the same structure to another
result1 = tsam.aggregate(df_wind, n_clusters=8)
result2 = result1.clustering.apply(df_all)
```

`apply()` reuses the stored clustering assignments and calls `run_pipeline()`
with those predefined assignments instead of clustering from scratch.

---

## Phase 1 — Prepare data

Turns the raw input into the candidate matrix the clustering stage consumes
(steps 1–2, plus optional `2a` / `2b`). Orchestrated by
[`prepare_data`][tsam.pipeline.orchestrator.prepare_data].

1. **Normalize** — scale every column to `[0, 1]` so no column dominates the
   distance. → [`normalize`][tsam.pipeline.normalize.normalize]
2. **Unstack to periods** — reshape the flat series into a
   `(period × timestep-feature)` matrix. →
   [`unstack_to_periods`][tsam.pipeline.periods.unstack_to_periods]
- **2a · Apply weights** *(optional, `weights`)* — bake per-column weights into
  a copy of the candidates so they influence clustering distance only.
- **2b · Add period-sum features** *(optional, `include_period_sums`)* — append
  per-period column sums as extra distance-only features. →
  [`add_period_sum_features`][tsam.pipeline.periods.add_period_sum_features]

**Milestone →** [`PreparedData`][tsam.pipeline.types.PreparedData] — normalized
data, period profiles, the candidate matrix, and the weight vector.

## Phase 2 — Cluster & post-process

Groups the periods, finalizes representatives, and computes how many original
periods each one stands for (steps 3–4, plus optional `3a` / `4a`). Orchestrated
by [`cluster_and_postprocess`][tsam.pipeline.orchestrator.cluster_and_postprocess].

3. **Cluster centers** — group periods and pick a representative for each. →
   [`cluster_periods`][tsam.pipeline.clustering.cluster_periods] (the
   [duration-curve][tsam.pipeline.clustering.cluster_sorted_periods] and
   [transfer][tsam.pipeline.clustering.use_predefined_assignments] variants
   handle `use_duration_curves` and `ClusteringResult.apply()`).
- **3a · Add extremes** *(optional, [`ExtremeConfig`][tsam.config.ExtremeConfig])*
  — inject extreme-value periods so peaks and troughs survive. →
  [`add_extreme_periods`][tsam.pipeline.extremes.add_extreme_periods]
4. **Trim · unweight · count** — strip the period-sum features, divide weights
   back out, count cluster occurrences, and correct the padded last period's
   weight.
- **4a · Rescale** *(optional, `preserve_column_means`)* — scale non-extreme
  centers so their occurrence-weighted means match the original totals. →
  [`rescale_representatives`][tsam.pipeline.rescale.rescale_representatives]

**Milestone →** [`ClusteringOutput`][tsam.pipeline.types.ClusteringOutput] —
representatives, cluster order, occurrence counts, and extreme/rescale metadata.

## Phase 3 — Format & reconstruct

Shapes the representatives into user-facing DataFrames, returns them to
original units, and rebuilds the full series with accuracy metrics
(steps 5–7, plus optional `5a`). Orchestrated by
[`format_and_reconstruct`][tsam.pipeline.orchestrator.format_and_reconstruct].

5. **Format representatives** — reshape the flat center vectors into a
   `(PeriodNum, TimeStep)` MultiIndex DataFrame.
- **5a · Segment** *(optional, [`SegmentConfig`][tsam.config.SegmentConfig])* —
  merge adjacent timesteps within each period into fewer segments. →
  [`segment_typical_periods`][tsam.pipeline.segment.segment_typical_periods]
6. **Denormalize** — convert the representatives back to the user's units. →
   [`denormalize`][tsam.pipeline.normalize.denormalize]
7. **Reconstruct + accuracy** — expand the typical periods back to a
   full-length series and score it. →
   [`reconstruct`][tsam.pipeline.accuracy.reconstruct],
   [`compute_accuracy`][tsam.pipeline.accuracy.compute_accuracy]

**Milestone →** [`FormattedOutput`][tsam.pipeline.types.FormattedOutput] —
denormalized typical periods, the reconstructed series, and optional
segmentation.

## Phase 4 — Assemble

Orchestrated by [`assemble_result`][tsam.pipeline.orchestrator.assemble_result].

8. **Assemble** — build the serializable, transferable
   [`ClusteringResult`][tsam.config.ClusteringResult] and pack it with the
   typical periods, counts, reconstruction, and metadata into the result that
   [`tsam.aggregate()`][tsam.aggregate] returns as an
   [`AggregationResult`][tsam.result.AggregationResult].

**Milestone →** [`PipelineResult`][tsam.pipeline.types.PipelineResult] — the
internal result that `tsam.aggregate()` wraps as an `AggregationResult`.

---

## Configuration reference

The behavior of every phase is driven by three user-facing config objects:

- [`ClusterConfig`][tsam.config.ClusterConfig] — clustering method, representation, weights, scaling.
- [`SegmentConfig`][tsam.config.SegmentConfig] — intra-period segmentation.
- [`ExtremeConfig`][tsam.config.ExtremeConfig] — which extreme periods to preserve, and how.

## Working with results

`run_pipeline()` produces an internal result that
[`tsam.aggregate()`][tsam.aggregate] wraps as the user-facing
[`AggregationResult`][tsam.result.AggregationResult] — see its page for the full
interface (representatives, counts, reconstruction, accuracy, transfer).

---

## Developer reference

??? info "`run_pipeline()` parameter sources"

    | Parameter | Source: `aggregate()` | Source: `apply()` |
    |---|---|---|
    | `data` | user input | user input |
    | `n_clusters` | user input | derived from assignments |
    | `n_timesteps_per_period` | `period_duration / resolution` | stored value |
    | `cluster` | user input or `ClusterConfig()` | `ClusterConfig(representation=...)` |
    | `extremes` | user input or `None` | `None` (handled via `predef`) |
    | `segments` | user input or `None` | reconstructed from stored fields |
    | `rescale_cluster_periods` | `preserve_column_means` | stored value |
    | `rescale_exclude_columns` | user input | stored value |
    | `predef` | `None` | built from stored assignments |

??? info "Config field consumption map"

    **ClusterConfig:**

    | Field | Step | Function |
    |---|---|---|
    | `method` | 3 | `cluster_periods()` |
    | `representation` | 3, 5a (fallback) | clustering, `segment_typical_periods()` |
    | `weights` | 2a | vectorized multiply → `weight_vector` |
    | `scale_by_column_means` | 1 | `normalize()` |
    | `use_duration_curves` | 3 | branch gate |
    | `include_period_sums` | 2b | `add_period_sum_features()` |
    | `solver` | 3 | `cluster_periods()` |

    **ExtremeConfig:** All fields consumed exclusively in step 3a by `add_extreme_periods()`.

    **SegmentConfig:** Both fields consumed exclusively in step 5a by `segment_typical_periods()`.

??? info "Output assembly: PipelineResult → AggregationResult"

    | PipelineResult field | AggregationResult property |
    |---|---|
    | `typical_periods` | `cluster_representatives` |
    | `cluster_counts` | `cluster_counts` |
    | `original_data` | `.original` |
    | `reconstructed_data` | `.reconstructed` |
    | `accuracy_indicators` | `.accuracy` (RMSE, MAE, duration RMSE) |
    | `clustering_result` | `.clustering` (for transfer/serialization) |
    | `segmented_df` | `.assignments` (segment_idx column) |

    Derived properties: `n_clusters`, `n_segments`, `cluster_assignments`, `residuals`, `plot`.

For where each module lives in the source tree, see the
[Components](components.md) page.
