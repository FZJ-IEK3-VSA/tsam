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
[API reference](../../api/index.md), which tracks the code. The public input and
output types live in the API reference —
[Configuration](../../api/configuration.md) and [Results](../../api/results.md).

---

## Overview

The diagram below traces the user-facing data flow on the left — a **time series**
and a **`Config`** go into [`aggregate()`][tsam.aggregate], which returns the
**clustered data** — through the four-phase `run_pipeline()` down the centre, with
the milestone dataclass passed between phases. The right column lists the
clustering, representation, and segmentation options that Phases 2 and 3 draw on.

![Pipeline data flow](../../assets/architecture/pipeline_diagram.svg)

---

## Entry points

Two ways in, both ending in `run_pipeline()`:

**[`tsam.aggregate()`][tsam.aggregate]** — the primary API. It validates inputs,
derives `n_timesteps_per_period` from `period_duration / temporal_resolution`,
and calls `run_pipeline()`. See its [API reference][tsam.aggregate] for all
parameters.

```python
result = tsam.aggregate(df, n_clusters=8, period_duration=24)
```

**[`ClusteringResult.apply()`][tsam.config.ClusteringResult]** — reuse a fitted
clustering on new data, skipping clustering in favor of the stored assignments:

```python
result1 = tsam.aggregate(df_wind, n_clusters=8)
result2 = result1.clustering.apply(df_all)
```

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

## Reference

Full signatures and options live in the API reference —
[Configuration](../../api/configuration.md),
[Results](../../api/results.md), and
[Pipeline internals](../../api/pipeline.md) (the phase and stage functions
above link straight into it). The source-tree module map is below.

??? info "Public surface"

    | Module | Responsibility |
    |--------|---------------|
    | [`api.py`](../../api/index.md) | `aggregate()` — the entry point: builds a `PipelineConfig`, runs the pipeline, wraps the output as an `AggregationResult`. |
    | [`config.py`](../../api/configuration.md) | Config dataclasses (`ClusterConfig`, `SegmentConfig`, `ExtremeConfig`, `Distribution`, `MinMaxMean`) plus the transfer object `ClusteringResult`. |
    | [`result.py`](../../api/results.md) | `AggregationResult`, `AccuracyMetrics`. |
    | [`tuning.py`, `hyperparametertuning.py`](../../api/tuning.md) | Sweep configurations and rank by accuracy (loop `aggregate()`). |
    | [`plot.py`](../../api/utilities.md) | Plotly-based visualization (lazy import). |
    | [`options.py`](../../api/utilities.md) | Global numerical options and tolerances. |

??? info "Pipeline internals"

    | Module | Responsibility |
    |--------|---------------|
    | `pipeline/orchestrator.py` | `run_pipeline()` plus the four phase functions and the glue with no dedicated stage module. |
    | `pipeline/normalize.py` | Scale columns to [0, 1] and invert it (`normalize` / `denormalize`). |
    | `pipeline/periods.py` | Reshape the flat series into a (period, timestep) matrix; optional period-sum features. |
    | `pipeline/clustering.py` | Group periods and pick representatives; dispatches to a `utils/` backend or scikit-learn. |
    | `pipeline/extremes.py` | Inject extreme-value periods into the cluster set. |
    | `pipeline/rescale.py` | Adjust representatives so column means match the original. |
    | `pipeline/segment.py` | Merge adjacent timesteps within a typical period. |
    | `pipeline/accuracy.py` | Reconstruct the full series and compute accuracy metrics. |
    | `pipeline/types.py` | Internal dataclasses: `PipelineConfig`, the phase milestones, `PipelineResult`. |
    | `period_aggregation.py` · `representations.py` | Clustering dispatch and representative computation (shared by clustering and segmentation). |
    | `utils/k_medoids_exact.py` · `utils/k_maxoids.py` | k-medoids (MILP) / k-maxoids solvers. |
    | `utils/duration_representation.py` | Duration-curve representation (for `distribution`). |
    | `utils/segmentation.py` | Constrained agglomerative segmentation. |
    | `weights.py` · `exceptions.py` | Weight validation; custom warnings. |
