# Components

The diagram below shows the internal structure of the `tsam` package: which modules exist, what each one is responsible for, and how they depend on each other.

## Component diagram

```kroki-mermaid
%%{init: {'flowchart': {'padding': 18}}}%%
graph LR
    subgraph publicapi["Public API"]
        api["api.py<br/>aggregate()"]
        legacy["timeseriesaggregation.py<br/>TimeSeriesAggregation<br/>(legacy)"]
    end

    subgraph cfgres["Config & Results"]
        cfg["config.py<br/>ClusterConfig<br/>ExtremeConfig<br/>SegmentConfig<br/>ClusteringResult"]
        res["result.py<br/>AggregationResult<br/>AccuracyMetrics"]
        types["pipeline/types.py<br/>PipelineConfig<br/>PipelineResult"]
    end

    subgraph orch["Pipeline orchestrator"]
        runp["pipeline package<br/>run_pipeline()"]
    end

    subgraph stages["Pipeline stages"]
        norm[normalize]
        per[periods]
        clu[clustering]
        ext[extremes]
        rsc[rescale]
        seg[segment]
        acc[accuracy]
    end

    subgraph backends["Clustering backends"]
        kmex["utils/k_medoids_exact<br/>(MILP)"]
        kmco["utils/k_medoids_contiguity"]
        kmax["utils/k_maxoids"]
        segu["utils/segmentation"]
        dur["utils/duration_representation"]
    end

    subgraph aux["Auxiliary"]
        tune["tuning.py<br/>hyperparameter search"]
        plot["plot.py<br/>(lazy import, plotly)"]
        opts["options.py<br/>global options"]
    end

    api --> cfg
    api --> types
    api --> runp
    api --> res
    legacy --> runp

    runp --> norm
    runp --> per
    runp --> clu
    runp --> ext
    runp --> rsc
    runp --> seg
    runp --> acc

    clu --> kmex
    clu --> kmco
    clu --> kmax
    clu --> dur
    seg --> segu

    tune --> api

    %% Kroki's styles_* config cannot target subgraph clusters, so style them
    %% theme-neutral here: transparent fill works on any background, mid-grey
    %% stroke stays visible in both light and dark mode.
    style publicapi fill:none,stroke:#8a8a8a
    style cfgres fill:none,stroke:#8a8a8a
    style orch fill:none,stroke:#8a8a8a
    style stages fill:none,stroke:#8a8a8a
    style backends fill:none,stroke:#8a8a8a
    style aux fill:none,stroke:#8a8a8a
```

## Component responsibilities

### Public API

| Module | Responsibility |
|--------|---------------|
| [`api.py`](../../api/tsam/api.md) | Modern, function-based entry point. `aggregate()` accepts a DataFrame plus config dataclasses, builds a `PipelineConfig`, calls `run_pipeline()`, and wraps the output as an `AggregationResult`. |
| [`timeseriesaggregation.py`](../../api/tsam/timeseriesaggregation.md) | Legacy class-based API (`TimeSeriesAggregation.create_typical_periods()`). Kept for backwards compatibility — also delegates to `run_pipeline()` underneath. See [ADR 0001](../decisions/0001-v4-pipeline.md). |

### Config & Results

| Module | Responsibility |
|--------|---------------|
| [`config.py`](../../api/tsam/config.md) | User-facing config dataclasses (`ClusterConfig`, `ExtremeConfig`, `SegmentConfig`) plus representation specs (`Distribution`, `MinMaxMean`) and the post-clustering `ClusteringResult`. |
| [`result.py`](../../api/tsam/result.md) | User-facing result objects (`AggregationResult`, `AccuracyMetrics`) returned from `aggregate()`. |
| [`pipeline/types.py`](../../api/tsam/pipeline/types.md) | Internal types: `PipelineConfig` (full configuration consumed by `run_pipeline`) and `PipelineResult` (raw pipeline output, later wrapped into `AggregationResult`). |

### Pipeline orchestrator

`pipeline/__init__.py` contains `run_pipeline()`, which executes the 16-step aggregation flow in four phases:

1. **`_prepare_data`** — normalize, unstack to periods, weight, augment with period sums.
2. **`_cluster_and_postprocess`** — cluster, add extremes, compute counts, rescale.
3. **`_format_and_reconstruct`** — format representatives, segment, denormalize, reconstruct, compute accuracy.
4. **`_assemble_result`** — build `ClusteringResult` and `PipelineResult`.

For the step-by-step walk-through, see the [Pipeline Guide](../pipeline_guide.md).

### Pipeline stages

Each stage is a small module of pure functions with explicit inputs and outputs. They have no shared state — all data flows through function arguments. This is the property that makes the pipeline testable and reorderable.

| Module | Stage | Notes |
|--------|-------|-------|
| `pipeline/normalize.py` | normalize / denormalize | Scale to [0, 1] per column. |
| `pipeline/periods.py` | unstack / period-sum features | Reshape flat time series → (period, timestep) matrix. |
| `pipeline/clustering.py` | cluster | Dispatches to a backend in `utils/`. |
| `pipeline/extremes.py` | extreme periods | Inject extreme-value periods into the cluster set. |
| `pipeline/rescale.py` | rescale | Adjust representatives so column means match the original. |
| `pipeline/segment.py` | segment | Merge adjacent timesteps within a typical period. |
| `pipeline/accuracy.py` | reconstruct + metrics | Compute MAE, RMSE, etc. between original and reconstructed. |

### Clustering backends

The `utils/` package holds the actual algorithm implementations. `clustering.py` dispatches to them based on `ClusterConfig.method`:

| Module | Method |
|--------|--------|
| `utils/k_medoids_exact.py` | `k_medoids_exact` — MILP formulation; needs a solver. |
| `utils/k_medoids_contiguity.py` | `k_medoids_contiguity` — contiguity-constrained k-medoids. |
| `utils/k_maxoids.py` | `k_maxoids` — maxoid-based variant. |
| `utils/segmentation.py` | segmentation algorithms used by `pipeline/segment.py`. |
| `utils/duration_representation.py` | duration-curve sorting used when `use_duration_curves=True`. |

Other methods (`hierarchical`, `k_means`, `k_medoids`) delegate to scipy/scikit-learn.

### Auxiliary

| Module | Responsibility |
|--------|---------------|
| `tuning.py`, `hyperparametertuning.py` | Sweep configurations and rank by accuracy. Calls `aggregate()` in a loop. |
| `plot.py` | Plotly-based visualization. Lazy-imported (heavy dependency). |
| `options.py` | Global numerical tolerances and limits (e.g. `min_weight`). |
| `weights.py` | Validation helpers for the user-supplied `weights` dict. |
| `exceptions.py` | Custom warning/exception types (e.g. `LegacyAPIWarning`). |

## Key dependency rules

These constraints keep the architecture coherent — please respect them when adding code:

1. **The pipeline package never imports the public API.** `pipeline/` is the engine; `api.py` is the steering wheel. The dependency is one-way.
2. **Pipeline stages don't import each other directly.** Stages communicate only through values passed by the orchestrator. This is what makes them testable in isolation and reorderable in principle.
3. **`utils/` knows nothing about `pipeline/types.py` or `config.py`.** Backends accept plain arrays/arguments. This keeps them reusable and easy to test.
4. **Legacy and modern APIs share one engine.** Both `api.aggregate()` and `TimeSeriesAggregation.create_typical_periods()` flow through `run_pipeline()`. Behavioural changes must be made in the pipeline, not duplicated.

## When this diagram changes

Update this page when:

- A new pipeline stage is added or removed.
- A new clustering backend is introduced.
- A module is split, merged, or moved between groups (Public API / Config / Pipeline / Backends / Auxiliary).
- A dependency rule above is intentionally broken (record *why* in an [ADR](../decisions/index.md)).

Routine refactors within a single module do **not** require updating this page.
