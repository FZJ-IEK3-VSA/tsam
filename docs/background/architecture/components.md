# Components

The diagram below shows the architectural components of `tsam` and how they collaborate. Each component is a logical building block with a stated responsibility; the responsibility table further down lists the specific modules implementing each one.

## Component diagram

![Component diagram](../../assets/architecture/components_diagram.svg)

## Component responsibilities

### Public API

| Module | Responsibility |
|--------|---------------|
| [`api.py`](../../api/index.md) | Modern, function-based entry point. `aggregate()` accepts a DataFrame plus config dataclasses, builds a `PipelineConfig`, calls `run_pipeline()`, and wraps the output as an `AggregationResult`. |


### Config & Results

| Module | Responsibility |
|--------|---------------|
| [`config.py`](../../api/configuration.md) | User-facing config dataclasses (`ClusterConfig`, `ExtremeConfig`, `SegmentConfig`) plus representation specs (`Distribution`, `MinMaxMean`) and the post-clustering `ClusteringResult`. |
| [`result.py`](../../api/results.md) | User-facing result objects (`AggregationResult`, `AccuracyMetrics`) returned from `aggregate()`. |
| [`pipeline/types.py`](../../api/pipeline.md) | Internal types: `PipelineConfig` (full configuration consumed by `run_pipeline`) and `PipelineResult` (raw pipeline output, later wrapped into `AggregationResult`). |

### Pipeline orchestrator

`pipeline/orchestrator.py` holds `run_pipeline()` and the four phase functions
it calls — `prepare_data`, `cluster_and_postprocess`, `format_and_reconstruct`,
`assemble_result`. The [Pipeline Guide](pipeline_guide.md) explains what each
phase does; the [Pipeline internals](../../api/pipeline.md) reference documents
the functions.

### Pipeline modules

Each pipeline module is a small file of pure functions with explicit inputs and
outputs — data flows through arguments rather than instance state, which is what
makes the pipeline testable and reorderable.

| Module | Responsibility |
|--------|---------------|
| `pipeline/normalize.py` | Scale columns to [0, 1] and invert it (`normalize` / `denormalize`). |
| `pipeline/periods.py` | Reshape the flat series into a (period, timestep) matrix; optional period-sum features. |
| `pipeline/clustering.py` | Group periods and pick representatives; dispatches to a `utils/` backend or scikit-learn. |
| `pipeline/extremes.py` | Inject extreme-value periods into the cluster set. |
| `pipeline/rescale.py` | Adjust representatives so column means match the original. |
| `pipeline/segment.py` | Merge adjacent timesteps within a typical period. |
| `pipeline/accuracy.py` | Reconstruct the full series and compute accuracy metrics. |
| `pipeline/orchestrator.py` | `run_pipeline()` plus the four phase functions and the glue with no dedicated stage module. |

### Clustering backends

The `utils/` package holds the actual algorithm implementations. `clustering.py` dispatches to them based on `ClusterConfig.method`:

| Module | Method |
|--------|--------|
| `utils/k_medoids_exact.py` | `kmedoids` — MILP formulation; needs a solver. |
| `utils/k_maxoids.py` | `kmaxoids` — maxoid-based variant. |
| `utils/segmentation.py` | segmentation algorithms used by `pipeline/segment.py`. |
| `utils/duration_representation.py` | duration-curve representation used by `representation="distribution"`, `"distribution_minmax"`, and `Distribution(...)`. |

`kmeans`, `hierarchical`, and `contiguous` delegate to scikit-learn. `averaging` is implemented in `period_aggregation.py` itself.

### Auxiliary

| Module | Responsibility |
|--------|---------------|
| `tuning.py`, `hyperparametertuning.py` | Sweep configurations and rank by accuracy. Calls `aggregate()` in a loop. |
| `plot.py` | Plotly-based visualization. Lazy-imported (heavy dependency). |
| `options.py` | Global numerical tolerances and limits (e.g. `min_weight`). |
| `weights.py` | Validation helpers for the user-supplied `weights` dict. |
| `exceptions.py` | Custom warning/exception types (e.g. `LegacyAPIWarning`). |
