# Components

The diagram below shows the architectural components of `tsam` and how they collaborate. Each component is a logical building block with a stated responsibility; the responsibility table further down lists the specific modules implementing each one.

## Component diagram

![Component diagram](../../assets/architecture/components_diagram.svg)

## Component responsibilities

### Public API

| Module | Responsibility |
|--------|---------------|
| [`api.py`](../../api/tsam/api.md) | Modern, function-based entry point. `aggregate()` accepts a DataFrame plus config dataclasses, builds a `PipelineConfig`, calls `run_pipeline()`, and wraps the output as an `AggregationResult`. |


### Config & Results

| Module | Responsibility |
|--------|---------------|
| [`config.py`](../../api/tsam/config.md) | User-facing config dataclasses (`ClusterConfig`, `ExtremeConfig`, `SegmentConfig`) plus representation specs (`Distribution`, `MinMaxMean`) and the post-clustering `ClusteringResult`. |
| [`result.py`](../../api/tsam/result.md) | User-facing result objects (`AggregationResult`, `AccuracyMetrics`) returned from `aggregate()`. |
| [`pipeline/types.py`](../../api/tsam/pipeline/types.md) | Internal types: `PipelineConfig` (full configuration consumed by `run_pipeline`) and `PipelineResult` (raw pipeline output, later wrapped into `AggregationResult`). |

### Pipeline orchestrator

`pipeline/__init__.py` contains `run_pipeline()`, which executes the eight-stage aggregation flow in four phases:

1. **`_prepare_data`** — normalize, unstack to periods, weight, augment with period sums.
2. **`_cluster_and_postprocess`** — cluster, add extremes, compute counts, rescale.
3. **`_format_and_reconstruct`** — format representatives, segment, denormalize, reconstruct.
4. **`_assemble_result`** — build `ClusteringResult` and `PipelineResult`.

For the step-by-step walk-through, see the [Pipeline Guide](pipeline_guide.md).

### Pipeline modules

Each pipeline module is a small file of pure functions with explicit inputs and outputs. Data flows through function arguments rather than instance attributes. This is the property that makes the pipeline testable and reorderable. The eight-step aggregation flow is illustrated in the [Pipeline Guide](pipeline_guide.md); the table below maps each step to the module that implements it.

| Module | Step(s) implemented | Notes |
|--------|---------------------|-------|
| `pipeline/normalize.py` | 1 Normalize · 6 Denormalize | Scale to [0, 1] per column. |
| `pipeline/periods.py` | 2 Unstack · 2b Period-sum features (optional) | Reshape flat time series → (period, timestep) matrix. |
| `pipeline/clustering.py` | 3 Cluster | Calls `period_aggregation.aggregate_periods()`, which dispatches to scikit-learn or to a backend in `utils/`. |
| `pipeline/extremes.py` | 3a Extremes (optional) | Inject extreme-value periods into the cluster set. |
| `pipeline/rescale.py` | 4a Rescale (optional) | Adjust representatives so column means match the original. |
| `pipeline/segment.py` | 5a Segment (optional) | Merge adjacent timesteps within a typical period. |
| `pipeline/accuracy.py` | 7 Reconstruct | Compute MAE, RMSE, etc. between original and reconstructed (accuracy is computed lazily on `PipelineResult`). |
| `pipeline/__init__.py` | 2a Apply weights · 4 Trim · unweight · count · 5 Format representatives · 8 Assemble | The `run_pipeline()` orchestrator and its glue helpers — the steps with no dedicated stage module live here. |

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
