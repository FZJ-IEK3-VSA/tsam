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

Each pipeline module is a small file of pure functions with explicit inputs and outputs. They have no shared state — all data flows through function arguments. This is the property that makes the pipeline testable and reorderable.

| Module | Stage | Notes |
|--------|-------|-------|
| `pipeline/normalize.py` | normalize / denormalize | Scale to [0, 1] per column. |
| `pipeline/periods.py` | unstack / period-sum features | Reshape flat time series → (period, timestep) matrix. |
| `pipeline/clustering.py` | cluster | Calls `period_aggregation.aggregate_periods()`, which dispatches to scikit-learn or to a backend in `utils/`. |
| `pipeline/extremes.py` | extreme periods | Inject extreme-value periods into the cluster set. |
| `pipeline/rescale.py` | rescale | Adjust representatives so column means match the original. |
| `pipeline/segment.py` | segment | Merge adjacent timesteps within a typical period. |
| `pipeline/accuracy.py` | reconstruct + metrics | Compute MAE, RMSE, etc. between original and reconstructed. |

### Clustering backends

The `utils/` package holds the actual algorithm implementations. `clustering.py` dispatches to them based on `ClusterConfig.method`:

| Module | Method |
|--------|--------|
| `utils/k_medoids_exact.py` | `kmedoids` — MILP formulation; needs a solver. |
| `utils/k_maxoids.py` | `kmaxoids` — maxoid-based variant. |
| `utils/segmentation.py` | segmentation algorithms used by `pipeline/segment.py`. |
| `utils/duration_representation.py` | duration-curve sorting used when `use_duration_curves=True`. |

`kmeans`, `hierarchical`, and `contiguous` delegate to scikit-learn. `averaging` is implemented in `period_aggregation.py` itself.

### Auxiliary

| Module | Responsibility |
|--------|---------------|
| `tuning.py`, `hyperparametertuning.py` | Sweep configurations and rank by accuracy. Calls `aggregate()` in a loop. |
| `plot.py` | Plotly-based visualization. Lazy-imported (heavy dependency). |
| `options.py` | Global numerical tolerances and limits (e.g. `min_weight`). |
| `weights.py` | Validation helpers for the user-supplied `weights` dict. |
| `exceptions.py` | Custom warning/exception types (e.g. `LegacyAPIWarning`). |
