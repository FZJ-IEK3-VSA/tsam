# Pipeline Data Flow

The diagram below maps how data flows through `run_pipeline()` — what shape the data has at each boundary, and which configuration switches affect which stage. It focuses on the *shape* of the data, not the logic of each step (which is covered in the [Pipeline Guide](../pipeline_guide.md)).

## High-level data flow

```kroki-mermaid
%%{init: {'flowchart': {'padding': 18}}}%%
flowchart TB
    input["pandas.DataFrame<br/>(T timesteps × C columns)"]

    subgraph prep["Phase 1: _prepare_data"]
        norm["normalize<br/>→ values in [0,1]"]
        unstack["unstack_to_periods<br/>→ (P periods, S steps, C cols)"]
        weight["apply weights<br/>→ weighted candidates"]
        augment["+ period-sum features<br/>(optional)"]
    end

    subgraph clust["Phase 2: _cluster_and_postprocess"]
        cluster["cluster<br/>→ K representatives<br/>+ cluster_order (length P)"]
        trim["trim eval features"]
        extremes["add extremes<br/>(optional)"]
        unweight["remove weights"]
        rescale["rescale<br/>(optional)"]
        partial["partial-period<br/>count adjustment"]
    end

    subgraph fmt["Phase 3: _format_and_reconstruct"]
        format["format to<br/>MultiIndex DataFrame"]
        segment["segment<br/>(optional)"]
        denorm["denormalize<br/>→ original units"]
        reconstruct["reconstruct<br/>full-length series<br/>+ accuracy metrics"]
    end

    subgraph assemble["Phase 4: _assemble_result"]
        result["PipelineResult"]
    end

    output["AggregationResult<br/>(wrapped by api.aggregate)"]

    input --> norm --> unstack --> weight --> augment
    augment --> cluster --> trim --> extremes --> unweight --> rescale --> partial
    partial --> format --> segment --> denorm --> reconstruct
    reconstruct --> result --> output
```

## Data shapes at each stage boundary

The diagram above hides the actual array shapes. This table gives the precise contract — useful when debugging or adding a new stage.

| Boundary | Shape | Notes |
|----------|-------|-------|
| Input | `DataFrame[T, C]` | T = total timesteps, C = number of columns. |
| After `normalize` | `DataFrame[T, C]` | Same shape, values in [0, 1] (or column-mean-centered). |
| After `unstack_to_periods` | `ndarray[P, S·C]` | P = T // S, S = `n_timesteps_per_period`. Each row is one period flattened across timesteps and columns. |
| After weighting | `ndarray[P, S·C]` | Same shape; each column-block scaled by its weight. |
| After period-sum augmentation | `ndarray[P, S·C + C]` | Extra C columns appended for clustering distance only. |
| After clustering | `list[ndarray]` (length K) + `ndarray[P]` | K representatives (each `S·C` long) plus a cluster-assignment vector. |
| After extremes | `list[ndarray]` (length K + E) + `ndarray[P]` | E extreme periods appended. `cluster_order` updated. |
| After rescale | same shapes | Representatives shifted so column means match the original. |
| After format | `DataFrame` with MultiIndex `(PeriodNum, TimeStep) × C` | The typical periods in normalized space. |
| After segment | `DataFrame` with `(PeriodNum, TimeStep, SegmentNum) × C` | Adjacent timesteps merged. |
| After denormalize | `DataFrame` in original units | What the user sees as `typical_periods`. |
| After reconstruct | `DataFrame[T, C]` | Full-length reconstruction for accuracy metrics. |

## Configuration → stage matrix

Each user-facing config knob affects one or more stages. Use this matrix to predict the impact of a config change.

| Config field | Stages affected |
|--------------|----------------|
| `n_clusters` | cluster, count, reconstruct |
| `n_timesteps_per_period` | unstack, every downstream stage |
| `ClusterConfig.method` | cluster (backend dispatch) |
| `ClusterConfig.weights` | weight, segment, unweight |
| `ClusterConfig.representation` | cluster, rescale |
| `ClusterConfig.include_period_sums` | augment, cluster, trim |
| `ClusterConfig.use_duration_curves` | cluster (alternative path) |
| `ClusterConfig.scale_by_column_means` | normalize, denormalize |
| `ExtremeConfig` (any field) | extremes |
| `SegmentConfig` (any field) | segment |
| `rescale_cluster_periods` | rescale |
| `rescale_exclude_columns` | rescale |
| `round_decimals` | denormalize, reconstruct |
| `numerical_tolerance` | bounds-check warnings |
| `predef` | cluster (bypass), extremes (bypass) |

## When this diagram changes

Update this page when:

- A new pipeline stage is added between existing stages.
- A data shape between stages changes (different dimensionality, different index structure).
- A new config field is added that affects a stage.

If only the *internals* of a stage change but inputs and outputs stay the same, you don't need to update this page — the [Pipeline Guide](../pipeline_guide.md) or the stage's docstring is the right place.
