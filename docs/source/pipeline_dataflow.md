# Pipeline Data Flow

This document describes the complete data flow through the `tsam` aggregation
pipeline: which functions are called, in what order, what each receives,
and where every parameter originates.

---

## Entry points

There are two ways into the pipeline. Both end up calling `run_pipeline()`.

### `api.aggregate()`

The primary user entry point. Validates inputs, applies defaults, computes
`n_timesteps_per_period` from `period_duration` / `temporal_resolution`, then
calls `run_pipeline()`.

```
aggregate(data, n_clusters, period_duration, temporal_resolution,
          cluster, segments, extremes, preserve_column_means,
          rescale_exclude_columns, round_decimals, numerical_tolerance)
    │
    ├── validates data, n_clusters, period_duration, temporal_resolution
    ├── defaults cluster to ClusterConfig()
    ├── validates segments.n_segments <= n_timesteps_per_period
    ├── validates extreme columns exist in data
    ├── validates weight columns exist in data
    │
    └── run_pipeline(data, n_clusters, n_timesteps_per_period,
                     cluster=cluster, extremes=extremes, segments=segments,
                     rescale_cluster_periods=preserve_column_means,
                     rescale_exclude_columns=..., round_decimals=...,
                     numerical_tolerance=..., temporal_resolution=...)
```

### `ClusteringResult.apply()`

The transfer entry point. Reconstructs a minimal `ClusterConfig` and a
`PredefParams` from the stored clustering, then calls `run_pipeline()`.

```
clustering_result.apply(data, temporal_resolution, round_decimals, numerical_tolerance)
    │
    ├── validates n_timesteps_per_period matches data
    ├── validates n_periods matches data
    │
    ├── builds: cluster = ClusterConfig(representation=self.representation)
    │           (weights=None, normalize_column_means=False — defaults)
    │
    ├── builds: predef = PredefParams(
    │       cluster_order, cluster_center_indices, extreme_cluster_idx,
    │       segment_order, segment_durations, segment_centers)
    │
    └── run_pipeline(data, n_clusters, n_timesteps_per_period,
                     cluster=cluster, segments=segments, predef=predef,
                     rescale_cluster_periods=self.preserve_column_means, ...)
```

**Key difference during transfer:** `weights=None` and
`normalize_column_means=False`, so `NormalizedData` stores no weights and
no mean-normalization. The clustering assignments carry the structural
information; denormalization is a plain `scaler.inverse_transform`.

---

## `run_pipeline()` parameter sources

| Parameter | Source: `aggregate()` | Source: `apply()` |
|---|---|---|
| `data` | user input | user input |
| `n_clusters` | user input | `self.n_clusters` (derived from assignments) |
| `n_timesteps_per_period` | `period_duration / resolution` | `self.n_timesteps_per_period` |
| `cluster` | user input or `ClusterConfig()` | `ClusterConfig(representation=...)` |
| `extremes` | user input or `None` | `None` (extremes handled via `predef`) |
| `segments` | user input or `None` | reconstructed from stored fields |
| `rescale_cluster_periods` | `preserve_column_means` | `self.preserve_column_means` |
| `rescale_exclude_columns` | user input | `self.rescale_exclude_columns` |
| `round_decimals` | user input | user input |
| `numerical_tolerance` | user input | user input |
| `temporal_resolution` | user input | stored or user input |
| `predef` | `None` | built from stored assignments |

---

## Pipeline steps

### Step 1: Normalize

```
normalize(data, cluster.weights, cluster.normalize_column_means)
    → NormalizedData
```

| Module | `pipeline/normalize.py` |
|---|---|
| **Input** | `data` (raw DataFrame) |
| **From `ClusterConfig`** | `weights`, `normalize_column_means` |
| **Output** | `NormalizedData` |

Operations:
1. Sort columns alphabetically
2. Cast to float
3. Fit `MinMaxScaler`, transform to [0, 1]
4. Store column means (`normalized_mean`)
5. If `normalize_column_means`: divide each column by its mean
6. Apply per-column `weights`

`NormalizedData` captures everything needed for all downstream steps:

| Field | Description | Downstream consumers |
|---|---|---|
| `values` | Normalized + weighted DataFrame | `unstack_to_periods`, `compute_accuracy` |
| `scaler` | Fitted `MinMaxScaler` | `denormalize` (both call sites) |
| `normalized_mean` | Column means before sameMean division | `denormalize` (when `normalize_column_means=True`) |
| `original_data` | Sorted, float-cast copy of input | `add_extreme_periods`, `rescale_representatives`, `reconstruct`, bounds check, output reindex |
| `weights` | Per-column weights (or `None`) | `denormalize`, `rescale_representatives`, `compute_accuracy` |
| `normalize_column_means` | Whether sameMean was applied | `denormalize`, `rescale_representatives` |

### Step 2: Unstack to periods

```
unstack_to_periods(norm_data.values, n_timesteps_per_period)
    → PeriodProfiles
```

| Module | `pipeline/periods.py` |
|---|---|
| **Input** | `norm_data.values` |
| **From `run_pipeline`** | `n_timesteps_per_period` |
| **Output** | `PeriodProfiles` |

Reshapes the flat time series into a matrix of period profiles. Pads with
repeated initial rows if the time series length is not an integer multiple
of the period length.

`PeriodProfiles` fields used downstream:

| Field | Downstream consumers |
|---|---|
| `profiles_dataframe` | `add_period_sum_features`, clustering functions, `add_extreme_periods`, `rescale_representatives` |
| `profiles_dataframe.values` (= `candidates`) | all three clustering functions |
| `column_index` | `_representatives_to_dataframe` |
| `time_index` | `PipelineResult` |
| `n_columns` | `cluster_sorted_periods` |

### Step 3: Add period sum features (optional)

```
if cluster.include_period_sums:
    add_period_sum_features(profiles_dataframe, candidates)
        → (augmented_candidates, n_extra)
```

| Module | `pipeline/periods.py` |
|---|---|
| **Gate** | `cluster.include_period_sums` |
| **Input** | `period_profiles.profiles_dataframe`, `candidates` |
| **Output** | augmented `candidates` array, count of extra features to trim later |

Appends per-column period sums as extra columns to bias clustering toward
preserving totals. The extra features are trimmed off representatives in
step 5.

### Step 4: Cluster

Three mutually exclusive branches, all returning the same triple:
`(cluster_centers, cluster_center_indices, cluster_order)`.

#### Branch A: Predefined assignments (`predef is not None`)

```
use_predefined_assignments(candidates, predef.cluster_order,
    predef.cluster_center_indices, representation_method,
    representation_dict, n_timesteps_per_period)
```

| Module | `pipeline/clustering.py` |
|---|---|
| **Input** | `candidates` |
| **From `PredefParams`** | `cluster_order`, `cluster_center_indices` |
| **From `ClusterConfig`** | `get_representation()` |
| **Derived** | `representation_dict` (built from `data.columns`, all "mean", sorted) |

#### Branch B: Standard clustering (`predef is None`, `use_duration_curves=False`)

```
cluster_periods(candidates, n_clusters, cluster.method,
    cluster.solver, representation_method, representation_dict,
    n_timesteps_per_period)
```

| Module | `pipeline/clustering.py` |
|---|---|
| **Input** | `candidates` |
| **From `ClusterConfig`** | `method`, `solver`, `get_representation()` |
| **Derived** | `representation_dict`, `n_clusters` |

#### Branch C: Duration curve clustering (`predef is None`, `use_duration_curves=True`)

```
cluster_sorted_periods(candidates, profiles_values, n_columns,
    n_clusters, cluster.method, cluster.solver,
    representation_method, representation_dict, n_timesteps_per_period)
```

| Module | `pipeline/clustering.py` |
|---|---|
| **Input** | `candidates`, `period_profiles.profiles_dataframe.values`, `period_profiles.n_columns` |
| **From `ClusterConfig`** | `method`, `solver`, `get_representation()`, `use_duration_curves` (gate) |

All three branches delegate to `periodAggregation.aggregatePeriods()` and
`representations.representations()` internally.

### Step 5: Trim eval features

```python
cluster_periods_list = [center[:del_cluster_params] for center in cluster_centers]
```

Inline. Removes period-sum features appended in step 3 from the
representative vectors.

### Step 6: Add extreme periods (optional)

```
if extremes is not None:
    add_extreme_periods(profiles_dataframe, cluster_periods_list,
        cluster_order, extremes.method, extremes.max_value,
        extremes.min_value, extremes.max_period, extremes.min_period,
        columns)
    → (cluster_periods_list, cluster_order, extreme_cluster_idx, extreme_periods_info)
```

| Module | `pipeline/extremes.py` |
|---|---|
| **Gate** | `extremes is not None` |
| **Input** | `period_profiles.profiles_dataframe`, `cluster_periods_list`, `cluster_order` |
| **From `ExtremeConfig`** | `method`, `max_value`, `min_value`, `max_period`, `min_period` |
| **Derived** | `columns` = `list(norm_data.original_data.columns)` |
| **Output** | mutated `cluster_periods_list`, `cluster_order`; new `extreme_cluster_idx`, `extreme_periods_info` |

Depending on `method`:
- **append**: adds extreme periods as new clusters at the end
- **new_cluster**: adds new clusters and reassigns nearby periods
- **replace**: overwrites relevant columns of nearest existing cluster

### Step 7: Compute cluster weights

```
_count_occurrences(cluster_order) → cluster_period_no_occur
```

Counts how many original periods are assigned to each cluster.

### Step 8: Rescale representatives (optional)

```
if rescale_cluster_periods:
    rescale_representatives(cluster_periods_list, cluster_period_no_occur,
        extreme_cluster_idx, profiles_dataframe, norm_data,
        n_timesteps_per_period, rescale_exclude_columns)
    → (rescaled_cluster_periods_list, rescale_deviations)
```

| Module | `pipeline/rescale.py` |
|---|---|
| **Gate** | `rescale_cluster_periods` |
| **Input** | `cluster_periods_list`, `cluster_period_no_occur`, `extreme_cluster_idx`, `period_profiles.profiles_dataframe` |
| **From `NormalizedData`** | `original_data` (upper bound), `weights` (scale upper bound), `normalize_column_means` (scale upper bound) |
| **From `run_pipeline`** | `n_timesteps_per_period`, `rescale_exclude_columns` |
| **Output** | rescaled array, per-column deviation dict |

Iteratively rescales non-extreme cluster representatives so each column's
weighted sum matches the original time series total. Extreme clusters are
excluded from rescaling.

### Step 9: Adjust for partial periods

```python
if len(data) % n_timesteps_per_period != 0:
    cluster_period_no_occur[last_cluster] -= fractional_adjustment
```

Inline. Reduces the last cluster's weight proportionally if the time series
does not divide evenly into periods.

### Step 10: Format representatives to DataFrame

```
_representatives_to_dataframe(cluster_periods_list, period_profiles.column_index)
    → normalized_typical_periods
```

Reshapes the list of flat 1-D representative vectors into a DataFrame
indexed by `(PeriodNum, TimeStep)` with the original column names.

### Step 11: Segment typical periods (optional)

```
if segments is not None:
    segment_typical_periods(normalized_typical_periods, segments.n_segments,
        n_timesteps_per_period, segment_representation,
        representation_dict, predef.segment_order,
        predef.segment_durations, predef.segment_centers)
    → (segmented_df, predicted_segmented_df, segment_center_indices)
```

| Module | `pipeline/segment.py` (wraps `utils/segmentation.py`) |
|---|---|
| **Gate** | `segments is not None` |
| **Input** | `normalized_typical_periods` |
| **From `SegmentConfig`** | `n_segments`, `representation` |
| **From `PredefParams`** | `segment_order`, `segment_durations`, `segment_centers` (if transfer) |
| **Derived** | `segment_representation` (falls back to `cluster.get_representation()`), `representation_dict` |
| **Output** | `segmented_df`, `predicted_segmented_df` (for reconstruction), `segment_center_indices` |

After segmentation, `normalized_typical_periods` is replaced with the
segmented version (dropping the `Original Start Step` index level).

### Step 12: Denormalize

```
denormalize(normalized_typical_periods, norm_data)
    → typical_periods

if round_decimals is not None:
    typical_periods.round(decimals=round_decimals)
```

| Module | `pipeline/normalize.py` |
|---|---|
| **Input** | `normalized_typical_periods` |
| **From `NormalizedData`** | `weights`, `normalize_column_means`, `normalized_mean`, `scaler` |
| **Output** | `typical_periods` (in original units) |

Operations:
1. Undo weights (divide each column by its weight)
2. Undo sameMean (multiply by `normalized_mean`)
3. Inverse-transform via stored `scaler`

Rounding is applied after denormalization as a formatting step.

### Step 13: Bounds check

```
_warn_if_out_of_bounds(typical_periods, norm_data.original_data, numerical_tolerance)
```

Warns if any column's max/min in `typical_periods` exceeds the
original data's max/min beyond `numerical_tolerance`.

### Step 14: Reconstruct and compute accuracy

```
reconstruct(normalized_typical_periods, cluster_order,
    period_profiles, norm_data, segmentation_active, predicted_segmented_df)
    → (reconstructed_data, normalized_predicted)

if round_decimals is not None:
    reconstructed_data.round(decimals=round_decimals)

compute_accuracy(norm_data.values, normalized_predicted, norm_data)
    → accuracy_df
```

| Module | `pipeline/accuracy.py` |
|---|---|

**`reconstruct()`:**

| Parameter | Source |
|---|---|
| `normalized_typical_periods` | step 10 (or step 11 if segmented) |
| `cluster_order` | step 4 (or step 6 if extremes modified it) |
| `period_profiles` | step 2 (`column_index`, `profiles_dataframe.index`) |
| `norm_data` | step 1 (`original_data` for trimming/index, `scaler`/`normalized_mean`/`normalize_column_means` via `denormalize`) |
| `segmentation_active` | `segments is not None` |
| `predicted_segmented_df` | step 11 (or `None`) |

Internally calls `denormalize(normalized_predicted, norm_data, apply_weights=False)`.
Weights are not undone during reconstruction because accuracy is measured
in the normalized-but-unweighted space.

**`compute_accuracy()`:**

| Parameter | Source |
|---|---|
| `normalized_original` | `norm_data.values` (from step 1) |
| `normalized_predicted` | returned by `reconstruct()` |
| `norm_data` | step 1 (reads `weights` to undo weighting before error calculation) |

Computes RMSE, MAE, and duration-curve RMSE per column. If weights are
present, divides the original normalized data by the weight to compare in
unweighted normalized space.

### Step 15: Build ClusteringResult

```
_build_clustering_result(cluster_center_indices, extreme_periods_info,
    extremes_config, cluster_order, segmented_df, segment_center_indices,
    n_timesteps_per_period, temporal_resolution, original_data,
    cluster_config, segment_config, rescale_cluster_periods,
    rescale_exclude_columns, extreme_cluster_idx)
    → ClusteringResult
```

Assembles all clustering metadata into a serializable `ClusteringResult`
for transfer/reuse. The three config objects (`ClusterConfig`,
`SegmentConfig`, `ExtremeConfig`) are stashed as reference fields.

### Step 16: Assemble PipelineResult

Restores original column order on output DataFrames (the pipeline sorts
columns alphabetically internally), then packs everything into a
`PipelineResult`.

---

## Config field consumption map

### `ClusterConfig`

| Field | Consumed in step | By function | Via |
|---|---|---|---|
| `method` | 4 | `cluster_periods()` / `cluster_sorted_periods()` | direct |
| `representation` | 4, 11 (fallback) | clustering functions, `segment_typical_periods()` | `cluster.get_representation()` |
| `weights` | 1 | `normalize()` | direct; then stored in `norm_data.weights` |
| `normalize_column_means` | 1 | `normalize()` | direct; then stored in `norm_data.normalize_column_means` |
| `use_duration_curves` | 4 | branch gate in `run_pipeline()` | direct |
| `include_period_sums` | 3 | `add_period_sum_features()` | direct |
| `solver` | 4 | `cluster_periods()` / `cluster_sorted_periods()` | direct |

After step 1, `weights` and `normalize_column_means` are never read from
`ClusterConfig` again. All downstream access goes through `NormalizedData`.

After step 4, `method`, `solver`, `use_duration_curves`, and
`include_period_sums` are never read again. The clustering phase is complete.

### `ExtremeConfig`

| Field | Consumed in step | By function |
|---|---|---|
| `method` | 6 | `add_extreme_periods()` |
| `max_value` | 6 | `add_extreme_periods()` |
| `min_value` | 6 | `add_extreme_periods()` |
| `max_period` | 6 | `add_extreme_periods()` |
| `min_period` | 6 | `add_extreme_periods()` |

All fields are consumed exclusively during step 6. Zero coupling to
normalization, clustering, rescaling, or reconstruction.

### `SegmentConfig`

| Field | Consumed in step | By function |
|---|---|---|
| `n_segments` | 11 | `segment_typical_periods()` |
| `representation` | 11 | `segment_typical_periods()` (via `segment_representation`) |

Both fields are consumed exclusively in step 11.

---

## Data object lifecycle

### `NormalizedData` — created once, read everywhere

```
Step  1  normalize()                    creates NormalizedData
Step  2  unstack_to_periods()           reads .values
Step  6  add_extreme_periods()          reads .original_data (for column list)
Step  8  rescale_representatives()      reads .original_data, .weights, .normalize_column_means
Step 12  denormalize()                  reads .weights, .normalize_column_means, .normalized_mean, .scaler
Step 13  _warn_if_out_of_bounds()       reads .original_data
Step 14  reconstruct()                  reads .original_data (length, index, columns)
         └─ denormalize()              reads .normalize_column_means, .normalized_mean, .scaler
                                        (apply_weights=False, so .weights is skipped)
Step 14  compute_accuracy()             reads .weights
Step 16  output reindex                 reads .original_data
```

### `PeriodProfiles` — created once, read by clustering and reshaping

```
Step  2  unstack_to_periods()           creates PeriodProfiles
Step  3  add_period_sum_features()      reads .profiles_dataframe
Step  4  clustering functions            reads .profiles_dataframe.values (= candidates), .n_columns
Step  6  add_extreme_periods()          reads .profiles_dataframe
Step  8  rescale_representatives()      reads .profiles_dataframe
Step 10  _representatives_to_dataframe() reads .column_index
Step 16  PipelineResult                  reads .time_index
```

### `PredefParams` — transfer-only, skips clustering

```
Step  4  use_predefined_assignments()   reads .cluster_order, .cluster_center_indices
Step  6  (fallback)                     reads .extreme_cluster_idx
Step 11  segment_typical_periods()      reads .segment_order, .segment_durations, .segment_centers
```

---

## Output assembly

`run_pipeline()` returns a `PipelineResult`, which `api._build_aggregation_result()`
converts into the user-facing `AggregationResult`:

```
PipelineResult                          AggregationResult
─────────────                           ─────────────────
typical_periods                    →    cluster_representatives (renamed index levels)
cluster_weights                    →    cluster_weights
n_timesteps_per_period             →    n_timesteps_per_period
time_index                         →    _time_index
original_data                      →    _original_data → .original (property)
clustering_duration                →    clustering_duration
rescale_deviations                 →    accuracy.rescale_deviations
segmented_df                       →    _segmented_df → .assignments (property, segment_idx)
reconstructed_data                 →    _reconstructed_data → .reconstructed (property)
accuracy_indicators                →    accuracy (AccuracyMetrics: .rmse, .mae, .rmse_duration)
clustering_result                  →    clustering (ClusteringResult)
                                        segment_durations (from clustering_result)
```

`AggregationResult` also provides derived properties:
- `n_clusters` — from `cluster_representatives` index
- `n_segments` — from `clustering.n_segments`
- `cluster_assignments` — from `clustering.cluster_assignments`
- `residuals` — `original - reconstructed`
- `assignments` — per-timestep mapping DataFrame
- `plot` — plotting accessor
