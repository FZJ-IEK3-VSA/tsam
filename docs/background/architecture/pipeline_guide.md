# Pipeline Guide

This guide walks through the tsam aggregation pipeline from start to finish.
It is written for two audiences:

- **Users** who want to understand what happens when they call `tsam.aggregate()`
  and how configuration choices affect the result.
- **Developers** who need to modify or extend the pipeline code.

Each section explains one pipeline step: *what* it does, *why* it exists,
and *where* the code lives.

---

## Overview

The diagram below shows the eight main pipeline stages and their key data
structures. Dashed boxes are optional stages activated by the named config
parameter. The detailed step-by-step walkthrough that follows uses the same
step numbering as the diagram.

![Pipeline data flow](../../assets/architecture/pipeline_diagram.svg)

**Legend — abbreviations:** T = input timesteps · C = columns (variables) · S = timesteps per period ·
P = T÷S periods · K = number of clusters · K' = extra extreme clusters · seg = segments per period ·
df = `pd.DataFrame` · arr = `np.ndarray`

**Dashed boxes** are optional stages, each activated by the named config parameter.
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

`apply()` reconstructs a `PredefParams` object from the stored clustering
assignments and calls `run_pipeline()` with those predefined assignments
instead of running clustering from scratch.

---

## Step-by-step walkthrough

### Step 1: Normalize

| | |
|---|---|
| **Module** | `pipeline/normalize.py` |
| **Function** | `normalize()` |
| **Config** | `ClusterConfig.scale_by_column_means` |
| **Produces** | `NormalizedData` |
| **Key fields** | `values: pd.DataFrame`, `scaler: MinMaxScaler`, `normalized_mean: pd.Series`, `scale_by_column_means: bool` |

This step prepares the raw data for clustering by removing scale
differences between columns.

**What happens:**

1. **Cast to float** (in case of integer columns).
2. **Min-max scale** each column to [0, 1] using scikit-learn's
   `MinMaxScaler`. The fitted scaler is stored for later inversion.
3. **Column-mean normalization** (optional, `scale_by_column_means=True`):
   divide each column by its mean so all columns have equal weight
   regardless of their typical magnitude. Useful when columns have very
   different average levels.

**Weights are NOT applied here.** Per-column weights are applied in step 2a
to a separate copy used only for clustering distance.
The `NormalizedData` produced here contains unweighted normalized values that
flow through all downstream steps (rescaling, denormalization, reconstruction,
accuracy) without any weight compensation.

**Why it matters:** Without normalization, columns with larger numeric ranges
dominate the clustering distance. A temperature column ranging 0–40 would
overshadow a solar capacity factor ranging 0–1.

**Developer note:** The `NormalizedData` object is the most widely-used
intermediate — it is read by nearly every subsequent step.

---

### Step 2: Unstack to periods

| | |
|---|---|
| **Module** | `pipeline/periods.py` |
| **Function** | `unstack_to_periods()` |
| **Consumes** | `NormalizedData.values` (`pd.DataFrame`) |
| **Produces** | `PeriodProfiles` |
| **Key fields** | `column_index: pd.MultiIndex`, `time_index: pd.Index`, `profiles_dataframe: pd.DataFrame`, `n_periods: int` |

Reshapes the flat time series into a matrix where each row is one period
and each column is `(attribute, timestep)`.

**Example:** With 365 days of hourly data for 3 columns, the input is a
(8760, 3) DataFrame. After unstacking with `n_timesteps_per_period=24`,
the profiles matrix is (365, 72) — each row is a 72-dimensional point
(3 columns × 24 hours).

If the time series length is not evenly divisible by the period length,
the last period is padded by repeating initial rows.

---

### Step 2a: Apply column weights (optional)

| | |
|---|---|
| **Module** | `pipeline/__init__.py` |
| **Functions** | `_build_weight_vector()` |
| **Config** | `weights` parameter of `aggregate()` (preferred); `ClusterConfig.weights` (deprecated) |
| **Consumes** | `PeriodProfiles.profiles_dataframe`, normalized column order |
| **Produces** | `weight_vector: np.ndarray | None`, weighted `candidates: np.ndarray`, optional `weighted_profiles_df: pd.DataFrame` |

If `weights` is provided, weights are baked directly into the
candidates array via vectorized multiply (`np.repeat` + broadcast). The
`weight_vector` (`np.ndarray`) is stored on `PreparedData` for later
unweighting.

This means:
- Weights influence *which* periods get grouped together (clustering distance),
  *which* period is chosen as representative (medoid/maxoid selection), and
  `new_cluster` extreme reassignment distances (step 3a).
- After extremes, all representatives are unweighted (step 4) before
  downstream steps (rescale, denormalization) which expect unweighted data.

If no weights are provided, candidates pass through unchanged.

---

### Step 2b: Add period-sum features (optional)

| | |
|---|---|
| **Module** | `pipeline/periods.py` |
| **Function** | `add_period_sum_features()` |
| **Config** | `ClusterConfig.include_period_sums` |
| **Consumes** | `profiles_dataframe: pd.DataFrame`, current `candidates: np.ndarray` |
| **Produces** | augmented `candidates: np.ndarray` plus `PreparedData.n_feature_cols` for later trimming |

Appends the per-column sum of each period as extra features. When weights
are active, the sums are appended to the **weighted** candidates (they are
clustering features). When no weights are active, they are appended to the
regular candidates. Either way, the extra columns are removed from the
cluster centers in step 4 — they only influence which periods get grouped
together.

At the end of steps 1–2b, `_prepare_data()` returns a `PreparedData` dataclass with:

- `norm_data: NormalizedData`
- `period_profiles: PeriodProfiles`
- `candidates: np.ndarray`
- `representation_dict: dict[str, str]`
- `n_feature_cols: int`
- `original_column_order: list[str]`
- `original_data: pd.DataFrame`
- `weight_vector: np.ndarray | None`
- `weighted_profiles_df: pd.DataFrame | None`

---

### Step 3: Cluster centers

| | |
|---|---|
| **Module** | `pipeline/clustering.py` |
| **Config** | `ClusterConfig.method`, `.representation`, `.solver`, `.use_duration_curves` |
| **Consumes** | weighted or unweighted `candidates: np.ndarray`, optional `PredefParams` |
| **Produces** | `cluster_centers: list[np.ndarray]`, `cluster_center_indices: list[int] | None`, `cluster_order: np.ndarray` |

This is the core step. It groups the period profiles into `n_clusters`
clusters and selects or computes a representative for each.

Candidates are already weighted (from step 2a). Representatives are
computed from weighted candidates; unweighting happens later (step 4).
The result: cluster assignments reflect weighted importance, but
typical-period values are still in the original normalized space.

**Clustering methods** (`ClusterConfig.method`):

| Method | Description |
|---|---|
| `"hierarchical"` | Agglomerative (Ward linkage). Default. Deterministic. |
| `"kmeans"` | K-means. Fast but non-deterministic (set random seed externally). |
| `"kmedoids"` | Exact k-medoids via MILP. Slow but optimal. Requires a solver. |
| `"kmaxoids"` | K-maxoids heuristic. |
| `"averaging"` | Simple period averaging (1 cluster = mean of all). |
| `"contiguous"` | Adjacent periods only (preserves temporal order). |

**Representation methods** (`ClusterConfig.representation`):

After clustering, each cluster needs a representative period. The choice
controls what the typical period looks like:

| Representation | Description |
|---|---|
| `"mean"` | Arithmetic mean of cluster members. |
| `"medoid"` | The real period closest to the cluster center. Default. |
| `"maxoid"` | The real period farthest from the center. |
| `"distribution"` | Duration-curve fit: sorts values to preserve the statistical distribution. |
| `"distribution_minmax"` | Like `"distribution"` but also preserves extreme values. |
| `"minmax_mean"` | Separate min/max/mean per column. |
| `Distribution(...)` | Fine-grained control over distribution representation. |
| `MinMaxMean(...)` | Fine-grained control over which columns get min/max treatment. |

**Duration-curve clustering** (`use_duration_curves=True`): Sorts each
period's values before clustering, so periods are grouped by value
distribution rather than temporal shape. Useful when the ordering within
a period doesn't matter (e.g., energy storage optimization).

**Transfer path:** When `predef` is provided (via `ClusteringResult.apply()`),
clustering is skipped entirely. The stored assignments and centers are
reused as-is.

---

### Step 3a: Add extreme periods (optional)

| | |
|---|---|
| **Module** | `pipeline/extremes.py` |
| **Function** | `add_extreme_periods()` |
| **Config** | `ExtremeConfig` |
| **Consumes** | weighted `profiles_dataframe`, `cluster_periods_list`, `cluster_order` |
| **Produces** | updated `cluster_periods_list`, `cluster_order`, `extreme_cluster_idx`, `extreme_periods_info` |

Extremes run in **weighted space**: when weights are active,
`profiles_dataframe` is weighted before being passed to `add_extreme_periods()`.
Extreme detection itself (per-column idxmax/idxmin) is weight-invariant, but
the `new_cluster` method's distance-based reassignment respects weights.
Extracted extreme profiles carry weights, which are removed uniformly in step 4.

Ensures that periods with extreme values (peak demand, minimum solar, etc.)
are explicitly represented in the output rather than averaged away by
clustering.

**Extreme types:**

| Config field | What it preserves |
|---|---|
| `max_value=["demand"]` | The period containing the single highest demand value. |
| `min_value=["solar"]` | The period containing the single lowest solar value. |
| `max_period=["demand"]` | The period with the highest average demand. |
| `min_period=["solar"]` | The period with the lowest average solar. |

**Methods** (`ExtremeConfig.method`):

| Method | Behavior |
|---|---|
| `"append"` | Adds extreme periods as new clusters (increases `n_clusters`). Default. |
| `"new_cluster"` | Like append, but also reassigns nearby periods to the new cluster. |
| `"replace"` | Overwrites the relevant column values in the nearest existing cluster center. |

---

### Step 4: Trim · unweight · count

| | |
|---|---|
| **Module** | `pipeline/__init__.py` |
| **Function** | `_cluster_and_postprocess()` |
| **Consumes** | (possibly weighted) `cluster_periods_list`, `cluster_order`, `weight_vector`, original data length |
| **Produces** | unweighted `cluster_periods_list`, final `cluster_counts: dict[int, float]` |

Four operations happen here in sequence:

1. **Trim** period-sum extras: if period-sum features were added in step 2b,
   the extra columns are stripped from each cluster center vector, restoring
   the original dimensionality.

2. **Unweight** all representatives (regular + extreme): divides weights back
   out of all representatives using the stored `weight_vector`. After this
   point, all data is in unweighted normalized space for rescale,
   denormalization, and reconstruction.

3. **Count** cluster occurrences: counts how many original periods are assigned
   to each cluster. The result is a dictionary like `{0: 45, 1: 52, 2: 38, ...}`.
   These weights are used for rescaling (step 4a) and for downstream optimization
   models — each typical period represents `weight` real periods.

4. **Adjust for partial periods**: if the time series doesn't divide evenly
   into periods (e.g., 8761 hours with 24-hour periods), the last period was
   padded in step 2. Here, its cluster weight is reduced proportionally so the
   total weight is correct.

---

### Step 4a: Rescale representatives (optional)

| | |
|---|---|
| **Module** | `pipeline/rescale.py` |
| **Function** | `rescale_representatives()` |
| **Config** | `preserve_column_means` (= `rescale_cluster_periods`) |
| **Consumes** | unweighted `cluster_periods_list`, `cluster_counts`, normalized profiles, original data |
| **Produces** | rescaled `cluster_periods_list`, `rescale_deviations: dict[str, dict]` |

**Problem:** Clustering can shift column means. If you aggregate
365 daily load profiles into 8 typical days, the weighted average of the
8 representatives may not match the original annual average.

**Solution:** Iteratively scale each column of each non-extreme cluster
center until the weighted sum matches the original total (within tolerance).
Values are clipped to `[0, scale_ub]` where `scale_ub` depends on
`scale_by_column_means` (ratio of max to mean). Because the data is
unweighted at this point, no weight compensation is needed for the
clipping bound.

Extreme clusters (from step 3a) are excluded from rescaling to preserve
their extreme values.

Columns listed in `rescale_exclude_columns` are also skipped — useful for
binary columns (0/1) that shouldn't be scaled.

At the end of steps 3–4a, `_cluster_and_postprocess()` returns a `ClusteringOutput` dataclass with:

- `cluster_periods_list: list[np.ndarray]`
- `cluster_order: np.ndarray`
- `cluster_counts: dict[int, float]`
- `cluster_center_indices: list[int] | None`
- `extreme_cluster_idx: list[int]`
- `extreme_periods_info: dict[str, dict]`
- `clustering_duration: float`
- `rescale_deviations: dict[str, dict]`

---

### Step 5: Format representatives

| | |
|---|---|
| **Function** | `_representatives_to_dataframe()` |
| **Consumes** | `cluster_periods_list`, `PeriodProfiles.column_index` |
| **Produces** | `normalized_typical_periods: pd.DataFrame` with a `(PeriodNum, TimeStep)` MultiIndex |

Reshapes the flat 1-D cluster center vectors back into a DataFrame with
a `(PeriodNum, TimeStep)` MultiIndex. This is the `normalized_typical_periods`
DataFrame used by subsequent steps.

---

### Step 5a: Segment typical periods (optional)

| | |
|---|---|
| **Module** | `pipeline/segment.py` |
| **Function** | `segment_typical_periods()` |
| **Config** | `SegmentConfig.n_segments`, `.representation` |
| **Consumes** | unweighted `normalized_typical_periods` (weights applied internally for segment boundary placement) |
| **Produces** | `segmented_df`, `predicted_segmented_df`, `segment_center_indices` |

**Problem:** Even after clustering, each typical period still has the
full temporal resolution (e.g., 24 hourly timesteps). Some optimization
models need fewer timesteps.

**Solution:** Within each typical period, adjacent timesteps with similar
values are merged into segments. If `n_segments=8`, each 24-hour period
is reduced to 8 segments of variable duration.

The segmentation uses the same clustering machinery (constrained
agglomerative clustering of adjacent timesteps) as the main clustering
step. The `representation` parameter controls how segment values are
computed (typically `"mean"`).

**Weights and segmentation:** `normalized_typical_periods` entering this step
is unweighted. Internally, `_apply_weights_df()` multiplies column weights into
a copy before calling `segment_typical_periods()`, so that high-weight columns
have more influence on where segment boundaries fall. `_remove_weights_df()` is
called on the outputs before denormalization.

After segmentation, the pipeline tracks two DataFrames:
- `segmented_normalized` — for denormalization (step 6).
- `predicted_segmented_df` — for reconstruction (step 7).

---

### Step 6: Denormalize

| | |
|---|---|
| **Module** | `pipeline/normalize.py` |
| **Function** | `denormalize()` |
| **Consumes** | `denorm_source: pd.DataFrame`, `NormalizedData` |
| **Produces** | `typical_periods: pd.DataFrame` in original units |

Inverts the transformations from step 1 to return values in original
units:

1. Undo column-mean normalization (multiply by stored mean).
2. Inverse min-max scaling (via the stored `MinMaxScaler`).

No weight removal is needed because weights were never baked into the data
passed to this step.

The output is `typical_periods` — the final representative periods in
the user's original units.

---

### Step 7: Reconstruct + accuracy

| | |
|---|---|
| **Module** | `pipeline/accuracy.py` |
| **Functions** | `reconstruct()`, `compute_accuracy()` |
| **Consumes** | representative DataFrame, `cluster_order`, `PeriodProfiles`, `NormalizedData`, original data |
| **Produces** | `reconstructed_data: pd.DataFrame`, `normalized_predicted: pd.DataFrame` (accuracy computed lazily on `PipelineResult`) |

**Bounds check:** Before reconstruction, any column whose max (or min) in
the typical periods exceeds the original data's range beyond `numerical_tolerance`
triggers a warning. This can happen with distribution representations or
aggressive rescaling.

**Reconstruct:** Expands the typical periods back into a full-length time
series by replacing each original period with its assigned cluster
representative. The result has the same shape as the input data.

**Accuracy:** The reconstructed and original series are stored in normalized
(unweighted) form as `normalized_predicted` on `PipelineResult`. Accuracy
metrics are computed lazily via `PipelineResult.accuracy_indicators` (a
`cached_property`) when first accessed. Computes per-column:

| Metric | Description |
|---|---|
| RMSE | Root mean square error. |
| MAE | Mean absolute error. |
| RMSE (duration) | RMSE on sorted (duration-curve) values — measures distribution fit. |

At the end of steps 5–7, `_format_and_reconstruct()` returns a `FormattedOutput` dataclass with:

- `typical_periods: pd.DataFrame`
- `reconstructed_data: pd.DataFrame`
- `normalized_predicted: pd.DataFrame`
- `segmented_df: pd.DataFrame | None`
- `segment_center_indices: list | None`

---

### Step 8: Assemble

| | |
|---|---|
| **Module** | `config.py` and `pipeline/__init__.py` |
| **Functions** | `_assemble_result()`, `ClusteringResult.from_pipeline()` |
| **Consumes** | `ClusteringOutput`, `FormattedOutput`, `PipelineConfig`, original data |
| **Produces** | `PipelineResult` → `AggregationResult` |

Assembles all clustering metadata into a `ClusteringResult` object.
This object is serializable (`.to_json()`, `.from_json()`) and supports
transfer via `.apply(new_data)`.

Key fields stored:
- `cluster_assignments` — which cluster each original period belongs to.
- `cluster_centers` — indices of medoid periods (if applicable).
- `segment_assignments`, `segment_durations` — segmentation structure.
- Config references for documentation.

The pipeline preserves the original column order and packs everything
into a `PipelineResult`, which `aggregate()` converts to the user-facing
`AggregationResult`.

`PipelineResult` is the final internal handoff. It contains the denormalized
representatives, cluster counts, timing and rescaling metadata, the
reconstructed series, and the `ClusteringResult` needed for transfer and
serialization.

---

## Working with results

### AggregationResult

The object returned by `tsam.aggregate()`:

```python
result = tsam.aggregate(df, n_clusters=8)

# Core outputs
result.cluster_representatives  # DataFrame (cluster × timestep)
result.cluster_counts           # {cluster_id: count}
result.cluster_assignments      # array of cluster IDs per original period

# Reconstruction
result.original        # original data
result.reconstructed   # reconstructed data
result.residuals       # original - reconstructed

# Accuracy
result.accuracy.rmse           # per-column RMSE
result.accuracy.mae            # per-column MAE
result.accuracy.rmse_duration  # per-column duration-curve RMSE
result.accuracy.summary        # combined DataFrame

# Metadata
result.n_clusters
result.n_timesteps_per_period
result.n_segments              # None if no segmentation
result.clustering_duration     # seconds

# Assignments detail
result.assignments  # DataFrame with period_idx, timestep_idx, cluster_idx, [segment_idx]

# Transfer
result.clustering.apply(new_data)  # apply same clustering to different data
result.clustering.to_json("clustering.json")  # save for later
```

---

## Configuration reference

### ClusterConfig

```python
from tsam import ClusterConfig

cluster = ClusterConfig(
    method="hierarchical",        # clustering algorithm
    representation="medoid",      # how to compute cluster centers
    scale_by_column_means=False,  # divide by column mean before clustering
    use_duration_curves=False,    # sort values within periods before clustering
    include_period_sums=False,    # add period sums as extra clustering features
    solver="highs",               # MILP solver (for kmedoids only)
)
```

### SegmentConfig

```python
from tsam import SegmentConfig

segments = SegmentConfig(
    n_segments=8,           # number of segments per period
    representation="mean",  # how to compute segment values
)
```

### ExtremeConfig

```python
from tsam import ExtremeConfig

extremes = ExtremeConfig(
    method="append",             # how to integrate extreme periods
    max_value=["demand"],        # preserve peak-value periods
    min_value=["solar"],         # preserve minimum-value periods
    max_period=["demand"],       # preserve highest-average periods
    min_period=[],               # preserve lowest-average periods
)
```

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

## Source file map

For developers navigating the codebase:

| File | Role |
|---|---|
| `src/tsam/api.py` | User-facing `aggregate()` function and result builder |
| `src/tsam/config.py` | `ClusterConfig`, `SegmentConfig`, `ExtremeConfig`, `ClusteringResult` |
| `src/tsam/result.py` | `AggregationResult`, `AccuracyMetrics` |
| `src/tsam/pipeline/__init__.py` | `run_pipeline()` — orchestrates all 8 steps |
| `src/tsam/pipeline/normalize.py` | `normalize()`, `denormalize()` |
| `src/tsam/pipeline/periods.py` | `unstack_to_periods()`, `add_period_sum_features()` |
| `src/tsam/pipeline/clustering.py` | `cluster_periods()`, `cluster_sorted_periods()`, `use_predefined_assignments()` |
| `src/tsam/pipeline/extremes.py` | `add_extreme_periods()` |
| `src/tsam/pipeline/rescale.py` | `rescale_representatives()` |
| `src/tsam/pipeline/segment.py` | `segment_typical_periods()` |
| `src/tsam/pipeline/accuracy.py` | `reconstruct()`, `compute_accuracy()` |
| `src/tsam/pipeline/types.py` | `PipelineConfig`, `PredefParams`, `NormalizedData`, `PeriodProfiles`, `PreparedData`, `ClusteringOutput`, `FormattedOutput`, `PipelineResult` |
