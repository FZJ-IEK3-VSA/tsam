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

A call to `tsam.aggregate()` produces an `AggregationResult` by running
16 sequential steps. The diagram below shows the high-level flow:

```
   data (DataFrame)
        │
        ▼
  ┌─────────────┐
  │  Normalize   │ Step 1 — scale to [0,1], apply weights & column-mean normalization
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Unstack    │ Step 2 — reshape flat time series into period × timestep matrix
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Augment    │ Step 3 — optionally append period-sum features for clustering
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Cluster    │ Step 4 — group similar periods into k clusters
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Trim       │ Step 5 — remove augmented features from cluster centers
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Extremes   │ Step 6 — optionally add/replace extreme-value periods
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Weights    │ Step 7 — count how many original periods each cluster represents
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Rescale    │ Step 8 — adjust representatives so column means match the original
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Partial    │ Step 9 — adjust weight for the last period if the series doesn't divide evenly
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Format     │ Step 10 — reshape flat vectors back into a MultiIndex DataFrame
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Segment    │ Step 11 — optionally reduce intra-period resolution
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Denormalize│ Step 12 — invert normalization back to original units
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Bounds     │ Step 13 — warn if aggregated values exceed original min/max
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Reconstruct│ Step 14 — expand typical periods back to full time series & measure accuracy
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Metadata   │ Step 15 — assemble ClusteringResult for serialization/transfer
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Return     │ Step 16 — pack everything into PipelineResult → AggregationResult
  └──────┴───────┘
```

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
| **Config** | `ClusterConfig.weights`, `ClusterConfig.normalize_column_means` |
| **Output** | `NormalizedData` |

This step prepares the raw data for clustering by removing scale
differences between columns.

**What happens:**

1. **Sort columns** alphabetically (deterministic column order).
2. **Cast to float** (in case of integer columns).
3. **Min-max scale** each column to [0, 1] using scikit-learn's
   `MinMaxScaler`. The fitted scaler is stored for later inversion.
4. **Column-mean normalization** (optional, `normalize_column_means=True`):
   divide each column by its mean so all columns have equal weight
   regardless of their typical magnitude. Useful when columns have very
   different average levels.
5. **Apply weights** (optional): multiply each column by a user-supplied
   weight factor to emphasize or de-emphasize it during clustering.

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
| **Output** | `PeriodProfiles` |

Reshapes the flat time series into a matrix where each row is one period
and each column is `(attribute, timestep)`.

**Example:** With 365 days of hourly data for 3 columns, the input is a
(8760, 3) DataFrame. After unstacking with `n_timesteps_per_period=24`,
the profiles matrix is (365, 72) — each row is a 72-dimensional point
(3 columns × 24 hours).

If the time series length is not evenly divisible by the period length,
the last period is padded by repeating initial rows.

---

### Step 3: Add period-sum features (optional)

| | |
|---|---|
| **Module** | `pipeline/periods.py` |
| **Function** | `add_period_sum_features()` |
| **Config** | `ClusterConfig.include_period_sums` |

Appends the per-column sum of each period as extra features in the
candidate matrix. This biases the clustering distance toward preserving
total energy/load per period.

The extra columns are removed from the cluster centers in step 5 — they
only influence which periods get grouped together.

---

### Step 4: Cluster

| | |
|---|---|
| **Module** | `pipeline/clustering.py` |
| **Config** | `ClusterConfig.method`, `.representation`, `.solver`, `.use_duration_curves` |
| **Output** | `cluster_centers`, `cluster_center_indices`, `cluster_order` |

This is the core step. It groups the period profiles into `n_clusters`
clusters and selects or computes a representative for each.

**Clustering methods** (`ClusterConfig.method`):

| Method | Description |
|---|---|
| `"hierarchical"` | Agglomerative (Ward linkage). Default. Deterministic. |
| `"kmeans"` | K-means. Fast but non-deterministic (set random seed externally). |
| `"kmedoids"` | Exact k-medoids via MILP. Slow but optimal. |
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

### Step 5: Trim augmented features

Inline in `run_pipeline()`.

If period-sum features were added in step 3, the extra columns are
stripped from each cluster center vector, restoring the original
dimensionality.

---

### Step 6: Add extreme periods (optional)

| | |
|---|---|
| **Module** | `pipeline/extremes.py` |
| **Function** | `add_extreme_periods()` |
| **Config** | `ExtremeConfig` |

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

### Step 7: Compute cluster weights

Inline in `run_pipeline()`.

Counts how many original periods are assigned to each cluster. The result
is a dictionary like `{0: 45, 1: 52, 2: 38, ...}`. These weights are
used for:

- Rescaling (step 8) — the weighted sum must match the original total.
- Downstream optimization models — each typical period represents
  `weight` real periods.

---

### Step 8: Rescale representatives (optional)

| | |
|---|---|
| **Module** | `pipeline/rescale.py` |
| **Function** | `rescale_representatives()` |
| **Config** | `preserve_column_means` (= `rescale_cluster_periods`) |

**Problem:** Clustering can shift column means. If you aggregate
365 daily load profiles into 8 typical days, the weighted average of the
8 representatives may not match the original annual average.

**Solution:** Iteratively scale each column of each non-extreme cluster
center until the weighted sum matches the original total (within tolerance).

Extreme clusters (from step 6) are excluded from rescaling to preserve
their extreme values.

Columns listed in `rescale_exclude_columns` are also skipped — useful for
binary columns (0/1) that shouldn't be scaled.

---

### Step 9: Adjust for partial periods

Inline in `run_pipeline()`.

If the time series doesn't divide evenly into periods (e.g., 8761 hours
with 24-hour periods), the last period is padded in step 2. Here, its
cluster weight is reduced proportionally so the total weight is correct.

---

### Step 10: Format representatives to DataFrame

| | |
|---|---|
| **Function** | `_representatives_to_dataframe()` |

Reshapes the flat 1-D cluster center vectors back into a DataFrame with
a `(PeriodNum, TimeStep)` MultiIndex. This is the `normalized_typical_periods`
DataFrame used by subsequent steps.

---

### Step 11: Segment typical periods (optional)

| | |
|---|---|
| **Module** | `pipeline/segment.py` |
| **Config** | `SegmentConfig.n_segments`, `.representation` |

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

After segmentation, the pipeline tracks two DataFrames:
- `segmented_normalized` — for denormalization (step 12).
- `predicted_segmented_df` — for reconstruction (step 14).

---

### Step 12: Denormalize

| | |
|---|---|
| **Module** | `pipeline/normalize.py` |
| **Function** | `denormalize()` |

Inverts the transformations from step 1 to return values in original
units:

1. Undo weights (divide by weight).
2. Undo column-mean normalization (multiply by stored mean).
3. Inverse min-max scaling (via the stored `MinMaxScaler`).

The output is `typical_periods` — the final representative periods in
the user's original units.

---

### Step 13: Bounds check

| | |
|---|---|
| **Function** | `_warn_if_out_of_bounds()` |

Warns if any column's max (or min) in the typical periods exceeds the
original data's range beyond `numerical_tolerance`. This can happen with
distribution representations or aggressive rescaling.

---

### Step 14: Reconstruct and compute accuracy

| | |
|---|---|
| **Module** | `pipeline/accuracy.py` |
| **Functions** | `reconstruct()`, `compute_accuracy()` |

**Reconstruct:** Expands the typical periods back into a full-length time
series by replacing each original period with its assigned cluster
representative. The result has the same shape as the input data.

**Accuracy:** Compares the reconstruction to the original (in normalized
space) and computes per-column:

| Metric | Description |
|---|---|
| RMSE | Root mean square error. |
| MAE | Mean absolute error. |
| RMSE (duration) | RMSE on sorted (duration-curve) values — measures distribution fit. |

---

### Step 15: Build ClusteringResult

| | |
|---|---|
| **Function** | `_build_clustering_result()` |

Assembles all clustering metadata into a `ClusteringResult` object.
This object is serializable (`.to_json()`, `.from_json()`) and supports
transfer via `.apply(new_data)`.

Key fields stored:
- `cluster_assignments` — which cluster each original period belongs to.
- `cluster_centers` — indices of medoid periods (if applicable).
- `segment_assignments`, `segment_durations` — segmentation structure.
- Config references for documentation.

---

### Step 16: Assemble PipelineResult

The pipeline restores the original column order (columns are sorted
alphabetically internally) and packs everything into a `PipelineResult`,
which `aggregate()` converts to the user-facing `AggregationResult`.

---

## Working with results

### AggregationResult

The object returned by `tsam.aggregate()`:

```python
result = tsam.aggregate(df, n_clusters=8)

# Core outputs
result.cluster_representatives  # DataFrame (cluster × timestep)
result.cluster_weights          # {cluster_id: count}
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
    weights=None,                 # per-column weights, e.g. {"solar": 2.0}
    normalize_column_means=False, # divide by column mean before clustering
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

## Source file map

For developers navigating the codebase:

| File | Role |
|---|---|
| `src/tsam/api.py` | User-facing `aggregate()` function and result builder |
| `src/tsam/config.py` | `ClusterConfig`, `SegmentConfig`, `ExtremeConfig`, `ClusteringResult` |
| `src/tsam/result.py` | `AggregationResult`, `AccuracyMetrics` |
| `src/tsam/pipeline/__init__.py` | `run_pipeline()` — orchestrates all 16 steps |
| `src/tsam/pipeline/normalize.py` | `normalize()`, `denormalize()` |
| `src/tsam/pipeline/periods.py` | `unstack_to_periods()`, `add_period_sum_features()` |
| `src/tsam/pipeline/clustering.py` | `cluster_periods()`, `cluster_sorted_periods()`, `use_predefined_assignments()` |
| `src/tsam/pipeline/extremes.py` | `add_extreme_periods()` |
| `src/tsam/pipeline/rescale.py` | `rescale_representatives()` |
| `src/tsam/pipeline/segment.py` | `segment_typical_periods()` |
| `src/tsam/pipeline/accuracy.py` | `reconstruct()`, `compute_accuracy()` |
| `src/tsam/pipeline/types.py` | `PipelineResult`, `PeriodProfiles`, `NormalizedData`, `PredefParams` |
| `src/tsam/timeseriesaggregation.py` | Legacy monolith (backward compatibility) |
