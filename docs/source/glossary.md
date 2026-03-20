# Glossary

This is the comprehensive glossary for tsam. It covers every concept in the
aggregation pipeline — from raw input to final output — and maps old (legacy)
names to the new API names.

For a quick overview of the most common terms, see the
[Getting Started glossary table](gettingStartedDoc.rst).

---

## 1. Input Domain

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Time Series** | The raw input: a DataFrame with a datetime index and one or more numeric columns. | `timeSeries` | `data` (param), `time_series` (internal) |
| **Column** | A single variable in the time series (e.g., "temperature", "demand"). | `column` | `column` |
| **Timestep** | A single discrete time point. Its position within a period is the **step index** (0 to `n_timesteps_per_period - 1`). | `TimeStep` | `timestep` |
| **Temporal Resolution** | Duration between consecutive timesteps (e.g., 1h, 15min). | `resolution` | `temporal_resolution` |

## 2. Period Structure

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Period** | A fixed-length contiguous time window (e.g., one day = 24h). The time series is divided into periods for clustering. | "period" | `period` |
| **Period Duration** | Length of one period in hours or as a timedelta string. | `hoursPerPeriod` | `period_duration` |
| **Timesteps per Period** | Number of timesteps in one period: `period_duration / temporal_resolution`. | `timeStepsPerPeriod` | `n_timesteps_per_period` |
| **Period Profile** | The vector representation of a single period: all columns' values at all timesteps, flattened into one row of length `n_columns * n_timesteps_per_period`. This is the "data point" that clustering operates on. | `candidates`, `normalizedPeriodlyProfiles` | `period_profiles` |
| **Unstacking** | Reshaping a flat time series (T rows) into a matrix of period profiles (P rows x features columns). | `unstackToPeriods()` | `unstack_to_periods()` |

## 3. Normalization & Weighting

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Normalization** | Min-max scaling each column independently to [0, 1], so columns with different units are comparable during clustering. | `_normalizeTimeSeries()` | `normalize()` |
| **Denormalization** | Inverse of normalization: scaling from [0, 1] back to original units using the stored scaler. | `_unnormalizeTimeSeries()` | `denormalize()` |
| **Scaler** | The fitted MinMaxScaler storing each column's min/max. Created once, used for both directions. | (not stored) | `scaler` |
| **Mean Normalization** | Optional: dividing each column by its normalized mean so all columns contribute equally to distances regardless of value distribution. | `sameMean` | `normalize_column_means` |
| **Column Weight** | A per-column multiplier baked directly into the candidate matrix for clustering. Higher weight = more influence on clustering distance, medoid/maxoid selection, and `new_cluster` extreme reassignment. Weights are divided back out after extremes (step 6b) so downstream steps (rescale, denormalization) see unweighted data. Weight of 0 is replaced by `MIN_WEIGHT` (1e-6). | `weightDict` | `weights` |
| **Weight Vector** | A `np.ndarray` aligned to column order, stored on `PreparedData`. Used to apply weights (step 2b) and remove them (step 6b). `None` when all weights are 1.0. | `normalizedPeriodlyProfiles` (misleading — included weights in data) | `weight_vector` |

## 4. Clustering

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Cluster** | A group of similar periods. Each cluster is represented by one **representative**. | "cluster" | `cluster` |
| **n_clusters** | Number of clusters to create. | `noTypicalPeriods` | `n_clusters` |
| **Cluster Method** | The partitioning algorithm. Options: averaging, kmeans, kmedoids, kmaxoids, hierarchical, contiguous. | `clusterMethod` | `method` (in `ClusterConfig`) |
| **Cluster Assignments** | Integer array mapping each original period (by index) to its cluster ID. Length = `n_periods`. | `clusterOrder`, `_clusterOrder` | `cluster_assignments` |
| **Duration Curve Clustering** | Variant where each period's timesteps are sorted descending before clustering, matching periods by value distribution rather than temporal shape. | `sortValues` | `use_duration_curves` |
| **Period Sum Features** | Optional: appending per-column period sums as extra features to bias clustering toward preserving totals. | `evalSumPeriods` | `include_period_sums` |

## 5. Representation

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Representative** | The single period profile standing in for an entire cluster. Also called **typical period** in user-facing docs. | `clusterCenters`, `clusterPeriods` (confusingly different things!) | `representatives` |
| **Representation Method** | Strategy for computing a representative. | `representationMethod` | `representation` |
| **Mean** | Representative = centroid (arithmetic mean of members). | `"meanRepresentation"` | `"mean"` |
| **Medoid** | Representative = actual member closest (Euclidean) to centroid. | `"medoidRepresentation"` | `"medoid"` |
| **Maxoid** | Representative = actual member most dissimilar (Euclidean) to other clusters. | `"maxoidRepresentation"` | `"maxoid"` |
| **Min/Max/Mean** | Per-column, representative = min, max, or mean of members at each timestep (configured by a dict). | `"minmaxmeanRepresentation"`, `representationDict` | `"minmax_mean"`, `representation_dict` |
| **Distribution** | Representative preserves the value distribution (duration curve shape). | `"durationRepresentation"` / `"distributionRepresentation"` | `"distribution"` |
| **Distribution Min/Max** | Like distribution, but adjusts min/max values to match cluster extremes while preserving sums. | `"distributionAndMinMaxRepresentation"` | `"distribution_minmax"` |
| **Representative Indices** | Indices of the original periods used as representatives (meaningful for medoid/maxoid; `None` for mean/distribution). | `clusterCenterIndices` | `representative_indices` |

## 6. Extreme Periods

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Extreme Period** | An original period containing a critical value that must be preserved (e.g., day with peak electricity demand). | "extreme period" | `extreme_period` |
| **Extreme Criterion** | What makes a period "extreme". Four types: | `addPeakMax`, `addPeakMin`, `addMeanMax`, `addMeanMin` | Fields in `ExtremeConfig`: |
| — **max_value** | Period containing the single highest value of a column. | `addPeakMax` | `max_value` |
| — **min_value** | Period containing the single lowest value of a column. | `addPeakMin` | `min_value` |
| — **max_period** | Period with the highest mean (total) of a column. | `addMeanMax` | `max_period` |
| — **min_period** | Period with the lowest mean (total) of a column. | `addMeanMin` | `min_period` |
| **Extreme Method** | How extreme periods are integrated into clustering. | `extremePeriodMethod` | `method` (in `ExtremeConfig`) |
| — **append** | Add as new single-member clusters. Original assignments unchanged. | `"append"` | `"append"` |
| — **new_cluster** | Add as new clusters and reassign nearby periods. | `"new_cluster_center"` | `"new_cluster"` |
| — **replace** | Overwrite relevant columns of nearest cluster's representative. Count unchanged. | `"replace_cluster_center"` | `"replace"` |
| **Extreme Cluster Indices** | Indices (into the representatives list) that are extreme-period clusters. Excluded from rescaling. | `extremeClusterIdx` | `extreme_cluster_indices` |

## 7. Rescaling

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Rescaling** | Iterative post-clustering adjustment: multiplicatively scales non-extreme representatives so the weighted sum of each column matches the original time series total. Ensures energy/mass conservation. | `_rescaleClusterPeriods()`, `rescaleClusterPeriods` (bool) | `rescale()` (function), `preserve_column_means` (parameter) |
| **Rescale Upper Bound** | Maximum allowed value during rescaling, derived from mean-normalization settings only (ratio of column max to column mean when `normalize_column_means` is active, otherwise 1.0). Column weights do not affect this bound. | `scale_ub` | `rescale_upper_bound` |
| **Rescale Deviation** | Remaining percentage difference after rescaling converges or hits max iterations. | `_rescaleDeviations` | `rescale_deviations` |
| **Rescale-Excluded Columns** | Columns exempt from rescaling (e.g., binary indicators). | `rescaleExcludeColumns` | `rescale_exclude_columns` |

## 8. Segmentation

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Segmentation** | Second-level aggregation *within* each representative: consecutive timesteps are grouped into segments via constrained agglomerative clustering. | `segmentation` (bool) | `segments` (param: `SegmentConfig` or `None`) |
| **Segment** | A group of consecutive timesteps within a period, represented by a single value per column. | "segment" | `segment` |
| **n_segments** | Number of segments per period. Must be <= `n_timesteps_per_period`. | `noSegments` | `n_segments` |
| **Segment Duration** | Number of original timesteps a segment spans. Variable across segments. | `segmentDuration` | `segment_duration` |
| **Segment Assignments** | Mapping from each timestep within a period to its segment ID. | `predefSegmentOrder` | `segment_assignments` |
| **Segment Representative** | The value representing one segment per column. | segment center | `segment_representative` |
| **Segment Representation** | Method for computing segment representatives. Defaults to cluster representation method. | `segmentRepresentationMethod` | `representation` (in `SegmentConfig`) |

## 9. Output / Results

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Typical Period** | User-facing synonym for **representative**. The representative profile denormalized to original units. | `typicalPeriods` | `cluster_representatives` (in `AggregationResult`) |
| **Cluster Count** | Number of original periods assigned to a cluster. May be fractional if time series length isn't an integer multiple of period duration. | `_clusterPeriodNoOccur` | `cluster_counts` |
| **Reconstruction** | Approximation of the original time series: replace each period with its cluster's representative. | `predictedData` | `reconstructed` |
| **Residuals** | Difference: original minus reconstruction. | (computed ad-hoc) | `residuals` |

## 10. Accuracy Metrics

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **RMSE** | Root Mean Square Error between normalized original and reconstructed, per column. | `indicatorRaw["RMSE"]` | `rmse` |
| **MAE** | Mean Absolute Error, per column. | `indicatorRaw["MAE"]` | `mae` |
| **Duration RMSE** | RMSE on sorted (duration curve) values, per column. Measures distribution preservation regardless of timing. | `indicatorRaw["RMSE_duration"]` | `rmse_duration` |

## 11. Transfer / Reuse

| Term | Definition | Old code | New code |
|------|-----------|----------|----------|
| **Clustering Result** | Serializable bundle of cluster assignments, representative indices, and segment structure. Saved to JSON, applied to new data. | `ClusteringResult` | `ClusteringResult` |
| **Apply / Transfer** | Using a saved clustering on a different dataset. Skips clustering; recomputes representatives and rescaling only. | `.apply()` | `.apply()` |

---

## Naming Conventions

1. **Parameters**: `snake_case`, matching glossary terms exactly
2. **Config dataclasses**: `PascalCase` + `Config` suffix (e.g., `ClusterConfig`, `ExtremeConfig`)
3. **Result dataclasses**: `PascalCase` + `Result`/`Metrics` suffix (e.g., `AggregationResult`)
4. **Pipeline functions**: `snake_case` verbs: `normalize()`, `denormalize()`, `unstack_to_periods()`, `cluster()`, `compute_representatives()`, `add_extreme_periods()`, `rescale()`, `segment()`
5. **Intermediate variables**: `snake_case` nouns from glossary: `period_profiles`, `weight_vector`, `cluster_assignments`, `representatives`, `extreme_cluster_indices`
6. **No abbreviations** except: `n_clusters`, `n_segments`, `RMSE`, `MAE`

## Ambiguities Resolved

| Old confusion | Resolution |
|---|---|
| `clusterCenters` vs `clusterPeriods` (trimmed centers) | Single name: `representatives`. Trim immediately after computation. |
| `clusterOrder` (not an ordering!) | `cluster_assignments` |
| `clusterPeriodNoOccur` (double negative) | `cluster_counts` |
| `typicalPeriods` vs `normalizedTypicalPeriods` | `representatives` (normalized internal) / `cluster_representatives` (denormalized output) |
| `candidates` (candidates for what?) | `period_profiles`; `candidates` (weighted in-place if weights active, unweighted after step 6b) |
| `sameMean` (cryptic bool) | `normalize_column_means` |
| `sortValues` (sort what?) | `use_duration_curves` |
| `evalSumPeriods` | `include_period_sums` |
| `addPeakMax` / `addMeanMin` (verb as param name) | `max_value` / `min_period` in `ExtremeConfig` |
| `representationMethod` vs `representation` | `representation` everywhere |
