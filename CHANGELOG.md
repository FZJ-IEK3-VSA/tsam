# ETHOS.TSAM Change Log

All notable changes to this project will be documented in this file.

New entries are automatically added by [release-please](https://github.com/googleapis/release-please) from conventional commit messages.

## [4.0.0] (unreleased)

### Architecture

* **Pipeline rewrite**: The monolithic `TimeSeriesAggregation` internals have been
  replaced with a stateless pipeline of pure functions in `src/tsam/pipeline/`.
  Both `tsam.aggregate()` and the legacy `TimeSeriesAggregation` class now delegate
  to the same `run_pipeline()` orchestrator.

* **snake_case identifiers**: All internal identifiers have been renamed from camelCase
  to snake_case. The legacy `TimeSeriesAggregation` class retains its original
  parameter names (both camelCase and snake_case accepted) for backward compatibility.

### Breaking Changes

* **Decoupled column weights from data pipeline**: `weights` now affects **only** the
  clustering distance calculation. Previously, weights were multiplied into the normalized
  data during preprocessing. This simplifies the pipeline and eliminates a class of
  weight-related bugs. Cluster *assignments* are identical to v3. The **only** configuration
  that produces different output is `hierarchical_weighted` — non-uniform weights with
  medoid representation.

* **Output column order preserved**: `cluster_representatives`, `reconstructed`, and `original`
  now return columns in the same order as the input DataFrame. Previously, columns were
  alphabetically sorted. (The legacy class preserves alphabetical sorting for backward compat.)

* **Renamed `cluster_weights` to `cluster_counts`**: `AggregationResult.cluster_weights`
  has been renamed to `cluster_counts` to avoid confusion with per-column clustering weights.
  The old name remains as a deprecated property emitting `FutureWarning`.

### Improvements

* **Versioned clustering JSON**: `ClusteringResult.to_json()` now includes a
  `"version"` field recording the tsam version that created the file.

* **Deterministic extreme period assignment**: The `new_cluster` extreme method now
  picks the closest extreme period by distance (deterministic) instead of the previous
  last-match-wins loop order.

See the [v3 to v4 migration guide](migration-guide.md#migration-v3-to-v4) for upgrade instructions.

## [3.2.0](https://github.com/FZJ-IEK3-VSA/tsam/compare/v3.1.2...v3.2.0) (2026-03-24)

This release moves the `weights` argument out of `ClusterConfig`and into `aggregate` (and similar methods), while deprecating the old usage inside `ClusterConfig`. The Parameter affects all aggregation steps and is now placed accordingly. Further, we added a new plotting method that allows you to inspect cluster members and their representation.


### Features

* Interactive cluster member visualization ([#159](https://github.com/FZJ-IEK3-VSA/tsam/issues/159)) ([61c6296](https://github.com/FZJ-IEK3-VSA/tsam/commit/61c6296e2a9c616b36af42ad8d22181652d5d291))
* Move weights to top-level aggregate() parameter ([#195](https://github.com/FZJ-IEK3-VSA/tsam/issues/195)) ([4f177d0](https://github.com/FZJ-IEK3-VSA/tsam/commit/4f177d0792e06373c23ab1eefc2f0794c7990675))

### Documentation

* Add ETHOS.TSAM branding, FZJ theme, and documentation update ([#194](https://github.com/FZJ-IEK3-VSA/tsam/issues/194)) ([d24a0a3](https://github.com/FZJ-IEK3-VSA/tsam/commit/d24a0a39971c8cf0c597956a9a3c4b64bc263e1d))
* extract glossary into standalone file ([d24a0a3](https://github.com/FZJ-IEK3-VSA/tsam/commit/d24a0a39971c8cf0c597956a9a3c4b64bc263e1d))
* improve codeblock in Getting Started: ([d24a0a3](https://github.com/FZJ-IEK3-VSA/tsam/commit/d24a0a39971c8cf0c597956a9a3c4b64bc263e1d))
* remove integrated software section and update legal notice ([#218](https://github.com/FZJ-IEK3-VSA/tsam/issues/218)) ([4c9cc71](https://github.com/FZJ-IEK3-VSA/tsam/commit/4c9cc71621b2fa9fd690211f6186b7bf5d9d2444))
* update images to README_assets v1.0.0 and add missing publication ([#215](https://github.com/FZJ-IEK3-VSA/tsam/issues/215)) ([e56a686](https://github.com/FZJ-IEK3-VSA/tsam/commit/e56a686ab621cdc14e3837a1095c678e7c4ec19f))

## [3.1.1](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v3.1.1)

ETHOS.TSAM v3.1.1 is the first stable v3 release (versions 3.0.0 and 3.1.0 were yanked from PyPI).
It introduces a modern functional API alongside significant improvements to performance,
plotting, hyperparameter tuning, and overall code quality.

See the [migration guide](migration-guide.md) for a complete guide on upgrading from v2.

### Breaking Changes

* **New functional API**: The primary interface is now `tsam.aggregate()` which returns an `AggregationResult` object
* **Configuration objects**: Clustering and segmentation options are now configured via `ClusterConfig`, `SegmentConfig`, and `ExtremeConfig` dataclasses
* **Segment representation default**: In v2, omitting `segmentRepresentationMethod` caused segments
  to silently inherit the cluster `representationMethod` (e.g. distribution). In v3,
  `SegmentConfig(representation=...)` defaults to `"mean"` independently. If you relied on the
  implicit inheritance, pass the representation explicitly:
  `SegmentConfig(n_segments=12, representation=Distribution(scope="global"))`
* **Removed methods**: The `reconstruct()` method has been removed; use the `reconstructed` property on `AggregationResult` instead
* **Renamed parameters**: Parameters have been renamed for consistency:

| Old (v2) | New (v3) |
|----------|----------|
| `noTypicalPeriods` | `n_clusters` |
| `hoursPerPeriod` | `period_duration` |
| `resolution` | `temporal_resolution` |
| `clusterMethod` | `cluster=ClusterConfig(method=...)` |
| `representationMethod` | `cluster=ClusterConfig(representation=...)` |
| `segmentation` + `noSegments` | `segments=SegmentConfig(n_segments=...)` |
| `sameMean` | `cluster=ClusterConfig(normalize_column_means=...)` |
| `rescaleClusterPeriods` | `preserve_column_means` |
| `sortValues` | `cluster=ClusterConfig(use_duration_curves=...)` |
| `evalSumPeriods` | `cluster=ClusterConfig(include_period_sums=...)` |
| `weightDict` | `weights` (top-level parameter) |
| `addPeakMax/Min`, etc. | `extremes=ExtremeConfig(max_value=..., ...)` |

### New Features

* **Modern functional API**: New `tsam.aggregate()` function returns an `AggregationResult` with properties:

    - `cluster_representatives`: DataFrame with aggregated typical periods
    - `cluster_assignments`: Which cluster each original period belongs to
    - `cluster_counts`: Occurrence count per cluster (fractional for partial periods)
    - `accuracy`: `AccuracyMetrics` object with RMSE, MAE, and duration curve RMSE
    - `reconstructed`: Reconstructed time series (cached property)
    - `residuals`: Difference between original and reconstructed
    - `original`: Access to original input data
    - `clustering`: `ClusteringResult` for serialization and transfer

* **Clustering transfer and serialization**: New `ClusteringResult` enables:

    - Save/load clustering via `to_json()` / `from_json()`
    - Apply same clustering to different data via `apply()`
    - Transfer clustering from one dataset to another (e.g., cluster on wind, apply to all columns)

* **Integrated plotting** via `result.plot` accessor with Plotly (replaces matplotlib):

    - `result.plot.compare()`: Compare original vs reconstructed (overlay, side-by-side, or duration curves)
    - `result.plot.residuals()`: Visualize reconstruction errors (time series, histogram, by period, or by timestep)
    - `result.plot.cluster_representatives()`: Plot typical periods with cluster weights
    - `result.plot.cluster_members()`: All original periods per cluster with representative highlighted, interactive slider
    - `result.plot.cluster_weights()`: Cluster weight distribution
    - `result.plot.accuracy()`: Accuracy metrics (RMSE, MAE, duration RMSE) per column
    - `result.plot.segment_durations()`: Average segment durations (when using segmentation)

* **Hyperparameter tuning module** `tsam.tuning` with:

    - `find_optimal_combination()`: Find best n_clusters/n_segments combination
    - `find_pareto_front()`: Compute Pareto front of accuracy vs. complexity
    - Support for parallel execution
    - New parameters: `segment_representation`, `extremes`, `preserve_column_means`, `round_decimals`, `numerical_tolerance`

* **Accuracy metrics**: `AccuracyMetrics` class with `.summary` property for convenient DataFrame output

* **Utility functions**: `tsam.unstack_to_periods()` for reshaping time series for heatmap visualization

* `Distribution` and `MinMaxMean` **representation objects** for `ClusterConfig` and
  `SegmentConfig`, providing a structured alternative to plain string representation names

### Improvements

* Segment center preservation for better accuracy when using medoid/maxoid segment representation
* Consistent semantic naming across the entire codebase
* Better handling of extreme periods with `n_clusters` edge cases
* Lazy loading of optional modules (`plot`, `tuning`) to reduce import time

### Bug Fixes

These bugs existed in v2.3.9:

* Fixed rescaling with segmentation (was applying rescaling twice)
* Fixed `predictOriginalData()` denormalization when using `sameMean=True` with segmentation
* Fixed segment label ordering bug: `AgglomerativeClustering` produces arbitrary cluster labels,
  which caused `durationRepresentation()` with `distributionPeriodWise=False` to allocate
  the global distribution differently when transferring a clustering. Segment clusters are now
  relabelled to temporal order after `fit_predict()`.
* Fixed non-deterministic sorting in `durationRepresentation()` across both code paths
  by using `kind="stable"` and `np.round(mean, 10)` before `argsort`, ensuring
  identical tie-breaking across platforms.

### Result consistency

The stable sort fix guarantees cross-platform reproducibility but changes tie-breaking
compared to v2.3.9. Four distribution-related configurations (`hierarchical_distribution`,
`hierarchical_distribution_minmax`, `distribution_global`, `distribution_minmax_global`)
produce slightly different results, but will be consistent across systems from now on. All statistical properties are preserved. The remaining
23 configurations are bit-for-bit identical to v2.3.9. See the
[migration guide](migration-guide.md) for details.

### Known Limitations

* **Clustering transfer with 'replace' extreme method**: The 'replace' extreme method
  creates a hybrid cluster representation where some columns use the medoid values
  and others use the extreme period values. This hybrid representation cannot be
  perfectly reproduced during transfer via `ClusteringResult.apply()`. Warnings
  are issued when saving (`to_json()`) or applying such a clustering. For exact
  transfer with extreme periods, use 'append' or 'new_cluster' extreme methods instead.

### Performance

Multiple vectorization optimizations replace pandas loops with numpy array operations,
providing **35--77x** end-to-end speedups over v2.3.9 for most configurations.

**Benchmarked across 27 configurations x 4 datasets against v2.3.9:**

* Hierarchical methods on real-world data: **35--60x faster**
* Distribution representation (cluster-wise): **35--55x faster**
* Averaging: up to **77x faster**
* Contiguous clustering: **50--54x faster**
* Distribution representation (global scope): **7--16x faster**
* Iterative methods (kmeans, kmedoids, kmaxoids): **1--6x faster** (core solver dominates)

**Key function-level optimizations:**

* **`predictOriginalData()`**: Vectorized indexing replaces per-period
  `.unstack()` loop (~290x function speedup).
* **`durationRepresentation()`**: Vectorized numpy 3D operations replace
  nested pandas loops (~8x function speedup).
* **`_rescaleClusterPeriods()`**: numpy 3D arrays replace pandas
  MultiIndex operations (~11x function speedup).
* **`_clusterSortedPeriods()`**: numpy 3D reshape + sort replaces
  per-column DataFrame sorting loop (~12x function speedup).

### Testing

* Regression test suite: 296 old/new API equivalence tests + 148 golden-file tests
  comparing both APIs against baselines generated with tsam v2.3.9.
* Benchmark suite (`benchmarks/bench.py`) for performance comparison across versions
  using pytest-benchmark.

### Deprecations

* **TimeSeriesAggregation class**: The legacy class-based API now emits a `LegacyAPIWarning` when instantiated. It will be removed in a future version. Users should migrate to the new `tsam.aggregate()` function.

* **unstackToPeriods function**: Deprecated in favor of `tsam.unstack_to_periods()`.

* **HyperTunedAggregations class**: The legacy hyperparameter tuning class in `tsam.hyperparametertuning` is deprecated. Use `tsam.tuning.find_optimal_combination()` or `tsam.tuning.find_pareto_front()` instead.

* **getNoPeriodsForDataReduction / getNoSegmentsForDataReduction**: Helper functions deprecated along with `HyperTunedAggregations`.

* To suppress warnings during migration:

    ```python
    import warnings
    from tsam import LegacyAPIWarning
    warnings.filterwarnings("ignore", category=LegacyAPIWarning)
    ```

### Legacy API

The class-based API remains available for backward compatibility but is deprecated:

```python
import tsam.timeseriesaggregation as tsam_legacy

aggregation = tsam_legacy.TimeSeriesAggregation(
    raw,
    noTypicalPeriods=8,
    hoursPerPeriod=24,
    clusterMethod='hierarchical',
)
typical_periods = aggregation.createTypicalPeriods()
```

## [2.3.9](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.9)

* Improved time series aggregation speed with segmentation (issue #96)
* Fixed issue #99

## [2.3.8](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.8)

* Enhanced time series aggregation speed with segmentation (issue #96)

## [2.3.7](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.7)

* Added Python 3.13 support
* Updated GitHub Actions workflow (ubuntu-20.04 to ubuntu-22.04)
* Resolved invalid escape sequence error (issue #90)

## [2.3.6](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.6)

* Migrated from `setup.py` to `pyproject.toml`
* Changed project layout from flat to source structure
* Updated installation documentation
* Fixed deprecation and future warnings (issue #91)

## [2.3.5](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.5)

* Re-release of v2.3.4 to fix GitHub/PyPI synchronization

## [2.3.4](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.4)

* Extended reporting for time series tolerance exceedances
* Added option to silence tolerance warnings (default threshold: 1e-13)

## [2.3.3](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.3)

* Dropped support for Python versions below 3.9
* Fixed deprecation warnings

## [2.3.2](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.2)

* Limited pandas version to below 3.0
* Silenced deprecation warnings

## [2.3.1](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.1)

* Accelerated rescale cluster periods functionality
* Updated documentation with autodeployment features

## [2.3.0](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.3.0)

* Fixed deprecated pandas functions
* Corrected distribution representation sum calculations
* Added segment representation capability
* Extended default example
* Switched CI infrastructure from Travis to GitHub workflows

## [2.2.2](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.2.2)

* Fixed Hypertuning class
* Adjusted the default MILP solver
* Reworked documentation

## [2.1.0](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.1.0)

* Added hyperparameter tuning meta class for identifying optimal time series aggregation parameters

## [2.0.1](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.0.1)

* Changed dependency of scikit-learn to make tsam conda-forge compatible

## [2.0.0](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v2.0.0)

* A new comprehensive structure that allows for free cross-combination of clustering algorithms and cluster representations (e.g., centroids or medoids)
* A novel cluster representation method that precisely replicates the original time series value distribution based on [Hoffmann, Kotzur and Stolten (2021)](https://arxiv.org/abs/2111.12072)
* Maxoids as representation algorithm which represents time series by outliers only based on Sifa and Bauckhage (2017): "Online k-Maxoids clustering"
* K-medoids contiguity: An algorithm based on Oehrlein and Hauner (2017) that accounts for contiguity constraints

## [1.1.2](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v1.1.2)

* Added first version of the k-medoid contiguity algorithm

## [1.1.1](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v1.1.1)

* Significantly increased test coverage
* Separation between clustering and representation (e.g., for Ward's hierarchical clustering, the representation by medoids or centroids can now be freely chosen)

## [1.1.0](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v1.1.0)

* Segmentation (clustering of adjacent time steps) according to Pineda et al. (2018)
* k-MILP: Extension of MILP-based k-medoids clustering for automatic identification of extreme periods according to Zatti et al. (2019)
* Option to dynamically choose whether clusters should be represented by their centroid or medoid
