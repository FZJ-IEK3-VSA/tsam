# ETHOS.TSAM Change Log

All notable changes to this project will be documented in this file.

New entries are automatically added by [release-please](https://github.com/googleapis/release-please) from conventional commit messages.

## [4.0.0](https://github.com/FZJ-IEK3-VSA/tsam/compare/v3.4.2...v4.0.0) (2026-08-02)

tsam v4 is a rewrite of the internals: aggregation is now a chain of stateless functions in `src/tsam/pipeline/`, the class-based `TimeSeriesAggregation` API has been removed — `tsam.aggregate()` is the single entry point — and every internal identifier moved from camelCase to snake_case. For the overwhelming majority of configurations the results are bit-identical to 3.4.2; the exceptions are listed below and in the [migration guide](https://tsam.readthedocs.io/en/latest/migration/v3-to-v4/), which says for each change whether — and what — you need to do.

The v4 line was squash-merged as a single commit ([#234](https://github.com/FZJ-IEK3-VSA/tsam/issues/234)); the entries below are reconstructed from the pull requests it contains.


### ⚠ BREAKING CHANGES

* the legacy `TimeSeriesAggregation` API and the v3 compatibility shims have been removed — use `tsam.aggregate()` instead ([#337](https://github.com/FZJ-IEK3-VSA/tsam/issues/337), [#338](https://github.com/FZJ-IEK3-VSA/tsam/issues/338))
* package structure reorganized (`utils/` → `algorithms/`, new `pipeline/`, public serializers) ([#338](https://github.com/FZJ-IEK3-VSA/tsam/issues/338))
* new pipeline architecture, weight decoupling, and snake_case API ([#176](https://github.com/FZJ-IEK3-VSA/tsam/issues/176))
* per-column `weights` are now a top-level argument of `aggregate()`; `ClusterConfig(weights=...)` raises `TypeError` ([#176](https://github.com/FZJ-IEK3-VSA/tsam/issues/176))
* cluster and segment representations are resolved independently — v3 silently discarded the cluster setting when both were set ([#436](https://github.com/FZJ-IEK3-VSA/tsam/issues/436))
* the duration representation preserves the integral and the min/max envelope ([#376](https://github.com/FZJ-IEK3-VSA/tsam/issues/376))
* `cluster_representatives`, `reconstructed`, and `original` return columns in input order instead of alphabetically sorted ([#234](https://github.com/FZJ-IEK3-VSA/tsam/issues/234))
* `MinMaxMean` naming a column that is not in the data — or the same column in both `min_columns` and `max_columns` — now raises `ValueError` instead of being silently ignored ([#234](https://github.com/FZJ-IEK3-VSA/tsam/issues/234))
* review follow-ups on the v4 pipeline ([#434](https://github.com/FZJ-IEK3-VSA/tsam/issues/434))

### Features

* new pipeline architecture ([#234](https://github.com/FZJ-IEK3-VSA/tsam/issues/234)) ([5d99d59](https://github.com/FZJ-IEK3-VSA/tsam/commit/5d99d594f192d6e41c10ba0e0573d5a57c80719f))
* concurrency-preserving distribution ordering ([#377](https://github.com/FZJ-IEK3-VSA/tsam/issues/377), [#400](https://github.com/FZJ-IEK3-VSA/tsam/issues/400))
* type annotations for all functions in `src/tsam` ([#339](https://github.com/FZJ-IEK3-VSA/tsam/issues/339), [#402](https://github.com/FZJ-IEK3-VSA/tsam/issues/402))
* **plot:** `compare()` gained `time_slice` and a color dimension ([#338](https://github.com/FZJ-IEK3-VSA/tsam/issues/338))
* **plot:** new cluster-representative plot ([#412](https://github.com/FZJ-IEK3-VSA/tsam/issues/412))

### Bug Fixes

* **result:** `ClusteringResult.apply()` is now faithful to the run it replays ([#438](https://github.com/FZJ-IEK3-VSA/tsam/issues/438))
* **algorithms:** representative selection breaks ties deterministically ([#439](https://github.com/FZJ-IEK3-VSA/tsam/issues/439))
* **pipeline:** restore v3 parity for period sums ([#436](https://github.com/FZJ-IEK3-VSA/tsam/issues/436))
* **representations:** correct maxoid variable name and document its scope ([#366](https://github.com/FZJ-IEK3-VSA/tsam/issues/366), [#419](https://github.com/FZJ-IEK3-VSA/tsam/issues/419))
* **docs:** disable `navigation.instant` so notebook plots render ([#388](https://github.com/FZJ-IEK3-VSA/tsam/issues/388))

### Deprecations

* `result.plot.cluster_weights()` is renamed to `result.plot.cluster_counts()`; the old name still works and emits a `FutureWarning`
* `Distribution(scope="cluster")` is renamed to `Distribution(scope="local")`; `"cluster"` is normalized and emits a `FutureWarning` ([#378](https://github.com/FZJ-IEK3-VSA/tsam/issues/378), [#382](https://github.com/FZJ-IEK3-VSA/tsam/issues/382), [#383](https://github.com/FZJ-IEK3-VSA/tsam/issues/383))

### Documentation

* documentation restructured along Diátaxis (Tutorials / How-to / Explanation / Reference) ([#412](https://github.com/FZJ-IEK3-VSA/tsam/issues/412))
* v4 API reference and architecture docs ([#331](https://github.com/FZJ-IEK3-VSA/tsam/issues/331))
* docstrings normalized to Google style ([#339](https://github.com/FZJ-IEK3-VSA/tsam/issues/339), [#379](https://github.com/FZJ-IEK3-VSA/tsam/issues/379), [#384](https://github.com/FZJ-IEK3-VSA/tsam/issues/384), [#416](https://github.com/FZJ-IEK3-VSA/tsam/issues/416))
* terminology in docs and notebooks aligned with the glossary ([#422](https://github.com/FZJ-IEK3-VSA/tsam/issues/422))
* branding and logos unified across README and the RTD landing page ([#433](https://github.com/FZJ-IEK3-VSA/tsam/issues/433))
* tuning-notebook animation starts at full resolution ([#421](https://github.com/FZJ-IEK3-VSA/tsam/issues/421))

### Already shipped in 3.x (v4 ports — not new when upgrading from 3.4.2)

* DatetimeIndex preserved through the aggregate/disaggregate round-trip ([#314](https://github.com/FZJ-IEK3-VSA/tsam/issues/314) — released in 3.4.0)
* column weights in the accuracy metrics ([#263](https://github.com/FZJ-IEK3-VSA/tsam/issues/263) — released in 3.3.0)

## [3.4.2](https://github.com/FZJ-IEK3-VSA/tsam/compare/v3.4.1...v3.4.2) (2026-07-22)


### Bug Fixes

* **ci:** run mypy as a local hook against the project environment ([#363](https://github.com/FZJ-IEK3-VSA/tsam/issues/363)) ([46aed9b](https://github.com/FZJ-IEK3-VSA/tsam/commit/46aed9b3152e81f707ba368bfc27bdba72012243))
* Conurrency and integral fix ([#373](https://github.com/FZJ-IEK3-VSA/tsam/issues/373)) ([09a736c](https://github.com/FZJ-IEK3-VSA/tsam/commit/09a736ca3e44c9b5a28f04c8bd151b95d5133b3b))
* plot cluster_representatives for segmented results ([#340](https://github.com/FZJ-IEK3-VSA/tsam/issues/340)) ([#357](https://github.com/FZJ-IEK3-VSA/tsam/issues/357)) ([12cf9bf](https://github.com/FZJ-IEK3-VSA/tsam/commit/12cf9bf825be3580e2ef5a6f0664f85090cae1db))

## [3.4.1](https://github.com/FZJ-IEK3-VSA/tsam/compare/v3.4.0...v3.4.1) (2026-05-31)


### Bug Fixes

* warn that aggregate() column order will change in v4 ([#330](https://github.com/FZJ-IEK3-VSA/tsam/issues/330)) ([97f6e6d](https://github.com/FZJ-IEK3-VSA/tsam/commit/97f6e6d984a0eb089547fe0c61f95353498de416))

## [3.4.0](https://github.com/FZJ-IEK3-VSA/tsam/compare/v3.3.0...v3.4.0) (2026-05-15)


### Features

* preserve DatetimeIndex through aggregate/disaggregate round-trip ([#267](https://github.com/FZJ-IEK3-VSA/tsam/issues/267)) ([8d0240a](https://github.com/FZJ-IEK3-VSA/tsam/commit/8d0240a2a436afccf73c1bcb90a8d1ceceae0034))


### Bug Fixes

* **ci:** gracefully skip GitHub release when already created by release-please ([#260](https://github.com/FZJ-IEK3-VSA/tsam/issues/260)) ([6fc668d](https://github.com/FZJ-IEK3-VSA/tsam/commit/6fc668da8f8d8827d2475ec47b44c1e74317dd01))
* handle missing columns in weightDict in accuracyIndicators ([#288](https://github.com/FZJ-IEK3-VSA/tsam/issues/288)) ([a475570](https://github.com/FZJ-IEK3-VSA/tsam/commit/a475570e67d19bc2c41bdcb4566d9c5ba5d5c758))

## [3.3.0](https://github.com/FZJ-IEK3-VSA/tsam/compare/v3.2.1...v3.3.0) (2026-03-30)


### Features

* AccuracyMetrics now exposes weighted_rmse, weighted_mae, and weighted_rmse_duration as pre-computed scalars ([#238](https://github.com/FZJ-IEK3-VSA/tsam/issues/238)) ([b70b819](https://github.com/FZJ-IEK3-VSA/tsam/commit/b70b81998c03473cc494834f5589ba7364cf5ff9))
* add disaggregate() method ([#245](https://github.com/FZJ-IEK3-VSA/tsam/issues/245)) ([b24e32e](https://github.com/FZJ-IEK3-VSA/tsam/commit/b24e32e4263b5d97c127b3e881d5d845228c01b9))


### Bug Fixes

* make LegacyAPIWarning visible by default before v4 removal ([#236](https://github.com/FZJ-IEK3-VSA/tsam/issues/236)) ([#237](https://github.com/FZJ-IEK3-VSA/tsam/issues/237)) ([37ff3d8](https://github.com/FZJ-IEK3-VSA/tsam/commit/37ff3d88a1f28b64bcb4615ba9842f81d8d8bb43))

## [3.2.1](https://github.com/FZJ-IEK3-VSA/tsam/compare/v3.2.0...v3.2.1) (2026-03-25)

### Bug Fixes

* use column weights in tuning RMSE objective ([#227](https://github.com/FZJ-IEK3-VSA/tsam/issues/227)) ([1ceee5c](https://github.com/FZJ-IEK3-VSA/tsam/commit/1ceee5c69856b61aed9eae3f5d5f713be8ac85e9)), closes [#226](https://github.com/FZJ-IEK3-VSA/tsam/issues/226)

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
[v2 to v3 migration guide](migration/v2-to-v3.md#result-consistency-and-reproducibility) for details.

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

## [2.3.9](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v.2.3.9)

* Improved time series aggregation speed with segmentation (issue #96)
* Fixed issue #99

## [2.3.8](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v.2.3.8)

* Enhanced time series aggregation speed with segmentation (issue #96)

## [2.3.7](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v.2.3.7)

* Added Python 3.13 support
* Updated GitHub Actions workflow (ubuntu-20.04 to ubuntu-22.04)
* Resolved invalid escape sequence error (issue #90)

## [2.3.6](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v.2.3.6)

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

## 1.1.0

* Segmentation (clustering of adjacent time steps) according to Pineda et al. (2018)
* k-MILP: Extension of MILP-based k-medoids clustering for automatic identification of extreme periods according to Zatti et al. (2019)
* Option to dynamically choose whether clusters should be represented by their centroid or medoid
