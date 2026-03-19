# Changelog

All notable changes to this project will be documented in this file.

New entries are automatically added by [release-please](https://github.com/googleapis/release-please) from conventional commit messages.

## [3.1.1](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v3.1.1)

tsam v3.1.1 is the first stable v3 release (versions 3.0.0 and 3.1.0 were yanked from PyPI).
It introduces a modern functional API alongside significant improvements to performance,
plotting, hyperparameter tuning, and overall code quality.

See the [migration guide](https://tsam.readthedocs.io/en/latest/migrationGuideDoc.html) for upgrading from v2.

### Breaking Changes

* **New functional API**: The primary interface is now `tsam.aggregate()` which returns an `AggregationResult` object
* **Configuration objects**: Clustering and segmentation options are now configured via `ClusterConfig`, `SegmentConfig`, and `ExtremeConfig` dataclasses
* **Segment representation default**: In v2, omitting `segmentRepresentationMethod` caused segments to silently inherit the cluster `representationMethod`. In v3, `SegmentConfig(representation=...)` defaults to `"mean"` independently.
* **Removed methods**: The `reconstruct()` method has been removed; use the `reconstructed` property on `AggregationResult` instead
* **Renamed parameters**: See migration guide for full mapping table

### New Features

* **Modern functional API**: `tsam.aggregate()` returns `AggregationResult` with `cluster_representatives`, `cluster_assignments`, `cluster_weights`, `accuracy`, `reconstructed`, `residuals`, `original`, `clustering` properties
* **Clustering transfer and serialization**: `ClusteringResult` with `to_json()` / `from_json()` / `apply()`
* **Integrated plotting** via `result.plot` accessor with Plotly (replaces matplotlib): `compare()`, `residuals()`, `cluster_representatives()`, `cluster_members()`, `cluster_weights()`, `accuracy()`, `segment_durations()`
* **Hyperparameter tuning module** `tsam.tuning` with `find_optimal_combination()` and `find_pareto_front()` with parallel execution support
* **Accuracy metrics**: `AccuracyMetrics` class with `.summary` property
* **Utility functions**: `tsam.unstack_to_periods()`
* `Distribution` and `MinMaxMean` representation objects for structured configuration

### Bug Fixes

* Fixed rescaling with segmentation (was applying rescaling twice)
* Fixed `predictOriginalData()` denormalization when using `sameMean=True` with segmentation
* Fixed segment label ordering bug causing non-deterministic `durationRepresentation()` with `distributionPeriodWise=False`
* Fixed non-deterministic sorting in `durationRepresentation()` with `kind="stable"` and `np.round(mean, 10)`

### Performance

35--77x end-to-end speedups over v2.3.9 for most configurations via numpy vectorization.

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

* New comprehensive structure for free cross-combination of clustering algorithms and cluster representations
* Novel cluster representation method replicating original time series value distribution (Hoffmann, Kotzur and Stolten, 2021)
* Maxoids as representation algorithm (Sifa and Bauckhage, 2017)
* K-medoids contiguity algorithm (Oehrlein and Hauner, 2017)

## [1.1.2](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v1.1.2)

* Added first version of the k-medoid contiguity algorithm

## [1.1.1](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v1.1.1)

* Significantly increased test coverage
* Separation between clustering and representation

## [1.1.0](https://github.com/FZJ-IEK3-VSA/tsam/releases/tag/v1.1.0)

* Segmentation according to Pineda et al. (2018)
* k-MILP extension for automatic identification of extreme periods (Zatti et al., 2019)
* Option to dynamically choose centroid or medoid representation
