##################
tsam's Change Log
##################

*********************
Release version 3.1.1
*********************

tsam v3.1.1 is the first stable v3 release (versions 3.0.0 and 3.1.0 were yanked from PyPI).
It introduces a modern functional API alongside significant improvements to performance,
plotting, hyperparameter tuning, and overall code quality.

See the :ref:`migration guide <migration_guide>` for a complete guide on upgrading from v2.

Breaking Changes
================

* **New functional API**: The primary interface is now ``tsam.aggregate()`` which returns an ``AggregationResult`` object
* **Configuration objects**: Clustering and segmentation options are now configured via ``ClusterConfig``, ``SegmentConfig``, and ``ExtremeConfig`` dataclasses
* **Segment representation default**: In v2, omitting ``segmentRepresentationMethod`` caused segments
  to silently inherit the cluster ``representationMethod`` (e.g. distribution). In v3,
  ``SegmentConfig(representation=...)`` defaults to ``"mean"`` independently. If you relied on the
  implicit inheritance, pass the representation explicitly:
  ``SegmentConfig(n_segments=12, representation=Distribution(scope="global"))``
* **Removed methods**: The ``reconstruct()`` method has been removed; use the ``reconstructed`` property on ``AggregationResult`` instead
* **Renamed parameters**: Parameters have been renamed for consistency:

  ==================================  ======================================================
  Old (v2)                            New (v3)
  ==================================  ======================================================
  ``noTypicalPeriods``                ``n_clusters``
  ``hoursPerPeriod``                  ``period_duration``
  ``resolution``                      ``temporal_resolution``
  ``clusterMethod``                   ``cluster=ClusterConfig(method=...)``
  ``representationMethod``            ``cluster=ClusterConfig(representation=...)``
  ``segmentation`` + ``noSegments``   ``segments=SegmentConfig(n_segments=...)``
  ``sameMean``                        ``cluster=ClusterConfig(normalize_column_means=...)``
  ``rescaleClusterPeriods``           ``preserve_column_means``
  ``sortValues``                      ``cluster=ClusterConfig(use_duration_curves=...)``
  ``evalSumPeriods``                  ``cluster=ClusterConfig(include_period_sums=...)``
  ``weightDict``                      ``cluster=ClusterConfig(weights=...)``
  ``addPeakMax/Min``, etc.            ``extremes=ExtremeConfig(max_value=..., ...)``
  ==================================  ======================================================

New Features
============

* **Modern functional API**: New ``tsam.aggregate()`` function returns an ``AggregationResult`` with properties:

  - ``cluster_representatives``: DataFrame with aggregated typical periods
  - ``cluster_assignments``: Which cluster each original period belongs to
  - ``cluster_weights``: Occurrence count per cluster
  - ``accuracy``: ``AccuracyMetrics`` object with RMSE, MAE, and duration curve RMSE
  - ``reconstructed``: Reconstructed time series (cached property)
  - ``residuals``: Difference between original and reconstructed
  - ``original``: Access to original input data
  - ``clustering``: ``ClusteringResult`` for serialization and transfer

* **Clustering transfer and serialization**: New ``ClusteringResult`` enables:

  - Save/load clustering via ``to_json()`` / ``from_json()``
  - Apply same clustering to different data via ``apply()``
  - Transfer clustering from one dataset to another (e.g., cluster on wind, apply to all columns)

* **Integrated plotting** via ``result.plot`` accessor with Plotly (replaces matplotlib):

  - ``result.plot.compare()``: Compare original vs reconstructed (duration curves)
  - ``result.plot.residuals()``: Visualize reconstruction errors
  - ``result.plot.heatmap()``: Heatmap of cluster representatives
  - ``result.plot.cluster_assignments()``: Visualize period-to-cluster mapping

* **Hyperparameter tuning module** ``tsam.tuning`` with:

  - ``find_optimal_combination()``: Find best n_clusters/n_segments combination
  - ``find_pareto_front()``: Compute Pareto front of accuracy vs. complexity
  - Support for parallel execution
  - New parameters: ``segment_representation``, ``extremes``, ``preserve_column_means``, ``round_decimals``, ``numerical_tolerance``

* **Accuracy metrics**: ``AccuracyMetrics`` class with ``.summary`` property for convenient DataFrame output

* **Utility functions**: ``tsam.unstack_to_periods()`` for reshaping time series for heatmap visualization

* ``Distribution`` and ``MinMaxMean`` **representation objects** for ``ClusterConfig`` and
  ``SegmentConfig``, providing a structured alternative to plain string representation names

Improvements
============

* Segment center preservation for better accuracy when using medoid/maxoid segment representation
* Consistent semantic naming across the entire codebase
* Better handling of extreme periods with ``n_clusters`` edge cases
* Lazy loading of optional modules (``plot``, ``tuning``) to reduce import time

Bug Fixes
=========

These bugs existed in v2.3.9:

* Fixed rescaling with segmentation (was applying rescaling twice)
* Fixed ``predictOriginalData()`` denormalization when using ``sameMean=True`` with segmentation
* Fixed segment label ordering bug: ``AgglomerativeClustering`` produces arbitrary cluster labels,
  which caused ``durationRepresentation()`` with ``distributionPeriodWise=False`` to allocate
  the global distribution differently when transferring a clustering. Segment clusters are now
  relabelled to temporal order after ``fit_predict()``.
* Fixed non-deterministic sorting in ``durationRepresentation()`` across both code paths
  by using ``kind="stable"`` and ``np.round(mean, 10)`` before ``argsort``, ensuring
  identical tie-breaking across platforms.

Result consistency
==================

The stable sort fix guarantees cross-platform reproducibility but changes tie-breaking
compared to v2.3.9. Four distribution-related configurations (``hierarchical_distribution``,
``hierarchical_distribution_minmax``, ``distribution_global``, ``distribution_minmax_global``)
produce slightly different results, but will be consistent across systems from now on. All statistical properties are preserved. The remaining
23 configurations are bit-for-bit identical to v2.3.9. See the
:ref:`migration guide <migration_guide>` for details.

Known Limitations
=================

* **Clustering transfer with 'replace' extreme method**: The 'replace' extreme method
  creates a hybrid cluster representation where some columns use the medoid values
  and others use the extreme period values. This hybrid representation cannot be
  perfectly reproduced during transfer via ``ClusteringResult.apply()``. Warnings
  are issued when saving (``to_json()``) or applying such a clustering. For exact
  transfer with extreme periods, use 'append' or 'new_cluster' extreme methods instead.

Performance
===========

Multiple vectorization optimizations replace pandas loops with numpy array operations,
providing **35--77x** end-to-end speedups over v2.3.9 for most configurations.

**Benchmarked across 27 configurations × 4 datasets against v2.3.9:**

* Hierarchical methods on real-world data: **35--60x faster**
* Distribution representation (cluster-wise): **35--55x faster**
* Averaging: up to **77x faster**
* Contiguous clustering: **50--54x faster**
* Distribution representation (global scope): **7--16x faster**
* Iterative methods (kmeans, kmedoids, kmaxoids): **1--6x faster** (core solver dominates)

**Key function-level optimizations:**

* **``predictOriginalData()``**: Vectorized indexing replaces per-period
  ``.unstack()`` loop (~290x function speedup).
* **``durationRepresentation()``**: Vectorized numpy 3D operations replace
  nested pandas loops (~8x function speedup).
* **``_rescaleClusterPeriods()``**: numpy 3D arrays replace pandas
  MultiIndex operations (~11x function speedup).
* **``_clusterSortedPeriods()``**: numpy 3D reshape + sort replaces
  per-column DataFrame sorting loop (~12x function speedup).

Testing
=======

* Regression test suite: 296 old/new API equivalence tests + 148 golden-file tests
  comparing both APIs against baselines generated with tsam v2.3.9.
* Benchmark suite (``benchmarks/bench.py``) for performance comparison across versions
  using pytest-benchmark.

Deprecations
============

* **TimeSeriesAggregation class**: The legacy class-based API now emits a ``LegacyAPIWarning`` when instantiated. It will be removed in a future version. Users should migrate to the new ``tsam.aggregate()`` function.

* **unstackToPeriods function**: Deprecated in favor of ``tsam.unstack_to_periods()``.

* **HyperTunedAggregations class**: The legacy hyperparameter tuning class in ``tsam.hyperparametertuning`` is deprecated. Use ``tsam.tuning.find_optimal_combination()`` or ``tsam.tuning.find_pareto_front()`` instead.

* **getNoPeriodsForDataReduction / getNoSegmentsForDataReduction**: Helper functions deprecated along with ``HyperTunedAggregations``.

* To suppress warnings during migration::

    import warnings
    from tsam import LegacyAPIWarning
    warnings.filterwarnings("ignore", category=LegacyAPIWarning)

Legacy API
==========

The class-based API remains available for backward compatibility but is deprecated::

    import tsam.timeseriesaggregation as tsam_legacy

    aggregation = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod='hierarchical',
    )
    typical_periods = aggregation.createTypicalPeriods()


*********************
Release version 2.3.9
*********************

* Improved time series aggregation speed with segmentation (issue #96)
* Fixed issue #99


*********************
Release version 2.3.8
*********************

* Enhanced time series aggregation speed with segmentation (issue #96)


*********************
Release version 2.3.7
*********************

* Added Python 3.13 support
* Updated GitHub Actions workflow (ubuntu-20.04 → ubuntu-22.04)
* Resolved invalid escape sequence error (issue #90)


*********************
Release version 2.3.6
*********************

* Migrated from ``setup.py`` to ``pyproject.toml``
* Changed project layout from flat to source structure
* Updated installation documentation
* Fixed deprecation and future warnings (issue #91)


*********************
Release version 2.3.5
*********************

* Re-release of v2.3.4 to fix GitHub/PyPI synchronization


*********************
Release version 2.3.4
*********************

* Extended reporting for time series tolerance exceedances
* Added option to silence tolerance warnings (default threshold: 1e-13)


*********************
Release version 2.3.3
*********************

* Dropped support for Python versions below 3.9
* Fixed deprecation warnings


*********************
Release version 2.3.2
*********************

* Limited pandas version to below 3.0
* Silenced deprecation warnings


*********************
Release version 2.3.1
*********************

* Accelerated rescale cluster periods functionality
* Updated documentation with autodeployment features


*********************
Release version 2.3.0
*********************

* Fixed deprecated pandas functions
* Corrected distribution representation sum calculations
* Added segment representation capability
* Extended default example
* Switched CI infrastructure from Travis to GitHub workflows


*********************
Release version 2.2.2
*********************

* Fixed Hypertuning class
* Adjusted the default MILP solver
* Reworked documentation


*********************
Release version 2.1.0
*********************

* Added hyperparameter tuning meta class for identifying optimal time series aggregation parameters


*********************
Release version 2.0.1
*********************

* Changed dependency of scikit-learn to make tsam conda-forge compatible


*********************
Release version 2.0.0
*********************

* A new comprehensive structure that allows for free cross-combination of clustering algorithms and cluster representations (e.g., centroids or medoids)
* A novel cluster representation method that precisely replicates the original time series value distribution based on `Hoffmann, Kotzur and Stolten (2021) <https://arxiv.org/abs/2111.12072>`_
* Maxoids as representation algorithm which represents time series by outliers only based on Sifa and Bauckhage (2017): "Online k-Maxoids clustering"
* K-medoids contiguity: An algorithm based on Oehrlein and Hauner (2017) that accounts for contiguity constraints


*********************
Release version 1.1.2
*********************

* Added first version of the k-medoid contiguity algorithm


*********************
Release version 1.1.1
*********************

* Significantly increased test coverage
* Separation between clustering and representation (e.g., for Ward's hierarchical clustering, the representation by medoids or centroids can now be freely chosen)


*********************
Release version 1.1.0
*********************

* Segmentation (clustering of adjacent time steps) according to Pineda et al. (2018)
* k-MILP: Extension of MILP-based k-medoids clustering for automatic identification of extreme periods according to Zatti et al. (2019)
* Option to dynamically choose whether clusters should be represented by their centroid or medoid
