##################
tsam's Change Log
##################

*********************
Release version 4.0.0
*********************

Breaking Changes
================

* **Decoupled column weights from data pipeline**: ``ClusterConfig.weights`` now affects
  **only** the clustering distance calculation.  Previously, weights were multiplied into
  the normalized data during preprocessing, which forced every downstream step to
  compensate (rescaling widened clip bounds by the weight factor, reconstruction divided
  out weights before denormalization, and accuracy computation divided out weights
  before comparison).

  **Where weights are applied now:**

  - A separate *weighted candidates* array is created after unstacking and used
    exclusively as the distance matrix for clustering (step 4).
  - When ``include_period_sums`` is enabled, period-sum features are appended to
    the weighted candidates only.
  - Cluster *representations* (medoid, mean, distribution, etc.) are computed from
    the **unweighted** candidates so that typical periods live in the original
    normalized space.

  **Where weights are no longer applied:**

  - ``normalize()`` no longer accepts or applies weights.
  - ``NormalizedData`` no longer carries a ``weights`` field.
  - ``denormalize()`` no longer has an ``apply_weights`` parameter.
  - ``rescale_representatives()`` no longer widens the clipping bound by the
    weight factor.
  - ``reconstruct()`` no longer divides out weights before denormalization.
  - ``compute_accuracy()`` no longer divides out weights before error comparison.

  This simplifies the pipeline, eliminates a class of weight-related bugs, and makes
  the semantics of ``ClusterConfig.weights`` match its documentation: *"per-column
  weights for clustering distance calculation"*.

  **Compatibility note:** Cluster *assignments* are identical to the previous version
  (the weighted distance matrix is mathematically equivalent).  With medoid or maxoid
  representation, the selected representative may differ in edge cases because the
  medoid is now chosen in the unweighted output space rather than the weighted
  clustering space.  The legacy ``TimeSeriesAggregation`` class is unchanged.

* **Output column order preserved**: ``cluster_representatives``, ``reconstructed``, and ``original``
  now return columns in the same order as the input DataFrame. Previously, columns were
  alphabetically sorted. Code that relied on sorted column order may need adjustment.

* **Renamed ``cluster_weights`` to ``cluster_counts``**: ``AggregationResult.cluster_weights``
  has been renamed to ``cluster_counts`` to avoid confusion with per-column clustering weights
  (``ClusterConfig.weights``). The old name remains as a deprecated property emitting
  ``FutureWarning``.

Improvements
============

* **Column weights preserved during transfer**: ``ClusteringResult.apply()`` now replays
  the original ``ClusterConfig.weights`` so that transferred results use the same
  clustering distance when recomputing representatives.

* **Fixed type annotation**: ``cluster_counts`` (formerly ``cluster_weights``) is now
  correctly annotated as ``dict[int, float]`` since partial-period adjustment can produce
  fractional values.

*********************
Release version 3.1.1
*********************

New Features
============

* Added ``Distribution`` and ``MinMaxMean`` representation objects for ``ClusterConfig`` and
  ``SegmentConfig``, providing a structured alternative to plain string representation names.

* Added migration guide (``docs/source/migrationGuideDoc.rst``) documenting the transition
  from the v2 class-based API to the v3 functional API.

Bug Fixes
=========

* Fixed segment label ordering bug: ``AgglomerativeClustering`` produces arbitrary cluster labels,
  which caused ``durationRepresentation()`` with ``distributionPeriodWise=False`` to allocate
  the global distribution differently when transferring a clustering. Segment clusters are now
  relabelled to temporal order after ``fit_predict()``.

* Fixed non-deterministic tie-breaking in vectorized ``durationRepresentation()``
  (``distributionPeriodWise=True`` path). The vectorized mean computation used C-contiguous
  memory layout, producing ~1e-16 floating-point differences from the original pandas-based code,
  which caused ``argsort`` to swap tied timesteps. Fixed by computing per-attribute means on
  F-contiguous arrays to match pandas' accumulation order. Verified against tsam v2.3.9
  golden baselines.

Testing & Benchmarks
====================

* Added extensive regression test suite (296 equivalence tests + 148 golden-file tests) comparing
  both APIs against golden baselines generated with tsam v2.3.9.

* Added benchmark suite (``benchmarks/bench.py``) for performance comparison across versions
  using pytest-benchmark. Current version is **50--80x faster** than v2.3.9 for hierarchical
  methods on real-world data.

*********************
Release version 3.1.0
*********************

* Removed ``preserve_n_clusters`` from ``ExtremeConfig``. Extreme periods are always added on top
  of ``n_clusters``.

* Removed ``matplotlib`` dependency; all plotting and notebooks now use ``plotly``

* The tdqm test pipeline has been reduced due to the number of versions and the maturity of the library.

* Renamed the examples for more clarity

*********************
Release version 3.0.0
*********************

tsam v3.0.0 is a major release introducing a modern, functional API alongside significant improvements to plotting, hyperparameter tuning, and overall code quality.

Breaking Changes
================

* **New functional API**: The primary interface is now ``tsam.aggregate()`` which returns an ``AggregationResult`` object
* **Configuration objects**: Clustering and segmentation options are now configured via ``ClusterConfig``, ``SegmentConfig``, and ``ExtremeConfig`` dataclasses
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
  - ``cluster_counts``: Occurrence count per cluster (fractional for partial periods)
  - ``accuracy``: ``AccuracyMetrics`` object with RMSE, MAE, and duration curve RMSE
  - ``reconstructed``: Reconstructed time series (cached property)
  - ``residuals``: Difference between original and reconstructed
  - ``original``: Access to original input data
  - ``clustering``: ``ClusteringResult`` for serialization and transfer

* **Clustering transfer and serialization**: New ``ClusteringResult`` enables:

  - Save/load clustering via ``to_json()`` / ``from_json()``
  - Apply same clustering to different data via ``apply()``
  - Transfer clustering from one dataset to another (e.g., cluster on wind, apply to all columns)

* **Integrated plotting** via ``result.plot`` accessor with Plotly:

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

Improvements
============

* Segment center preservation for better accuracy when using medoid/maxoid segment representation
* Consistent semantic naming across the entire codebase
* Better handling of extreme periods with ``n_clusters`` edge cases
* Fixed rescaling with segmentation (was applying rescaling twice)
* Fixed ``predict_original_data()`` denormalization when using ``same_mean=True`` with segmentation
* Fixed non-deterministic sorting in duration representation by using stable sort, ensuring reproducible results across environments
* Lazy loading of optional modules (``plot``, ``tuning``) to reduce import time

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

Multiple vectorization optimizations replace pandas loops with numpy array operations, significantly improving performance for datasets with many columns.

**Function-level optimizations:**

* **``predict_original_data()``** (reconstruction step, always used):
  Replaced per-period ``.unstack()`` loop with single vectorized indexing.
  Function speedup: ~650ms → ~2ms (**290x**).

* **``_rescale_cluster_periods()``** (only when ``preserve_column_means=True``):
  Replaced pandas MultiIndex operations with numpy 3D array operations.
  Function speedup: ~400ms → ~36ms (**11x**).

* **``_cluster_sorted_periods()``** (only when ``use_duration_curves=True``):
  Replaced per-column DataFrame sorting loop with numpy 3D reshape + sort.
  Sorting step speedup: ~291ms → ~25ms (**12x**).

* **``duration_representation()``** (only when ``representation='distribution'``):
  Replaced nested loops with pandas MultiIndex indexing with numpy 3D operations.
  Function speedup: ~220ms → ~26ms (**8x**).

**Combined workflow benchmarks** (8760 hours, 4 columns):

  =============================================  ==========  ==========  =========
  Workflow                                       Before      After       Speedup
  =============================================  ==========  ==========  =========
  Basic (cluster + reconstruct)                  1244 ms     20 ms       **64x**
  All options combined                           1268 ms     29 ms       **45x**
  =============================================  ==========  ==========  =========

Deprecations
============

* **TimeSeriesAggregation class**: The legacy class-based API now emits a ``LegacyAPIWarning`` when instantiated. It will be removed in a future version. Users should migrate to the new ``tsam.aggregate()`` function.

* **unstackToPeriods function**: Deprecated in favor of ``tsam.unstack_to_periods()``.

* **HyperTunedAggregations class**: The legacy hyperparameter tuning class in ``tsam.hyperparametertuning`` is deprecated. Use ``tsam.tuning.find_optimal_combination()`` or ``tsam.tuning.find_pareto_front()`` instead.

* **get_no_periods_for_data_reduction / get_no_segments_for_data_reduction**: Helper functions deprecated along with ``HyperTunedAggregations``.

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
        no_typical_periods=8,
        hours_per_period=24,
        cluster_method='hierarchical',
    )
    typical_periods = aggregation.create_typical_periods()

Quick Migration Example
=======================

**Before (v2)**::

    import tsam.timeseriesaggregation as tsam

    agg = tsam.TimeSeriesAggregation(
        df,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod='hierarchical',
        representationMethod='distributionAndMinMaxRepresentation',
        segmentation=True,
        noSegments=12,
    )
    typical = agg.createTypicalPeriods()
    reconstructed = agg.predictOriginalData()

**After (v3)**::

    import tsam
    from tsam import ClusterConfig, SegmentConfig

    result = tsam.aggregate(
        df,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method='hierarchical',
            representation='distribution_minmax',
        ),
        segments=SegmentConfig(n_segments=12),
    )
    typical = result.cluster_representatives
    reconstructed = result.reconstructed


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
