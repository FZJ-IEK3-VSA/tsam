.. _migration_guide:

############################
Migrating from tsam v2 to v3
############################

tsam v3 replaces the class-based API with a functional API.
The old ``TimeSeriesAggregation`` class still works but is deprecated
and will be removed in a future release.

This guide covers every change you need to make.

***********************
Quick before-and-after
***********************

**v2**::

    import tsam.timeseriesaggregation as tsam

    agg = tsam.TimeSeriesAggregation(
        df,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod='hierarchical',
        representationMethod='distributionAndMinMaxRepresentation',
        segmentation=True,
        noSegments=12,
        rescaleClusterPeriods=True,
        addPeakMax=['demand'],
    )
    representatives = agg.createTypicalPeriods()
    reconstructed = agg.predictOriginalData()
    accuracy = agg.accuracyIndicators()

**v3**::

    import tsam
    from tsam import ClusterConfig, SegmentConfig, ExtremeConfig

    result = tsam.aggregate(
        df,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method='hierarchical',
            representation='distribution_minmax',
        ),
        segments=SegmentConfig(n_segments=12),
        preserve_column_means=True,
        extremes=ExtremeConfig(max_value=['demand']),
    )
    representatives = result.cluster_representatives
    reconstructed = result.reconstructed
    accuracy = result.accuracy.summary


***********************
Parameter mapping
***********************

The table below maps every old parameter to its v3 equivalent.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Old (v2)
     - New (v3)
     - Notes
   * - ``timeSeries``
     - ``data``
     - Renamed.
   * - ``noTypicalPeriods``
     - ``n_clusters``
     -
   * - ``hoursPerPeriod``
     - ``period_duration``
     - Also accepts strings (``'24h'``, ``'1d'``).
   * - ``resolution``
     - ``temporal_resolution``
     - Also accepts strings (``'1h'``, ``'15min'``).
   * - ``clusterMethod``
     - ``ClusterConfig(method=...)``
     - See :ref:`cluster method values <migration_cluster_methods>`.
   * - ``representationMethod``
     - ``ClusterConfig(representation=...)``
     - See :ref:`representation values <migration_representation_methods>`.
   * - ``weightDict``
     - ``ClusterConfig(weights=...)``
     -
   * - ``sameMean``
     - ``ClusterConfig(normalize_column_means=...)``
     -
   * - ``sortValues``
     - ``ClusterConfig(use_duration_curves=...)``
     -
   * - ``evalSumPeriods``
     - ``ClusterConfig(include_period_sums=...)``
     -
   * - ``solver``
     - ``ClusterConfig(solver=...)``
     -
   * - ``segmentation``
     - Pass a ``SegmentConfig`` or omit it.
     - No boolean flag needed.
   * - ``noSegments``
     - ``SegmentConfig(n_segments=...)``
     -
   * - ``segmentRepresentationMethod``
     - ``SegmentConfig(representation=...)``
     - Uses short names (see below).
   * - ``rescaleClusterPeriods``
     - ``preserve_column_means``
     - Top-level kwarg of ``aggregate()``.
   * - ``rescaleExcludeColumns``
     - ``rescale_exclude_columns``
     -
   * - ``roundOutput``
     - ``round_decimals``
     -
   * - ``numericalTolerance``
     - ``numerical_tolerance``
     -
   * - ``extremePeriodMethod``
     - ``ExtremeConfig(method=...)``
     - See :ref:`extreme method values <migration_extreme_methods>`.
   * - ``addPeakMax``
     - ``ExtremeConfig(max_value=...)``
     -
   * - ``addPeakMin``
     - ``ExtremeConfig(min_value=...)``
     -
   * - ``addMeanMax``
     - ``ExtremeConfig(max_period=...)``
     -
   * - ``addMeanMin``
     - ``ExtremeConfig(min_period=...)``
     -
   * - ``distributionPeriodWise``
     - ``Distribution(scope="cluster"|"global")``
     - See :ref:`representation objects <migration_representation_objects>`.
   * - ``representationDict``
     - ``MinMaxMean(max_columns=[...], min_columns=[...])``
     - See :ref:`representation objects <migration_representation_objects>`.


.. _migration_cluster_methods:

Cluster method values
=====================

.. list-table::
   :header-rows: 1
   :widths: 40 40

   * - Old (v2)
     - New (v3)
   * - ``'averaging'``
     - ``'averaging'``
   * - ``'k_means'``
     - ``'kmeans'``
   * - ``'k_medoids'``
     - ``'kmedoids'``
   * - ``'k_maxoids'``
     - ``'kmaxoids'``
   * - ``'hierarchical'``
     - ``'hierarchical'``
   * - ``'adjacent_periods'``
     - ``'contiguous'``


.. _migration_representation_methods:

Representation method values
============================

.. list-table::
   :header-rows: 1
   :widths: 50 40

   * - Old (v2)
     - New (v3)
   * - ``'meanRepresentation'``
     - ``'mean'``
   * - ``'medoidRepresentation'``
     - ``'medoid'``
   * - ``'maxoidRepresentation'``
     - ``'maxoid'``
   * - ``'distributionRepresentation'``
     - ``'distribution'``
   * - ``'durationRepresentation'``
     - ``'distribution'`` (both old parameters meant the same)
   * - ``'distributionAndMinMaxRepresentation'``
     - ``'distribution_minmax'``
   * - ``'minmaxmeanRepresentation'``
     - ``'minmax_mean'``


.. _migration_representation_objects:

Typed representation objects
============================

For ``distribution``, ``distribution_minmax``, and ``minmax_mean``
representations, v3 offers typed objects that expose options previously
controlled by separate parameters (``distributionPeriodWise``,
``representationDict``). Plain string shortcuts still work for the
common cases.

**Distribution with global scope** (``distributionPeriodWise=False``):

*v2*::

    agg = tsam.TimeSeriesAggregation(
        df,
        noTypicalPeriods=8,
        representationMethod='distributionRepresentation',
        distributionPeriodWise=False,
    )

*v3*::

    from tsam import Distribution

    result = tsam.aggregate(
        df,
        n_clusters=8,
        cluster=ClusterConfig(
            representation=Distribution(scope="global"),
        ),
    )

**Distribution with min/max preservation and global scope**:

*v2*::

    agg = tsam.TimeSeriesAggregation(
        df,
        noTypicalPeriods=8,
        representationMethod='distributionAndMinMaxRepresentation',
        distributionPeriodWise=False,
    )

*v3*::

    from tsam import Distribution

    result = tsam.aggregate(
        df,
        n_clusters=8,
        cluster=ClusterConfig(
            representation=Distribution(scope="global", preserve_minmax=True),
        ),
    )

**Per-column min/max/mean** (``representationDict``):

*v2*::

    agg = tsam.TimeSeriesAggregation(
        df,
        noTypicalPeriods=8,
        representationMethod='minmaxmeanRepresentation',
        representationDict={'GHI': 'max', 'T': 'min', 'Wind': 'mean', 'Load': 'min'},
    )

*v3*::

    from tsam import MinMaxMean

    result = tsam.aggregate(
        df,
        n_clusters=8,
        cluster=ClusterConfig(
            representation=MinMaxMean(
                max_columns=['GHI'],
                min_columns=['T', 'Load'],
            ),
        ),
    )

Columns not listed in ``max_columns`` or ``min_columns`` default to mean.

.. note::
   The string shortcuts ``"distribution"``, ``"distribution_minmax"``, and
   ``"minmax_mean"`` remain valid and are equivalent to:

   - ``"distribution"`` → ``Distribution()``
   - ``"distribution_minmax"`` → ``Distribution(preserve_minmax=True)``
   - ``"minmax_mean"`` → ``MinMaxMean()`` (all columns default to mean)


.. _migration_extreme_methods:

Extreme method values
=====================

.. list-table::
   :header-rows: 1
   :widths: 40 40

   * - Old (v2)
     - New (v3)
   * - ``'None'``
     - Omit the ``extremes`` parameter entirely.
   * - ``'append'``
     - ``'append'``
   * - ``'replace_cluster_center'``
     - ``'replace'``
   * - ``'new_cluster_center'``
     - ``'new_cluster'``


.. _migration_defaults:

***********************
Default changes
***********************

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Parameter
     - Old default
     - New default
     - Impact
   * - ``n_clusters``
     - 10
     - *required*
     - Code that relied on the default must now pass a value explicitly.

***********************
Accessing results
***********************

The old API returned raw DataFrames and arrays from methods you had to
call in sequence. The new API returns a single ``AggregationResult``
object with everything attached.

.. list-table::
   :header-rows: 1
   :widths: 45 45

   * - Old (v2)
     - New (v3)
   * - ``agg.createTypicalPeriods()``
     - ``result.cluster_representatives``
   * - ``agg.predictOriginalData()``
     - ``result.reconstructed``
   * - ``agg.accuracyIndicators()``
     - ``result.accuracy.summary``
   * - ``agg.clusterOrder``
     - ``result.cluster_assignments``
   * - ``agg.clusterPeriodNoOccur``
     - ``result.cluster_weights``
   * - ``agg.clusterCenterIndices``
     - ``result.clustering.cluster_centers``
   * - ``agg.timeSeries``
     - ``result.original``
   * - *(no equivalent)*
     - ``result.residuals``
   * - *(no equivalent)*
     - ``result.plot.compare()``

The ``cluster_representatives`` DataFrame now uses a
``MultiIndex(cluster, timestep)`` instead of
``MultiIndex(PeriodNum, TimeStep)``.


***********************
Clustering transfer
***********************

Reusing a clustering on new data used to require manually passing
``predefClusterOrder``, ``predefClusterCenterIndices``, etc.
In v3 this is a single method call::

    # Cluster on one dataset
    result = tsam.aggregate(df_wind, n_clusters=8)

    # Apply same clustering to another dataset
    result_all = result.clustering.apply(df_all)

You can also save and load clusterings::

    result.clustering.to_json("clustering.json")

    from tsam import ClusteringResult
    clustering = ClusteringResult.from_json("clustering.json")
    result = clustering.apply(df)


***********************
Plotting
***********************

Plotting has moved from ``matplotlib`` to ``plotly``.
Instead of calling separate functions, use the ``result.plot`` accessor::

    result.plot.compare()               # Duration curves: original vs reconstructed
    result.plot.residuals()             # Reconstruction errors
    result.plot.heatmap()               # Heatmap of cluster representatives
    result.plot.cluster_assignments()   # Period-to-cluster mapping
    result.plot.cluster_weights()       # Cluster occurrence counts
    result.plot.accuracy()              # Accuracy metrics bar chart


***********************
Hyperparameter tuning
***********************

The ``HyperTunedAggregations`` class is replaced by two functions in
``tsam.tuning``.

``identifyOptimalSegmentPeriodCombination`` → ``find_optimal_combination``
==========================================================================

**v2**::

    from tsam.hyperparametertuning import HyperTunedAggregations
    import tsam.timeseriesaggregation as tsam_legacy

    agg = HyperTunedAggregations(
        tsam_legacy.TimeSeriesAggregation(
            df,
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="meanRepresentation",
            segmentation=True,
        )
    )
    segments, periods, rmse = agg.identifyOptimalSegmentPeriodCombination(
        dataReduction=0.01,
    )

**v3**::

    import tsam
    from tsam import ClusterConfig

    result = tsam.tuning.find_optimal_combination(
        df,
        data_reduction=0.01,
        period_duration=24,
        cluster=ClusterConfig(method="hierarchical"),
        segment_representation="mean",
    )
    segments = result.n_segments
    periods = result.n_clusters
    rmse = result.rmse
    best = result.best_result          # AggregationResult

``identifyParetoOptimalAggregation`` → ``find_pareto_front``
=============================================================

**v2**::

    agg.identifyParetoOptimalAggregation(untilTotalTimeSteps=500)
    for a in agg.aggregationHistory:
        print(a.totalAccuracyIndicators()["RMSE"])

**v3**::

    pareto = tsam.tuning.find_pareto_front(
        df,
        period_duration=24,
        max_timesteps=500,
        cluster=ClusterConfig(method="hierarchical"),
        segment_representation="mean",
    )
    print(pareto.summary)              # DataFrame of all tested configs
    pareto.plot()                      # Interactive Plotly visualization

The ``TuningResult`` returned by both functions also supports
``find_by_timesteps(target)`` and ``find_by_rmse(threshold)`` for
querying specific configurations, and iteration via ``for r in result``.

Helper functions
================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old (v2)
     - New (v3)
   * - ``getNoPeriodsForDataReduction(n, segs, red)``
     - ``tsam.tuning.find_clusters_for_reduction(n, segs, red)``
   * - ``getNoSegmentsForDataReduction(n, periods, red)``
     - ``tsam.tuning.find_segments_for_reduction(n, periods, red)``

New capabilities
================

- **Parallel execution**: Pass ``n_jobs=-1`` to use all CPU cores.
- **Targeted exploration**: ``find_pareto_front`` accepts a ``timesteps``
  sequence (e.g., ``range(10, 500, 10)``) for faster targeted search
  instead of full steepest descent.
- **Built-in visualization**: ``result.plot()`` shows an interactive
  RMSE-vs-timesteps chart.


***********************
Performance
***********************

tsam v3 is significantly faster than v2.3.9, primarily due to replacing
pandas loops with vectorized numpy operations.

.. list-table:: Speedup vs v2.3.9 (selected configurations)
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Configuration
     - constant
     - testdata
     - wide
     - with_zero_col
   * - hierarchical (default)
     - 2x
     - 44x
     - 25x
     - 42x
   * - hierarchical (distribution)
     - 5x
     - 55x
     - 35x
     - 51x
   * - averaging
     - 5x
     - 77x
     - 66x
     - 74x
   * - contiguous
     - 5x
     - 54x
     - 50x
     - 53x
   * - distribution (global)
     - 2x
     - 16x
     - 7x
     - 13x
   * - kmeans
     - 1.4x
     - 4x
     - 6x
     - 6x
   * - kmaxoids
     - 1.3x
     - 1.4x
     - 1.4x
     - 1.4x

Key optimizations:

- **``predictOriginalData()``**: Vectorized indexing replaces per-period
  ``.unstack()`` loop (~290x function speedup).
- **``durationRepresentation()``**: numpy 3D operations replace nested
  pandas loops (~8x function speedup, contributing to the distribution
  config gains above).
- **``_rescaleClusterPeriods()``**: numpy 3D arrays replace pandas
  MultiIndex operations (~11x function speedup).

Iterative methods (kmeans, kmedoids, kmaxoids) show modest gains because
the solver itself dominates runtime.

Use ``benchmarks/bench.py`` to run your own comparisons::

    pytest benchmarks/bench.py --benchmark-save=my_run


*************************************
Result consistency and reproducibility
*************************************

Consistency with v2.3.9
=======================

Cross-platform reproducibility
==============================

v2.3.9 used numpy's default unstable sort (``introsort``) in
``durationRepresentation()``, which does not guarantee a specific order
for tied values. In practice, this caused different results on different
platforms (macOS vs Linux vs Windows) for distribution representations.

v3 fixes this by using ``kind="stable"`` (mergesort) for all sorting
operations and rounding floating-point means to 10 decimal places before
tie-breaking. This guarantees **identical results across macOS, Linux,
and Windows** for all configurations.

Consistency with v2.3.9
=======================

As a consequence of the stable sort fix, 4 distribution-related
configurations produce slightly different results compared to v2.3.9:

- ``hierarchical_distribution``
- ``hierarchical_distribution_minmax``
- ``distribution_global``
- ``distribution_minmax_global``

The stable sort breaks ties by position rather than arbitrarily, and
rounding absorbs ~1e-16 floating-point noise that previously created
artificial ordering among effectively-equal means. This changes the
assignment of representative values to time steps, but preserves all
statistical properties (same distribution, same min/max, same weighted
mean).

All other 23 configurations (hierarchical with medoid/mean/maxoid,
averaging, contiguous, kmeans, kmedoids, kmaxoids, minmaxmean,
segmentation, extremes) are bit-for-bit identical to v2.3.9.

Going forward
=============

Result stability is enforced by two test layers:

1. **Golden regression tests** (``test/test_golden_regression.py``):
   148 tests compare both APIs against stored CSV baselines. Any code
   change that alters output values will fail these tests.

2. **Old/new API equivalence tests** (``test/test_old_new_equivalence.py``):
   296 tests verify that the legacy ``TimeSeriesAggregation`` class and
   the new ``tsam.aggregate()`` function produce identical results.

If a future release intentionally changes results (e.g., improved
algorithm), the golden files will be regenerated and the change
documented in the changelog.


***********************
Suppressing warnings
***********************

During migration you can silence the deprecation warnings::

    import warnings
    from tsam import LegacyAPIWarning

    warnings.filterwarnings("ignore", category=LegacyAPIWarning)


***********************
Removed parameters
***********************

``prepareEnersysInput()``
    Removed. Access result properties directly instead.
