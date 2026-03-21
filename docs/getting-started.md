# Getting started

## Basic Workflow

Run the aggregation and access the results:

=== "v3"

    ```python
    import pandas as pd
    import tsam
    from tsam import ClusterConfig, SegmentConfig

    df = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)

    # Configure and run aggregation
    result = tsam.aggregate(
        df,
        n_clusters=8,
        period_duration='1D',
        cluster=ClusterConfig(
            method='hierarchical',
            representation='distribution_minmax',
        ),
        segments=SegmentConfig(n_segments=8),
    )

    # Access results
    cluster_representatives = result.cluster_representatives
    print(f"RMSE: {result.accuracy.rmse.mean():.4f}")
    reconstructed = result.reconstructed
    cluster_representatives.to_csv('cluster_representatives.csv')
    ```

=== "Legacy"

    ```python
    import pandas as pd
    import tsam.timeseriesaggregation as tsam_legacy

    df = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)

    # Configure and run aggregation
    aggregation = tsam_legacy.TimeSeriesAggregation(
        df,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod='hierarchical',
        representationMethod="distributionAndMinMaxRepresentation",
        segmentation=True, noSegments=8,
    )

    # Access results
    cluster_representatives = aggregation.createTypicalPeriods()
    print(f"RMSE: {aggregation.accuracyIndicators()['RMSE'].mean():.4f}")
    reconstructed = aggregation.predictOriginalData()
    cluster_representatives.to_csv('cluster_representatives.csv')
    ```

## Hypertuned Aggregation

If you don't know which number of periods or segments to choose, specify a target data reduction and let the tuner search for the best configuration:

=== "v3"

    ```python
    import pandas as pd
    from tsam.tuning import find_optimal_combination

    df = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)

    result = find_optimal_combination(
        df,
        data_reduction=0.02,
        period_duration='1D',
        n_jobs=-1,
        show_progress=True,
    )

    print(f"Optimal: {result.n_clusters} periods x {result.n_segments} segments")
    print(f"RMSE: {result.rmse:.4f}")
    cluster_representatives = result.best_result.cluster_representatives
    ```

=== "Legacy"

    ```python
    import pandas as pd
    import tsam.timeseriesaggregation as tsam_legacy
    from tsam.hyperparametertuning import HyperTunedAggregations

    df = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)

    aggregation = tsam_legacy.TimeSeriesAggregation(
        df,
        hoursPerPeriod=24,
        segmentation=True,
    )
    tuner = HyperTunedAggregations(aggregation)
    noSegments, noTypicalPeriods, RMSE = tuner.identifyOptimalSegmentPeriodCombination(
        dataReduction=0.02,
    )

    print(f"Optimal: {noTypicalPeriods} periods x {noSegments} segments")
    print(f"RMSE: {RMSE:.4f}")
    ```

!!! tip
    Tuning can be time consuming. Run it once, save the resulting period and segment counts, and use them as fixed values in production.

For exploring the full Pareto front of period/segment combinations:

=== "v3"

    !!! tip
        Use `timesteps` to only evaluate specific timestep counts instead of the full search space for huge speedups.

    ```python
    import pandas as pd
    from tsam.tuning import find_pareto_front

    df = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)

    pareto = find_pareto_front(
        df,
        timesteps=range(10, 500, 50),
        n_jobs=-1,
    )
    pareto.summary
    pareto.plot()
    ```

=== "Legacy"

    ```python
    import pandas as pd
    import tsam.timeseriesaggregation as tsam_legacy
    from tsam.hyperparametertuning import HyperTunedAggregations

    df = pd.read_csv('testdata.csv', index_col=0, parse_dates=True)

    aggregation = tsam_legacy.TimeSeriesAggregation(
        df,
        hoursPerPeriod=24,
        segmentation=True,
    )
    tuner = HyperTunedAggregations(aggregation)
    tuner.identifyParetoOptimalAggregation(untilTotalTimeSteps=500)
    # Results in tuner._periodHistory, tuner._segmentHistory, tuner._RMSEHistory
    ```

See the [tuning notebook](notebooks/tuning.ipynb) for a detailed walkthrough, and the [scientific paper](https://www.sciencedirect.com/science/article/abs/pii/S0306261922004342) for the methodology behind it.
