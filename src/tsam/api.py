"""New simplified API for tsam aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsam.config import (
    EXTREME_METHOD_MAPPING,
    METHOD_MAPPING,
    REPRESENTATION_MAPPING,
    ClusterConfig,
    ExtremeConfig,
    PredefinedConfig,
    SegmentConfig,
)
from tsam.result import AccuracyMetrics, AggregationResult
from tsam.timeseriesaggregation import TimeSeriesAggregation


def aggregate(
    data: pd.DataFrame,
    n_periods: int,
    *,
    period_hours: int = 24,
    resolution: float | None = None,
    cluster: ClusterConfig | None = None,
    segments: SegmentConfig | None = None,
    extremes: ExtremeConfig | None = None,
    predefined: PredefinedConfig | dict | None = None,
    rescale: bool = True,
    round_decimals: int | None = None,
) -> AggregationResult:
    """Aggregate time series data into typical periods.

    This function reduces a time series dataset to a smaller set of
    representative "typical periods" using clustering algorithms.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data with a datetime index.
        Each column represents a different variable (e.g., solar, wind, demand).
        The index should be a DatetimeIndex with regular intervals.

    n_periods : int
        Number of typical periods (clusters) to create.
        Higher values = more accuracy but less data reduction.
        Typical range: 4-20 for energy system models.

    period_hours : int, default 24
        Length of each period in hours.
        Common values:
        - 24: Daily periods (most common)
        - 168: Weekly periods
        - 1: Hourly periods (for sub-hourly data)

    resolution : float, optional
        Time resolution of input data in hours.
        If not provided, inferred from the datetime index.
        Examples: 1.0 (hourly), 0.25 (15-minute), 0.5 (30-minute)

    cluster : ClusterConfig, optional
        Clustering configuration. If not provided, uses defaults:
        - method: "hierarchical"
        - representation: "medoid"

    segments : SegmentConfig, optional
        Segmentation configuration for reducing temporal resolution
        within periods. If not provided, no segmentation is applied.

    extremes : ExtremeConfig, optional
        Configuration for preserving extreme periods.
        If not provided, no extreme period handling is applied.

    predefined : PredefinedConfig or dict, optional
        Predefined assignments from a previous aggregation result.
        Use `result.predefined` to get this, or load from JSON with
        `PredefinedConfig.from_dict()`. Overrides cluster/segment assignments.

    rescale : bool, default True
        Rescale typical periods to match the original data's mean.
        Preserves total energy/load values across the aggregation.

    round_decimals : int, optional
        Round output values to this many decimal places.
        If not provided, no rounding is applied.

    Returns
    -------
    AggregationResult
        Object containing:
        - typical_periods: DataFrame with aggregated periods
        - cluster_assignments: Which cluster each original period belongs to
        - cluster_weights: Occurrence count per cluster
        - accuracy: RMSE, MAE metrics
        - Methods: reconstruct(), to_dict()

    Raises
    ------
    ValueError
        If input data is invalid or parameters are inconsistent.
    TypeError
        If parameter types are incorrect.

    Examples
    --------
    Basic usage with defaults:

    >>> import tsam
    >>> result = tsam.aggregate(df, n_periods=8)
    >>> typical = result.typical_periods

    With custom clustering:

    >>> from tsam import aggregate, ClusterConfig
    >>> result = aggregate(
    ...     df,
    ...     n_periods=8,
    ...     cluster=ClusterConfig(method="kmeans", representation="mean"),
    ... )

    With segmentation (reduce to 12 timesteps per period):

    >>> from tsam import aggregate, SegmentConfig
    >>> result = aggregate(
    ...     df,
    ...     n_periods=8,
    ...     segments=SegmentConfig(n_segments=12),
    ... )

    Preserving peak demand periods:

    >>> from tsam import aggregate, ExtremeConfig
    >>> result = aggregate(
    ...     df,
    ...     n_periods=8,
    ...     extremes=ExtremeConfig(max_value=["demand"]),
    ... )

    Transferring assignments to new data:

    >>> result1 = aggregate(df_wind, n_periods=8)
    >>> result2 = aggregate(df_all, n_periods=8, predefined=result1.predefined)

    See Also
    --------
    ClusterConfig : Clustering algorithm configuration
    SegmentConfig : Temporal segmentation configuration
    ExtremeConfig : Extreme period preservation configuration
    AggregationResult : Result object with all outputs
    """
    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")

    if not isinstance(n_periods, int) or n_periods < 1:
        raise ValueError(f"n_periods must be a positive integer, got {n_periods}")

    if not isinstance(period_hours, int) or period_hours < 1:
        raise ValueError(f"period_hours must be a positive integer, got {period_hours}")

    # Apply defaults
    if cluster is None:
        cluster = ClusterConfig()

    # Apply predefined overrides
    if predefined is not None:
        # Convert dict to PredefinedConfig if needed
        if isinstance(predefined, dict):
            predefined = PredefinedConfig.from_dict(predefined)

        # Override cluster config with predefined values
        cluster_kwargs = {
            "method": cluster.method,
            "representation": cluster.representation,
            "weights": cluster.weights,
            "normalize_means": cluster.normalize_means,
            "use_duration_curves": cluster.use_duration_curves,
            "include_period_sums": cluster.include_period_sums,
            "solver": cluster.solver,
            "predef_cluster_order": predefined.cluster_order,
        }
        if predefined.cluster_centers is not None:
            cluster_kwargs["predef_cluster_centers"] = predefined.cluster_centers
        cluster = ClusterConfig(**cluster_kwargs)  # type: ignore[arg-type]

        # Override segment config with predefined values if present
        if (
            predefined.segment_order is not None
            and predefined.segment_durations is not None
            and len(predefined.segment_durations) > 0
        ):
            if segments is None:
                # Infer n_segments from the predefined data
                n_segments = len(predefined.segment_durations[0])
                segments = SegmentConfig(
                    n_segments=n_segments,
                    predef_segment_order=predefined.segment_order,
                    predef_segment_durations=predefined.segment_durations,
                )
            else:
                # Merge with existing segment config
                segments = SegmentConfig(
                    n_segments=segments.n_segments,
                    representation=segments.representation,
                    predef_segment_order=predefined.segment_order,
                    predef_segment_durations=predefined.segment_durations,
                    predef_segment_centers=segments.predef_segment_centers,
                )

    # Validate segments against data
    if segments is not None:
        # Calculate timesteps per period
        if resolution is not None:
            timesteps_per_period = int(period_hours / resolution)
        else:
            # Infer resolution from data index
            if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
                inferred_resolution = (
                    data.index[1] - data.index[0]
                ).total_seconds() / 3600
                timesteps_per_period = int(period_hours / inferred_resolution)
            else:
                # Fall back to assuming hourly resolution
                timesteps_per_period = period_hours

        if segments.n_segments > timesteps_per_period:
            raise ValueError(
                f"n_segments ({segments.n_segments}) cannot exceed "
                f"timesteps per period ({timesteps_per_period})"
            )

    # Validate extreme columns exist in data
    if extremes is not None:
        all_extreme_cols = (
            extremes.max_value
            + extremes.min_value
            + extremes.max_period
            + extremes.min_period
        )
        missing = set(all_extreme_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Extreme period columns not found in data: {missing}")

    # Validate weight columns exist
    if cluster.weights is not None:
        missing = set(cluster.weights.keys()) - set(data.columns)
        if missing:
            raise ValueError(f"Weight columns not found in data: {missing}")

    # Build old API parameters
    old_params = _build_old_params(
        data=data,
        n_periods=n_periods,
        period_hours=period_hours,
        resolution=resolution,
        cluster=cluster,
        segments=segments,
        extremes=extremes,
        rescale=rescale,
        round_decimals=round_decimals,
    )

    # Run aggregation using old implementation
    agg = TimeSeriesAggregation(**old_params)
    typical_periods = agg.createTypicalPeriods()

    # Build accuracy metrics
    accuracy_df = agg.accuracyIndicators()
    accuracy = AccuracyMetrics(
        rmse=accuracy_df["RMSE"],
        mae=accuracy_df["MAE"],
        rmse_duration=accuracy_df["RMSE_duration"],
    )

    # Build result object
    return AggregationResult(
        typical_periods=typical_periods,
        cluster_assignments=np.array(agg.clusterOrder),
        cluster_weights=dict(agg.clusterPeriodNoOccur),
        n_periods=len(agg.clusterPeriodIdx),
        n_timesteps_per_period=agg.timeStepsPerPeriod,
        n_segments=segments.n_segments if segments else None,
        segment_durations=agg.segmentDurationDict if segments else None,
        cluster_center_indices=np.array(agg.clusterCenterIndices)
        if agg.clusterCenterIndices is not None
        else None,
        accuracy=accuracy,
        clustering_duration=getattr(agg, "clusteringDuration", 0.0),
        _aggregation=agg,
    )


def _build_old_params(
    data: pd.DataFrame,
    n_periods: int,
    period_hours: int,
    resolution: float | None,
    cluster: ClusterConfig,
    segments: SegmentConfig | None,
    extremes: ExtremeConfig | None,
    rescale: bool,
    round_decimals: int | None,
) -> dict:
    """Build parameters for the old TimeSeriesAggregation API."""
    params: dict = {
        "timeSeries": data,
        "noTypicalPeriods": n_periods,
        "hoursPerPeriod": period_hours,
        "rescaleClusterPeriods": rescale,
    }

    if resolution is not None:
        params["resolution"] = resolution

    if round_decimals is not None:
        params["roundOutput"] = round_decimals

    # Cluster config
    method = METHOD_MAPPING.get(cluster.method)
    if method is None:
        raise ValueError(
            f"Unknown cluster method: {cluster.method!r}. "
            f"Valid options: {list(METHOD_MAPPING.keys())}"
        )
    params["clusterMethod"] = method

    representation = cluster.get_representation()
    rep_mapped = REPRESENTATION_MAPPING.get(representation)
    if rep_mapped is None:
        raise ValueError(
            f"Unknown representation method: {representation!r}. "
            f"Valid options: {list(REPRESENTATION_MAPPING.keys())}"
        )
    params["representationMethod"] = rep_mapped
    params["sortValues"] = cluster.use_duration_curves
    params["sameMean"] = cluster.normalize_means
    params["evalSumPeriods"] = cluster.include_period_sums
    params["solver"] = cluster.solver

    if cluster.weights is not None:
        params["weightDict"] = cluster.weights

    if cluster.predef_cluster_order is not None:
        params["predefClusterOrder"] = list(cluster.predef_cluster_order)

    if cluster.predef_cluster_centers is not None:
        params["predefClusterCenterIndices"] = list(cluster.predef_cluster_centers)

    # Segmentation config
    if segments is not None:
        params["segmentation"] = True
        params["noSegments"] = segments.n_segments
        params["segmentRepresentationMethod"] = REPRESENTATION_MAPPING.get(
            segments.representation, "meanRepresentation"
        )

        # Predefined segment parameters
        if segments.predef_segment_order is not None:
            params["predefSegmentOrder"] = [
                list(s) for s in segments.predef_segment_order
            ]
        if segments.predef_segment_durations is not None:
            params["predefSegmentDurations"] = [
                list(s) for s in segments.predef_segment_durations
            ]
        if segments.predef_segment_centers is not None:
            params["predefSegmentCenters"] = [
                list(s) for s in segments.predef_segment_centers
            ]
    else:
        params["segmentation"] = False

    # Extreme config
    if extremes is not None and extremes.has_extremes():
        params["extremePeriodMethod"] = EXTREME_METHOD_MAPPING[extremes.method]
        params["addPeakMax"] = extremes.max_value
        params["addPeakMin"] = extremes.min_value
        params["addMeanMax"] = extremes.max_period
        params["addMeanMin"] = extremes.min_period
    else:
        params["extremePeriodMethod"] = "None"

    return params
