"""New simplified API for tsam aggregation."""

from __future__ import annotations

import warnings
from typing import cast

import pandas as pd

from tsam.config import (
    EXTREME_METHOD_MAPPING,
    METHOD_MAPPING,
    REPRESENTATION_MAPPING,
    ClusterConfig,
    ClusteringResult,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    Representation,
    SegmentConfig,
)
from tsam.exceptions import LegacyAPIWarning
from tsam.result import AccuracyMetrics, AggregationResult
from tsam.timeseriesaggregation import TimeSeriesAggregation, unstackToPeriods


def _parse_duration_hours(value: int | float | str, param_name: str) -> float:
    """Parse a duration value to hours.

    Accepts:
    - int/float: interpreted as hours (e.g., 24 → 24.0 hours)
    - str: pandas Timedelta string (e.g., '24h', '1d', '15min')

    Returns duration in hours as float.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            td = pd.Timedelta(value)
            return td.total_seconds() / 3600
        except ValueError as e:
            raise ValueError(
                f"{param_name}: invalid duration string '{value}': {e}"
            ) from e
    raise TypeError(
        f"{param_name} must be int, float, or string, got {type(value).__name__}"
    )


def aggregate(
    data: pd.DataFrame,
    n_clusters: int,
    *,
    period_duration: int | float | str = 24,
    temporal_resolution: float | str | None = None,
    cluster: ClusterConfig | None = None,
    segments: SegmentConfig | None = None,
    extremes: ExtremeConfig | None = None,
    preserve_column_means: bool = True,
    rescale_exclude_columns: list[str] | None = None,
    round_decimals: int | None = None,
    numerical_tolerance: float = 1e-13,
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

    n_clusters : int
        Number of clusters (typical periods) to create.
        Higher values = more accuracy but less data reduction.
        Typical range: 4-20 for energy system models.

    period_duration : int, float, or str, default 24
        Length of each period. Accepts:
        - int/float: hours (e.g., 24 for daily, 168 for weekly)
        - str: pandas Timedelta string (e.g., '24h', '1d', '1w')

    temporal_resolution : float or str, optional
        Time resolution of input data. Accepts:
        - float: hours (e.g., 1.0 for hourly, 0.25 for 15-minute)
        - str: pandas Timedelta string (e.g., '1h', '15min', '30min')
        If not provided, inferred from the datetime index.

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

    preserve_column_means : bool, default True
        Rescale typical periods so each column's weighted mean matches
        the original data's mean. Ensures total energy/load is preserved
        when weights represent occurrence counts.

    rescale_exclude_columns : list[str], optional
        Column names to exclude from rescaling when preserve_column_means=True.
        Useful for binary/indicator columns (0/1 values) that should not be
        rescaled. If None (default), all columns are rescaled.

    round_decimals : int, optional
        Round output values to this many decimal places.
        If not provided, no rounding is applied.

    numerical_tolerance : float, default 1e-13
        Tolerance for numerical precision issues.
        Controls when warnings are raised for aggregated values exceeding
        the original time series bounds. Increase this value to silence
        warnings caused by floating-point precision errors.

    Returns
    -------
    AggregationResult
        Object containing:
        - cluster_representatives: DataFrame with aggregated periods
        - cluster_assignments: Which cluster each original period belongs to
        - cluster_weights: Occurrence count per cluster
        - accuracy: RMSE, MAE metrics
        - Methods: to_dict()

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
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> typical = result.cluster_representatives

    With custom clustering:

    >>> from tsam import aggregate, ClusterConfig
    >>> result = aggregate(
    ...     df,
    ...     n_clusters=8,
    ...     cluster=ClusterConfig(method="kmeans", representation="mean"),
    ... )

    With segmentation (reduce to 12 timesteps per period):

    >>> from tsam import aggregate, SegmentConfig
    >>> result = aggregate(
    ...     df,
    ...     n_clusters=8,
    ...     segments=SegmentConfig(n_segments=12),
    ... )

    Preserving peak demand periods:

    >>> from tsam import aggregate, ExtremeConfig
    >>> result = aggregate(
    ...     df,
    ...     n_clusters=8,
    ...     extremes=ExtremeConfig(max_value=["demand"]),
    ... )

    Transferring assignments to new data:

    >>> result1 = aggregate(df_wind, n_clusters=8)
    >>> result2 = result1.clustering.apply(df_all)

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

    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError(f"n_clusters must be a positive integer, got {n_clusters}")

    # Parse duration parameters to hours
    period_duration = _parse_duration_hours(period_duration, "period_duration")
    if period_duration <= 0:
        raise ValueError(f"period_duration must be positive, got {period_duration}")

    temporal_resolution = (
        _parse_duration_hours(temporal_resolution, "temporal_resolution")
        if temporal_resolution is not None
        else None
    )
    if temporal_resolution is not None and temporal_resolution <= 0:
        raise ValueError(
            f"temporal_resolution must be positive, got {temporal_resolution}"
        )

    # Apply defaults
    if cluster is None:
        cluster = ClusterConfig()

    # Validate segments against data
    if segments is not None:
        # Calculate timesteps per period
        if temporal_resolution is not None:
            timesteps_per_period = int(period_duration / temporal_resolution)
        else:
            # Infer resolution from data index
            if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
                inferred_resolution = (
                    data.index[1] - data.index[0]
                ).total_seconds() / 3600
                timesteps_per_period = int(period_duration / inferred_resolution)
            else:
                # Fall back to assuming hourly resolution
                timesteps_per_period = int(period_duration)

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
        n_clusters=n_clusters,
        period_duration=period_duration,
        temporal_resolution=temporal_resolution,
        cluster=cluster,
        segments=segments,
        extremes=extremes,
        preserve_column_means=preserve_column_means,
        rescale_exclude_columns=rescale_exclude_columns,
        round_decimals=round_decimals,
        numerical_tolerance=numerical_tolerance,
    )

    # Run aggregation using old implementation (suppress deprecation warning for internal use)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyAPIWarning)
        agg = TimeSeriesAggregation(**old_params)
        cluster_representatives = agg.createTypicalPeriods()

    # Rename index levels for consistency with new API terminology
    cluster_representatives = cluster_representatives.rename_axis(
        index={"PeriodNum": "cluster", "TimeStep": "timestep"}
    )

    # Build accuracy metrics
    accuracy_df = agg.accuracyIndicators()

    # Build rescale deviations DataFrame
    rescale_deviations_dict = getattr(agg, "_rescaleDeviations", {})
    if rescale_deviations_dict:
        rescale_deviations = pd.DataFrame.from_dict(
            rescale_deviations_dict, orient="index"
        )
        rescale_deviations.index.name = "column"
    else:
        rescale_deviations = pd.DataFrame(
            columns=["deviation_pct", "converged", "iterations"]
        )

    accuracy = AccuracyMetrics(
        rmse=accuracy_df["RMSE"],
        mae=accuracy_df["MAE"],
        rmse_duration=accuracy_df["RMSE_duration"],
        rescale_deviations=rescale_deviations,
    )

    # Build ClusteringResult
    clustering_result = _build_clustering_result(
        agg=agg,
        n_segments=segments.n_segments if segments else None,
        cluster_config=cluster,
        segment_config=segments,
        extremes_config=extremes,
        preserve_column_means=preserve_column_means,
        rescale_exclude_columns=rescale_exclude_columns,
        temporal_resolution=temporal_resolution,
    )

    # Compute segment_durations as tuple of tuples
    segment_durations_tuple = None
    if segments and hasattr(agg, "segmentedNormalizedTypicalPeriods"):
        segmented_df = agg.segmentedNormalizedTypicalPeriods
        segment_durations_tuple = tuple(
            tuple(
                int(seg_dur)
                for _seg_step, seg_dur, _orig_start in segmented_df.loc[
                    period_idx
                ].index
            )
            for period_idx in segmented_df.index.get_level_values(0).unique()
        )

    # Build result object
    return AggregationResult(
        cluster_representatives=cluster_representatives,
        cluster_weights=dict(agg.clusterPeriodNoOccur),
        n_timesteps_per_period=agg.timeStepsPerPeriod,
        segment_durations=segment_durations_tuple,
        accuracy=accuracy,
        clustering_duration=getattr(agg, "clusteringDuration", 0.0),
        clustering=clustering_result,
        is_transferred=False,
        _aggregation=agg,
    )


def _build_clustering_result(
    agg: TimeSeriesAggregation,
    n_segments: int | None,
    cluster_config: ClusterConfig,
    segment_config: SegmentConfig | None,
    extremes_config: ExtremeConfig | None,
    preserve_column_means: bool,
    rescale_exclude_columns: list[str] | None,
    temporal_resolution: float | None,
) -> ClusteringResult:
    """Build ClusteringResult from a TimeSeriesAggregation object."""
    # Get cluster centers (convert to Python ints for JSON serialization)
    # Handle extreme periods based on method:
    # - new_cluster/append: append extreme period indices (creates additional clusters)
    # - replace: keep original cluster centers
    #   Note: replace creates a hybrid representation (some columns from medoid, some
    #   from extreme period) that cannot be perfectly reproduced during transfer
    cluster_centers: tuple[int, ...] | None = None
    if agg.clusterCenterIndices is not None:
        center_indices = [int(x) for x in agg.clusterCenterIndices]

        if (
            hasattr(agg, "extremePeriods")
            and agg.extremePeriods
            and extremes_config is not None
            and extremes_config.method in ("new_cluster", "append")
        ):
            # Add extreme period indices as new cluster centers
            for period_type in agg.extremePeriods:
                center_indices.append(int(agg.extremePeriods[period_type]["stepNo"]))

        cluster_centers = tuple(center_indices)

    # Compute segment data if segmentation was used
    segment_assignments: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None
    segment_centers: tuple[tuple[int, ...], ...] | None = None

    if n_segments is not None and hasattr(agg, "segmentedNormalizedTypicalPeriods"):
        segmented_df = agg.segmentedNormalizedTypicalPeriods
        segment_assignments_list = []
        segment_durations_list = []

        for period_idx in segmented_df.index.get_level_values(0).unique():
            period_data = segmented_df.loc[period_idx]
            # Index levels: Segment Step, Segment Duration, Original Start Step
            assignments = []
            durations = []
            for seg_step, seg_dur, _orig_start in period_data.index:
                assignments.extend([int(seg_step)] * int(seg_dur))
                durations.append(int(seg_dur))
            segment_assignments_list.append(tuple(assignments))
            segment_durations_list.append(tuple(durations))

        segment_assignments = tuple(segment_assignments_list)
        segment_durations = tuple(segment_durations_list)

        # Extract segment center indices (only available for medoid/maxoid representations)
        if (
            hasattr(agg, "segmentCenterIndices")
            and agg.segmentCenterIndices is not None
        ):
            # Check if any period has center indices (None for mean representation)
            if all(pc is not None for pc in agg.segmentCenterIndices):
                segment_centers = tuple(
                    tuple(int(x) for x in period_centers)
                    for period_centers in agg.segmentCenterIndices
                )

    # Extract representation from configs
    representation = cluster_config.get_representation()
    segment_representation = segment_config.representation if segment_config else None

    # Extract extreme cluster indices if extremes were used
    extreme_cluster_indices: tuple[int, ...] | None = None
    if hasattr(agg, "extremeClusterIdx") and agg.extremeClusterIdx:
        extreme_cluster_indices = tuple(int(x) for x in agg.extremeClusterIdx)

    return ClusteringResult(
        period_duration=agg.hoursPerPeriod,
        cluster_assignments=tuple(int(x) for x in agg.clusterOrder),
        cluster_centers=cluster_centers,
        segment_assignments=segment_assignments,
        segment_durations=segment_durations,
        segment_centers=segment_centers,
        preserve_column_means=preserve_column_means,
        rescale_exclude_columns=tuple(rescale_exclude_columns)
        if rescale_exclude_columns
        else None,
        representation=representation,
        segment_representation=segment_representation,
        temporal_resolution=temporal_resolution,
        n_timesteps_per_period=agg.timeStepsPerPeriod,
        extreme_cluster_indices=extreme_cluster_indices,
        cluster_config=cluster_config,
        segment_config=segment_config,
        extremes_config=extremes_config,
    )


def _apply_representation_params(
    params: dict, representation: Representation, columns: list[str]
) -> None:
    """Apply representation parameters to the old API params dict.

    Handles both string shortcuts and typed representation objects
    (Distribution, MinMaxMean).
    """
    if isinstance(representation, Distribution):
        if representation.preserve_minmax:
            params["representationMethod"] = "distributionAndMinMaxRepresentation"
        else:
            params["representationMethod"] = "distributionRepresentation"
        params["distributionPeriodWise"] = representation.scope == "cluster"
    elif isinstance(representation, MinMaxMean):
        params["representationMethod"] = "minmaxmeanRepresentation"
        # Build representationDict: columns not in max/min default to mean
        rep_dict: dict[str, str] = {}
        max_set = set(representation.max_columns)
        min_set = set(representation.min_columns)
        for col in columns:
            if col in max_set:
                rep_dict[col] = "max"
            elif col in min_set:
                rep_dict[col] = "min"
            else:
                rep_dict[col] = "mean"
        params["representationDict"] = rep_dict
    else:
        # String representation
        rep_mapped = REPRESENTATION_MAPPING.get(representation)
        if rep_mapped is None:
            raise ValueError(
                f"Unknown representation method: {representation!r}. "
                f"Valid options: {list(REPRESENTATION_MAPPING.keys())}"
            )
        params["representationMethod"] = rep_mapped


def _build_old_params(
    data: pd.DataFrame,
    n_clusters: int,
    period_duration: float,
    temporal_resolution: float | None,
    cluster: ClusterConfig,
    segments: SegmentConfig | None,
    extremes: ExtremeConfig | None,
    preserve_column_means: bool,
    rescale_exclude_columns: list[str] | None,
    round_decimals: int | None,
    numerical_tolerance: float,
    *,
    # Predefined parameters (used internally by ClusteringResult.apply())
    predef_cluster_assignments: tuple[int, ...] | None = None,
    predef_cluster_centers: tuple[int, ...] | None = None,
    predef_extreme_cluster_indices: tuple[int, ...] | None = None,
    predef_segment_assignments: tuple[tuple[int, ...], ...] | None = None,
    predef_segment_durations: tuple[tuple[int, ...], ...] | None = None,
    predef_segment_centers: tuple[tuple[int, ...], ...] | None = None,
) -> dict:
    """Build parameters for the old TimeSeriesAggregation API."""
    params: dict = {
        "timeSeries": data,
        "noTypicalPeriods": n_clusters,
        "hoursPerPeriod": period_duration,
        "rescaleClusterPeriods": preserve_column_means,
        "rescaleExcludeColumns": rescale_exclude_columns,
        "numericalTolerance": numerical_tolerance,
    }

    if temporal_resolution is not None:
        params["resolution"] = temporal_resolution

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
    _apply_representation_params(params, representation, data.columns.tolist())
    params["sortValues"] = cluster.use_duration_curves
    params["sameMean"] = cluster.normalize_column_means
    params["evalSumPeriods"] = cluster.include_period_sums
    params["solver"] = cluster.solver

    if cluster.weights is not None:
        params["weightDict"] = cluster.weights

    if predef_cluster_assignments is not None:
        params["predefClusterOrder"] = list(predef_cluster_assignments)

    if predef_cluster_centers is not None:
        params["predefClusterCenterIndices"] = list(predef_cluster_centers)

    if predef_extreme_cluster_indices is not None:
        params["predefExtremeClusterIdx"] = list(predef_extreme_cluster_indices)

    # Segmentation config
    if segments is not None:
        params["segmentation"] = True
        params["noSegments"] = segments.n_segments
        params["segmentRepresentationMethod"] = REPRESENTATION_MAPPING.get(
            segments.representation, "meanRepresentation"
        )

        # Predefined segment parameters (from ClusteringResult)
        if predef_segment_assignments is not None:
            params["predefSegmentOrder"] = [list(s) for s in predef_segment_assignments]
        if predef_segment_durations is not None:
            params["predefSegmentDurations"] = [
                list(s) for s in predef_segment_durations
            ]
        if predef_segment_centers is not None:
            params["predefSegmentCenters"] = [list(s) for s in predef_segment_centers]
    else:
        params["segmentation"] = False

    # Extreme config
    if extremes is not None and extremes.has_extremes():
        params["extremePeriodMethod"] = EXTREME_METHOD_MAPPING[extremes.method]
        params["addPeakMax"] = extremes.max_value
        params["addPeakMin"] = extremes.min_value
        params["addMeanMax"] = extremes.max_period
        params["addMeanMin"] = extremes.min_period
        params["extremePreserveNumClusters"] = extremes._effective_preserve_n_clusters
    else:
        params["extremePeriodMethod"] = "None"

    return params


def unstack_to_periods(
    data: pd.DataFrame,
    period_duration: int | float | str = 24,
) -> pd.DataFrame:
    """Reshape time series data into period structure for visualization.

    Transforms a flat time series into a DataFrame with periods as rows and
    timesteps as a MultiIndex level, suitable for creating heatmaps with plotly.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index.
    period_duration : int, float, or str, default 24
        Length of each period. Accepts:
        - int/float: hours (e.g., 24 for daily, 168 for weekly)
        - str: pandas Timedelta string (e.g., '24h', '1d', '1w')

    Returns
    -------
    pd.DataFrame
        Reshaped data with shape (n_periods, n_timesteps_per_period) for each column.
        Suitable for ``px.imshow(result["column"].values.T)`` to create heatmaps.

    Examples
    --------
    >>> import tsam
    >>> import plotly.express as px
    >>>
    >>> # Reshape data for heatmap visualization
    >>> unstacked = tsam.unstack_to_periods(df, period_duration=24)
    >>>
    >>> # Create heatmap with plotly
    >>> px.imshow(
    ...     unstacked["Load"].values.T,
    ...     labels={"x": "Day", "y": "Hour", "color": "Load"},
    ...     title="Load Heatmap"
    ... )
    """
    period_hours = _parse_duration_hours(period_duration, "period_duration")

    # Infer timestep resolution from data index
    timestep_hours = 1.0  # Default to hourly
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        timestep_hours = (data.index[1] - data.index[0]).total_seconds() / 3600

    # Calculate timesteps per period
    timesteps_per_period = round(period_hours / timestep_hours)
    if timesteps_per_period < 1:
        raise ValueError(
            f"period_duration ({period_hours}h) is smaller than "
            f"data timestep resolution ({timestep_hours}h)"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyAPIWarning)
        unstacked, _ = unstackToPeriods(data.copy(), timesteps_per_period)
    return cast("pd.DataFrame", unstacked)
