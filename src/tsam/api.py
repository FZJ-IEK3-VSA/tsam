"""New simplified API for tsam aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from tsam.config import (
    REPRESENTATION_MAPPING,
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    Representation,
    SegmentConfig,
)
from tsam.pipeline import run_pipeline
from tsam.result import AccuracyMetrics, AggregationResult

if TYPE_CHECKING:
    from tsam.pipeline.types import PipelineResult


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

    # Compute n_timesteps_per_period
    if temporal_resolution is not None:
        resolution = temporal_resolution
    else:
        # Infer resolution from data index
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
            resolution = (data.index[1] - data.index[0]).total_seconds() / 3600
        else:
            resolution = 1.0  # Default to hourly

    n_timesteps_per_period = int(period_duration / resolution)

    # Validate segments against data
    if segments is not None:
        if segments.n_segments > n_timesteps_per_period:
            raise ValueError(
                f"n_segments ({segments.n_segments}) cannot exceed "
                f"timesteps per period ({n_timesteps_per_period})"
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

    # Run pipeline
    result = run_pipeline(
        data=data,
        n_clusters=n_clusters,
        n_timesteps_per_period=n_timesteps_per_period,
        cluster=cluster,
        extremes=extremes if extremes and extremes.has_extremes() else None,
        segments=segments,
        rescale_cluster_periods=preserve_column_means,
        rescale_exclude_columns=rescale_exclude_columns,
        round_decimals=round_decimals,
        numerical_tolerance=numerical_tolerance,
        temporal_resolution=temporal_resolution,
    )

    return _build_aggregation_result(result, is_transferred=False)


def _build_aggregation_result(
    result: PipelineResult,
    is_transferred: bool,
) -> AggregationResult:
    """Convert PipelineResult to the user-facing AggregationResult."""
    # Rename index levels
    cluster_representatives = result.typical_periods.rename_axis(
        index={"PeriodNum": "cluster", "TimeStep": "timestep"}
    )

    # Build rescale deviations DataFrame
    if result.rescale_deviations:
        rescale_deviations = pd.DataFrame.from_dict(
            result.rescale_deviations, orient="index"
        )
        rescale_deviations.index.name = "column"
    else:
        rescale_deviations = pd.DataFrame(
            columns=["deviation_pct", "converged", "iterations"]
        )

    accuracy = AccuracyMetrics(
        rmse=result.accuracy_indicators["RMSE"],
        mae=result.accuracy_indicators["MAE"],
        rmse_duration=result.accuracy_indicators["RMSE_duration"],
        rescale_deviations=rescale_deviations,
    )

    # Get segment_durations from ClusteringResult
    segment_durations = result.clustering_result.segment_durations

    return AggregationResult(
        cluster_representatives=cluster_representatives,
        cluster_weights=result.cluster_weights,
        n_timesteps_per_period=result.n_timesteps_per_period,
        segment_durations=segment_durations,
        accuracy=accuracy,
        clustering_duration=result.clustering_duration,
        clustering=result.clustering_result,
        is_transferred=is_transferred,
        _original_data=result.original_data,
        _reconstructed_data=result.reconstructed_data,
        _time_index=result.time_index,
        _segmented_df=result.segmented_df,
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

    from tsam.pipeline.periods import unstack_to_periods as _unstack

    profiles = _unstack(data.copy(), timesteps_per_period)
    return cast("pd.DataFrame", profiles.profiles_dataframe)
