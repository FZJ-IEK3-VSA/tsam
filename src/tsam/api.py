"""New simplified API for tsam aggregation."""

from __future__ import annotations

from typing import cast

import pandas as pd

from tsam.config import (
    EXTREME_METHOD_MAPPING,
    METHOD_MAPPING,
    REPRESENTATION_MAPPING,
    ClusterConfig,
    ClusteringResult,
    ExtremeConfig,
    SegmentConfig,
)
from tsam.pipeline import run_pipeline
from tsam.result import AccuracyMetrics, AggregationResult


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

    # Map config names to old API names
    method = METHOD_MAPPING.get(cluster.method)
    if method is None:
        raise ValueError(
            f"Unknown cluster method: {cluster.method!r}. "
            f"Valid options: {list(METHOD_MAPPING.keys())}"
        )

    representation = cluster.get_representation()
    rep_mapped = REPRESENTATION_MAPPING.get(representation)
    if rep_mapped is None:
        raise ValueError(
            f"Unknown representation method: {representation!r}. "
            f"Valid options: {list(REPRESENTATION_MAPPING.keys())}"
        )

    # Map segment representation
    seg_rep_mapped = None
    if segments is not None:
        seg_rep_mapped = REPRESENTATION_MAPPING.get(
            segments.representation, "meanRepresentation"
        )

    # Map extreme method
    if extremes is not None and extremes.has_extremes():
        extreme_method = EXTREME_METHOD_MAPPING[extremes.method]
        extreme_preserve = extremes._effective_preserve_n_clusters
    else:
        extreme_method = "None"
        extreme_preserve = False

    # Run pipeline
    result = run_pipeline(
        data=data,
        n_clusters=n_clusters,
        n_timesteps_per_period=n_timesteps_per_period,
        cluster_method=method,
        representation_method=rep_mapped,
        representation_dict=None,  # Will be built from defaults inside pipeline
        distribution_period_wise=True,
        solver=cluster.solver,
        use_duration_curves=cluster.use_duration_curves,
        normalize_column_means=cluster.normalize_column_means,
        include_period_sums=cluster.include_period_sums,
        weights=cluster.weights,
        rescale_cluster_periods=preserve_column_means,
        rescale_exclude_columns=rescale_exclude_columns,
        extreme_period_method=extreme_method,
        extreme_preserve_n_clusters=extreme_preserve,
        add_peak_max=extremes.max_value if extremes else [],
        add_peak_min=extremes.min_value if extremes else [],
        add_mean_max=extremes.max_period if extremes else [],
        add_mean_min=extremes.min_period if extremes else [],
        segmentation=segments is not None,
        n_segments=segments.n_segments if segments else 10,
        segment_representation_method=seg_rep_mapped,
        round_decimals=round_decimals,
        numerical_tolerance=numerical_tolerance,
    )

    # Build cluster_representatives with renamed index
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

    # Build ClusteringResult
    clustering_result = _build_clustering_result(
        result=result,
        n_segments=segments.n_segments if segments else None,
        cluster_config=cluster,
        segment_config=segments,
        extremes_config=extremes,
        preserve_column_means=preserve_column_means,
        rescale_exclude_columns=rescale_exclude_columns,
        temporal_resolution=temporal_resolution,
        extreme_method=extreme_method,
    )

    # Compute segment_durations as tuple of tuples
    segment_durations_tuple = None
    if segments and result.segmented_df is not None:
        segmented_df = result.segmented_df
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
        cluster_weights=result.cluster_weights,
        n_timesteps_per_period=result.n_timesteps_per_period,
        segment_durations=segment_durations_tuple,
        accuracy=accuracy,
        clustering_duration=result.clustering_duration,
        clustering=clustering_result,
        is_transferred=False,
        _original_data=result.original_data,
        _reconstructed_data=result.reconstructed_data,
        _time_index=result.time_index,
        _segmented_df=result.segmented_df,
    )


def _build_clustering_result(
    result,
    n_segments: int | None,
    cluster_config: ClusterConfig,
    segment_config: SegmentConfig | None,
    extremes_config: ExtremeConfig | None,
    preserve_column_means: bool,
    rescale_exclude_columns: list[str] | None,
    temporal_resolution: float | None,
    extreme_method: str = "None",
) -> ClusteringResult:
    """Build ClusteringResult from a PipelineResult."""
    # Get cluster centers
    cluster_centers: tuple[int, ...] | None = None
    if result.cluster_center_indices is not None:
        center_indices = [int(x) for x in result.cluster_center_indices]

        if (
            result.extreme_periods_info
            and extremes_config is not None
            and extremes_config.method in ("new_cluster", "append")
        ):
            for period_type in result.extreme_periods_info:
                center_indices.append(
                    int(result.extreme_periods_info[period_type]["stepNo"])
                )

        cluster_centers = tuple(center_indices)

    # Compute segment data if segmentation was used
    segment_assignments: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None
    segment_centers: tuple[tuple[int, ...], ...] | None = None

    if n_segments is not None and result.segmented_df is not None:
        segmented_df = result.segmented_df
        segment_assignments_list = []
        segment_durations_list = []

        for period_idx in segmented_df.index.get_level_values(0).unique():
            period_data = segmented_df.loc[period_idx]
            assignments = []
            durations = []
            for seg_step, seg_dur, _orig_start in period_data.index:
                assignments.extend([int(seg_step)] * int(seg_dur))
                durations.append(int(seg_dur))
            segment_assignments_list.append(tuple(assignments))
            segment_durations_list.append(tuple(durations))

        segment_assignments = tuple(segment_assignments_list)
        segment_durations = tuple(segment_durations_list)

        if result.segment_center_indices is not None:
            if all(pc is not None for pc in result.segment_center_indices):
                segment_centers = tuple(
                    tuple(int(x) for x in period_centers)
                    for period_centers in result.segment_center_indices
                )

    # Extract representation from configs
    representation = cluster_config.get_representation()
    segment_representation = segment_config.representation if segment_config else None

    # Extract extreme cluster indices
    extreme_cluster_indices: tuple[int, ...] | None = None
    if result.extreme_cluster_indices:
        extreme_cluster_indices = tuple(int(x) for x in result.extreme_cluster_indices)

    return ClusteringResult(
        period_duration=result.n_timesteps_per_period
        * (
            temporal_resolution
            if temporal_resolution is not None
            else _infer_resolution(result.original_data)
        ),
        cluster_assignments=tuple(int(x) for x in result.cluster_assignments),
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
        n_timesteps_per_period=result.n_timesteps_per_period,
        extreme_cluster_indices=extreme_cluster_indices,
        cluster_config=cluster_config,
        segment_config=segment_config,
        extremes_config=extremes_config,
    )


def _infer_resolution(data: pd.DataFrame) -> float:
    """Infer temporal resolution from data index."""
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        return (data.index[1] - data.index[0]).total_seconds() / 3600
    return 1.0


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
