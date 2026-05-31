"""Result classes for tsam aggregation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from tsam.commons import (
    infer_resolution,
    time_index_from_dict,
    time_index_to_dict,
)
from tsam.config import (
    ClusterConfig,
    ExtremeConfig,
    SegmentConfig,
    representation_from_dict,
    representation_to_dict,
)

if TYPE_CHECKING:
    from tsam.config import Representation
    from tsam.plot import ResultPlotAccessor


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing aggregated to original time series.

    Attributes
    ----------
    rmse : pd.Series
        Root Mean Square Error per column, comparing the original and
        reconstructed time series point-by-point over time.
    mae : pd.Series
        Mean Absolute Error per column, comparing the original and
        reconstructed time series point-by-point over time.
    rmse_duration : pd.Series
        RMSE on duration curves per column. Duration curves are created
        by sorting values in descending order, so this metric captures
        how well the aggregation preserves the overall value distribution
        regardless of temporal ordering.
    rescale_deviations : pd.DataFrame
        Rescaling deviation information per column. Contains columns:
        - deviation_pct: Final deviation percentage after rescaling
        - converged: Whether rescaling converged within max iterations
        - iterations: Number of iterations used
        Only populated if rescaling was enabled, otherwise empty DataFrame.
    weighted_rmse : float
        Weighted root-mean-square of per-column RMSE values:
        ``sqrt(sum(rmse_i² * w_i) / sum(w_i))``.
        Equals the RMSE over all pooled (weighted) residuals.
        With uniform weights this matches the old ``totalAccuracyIndicators()["RMSE"]``.
    weighted_mae : float
        Weighted arithmetic mean of per-column MAE values:
        ``sum(mae_i * w_i) / sum(w_i)``.
    weighted_rmse_duration : float
        Weighted root-mean-square of per-column duration-curve RMSE values:
        ``sqrt(sum(rmse_dur_i² * w_i) / sum(w_i))``.
    """

    rmse: pd.Series
    mae: pd.Series
    rmse_duration: pd.Series
    rescale_deviations: pd.DataFrame
    weighted_rmse: float
    weighted_mae: float
    weighted_rmse_duration: float

    @property
    def summary(self) -> pd.DataFrame:
        """Summary DataFrame with all metrics per column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: rmse, mae, rmse_duration, and deviation_pct
            (if rescaling was enabled). Index is the original column names.
        """
        df = pd.DataFrame(
            {
                "rmse": self.rmse,
                "mae": self.mae,
                "rmse_duration": self.rmse_duration,
            }
        )
        if not self.rescale_deviations.empty:
            df["deviation_pct"] = self.rescale_deviations["deviation_pct"]
        return df

    def __repr__(self) -> str:
        rescale_info = ""
        if not self.rescale_deviations.empty:
            n_failed = (~self.rescale_deviations["converged"]).sum()
            if n_failed > 0:
                max_dev = self.rescale_deviations["deviation_pct"].max()
                rescale_info = f",\n  rescale_failures={n_failed} (max {max_dev:.2f}%)"
        return (
            f"AccuracyMetrics(\n"
            f"  rmse={self.weighted_rmse:.4f} (weighted),\n"
            f"  mae={self.weighted_mae:.4f} (weighted),\n"
            f"  rmse_duration={self.weighted_rmse_duration:.4f} (weighted){rescale_info}\n"
            f")"
        )


@dataclass
class AggregationResult:
    """Result of time series aggregation.

    This class holds all outputs from the aggregation process and provides
    convenient methods for accessing and exporting the results.

    Attributes
    ----------
    cluster_representatives : pd.DataFrame
        The aggregated typical periods with MultiIndex (cluster, timestep).
        Each row represents one timestep in one cluster representative.

    cluster_assignments : np.ndarray
        Which cluster each original period belongs to.
        Length equals the number of original periods.
        Values are cluster indices (0 to n_clusters-1).

    cluster_counts : dict[int, float]
        How many original periods each cluster represents.
        Keys are cluster indices, values are occurrence counts.
        Values can be fractional due to partial-period adjustment.

    n_clusters : int
        Number of clusters (typical periods).

    n_timesteps_per_period : int
        Number of timesteps in each period.

    n_segments : int | None
        Number of segments per period if segmentation was used, else None.

    segment_durations : tuple[tuple[int, ...], ...] | None
        Duration (in timesteps) for each segment in each typical period.
        Outer tuple has one entry per typical period, inner tuple has
        duration for each segment. Use for transferring to another aggregation.

    accuracy : AccuracyMetrics
        Accuracy metrics comparing reconstructed to original data.

    clustering_duration : float
        Time taken for clustering in seconds.

    is_transferred : bool
        Whether this result was created by applying a transferred clustering
        (via ``ClusteringResult.apply()``) rather than by clustering this data directly.

    Examples
    --------
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> result.cluster_representatives
                        solar  wind  demand
    cluster timestep
    0       0           0.12   0.45   0.78
            1           0.15   0.42   0.82
    ...

    >>> result.cluster_counts
    {0: 45, 1: 52, 2: 38, ...}

    >>> result.accuracy.rmse
    solar     0.023
    wind      0.041
    demand    0.015
    dtype: float64
    """

    cluster_representatives: pd.DataFrame
    cluster_counts: dict[int, float]
    n_timesteps_per_period: int
    segment_durations: tuple[tuple[int, ...], ...] | None
    clustering_duration: float
    clustering: ClusteringResult
    is_transferred: bool
    _original_data: pd.DataFrame = field(repr=False, compare=False)
    _reconstructed_data: pd.DataFrame = field(repr=False, compare=False)
    _accuracy_metrics: AccuracyMetrics | None = field(
        default=None, repr=False, compare=False
    )
    _norm_values: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    _normalized_predicted: pd.DataFrame | None = field(
        default=None, repr=False, compare=False
    )
    _rescale_deviations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["deviation_pct", "converged", "iterations"]
        ),
        repr=False,
        compare=False,
    )
    _segmented_df: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    _weights: dict[str, float] | None = field(default=None, repr=False, compare=False)

    @property
    def _time_index(self) -> pd.Index:
        padded_length = self.clustering.n_original_periods * self.n_timesteps_per_period
        time_index = self.clustering.time_index
        if time_index is None:
            return pd.RangeIndex(stop=padded_length)
        if len(time_index) == padded_length:
            return time_index
        # Input length wasn't a multiple of n_timesteps_per_period; extend to the
        # padded length the pipeline produced internally.
        freq = pd.infer_freq(time_index)
        if freq is not None:
            return pd.date_range(time_index[0], periods=padded_length, freq=freq)
        return pd.RangeIndex(stop=padded_length)

    @cached_property
    def accuracy(self) -> AccuracyMetrics:
        """Accuracy metrics comparing reconstructed to original data.

        Computed lazily on first access.
        """
        if self._accuracy_metrics is not None:
            return self._accuracy_metrics
        from tsam.api import _weighted_mean, _weighted_rms
        from tsam.pipeline.accuracy import compute_accuracy

        assert self._norm_values is not None and self._normalized_predicted is not None
        accuracy_df = compute_accuracy(self._norm_values, self._normalized_predicted)
        return AccuracyMetrics(
            rmse=accuracy_df["RMSE"],
            mae=accuracy_df["MAE"],
            rmse_duration=accuracy_df["RMSE_duration"],
            rescale_deviations=self._rescale_deviations,
            weighted_rmse=_weighted_rms(accuracy_df["RMSE"], self._weights),
            weighted_mae=_weighted_mean(accuracy_df["MAE"], self._weights),
            weighted_rmse_duration=_weighted_rms(
                accuracy_df["RMSE_duration"], self._weights
            ),
        )

    @cached_property
    def n_clusters(self) -> int:
        """Number of clusters (typical periods).

        Derived from the cluster_representatives DataFrame index,
        which is the authoritative source. Note: cluster_counts may
        have more entries than actual cluster IDs due to tsam quirks.
        """
        return self.cluster_representatives.index.get_level_values(0).nunique()

    @cached_property
    def n_segments(self) -> int | None:
        """Number of segments per period if segmentation was used, else None."""
        return self.clustering.n_segments

    @cached_property
    def cluster_assignments(self) -> np.ndarray:
        """Which cluster each original period belongs to.

        Length equals the number of original periods.
        Values are cluster indices (0 to n_clusters-1).
        """
        return np.array(self.clustering.cluster_assignments)

    def __repr__(self) -> str:
        seg_info = f", n_segments={self.n_segments}" if self.n_segments else ""
        transferred_info = ", is_transferred=True" if self.is_transferred else ""
        return (
            f"AggregationResult(\n"
            f"  n_clusters={self.n_clusters},\n"
            f"  n_timesteps_per_period={self.n_timesteps_per_period}{seg_info}{transferred_info},\n"
            f"  accuracy={self.accuracy}\n"
            f")"
        )

    @cached_property
    def original(self) -> pd.DataFrame:
        """Original time series data.

        Returns
        -------
        pd.DataFrame
            The original input time series with datetime index.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> result.original.shape == df.shape
        True
        """
        return self._original_data

    @cached_property
    def reconstructed(self) -> pd.DataFrame:
        """Reconstructed time series from typical periods.

        Each original period is replaced by its assigned cluster representative.

        Returns
        -------
        pd.DataFrame
            Reconstructed time series with same shape as original.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> result.reconstructed.shape == df.shape
        True
        """
        return self._reconstructed_data

    def disaggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expand typical-period data back to the original time series length.

        Each original period is replaced by its assigned cluster representative
        from ``data``. The result uses the original datetime index.

        Parameters
        ----------
        data : pd.DataFrame
            Typical-period data matching ``cluster_representatives``:

            - ``(cluster, timestep)`` MultiIndex for non-segmented, or
            - ``(cluster, segment, duration)`` MultiIndex for segmented.

        Returns
        -------
        pd.DataFrame
            Disaggregated data with the original datetime index.
            For segmented input, non-segment-start timesteps are NaN.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> optimized = run_optimization(result.cluster_representatives)
        >>> full_year = result.disaggregate(optimized)
        """
        expanded = self.clustering.disaggregate(data)
        # Trim to original length (last period may be padded) and restore datetime index
        expanded = expanded.iloc[: len(self.original)]
        expanded.index = self.original.index
        return cast("pd.DataFrame", expanded)

    @cached_property
    def residuals(self) -> pd.DataFrame:
        """Residuals (original - reconstructed).

        Positive values indicate the original exceeded the reconstruction.

        Returns
        -------
        pd.DataFrame
            Residual time series with same shape as original.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> result.residuals.mean()  # Should be close to zero
        """
        return cast("pd.DataFrame", self.original - self.reconstructed)

    def to_dict(self) -> dict:
        """Export results as a dictionary for serialization.

        Returns
        -------
        dict
            Dictionary containing all result data in serializable format.
        """
        return {
            "cluster_representatives": self.cluster_representatives.to_dict(),
            "cluster_assignments": self.cluster_assignments.tolist(),
            "cluster_counts": self.cluster_counts,
            "n_clusters": self.n_clusters,
            "n_timesteps_per_period": self.n_timesteps_per_period,
            "n_segments": self.n_segments,
            "segment_durations": self.segment_durations,
            "clustering": self.clustering.to_dict(),
            "accuracy": {
                "rmse": self.accuracy.rmse.to_dict(),
                "mae": self.accuracy.mae.to_dict(),
                "rmse_duration": self.accuracy.rmse_duration.to_dict(),
                "rescale_deviations": self.accuracy.rescale_deviations.to_dict(),
                "weighted_rmse": self.accuracy.weighted_rmse,
                "weighted_mae": self.accuracy.weighted_mae,
                "weighted_rmse_duration": self.accuracy.weighted_rmse_duration,
            },
            "clustering_duration": self.clustering_duration,
        }

    @property
    def timestep_index(self) -> list[int]:
        """Get the timestep or segment indices.

        Returns
        -------
        list[int]
            List of indices [0, 1, ..., n-1] where n is n_segments
            if segmentation was used, otherwise n_timesteps_per_period.
        """
        n = self.n_segments if self.n_segments else self.n_timesteps_per_period
        return list(range(n))

    @property
    def period_index(self) -> list[int]:
        """Get the period (cluster) indices.

        Returns the actual cluster IDs from the cluster_representatives
        DataFrame, which is the authoritative source.

        Returns
        -------
        list[int]
            Sorted list of cluster indices present in cluster_representatives.
        """
        return sorted(self.cluster_representatives.index.get_level_values(0).unique())

    @cached_property
    def assignments(self) -> pd.DataFrame:
        """Get timestep-level assignment information.

        Returns a DataFrame with one row per original timestep containing
        assignment information for transferring results to another aggregation.

        Columns
        -------
        period_idx : int
            Index of the original period (0-indexed, 0 to n_original_periods-1).
        timestep_idx : int
            Timestep index within the period (0 to n_timesteps_per_period-1).
        cluster_idx : int
            Which cluster this period is assigned to (0 to n_clusters-1).
        segment_idx : int (only if segmentation was used)
            Which segment this timestep belongs to within its period.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by original time index with assignment columns.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> result.assignments.head()
                             period_idx  timestep_idx  cluster_idx
        2010-01-01 00:00:00          0             0            3
        2010-01-01 01:00:00          0             1            3
        ...

        >>> # Save and reload assignments
        >>> result.assignments.to_csv("assignments.csv")
        """
        # Build period_idx and timestep_idx for each original timestep
        period_indices = []
        timestep_indices = []
        cluster_indices = []

        for orig_period_idx, cluster_idx in enumerate(self.cluster_assignments):
            for timestep in range(self.n_timesteps_per_period):
                period_indices.append(orig_period_idx)
                timestep_indices.append(timestep)
                cluster_indices.append(cluster_idx)

        result_df = pd.DataFrame(
            {
                "period_idx": period_indices,
                "timestep_idx": timestep_indices,
                "cluster_idx": cluster_indices,
            },
            index=self._time_index,
        )

        # Add segment_idx if segmentation was used
        if self.n_segments is not None and self._segmented_df is not None:
            segment_indices = []
            for cluster_idx in self.cluster_assignments:
                segment_data = self._segmented_df.loc[cluster_idx]
                segment_steps = segment_data.index.get_level_values(0)
                segment_durations = segment_data.index.get_level_values(1)
                segment_indices.extend(
                    np.repeat(segment_steps, segment_durations).tolist()
                )
            result_df["segment_idx"] = segment_indices

        return result_df

    @cached_property
    def plot(self) -> ResultPlotAccessor:
        """Access plotting methods.

        Returns a plotting accessor with methods for visualizing the results.

        Returns
        -------
        ResultPlotAccessor
            Accessor with plotting methods.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> result.plot.compare()  # Compare original vs reconstructed
        >>> result.plot.residuals()  # View reconstruction errors
        >>> result.plot.cluster_representatives()
        >>> result.plot.cluster_members()  # All periods per cluster
        >>> result.plot.cluster_weights()
        >>> result.plot.accuracy()
        """
        from tsam.plot import ResultPlotAccessor

        return ResultPlotAccessor(self)


def _get_version() -> str:
    """Get tsam version string for ClusteringResult."""
    import importlib.metadata

    try:
        return importlib.metadata.version("tsam")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _validate_disaggregate_input(
    data: pd.DataFrame,
    clustering: ClusteringResult,
    *,
    is_segmented: bool,
) -> pd.DataFrame:
    """Validate and normalize input for disaggregation.

    Checks that the MultiIndex structure, cluster IDs, and timestep/segment
    counts match the clustering. For segmented data (3+ index levels), returns
    a copy with only the first two levels (cluster, segment).

    Returns the (possibly level-dropped) DataFrame ready for disaggregation.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels < 2:
        raise ValueError(
            "data must have a MultiIndex with at least 2 levels "
            "(cluster, timestep) or (cluster, segment, duration), "
            f"got {type(data.index).__name__}"
            + (
                f" with {data.index.nlevels} levels"
                if isinstance(data.index, pd.MultiIndex)
                else ""
            )
        )

    if is_segmented:
        data = data.droplevel(list(range(2, data.index.nlevels)))

    # Validate cluster IDs
    data_clusters = set(data.index.get_level_values(0).unique())
    expected_clusters = set(clustering.cluster_assignments)
    if data_clusters != expected_clusters:
        missing = expected_clusters - data_clusters
        extra = data_clusters - expected_clusters
        parts = []
        if missing:
            parts.append(f"missing clusters {sorted(missing)}")
        if extra:
            parts.append(f"unexpected clusters {sorted(extra)}")
        raise ValueError(
            f"Cluster IDs in data do not match this clustering: "
            f"{', '.join(parts)}. "
            f"Expected {sorted(expected_clusters)}, got {sorted(data_clusters)}."
        )

    # Validate second level count per cluster
    if is_segmented:
        expected = clustering.n_segments
        kind = "segments"
    else:
        expected = clustering.n_timesteps_per_period
        kind = "timesteps"

    for cluster in data.index.get_level_values(0).unique():
        n_in_cluster = len(data.loc[cluster])
        if n_in_cluster != expected:
            raise ValueError(
                f"cluster {cluster} has {n_in_cluster} {kind}, expected {expected}"
            )

    return data


def _expand_segments_to_timesteps(
    data: pd.DataFrame,
    segment_durations: tuple[tuple[int, ...], ...],
) -> pd.DataFrame:
    """Expand segmented typical-period data to full timestep resolution.

    Segment values are placed at the first timestep of each segment.
    All other timesteps are NaN. Callers can ``.ffill()`` the result
    to get a step function if needed.

    Parameters
    ----------
    data : pd.DataFrame
        Segmented data with ``(cluster, segment)`` MultiIndex.
    segment_durations : tuple of tuples
        Duration per segment per cluster.  ``segment_durations[i][j]`` is the
        number of timesteps for cluster *i*, segment *j*.

    Returns
    -------
    pd.DataFrame
        Data with ``(cluster, timestep)`` MultiIndex at full resolution.
        Only the first timestep of each segment has values; the rest are NaN.
    """
    clusters = data.index.get_level_values(0).unique()
    # Map cluster IDs to their segment durations. segment_durations is ordered
    # by unique cluster ID (sorted), not by positional index — so we zip with
    # the sorted unique cluster IDs from cluster_assignments to build the lookup.
    durations_by_cluster = dict(zip(sorted(set(clusters)), segment_durations))
    parts = []
    for cluster in clusters:
        cluster_data = data.loc[cluster]
        durations = durations_by_cluster[cluster]
        n_timesteps = sum(durations)

        values = np.full((n_timesteps, len(data.columns)), np.nan)
        pos = 0
        for seg_idx, d in enumerate(durations):
            values[pos] = cluster_data.values[seg_idx]
            pos += d

        idx = pd.MultiIndex.from_arrays([[cluster] * n_timesteps, range(n_timesteps)])
        parts.append(pd.DataFrame(values, index=idx, columns=data.columns))

    return pd.concat(parts)


def _expand_periods(
    data: pd.DataFrame,
    cluster_assignments: tuple[int, ...],
) -> pd.DataFrame:
    """Expand typical-period data to original time series length.

    Selects rows from ``data`` according to ``cluster_assignments``, mapping
    each original period to its cluster representative.

    Parameters
    ----------
    data : pd.DataFrame
        Typical-period data with ``(cluster, timestep)`` MultiIndex.
    cluster_assignments : tuple of int
        Cluster assignment for each original period.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with integer index, one row per original timestep.
    """
    unstacked = data.unstack(level=1)  # rows=cluster, cols=(col, timestep)
    expanded = unstacked.loc[list(cluster_assignments)]
    expanded.index = range(len(cluster_assignments))
    # Use level=-1 to always stack the timestep level (last), which is correct
    # even when the original columns are a MultiIndex.
    result: pd.DataFrame = expanded.stack(future_stack=True, level=-1)  # type: ignore[assignment]
    result.index = pd.RangeIndex(len(result))
    return result


@dataclass(frozen=True)
class ClusteringResult:
    """Clustering assignments that can be saved/loaded and applied to new data.

    This class bundles all clustering and segmentation assignments from an
    aggregation, enabling:
    - Simple IO via to_json()/from_json()
    - Applying the same clustering to different datasets via apply()
    - Preserving the parameters used to create the clustering

    Get this from `result.clustering` after running an aggregation.

    The first group of attributes is the *transfer state* consumed by
    `apply()`; the trailing ``*_config`` attributes are kept for reference only
    and are not used when re-applying.

    Attributes
    ----------
    period_duration : float
        Length of each period in hours (e.g., 24 for daily periods).
    cluster_assignments : tuple[int, ...]
        Cluster assignments for each original period. Length equals the number
        of original periods in the data.
    n_timesteps_per_period : int
        Number of timesteps in each period. Used to validate that new data has
        compatible structure when calling `apply()`.
    cluster_centers : tuple[int, ...], optional
        Indices of original periods used as cluster centers. If not provided,
        centers are recalculated when applying.
    segment_assignments : tuple[tuple[int, ...], ...], optional
        Segment assignments per timestep, per typical period. Only present if
        segmentation was used.
    segment_durations : tuple[tuple[int, ...], ...], optional
        Duration (in timesteps) per segment, per typical period. Required if
        ``segment_assignments`` is present.
    segment_centers : tuple[tuple[int, ...], ...], optional
        Indices of timesteps used as segment centers, per typical period.
        Required for fully deterministic segment replication.
    preserve_column_means : bool, default True
        Whether to rescale typical periods to match original data means.
    rescale_exclude_columns : tuple[str, ...], optional
        Column names to exclude from rescaling. Useful for binary columns.
    representation : str, default "medoid"
        How to compute typical periods from cluster members.
    segment_representation : str, optional
        How to compute segment values. Only used if segmentation is present.
    temporal_resolution : float, optional
        Time resolution of input data in hours. If not provided, inferred.
    cluster_config : ClusterConfig, optional
        Reference only. Clustering configuration used to create this result.
    segment_config : SegmentConfig, optional
        Reference only. Segmentation configuration used to create this result.
    extremes_config : ExtremeConfig, optional
        Reference only. Extreme-period configuration used to create this result.

    Examples
    --------
    >>> result = tsam.aggregate(df_wind, n_clusters=8)
    >>> clustering = result.clustering
    >>> clustering.to_json("clustering.json")  # save
    >>> clustering = ClusteringResult.from_json("clustering.json")  # load
    >>> result2 = clustering.apply(df_all)  # apply to new data
    """

    # === Transfer fields (used by apply()) ===
    period_duration: float
    cluster_assignments: tuple[int, ...]
    n_timesteps_per_period: int
    cluster_centers: tuple[int, ...] | None = None
    segment_assignments: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None
    segment_centers: tuple[tuple[int, ...], ...] | None = None
    preserve_column_means: bool = True
    rescale_exclude_columns: tuple[str, ...] | None = None
    representation: Representation = "medoid"
    segment_representation: Representation | None = None
    temporal_resolution: float | None = None
    extreme_cluster_indices: tuple[int, ...] | None = None
    weights: dict[str, float] | None = None

    # === Index fields (for disaggregate() round-trip) ===
    time_index: pd.DatetimeIndex | None = None

    # === Reference fields (for documentation, not used by apply()) ===
    cluster_config: ClusterConfig | None = None
    segment_config: SegmentConfig | None = None
    extremes_config: ExtremeConfig | None = None

    # === Format version ===
    version: str | None = None

    def __post_init__(self) -> None:
        if self.segment_assignments is not None and self.segment_durations is None:
            raise ValueError(
                "segment_durations must be provided when segment_assignments is specified"
            )
        if self.segment_durations is not None and self.segment_assignments is None:
            raise ValueError(
                "segment_assignments must be provided when segment_durations is specified"
            )
        if self.segment_centers is not None and self.segment_assignments is None:
            raise ValueError(
                "segment_assignments must be provided when segment_centers is specified"
            )

    @classmethod
    def from_pipeline(
        cls,
        *,
        cluster_center_indices: list | None,
        extreme_periods_info: dict,
        extremes_config: ExtremeConfig | None,
        cluster_order: list | np.ndarray,
        segmented_df: pd.DataFrame | None,
        segment_center_indices: list | None,
        n_timesteps_per_period: int,
        temporal_resolution: float | None,
        original_data: pd.DataFrame,
        cluster_config: ClusterConfig,
        segment_config: SegmentConfig | None,
        rescale_cluster_periods: bool,
        rescale_exclude_columns: list[str] | None,
        extreme_cluster_idx: list[int],
        weights: dict[str, float] | None = None,
        time_index: pd.DatetimeIndex | None = None,
    ) -> ClusteringResult:
        """Build a ClusteringResult from pipeline intermediate data."""
        # Get cluster centers
        cluster_centers: tuple[int, ...] | None = None
        if cluster_center_indices is not None:
            center_indices = [int(x) for x in cluster_center_indices]

            if (
                extreme_periods_info
                and extremes_config is not None
                and extremes_config.method in ("new_cluster", "append")
            ):
                for period_type in extreme_periods_info:
                    center_indices.append(
                        int(extreme_periods_info[period_type]["step_no"])
                    )

            cluster_centers = tuple(center_indices)

        # Compute segment data if segmentation was used
        segment_assignments: tuple[tuple[int, ...], ...] | None = None
        segment_durations: tuple[tuple[int, ...], ...] | None = None
        segment_centers: tuple[tuple[int, ...], ...] | None = None

        if segment_config is not None and segmented_df is not None:
            segment_assignments, segment_durations, segment_centers = (
                cls._extract_segment_data(segmented_df, segment_center_indices)
            )

        # Extract representation from configs
        representation = cluster_config.get_representation()
        segment_representation = (
            segment_config.representation if segment_config else None
        )

        # Extract extreme cluster indices
        extreme_cluster_indices_tuple: tuple[int, ...] | None = None
        if extreme_cluster_idx:
            extreme_cluster_indices_tuple = tuple(int(x) for x in extreme_cluster_idx)

        # Compute period_duration
        effective_resolution = (
            temporal_resolution
            if temporal_resolution is not None
            else infer_resolution(original_data)
        )
        period_duration = n_timesteps_per_period * effective_resolution

        return cls(
            period_duration=period_duration,
            cluster_assignments=tuple(int(x) for x in cluster_order),
            cluster_centers=cluster_centers,
            segment_assignments=segment_assignments,
            segment_durations=segment_durations,
            segment_centers=segment_centers,
            preserve_column_means=rescale_cluster_periods,
            rescale_exclude_columns=tuple(rescale_exclude_columns)
            if rescale_exclude_columns
            else None,
            representation=representation,
            segment_representation=segment_representation,
            temporal_resolution=temporal_resolution,
            n_timesteps_per_period=n_timesteps_per_period,
            extreme_cluster_indices=extreme_cluster_indices_tuple,
            weights=dict(weights) if weights else None,
            time_index=time_index,
            cluster_config=cluster_config,
            segment_config=segment_config,
            extremes_config=extremes_config,
            version=_get_version(),
        )

    @staticmethod
    def _extract_segment_data(
        segmented_df: pd.DataFrame,
        segment_center_indices: list | None,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...] | None,
    ]:
        """Extract segment assignments, durations, and centers from a segmented DataFrame."""
        assignments_list = []
        durations_list = []

        for period_idx in segmented_df.index.get_level_values(0).unique():
            period_data = segmented_df.loc[period_idx]
            assignments = []
            durations = []
            for seg_step, seg_dur, _orig_start in period_data.index:
                assignments.extend([int(seg_step)] * int(seg_dur))
                durations.append(int(seg_dur))
            assignments_list.append(tuple(assignments))
            durations_list.append(tuple(durations))

        centers: tuple[tuple[int, ...], ...] | None = None
        if segment_center_indices is not None:
            if all(pc is not None for pc in segment_center_indices):
                centers = tuple(
                    tuple(int(x) for x in period_centers)
                    for period_centers in segment_center_indices
                )

        return tuple(assignments_list), tuple(durations_list), centers

    @property
    def n_clusters(self) -> int:
        """Number of clusters (typical periods)."""
        return len(set(self.cluster_assignments))

    @property
    def n_original_periods(self) -> int:
        """Number of original periods in the source data."""
        return len(self.cluster_assignments)

    @property
    def n_segments(self) -> int | None:
        """Number of segments per period, or None if no segmentation."""
        if self.segment_durations is None:
            return None
        return len(self.segment_durations[0])

    def __repr__(self) -> str:
        has_centers = self.cluster_centers is not None
        has_segments = self.segment_assignments is not None

        lines = [
            "ClusteringResult(",
            f"  period_duration={self.period_duration},",
            f"  n_original_periods={self.n_original_periods},",
            f"  n_clusters={self.n_clusters},",
            f"  has_cluster_centers={has_centers},",
        ]

        if has_segments:
            n_segments = len(self.segment_durations[0]) if self.segment_durations else 0
            n_timesteps = (
                len(self.segment_assignments[0]) if self.segment_assignments else 0
            )
            has_seg_centers = self.segment_centers is not None
            lines.append(f"  n_segments={n_segments},")
            lines.append(f"  n_timesteps_per_period={n_timesteps},")
            lines.append(f"  has_segment_centers={has_seg_centers},")

        lines.append(")")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a readable DataFrame.

        Returns a DataFrame with one row per original period showing
        cluster assignments.

        Returns
        -------
        pd.DataFrame
            DataFrame with cluster_assignments indexed by original period.
        """
        df = pd.DataFrame(
            {"cluster": list(self.cluster_assignments)},
            index=pd.RangeIndex(len(self.cluster_assignments), name="original_period"),
        )

        if self.cluster_centers is not None:
            center_set = set(self.cluster_centers)
            df["is_center"] = [
                i in center_set for i in range(len(self.cluster_assignments))
            ]

        return df

    def segment_dataframe(self) -> pd.DataFrame | None:
        """Get segment structure as a readable DataFrame.

        Returns a DataFrame showing segment durations per typical period.
        Returns None if no segmentation is defined.

        Returns
        -------
        pd.DataFrame | None
            DataFrame with typical periods as rows and segments as columns,
            values are segment durations in timesteps.
        """
        if self.segment_durations is None:
            return None

        n_clusters = len(self.segment_durations)
        n_segments = len(self.segment_durations[0])

        return pd.DataFrame(
            list(self.segment_durations),
            index=pd.RangeIndex(n_clusters, name="cluster"),
            columns=pd.RangeIndex(n_segments, name="segment"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Transfer fields (always included)
        result: dict[str, Any] = {
            "version": self.version or _get_version(),
            "period_duration": self.period_duration,
            "cluster_assignments": list(self.cluster_assignments),
            "n_timesteps_per_period": self.n_timesteps_per_period,
            "preserve_column_means": self.preserve_column_means,
            "representation": representation_to_dict(self.representation),
        }
        if self.cluster_centers is not None:
            result["cluster_centers"] = list(self.cluster_centers)
        if self.segment_assignments is not None:
            result["segment_assignments"] = [list(s) for s in self.segment_assignments]
        if self.segment_durations is not None:
            result["segment_durations"] = [list(s) for s in self.segment_durations]
        if self.segment_centers is not None:
            result["segment_centers"] = [list(s) for s in self.segment_centers]
        if self.rescale_exclude_columns is not None:
            result["rescale_exclude_columns"] = list(self.rescale_exclude_columns)
        if self.segment_representation is not None:
            result["segment_representation"] = representation_to_dict(
                self.segment_representation
            )
        if self.temporal_resolution is not None:
            result["temporal_resolution"] = self.temporal_resolution
        if self.extreme_cluster_indices is not None:
            result["extreme_cluster_indices"] = list(self.extreme_cluster_indices)
        if self.weights is not None:
            result["weights"] = self.weights
        if self.time_index is not None:
            result["time_index"] = time_index_to_dict(self.time_index)
        # Reference fields (optional, for documentation)
        if self.cluster_config is not None:
            result["cluster_config"] = self.cluster_config.to_dict()
        if self.segment_config is not None:
            result["segment_config"] = self.segment_config.to_dict()
        if self.extremes_config is not None:
            result["extremes_config"] = self.extremes_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ClusteringResult:
        """Create from dictionary (e.g., loaded from JSON)."""
        # Transfer fields
        rep_data = data.get("representation", "medoid")
        seg_rep_data = data.get("segment_representation")
        kwargs: dict[str, Any] = {
            "period_duration": data["period_duration"],
            "cluster_assignments": tuple(data["cluster_assignments"]),
            "n_timesteps_per_period": data["n_timesteps_per_period"],
            "preserve_column_means": data.get("preserve_column_means", True),
            "representation": representation_from_dict(rep_data),
            "version": data.get("version"),
        }
        if "cluster_centers" in data:
            kwargs["cluster_centers"] = tuple(data["cluster_centers"])
        if "segment_assignments" in data:
            kwargs["segment_assignments"] = tuple(
                tuple(s) for s in data["segment_assignments"]
            )
        if "segment_durations" in data:
            kwargs["segment_durations"] = tuple(
                tuple(s) for s in data["segment_durations"]
            )
        if "segment_centers" in data:
            kwargs["segment_centers"] = tuple(tuple(s) for s in data["segment_centers"])
        if "rescale_exclude_columns" in data:
            kwargs["rescale_exclude_columns"] = tuple(data["rescale_exclude_columns"])
        if seg_rep_data is not None:
            kwargs["segment_representation"] = representation_from_dict(seg_rep_data)
        if "temporal_resolution" in data:
            kwargs["temporal_resolution"] = data["temporal_resolution"]
        if "extreme_cluster_indices" in data:
            kwargs["extreme_cluster_indices"] = tuple(data["extreme_cluster_indices"])
        if "weights" in data:
            kwargs["weights"] = data["weights"]
        raw_time_index = data.get("time_index")
        if raw_time_index is not None:
            kwargs["time_index"] = time_index_from_dict(raw_time_index)
        # Reference fields
        if "cluster_config" in data:
            kwargs["cluster_config"] = ClusterConfig.from_dict(data["cluster_config"])
        if "segment_config" in data:
            kwargs["segment_config"] = SegmentConfig.from_dict(data["segment_config"])
        if "extremes_config" in data:
            kwargs["extremes_config"] = ExtremeConfig.from_dict(data["extremes_config"])
        return cls(**kwargs)

    def to_json(self, path: str) -> None:
        """Save clustering result to a JSON file.

        Parameters
        ----------
        path : str
            File path to save to.

        Notes
        -----
        If the clustering used the 'replace' extreme method, a warning will be
        issued because the saved clustering cannot be perfectly reproduced when
        loaded and applied later. See :meth:`apply` for details.

        Examples
        --------
        >>> result.clustering.to_json("clustering.json")
        """
        import json

        # Warn if using replace extreme method (transfer is not exact)
        if (
            self.extremes_config is not None
            and self.extremes_config.method == "replace"
        ):
            warnings.warn(
                "Saving a clustering that used the 'replace' extreme method. "
                "The 'replace' method creates a hybrid cluster representation "
                "(some columns from the medoid, some from the extreme period) that "
                "cannot be perfectly reproduced when loaded and applied later. "
                "For exact transfer, use 'append' or 'new_cluster' extreme methods.",
                UserWarning,
                stacklevel=2,
            )

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> ClusteringResult:
        """Load clustering result from a JSON file.

        Parameters
        ----------
        path : str
            File path to load from.

        Returns
        -------
        ClusteringResult
            Loaded clustering result.

        Examples
        --------
        >>> clustering = ClusteringResult.from_json("clustering.json")
        >>> result = clustering.apply(new_data)
        """
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))

    def disaggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expand typical-period data back to the original time series length.

        Each original period is replaced by its assigned cluster representative
        from ``data``. For segmented data, segments are first expanded back to
        full timesteps using the stored segment durations, then periods are
        mapped back using cluster assignments.

        Parameters
        ----------
        data : pd.DataFrame
            Typical-period data with one of:

            - A ``(cluster, timestep)`` MultiIndex — works for any clustering,
              segmented or not. Periods are expanded directly.
            - A ``(cluster, segment, duration)`` MultiIndex — segments are
              expanded to timesteps first (NaN between segment starts),
              then periods are expanded.

        Returns
        -------
        pd.DataFrame
            Disaggregated data with integer-indexed rows
            (one row per original timestep). For segmented input,
            non-segment-start timesteps are NaN — use ``.ffill()``
            for a step function.

        Raises
        ------
        ValueError
            If the index structure, cluster IDs, or number of timesteps/segments
            do not match this clustering.

        Examples
        --------
        >>> clustering = ClusteringResult.from_json("clustering.json")
        >>> result = clustering.apply(df)
        >>> optimized = run_optimization(result.cluster_representatives)
        >>> full_year = clustering.disaggregate(optimized)
        """
        is_segmented_input = data.index.nlevels > 2
        is_segmented_clustering = self.segment_durations is not None

        if is_segmented_input and not is_segmented_clustering:
            raise ValueError(
                "data has segment-level index (3+ levels) but this clustering "
                "has no segmentation"
            )
        if is_segmented_clustering and not is_segmented_input:
            raise ValueError(
                "this clustering uses segmentation but data has a "
                "(cluster, timestep) index — pass segment-level data with a "
                "(cluster, segment, duration) index instead"
            )

        data = _validate_disaggregate_input(data, self, is_segmented=is_segmented_input)

        if is_segmented_input:
            data = _expand_segments_to_timesteps(data, self.segment_durations)  # type: ignore[arg-type]

        result = _expand_periods(data, self.cluster_assignments)

        if self.time_index is not None and len(self.time_index) == len(result):
            result.index = self.time_index

        return result

    def apply(
        self,
        data: pd.DataFrame,
        *,
        temporal_resolution: float | None = None,
        round_decimals: int | None = None,
        numerical_tolerance: float = 1e-13,
    ) -> AggregationResult:
        """Apply this clustering to new data.

        Uses the stored cluster assignments and transfer fields to aggregate
        a different dataset with the same clustering structure deterministically.

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data with a datetime index.
            Must have the same number of periods as the original data.

        temporal_resolution : float, optional
            Time resolution of input data in hours.
            If not provided, uses stored temporal_resolution or infers from data index.

        round_decimals : int, optional
            Round output values to this many decimal places.

        numerical_tolerance : float, default 1e-13
            Tolerance for numerical precision issues.

        Returns
        -------
        AggregationResult
            Aggregation result using this clustering.

        Notes
        -----
        **Extreme period transfer limitations:**

        The 'replace' extreme method creates a hybrid cluster representation where
        some columns use the medoid values and others use the extreme period values.
        This hybrid representation cannot be perfectly reproduced during transfer.
        When applying a clustering that used 'replace', a warning will be issued
        and the transferred result will use the medoid representation for all columns.

        For exact transfer with extreme periods, use 'append' or 'new_cluster'
        extreme methods instead.

        Examples
        --------
        >>> # Cluster on wind data, apply to full dataset
        >>> result_wind = tsam.aggregate(df_wind, n_clusters=8)
        >>> result_all = result_wind.clustering.apply(df_all)

        >>> # Load saved clustering and apply
        >>> clustering = ClusteringResult.from_json("clustering.json")
        >>> result = clustering.apply(df)
        """
        from tsam.api import _build_aggregation_result
        from tsam.pipeline import run_pipeline
        from tsam.pipeline.types import PipelineConfig, PredefParams

        # Warn if using replace extreme method (transfer is not exact)
        if (
            self.extremes_config is not None
            and self.extremes_config.method == "replace"
        ):
            warnings.warn(
                "The 'replace' extreme method creates a hybrid cluster representation "
                "(some columns from the cluster representative, some from the extreme period) "
                "that cannot be perfectly reproduced during transfer. The transferred result "
                "will use the stored cluster center periods directly, without the extreme "
                "value injection that was applied during the original aggregation. "
                "For exact transfer, use 'append' or 'new_cluster' extreme methods.",
                UserWarning,
                stacklevel=2,
            )

        # Use stored temporal_resolution if not provided
        effective_temporal_resolution = (
            temporal_resolution
            if temporal_resolution is not None
            else self.temporal_resolution
        )

        # Validate n_timesteps_per_period matches data
        if effective_temporal_resolution is None:
            inferred = infer_resolution(data)
        else:
            inferred = effective_temporal_resolution

        inferred_timesteps = int(self.period_duration / inferred)
        if inferred_timesteps != self.n_timesteps_per_period:
            raise ValueError(
                f"Data has {inferred_timesteps} timesteps per period "
                f"(period_duration={self.period_duration}h, timestep={inferred}h), "
                f"but clustering expects {self.n_timesteps_per_period} timesteps per period"
            )

        # Validate number of periods matches
        n_periods_in_data = len(data) // self.n_timesteps_per_period
        if n_periods_in_data != self.n_original_periods:
            raise ValueError(
                f"Data has {n_periods_in_data} periods, "
                f"but clustering expects {self.n_original_periods} periods"
            )

        # Build minimal ClusterConfig with just the representation.
        cluster = ClusterConfig(representation=self.representation)

        # Validate (and normalize) the stored weights against the new data
        from tsam.weights import validate_weights

        validated_weights = validate_weights(data.columns, self.weights)

        # Use stored segment config if available, otherwise build from transfer fields
        segments: SegmentConfig | None = None
        if self.segment_assignments is not None and self.segment_durations is not None:
            n_segments_val = len(self.segment_durations[0])
            segments = self.segment_config or SegmentConfig(
                n_segments=n_segments_val,
                representation=self.segment_representation or "mean",
            )

        # Run pipeline with predefined parameters
        predef = PredefParams(
            cluster_order=list(self.cluster_assignments),
            cluster_center_indices=list(self.cluster_centers)
            if self.cluster_centers
            else None,
            extreme_cluster_idx=list(self.extreme_cluster_indices)
            if self.extreme_cluster_indices
            else None,
            segment_order=[list(s) for s in self.segment_assignments]
            if self.segment_assignments
            else None,
            segment_durations=[list(s) for s in self.segment_durations]
            if self.segment_durations
            else None,
            segment_centers=[list(s) for s in self.segment_centers]
            if self.segment_centers
            else None,
        )

        cfg = PipelineConfig(
            n_clusters=self.n_clusters,
            n_timesteps_per_period=self.n_timesteps_per_period,
            cluster=cluster,
            weights=validated_weights,
            segments=segments,
            rescale_cluster_periods=self.preserve_column_means,
            rescale_exclude_columns=list(self.rescale_exclude_columns)
            if self.rescale_exclude_columns
            else None,
            round_decimals=round_decimals,
            numerical_tolerance=numerical_tolerance,
            temporal_resolution=effective_temporal_resolution,
            predef=predef,
        )

        result = run_pipeline(data=data, cfg=cfg)

        return _build_aggregation_result(result, is_transferred=True)
