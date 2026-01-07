"""Result classes for tsam aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tsam.config import PredefinedConfig
    from tsam.plot import ResultPlotAccessor
    from tsam.timeseriesaggregation import TimeSeriesAggregation


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing aggregated to original time series.

    Attributes
    ----------
    rmse : pd.Series
        Root Mean Square Error per column.
    mae : pd.Series
        Mean Absolute Error per column.
    rmse_duration : pd.Series
        RMSE on duration curves (sorted values) per column.
    """

    rmse: pd.Series
    mae: pd.Series
    rmse_duration: pd.Series

    def __repr__(self) -> str:
        return (
            f"AccuracyMetrics(\n"
            f"  rmse={self.rmse.mean():.4f} (mean),\n"
            f"  mae={self.mae.mean():.4f} (mean),\n"
            f"  rmse_duration={self.rmse_duration.mean():.4f} (mean)\n"
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

    cluster_weights : dict[int, int]
        How many original periods each cluster represents.
        Keys are cluster indices, values are occurrence counts.

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

    cluster_centers : np.ndarray | None
        Indices of original periods used as cluster centers.
        These are the "representative" periods for each cluster.

    accuracy : AccuracyMetrics
        Accuracy metrics comparing reconstructed to original data.

    clustering_duration : float
        Time taken for clustering in seconds.

    Examples
    --------
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> result.cluster_representatives
                        solar  wind  demand
    cluster timestep
    0       0           0.12   0.45   0.78
            1           0.15   0.42   0.82
    ...

    >>> result.cluster_weights
    {0: 45, 1: 52, 2: 38, ...}

    >>> result.accuracy.rmse
    solar     0.023
    wind      0.041
    demand    0.015
    dtype: float64
    """

    cluster_representatives: pd.DataFrame
    cluster_assignments: np.ndarray
    cluster_weights: dict[int, int]
    n_clusters: int
    n_timesteps_per_period: int
    n_segments: int | None
    segment_durations: tuple[tuple[int, ...], ...] | None
    cluster_centers: np.ndarray | None
    accuracy: AccuracyMetrics
    clustering_duration: float
    _aggregation: TimeSeriesAggregation = field(repr=False, compare=False)

    def __repr__(self) -> str:
        seg_info = f", n_segments={self.n_segments}" if self.n_segments else ""
        return (
            f"AggregationResult(\n"
            f"  n_clusters={self.n_clusters},\n"
            f"  n_timesteps_per_period={self.n_timesteps_per_period}{seg_info},\n"
            f"  accuracy={self.accuracy}\n"
            f")"
        )

    def reconstruct(self) -> pd.DataFrame:
        """Reconstruct the original time series from typical periods.

        Returns a DataFrame with the same shape as the original input,
        where each period is replaced by its assigned typical period.

        Returns
        -------
        pd.DataFrame
            Reconstructed time series.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8)
        >>> reconstructed = result.reconstruct()
        >>> reconstructed.shape == df.shape
        True
        """
        return cast("pd.DataFrame", self._aggregation.predictOriginalData())

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
            "cluster_weights": self.cluster_weights,
            "n_clusters": self.n_clusters,
            "n_timesteps_per_period": self.n_timesteps_per_period,
            "n_segments": self.n_segments,
            "segment_durations": [list(s) for s in self.segment_durations]
            if self.segment_durations is not None
            else None,
            "segment_assignments": self.segment_assignments,
            "cluster_centers": self.cluster_centers.tolist()
            if self.cluster_centers is not None
            else None,
            "accuracy": {
                "rmse": self.accuracy.rmse.to_dict(),
                "mae": self.accuracy.mae.to_dict(),
                "rmse_duration": self.accuracy.rmse_duration.to_dict(),
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

        Returns
        -------
        list[int]
            List of indices [0, 1, ..., n_clusters-1].
        """
        return list(range(self.n_clusters))

    @property
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
        agg = self._aggregation

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
            index=agg.timeIndex,
        )

        # Add segment_idx if segmentation was used
        if self.n_segments is not None and hasattr(
            agg, "segmentedNormalizedTypicalPeriods"
        ):
            segment_indices = []
            for cluster_idx in self.cluster_assignments:
                # Get segment structure for this cluster's typical period
                segment_data = agg.segmentedNormalizedTypicalPeriods.loc[cluster_idx]
                # Segment Step is level 0, Segment Duration is level 1
                segment_steps = segment_data.index.get_level_values(0)
                segment_durations = segment_data.index.get_level_values(1)
                # Repeat each segment index by its duration
                segment_indices.extend(
                    np.repeat(segment_steps, segment_durations).tolist()
                )
            result_df["segment_idx"] = segment_indices

        return result_df

    @property
    def segment_assignments(self) -> tuple[tuple[int, ...], ...] | None:
        """Get segment assignments per typical period for transfer.

        Returns the segment index for each timestep within each typical period.
        This can be passed to another aggregation's SegmentConfig.predef_segment_assignments
        to reproduce the same segmentation.

        Returns
        -------
        tuple[tuple[int, ...], ...] | None
            Tuple of tuples, one per typical period. Each inner tuple contains
            segment indices (0 to n_segments-1) for each timestep in that period.
            Returns None if segmentation was not used.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8, segments=SegmentConfig(n_segments=6))
        >>> result.segment_assignments[0]
        (0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5)

        >>> # Transfer to another aggregation
        >>> result2 = tsam.aggregate(
        ...     other_data,
        ...     segments=SegmentConfig(
        ...         n_segments=6,
        ...         predef_segment_assignments=result.segment_assignments,
        ...         predef_segment_durations=result.segment_durations,
        ...     ),
        ... )
        """
        if self.n_segments is None:
            return None

        agg = self._aggregation
        if not hasattr(agg, "segmentedNormalizedTypicalPeriods"):
            return None

        result = []
        segmented_df = agg.segmentedNormalizedTypicalPeriods

        for period_idx in segmented_df.index.get_level_values(0).unique():
            period_data = segmented_df.loc[period_idx]
            # Reconstruct full assignment from segment structure
            # Index levels: Segment Step, Segment Duration, Original Start Step
            assignments = []
            for seg_step, seg_dur, _orig_start in period_data.index:
                assignments.extend([int(seg_step)] * int(seg_dur))
            result.append(tuple(assignments))

        return tuple(result)

    @property
    def predefined(self) -> PredefinedConfig:
        """Get predefined state for transferring to another aggregation.

        Returns a PredefinedConfig containing all assignment information needed
        to reproduce this aggregation's clustering and segmentation on new data.

        Returns
        -------
        PredefinedConfig
            Config object with cluster_assignments, cluster_centers (optional),
            segment_assignments (if segmentation), segment_durations (if segmentation).

        Examples
        --------
        >>> result = tsam.aggregate(df, n_clusters=8, segments=SegmentConfig(n_segments=6))

        >>> # Apply directly to new data
        >>> result2 = tsam.aggregate(new_data, n_clusters=8, predefined=result.predefined)

        >>> # Save to file
        >>> import json
        >>> with open("predefined.json", "w") as f:
        ...     json.dump(result.predefined.to_dict(), f)

        >>> # Load and apply
        >>> with open("predefined.json") as f:
        ...     predefined = PredefinedConfig.from_dict(json.load(f))
        >>> result2 = tsam.aggregate(new_data, n_clusters=8, predefined=predefined)
        """
        from tsam.config import PredefinedConfig

        return PredefinedConfig(
            cluster_assignments=tuple(self.cluster_assignments.tolist()),
            cluster_centers=tuple(self.cluster_centers.tolist())
            if self.cluster_centers is not None
            else None,
            segment_assignments=self.segment_assignments,
            segment_durations=self.segment_durations,
        )

    @property
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
        >>> result.plot.heatmap(column="Load")
        >>> result.plot.duration_curve()
        >>> result.plot.cluster_representatives()
        >>> result.plot.cluster_weights()
        >>> result.plot.accuracy()
        """
        from tsam.plot import ResultPlotAccessor

        # Get original data from the internal aggregation object
        original_data = getattr(self._aggregation, "timeSeries", None)
        return ResultPlotAccessor(self, original_data=original_data)
