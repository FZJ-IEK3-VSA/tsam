"""Result classes for tsam aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tsam.config import ClusteringResult
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
    rescale_deviations : pd.DataFrame
        Rescaling deviation information per column. Contains columns:
        - deviation_pct: Final deviation percentage after rescaling
        - converged: Whether rescaling converged within max iterations
        - iterations: Number of iterations used
        Only populated if rescaling was enabled, otherwise empty DataFrame.
    """

    rmse: pd.Series
    mae: pd.Series
    rmse_duration: pd.Series
    rescale_deviations: pd.DataFrame

    def __repr__(self) -> str:
        rescale_info = ""
        if not self.rescale_deviations.empty:
            n_failed = (~self.rescale_deviations["converged"]).sum()
            if n_failed > 0:
                max_dev = self.rescale_deviations["deviation_pct"].max()
                rescale_info = f",\n  rescale_failures={n_failed} (max {max_dev:.2f}%)"
        return (
            f"AccuracyMetrics(\n"
            f"  rmse={self.rmse.mean():.4f} (mean),\n"
            f"  mae={self.mae.mean():.4f} (mean),\n"
            f"  rmse_duration={self.rmse_duration.mean():.4f} (mean){rescale_info}\n"
            f")"
        )


@dataclass
class AggregationResult:
    """Result of time series aggregation.

    This class holds all outputs from the aggregation process and provides
    convenient methods for accessing and exporting the results.

    Attributes
    ----------
    typical_periods : pd.DataFrame
        The aggregated typical periods with MultiIndex (period, timestep).
        Each row represents one timestep in one typical period.

    cluster_assignments : np.ndarray
        Which cluster each original period belongs to.
        Length equals the number of original periods.
        Values are cluster indices (0 to n_periods-1).

    cluster_weights : dict[int, int]
        How many original periods each cluster represents.
        Keys are cluster indices, values are occurrence counts.

    n_periods : int
        Number of typical periods (clusters).

    n_timesteps_per_period : int
        Number of timesteps in each period.

    n_segments : int | None
        Number of segments per period if segmentation was used, else None.

    segment_durations : dict[int, float] | None
        Duration of each segment if segmentation was used.
        Keys are segment indices, values are durations in hours.

    accuracy : AccuracyMetrics
        Accuracy metrics comparing reconstructed to original data.

    clustering_duration : float
        Time taken for clustering in seconds.

    Examples
    --------
    >>> result = tsam.aggregate(df, n_periods=8)
    >>> result.typical_periods
                        solar  wind  demand
    period timestep
    0      0           0.12   0.45   0.78
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

    typical_periods: pd.DataFrame
    cluster_weights: dict[int, int]
    n_timesteps_per_period: int
    segment_durations: dict[int, float] | None
    accuracy: AccuracyMetrics
    clustering_duration: float
    clustering: ClusteringResult
    _aggregation: TimeSeriesAggregation = field(repr=False, compare=False)

    @cached_property
    def n_periods(self) -> int:
        """Number of typical periods (clusters)."""
        return self.clustering.n_periods

    @cached_property
    def n_segments(self) -> int | None:
        """Number of segments per period if segmentation was used, else None."""
        return self.clustering.n_segments

    @cached_property
    def cluster_assignments(self) -> np.ndarray:
        """Which cluster each original period belongs to.

        Length equals the number of original periods.
        Values are cluster indices (0 to n_periods-1).
        """
        return np.array(self.clustering.cluster_order)

    def __repr__(self) -> str:
        seg_info = f", n_segments={self.n_segments}" if self.n_segments else ""
        return (
            f"AggregationResult(\n"
            f"  n_periods={self.n_periods},\n"
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
        >>> result = tsam.aggregate(df, n_periods=8)
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
            "typical_periods": self.typical_periods.to_dict(),
            "cluster_assignments": self.cluster_assignments.tolist(),
            "cluster_weights": self.cluster_weights,
            "n_periods": self.n_periods,
            "n_timesteps_per_period": self.n_timesteps_per_period,
            "n_segments": self.n_segments,
            "segment_durations": self.segment_durations,
            "clustering": self.clustering.to_dict(),
            "accuracy": {
                "rmse": self.accuracy.rmse.to_dict(),
                "mae": self.accuracy.mae.to_dict(),
                "rmse_duration": self.accuracy.rmse_duration.to_dict(),
                "rescale_deviations": self.accuracy.rescale_deviations.to_dict(),
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
            List of indices [0, 1, ..., n_periods-1].
        """
        return list(range(self.n_periods))

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
            Which cluster this period is assigned to (0 to n_periods-1).
        segment_idx : int (only if segmentation was used)
            Which segment this timestep belongs to within its period.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by original time index with assignment columns.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_periods=8)
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
    def plot(self) -> ResultPlotAccessor:
        """Access plotting methods.

        Returns a plotting accessor with methods for visualizing the results.

        Returns
        -------
        ResultPlotAccessor
            Accessor with plotting methods.

        Examples
        --------
        >>> result = tsam.aggregate(df, n_periods=8)
        >>> result.plot.heatmap(column="Load")
        >>> result.plot.duration_curve()
        >>> result.plot.typical_periods()
        >>> result.plot.cluster_weights()
        >>> result.plot.accuracy()
        """
        from tsam.plot import ResultPlotAccessor

        # Get original data from the internal aggregation object
        original_data = getattr(self._aggregation, "timeSeries", None)
        return ResultPlotAccessor(self, original_data=original_data)
