"""Milestone dataclasses for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    from tsam.config import (
        ClusterConfig,
        ExtremeConfig,
        SegmentConfig,
    )
    from tsam.result import ClusteringResult


ExtremeKind = Literal["max", "min", "mean_max", "mean_min"]


@dataclass
class ExtremePeriod:
    """An original period kept intact because of an extreme value it holds.

    Produced by `add_extreme_periods`, one per extreme that survives the
    redundancy check. Mutable, because which cluster the period ends up in is
    only settled once the integration method has run.

    Attributes:
        column: Data column whose extreme this period holds.
        kind: Which extreme it is — the column's single highest (``"max"``) or
            lowest (``"min"``) value, or its highest (``"mean_max"``) or lowest
            (``"mean_min"``) period average.
        step_no: Index of the period in the period-profile matrix.
        profile: The period's profile, in the same (weighted) space as the
            cluster centers it joins.
        new_cluster_no: Cluster created for this period by the
            ``"new_cluster"`` method; ``None`` for the other methods, which do
            not create one.
    """

    column: str | tuple
    kind: ExtremeKind
    step_no: int
    profile: np.ndarray
    new_cluster_no: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """All non-data parameters for a pipeline run.

    Attributes:
        n_clusters: Number of clusters (typical periods) to form.
        n_timesteps_per_period: Timesteps in one period.
        cluster: Clustering configuration.
        weights: Per-column weights applied to the clustering distance.
        extremes: Extreme-period configuration, if any.
        segments: Segmentation configuration, if any.
        rescale_cluster_periods: Whether to rescale representatives to match
            the original column means.
        rescale_exclude_columns: Columns to skip during rescaling.
        round_decimals: Number of decimals to round outputs to, if set.
        numerical_tolerance: Tolerance for the output bounds check.
        temporal_resolution: Time resolution of one timestep, if provided.
        predef: Predefined assignments for the transfer path, if any.
    """

    n_clusters: int
    n_timesteps_per_period: int
    cluster: ClusterConfig
    weights: dict[str, float] | None = None
    extremes: ExtremeConfig | None = None
    segments: SegmentConfig | None = None
    rescale_cluster_periods: bool = True
    rescale_exclude_columns: list[str] | None = None
    round_decimals: int | None = None
    numerical_tolerance: float = 1e-13
    temporal_resolution: float | None = None
    predef: PredefParams | None = None


@dataclass(frozen=True)
class PredefParams:
    """Predefined assignments for transfer/apply (skip clustering).

    Attributes:
        cluster_order: Per-period cluster assignment to reuse.
        cluster_center_indices: Stored medoid period indices, if saved.
        extreme_cluster_idx: Indices of the extreme clusters, if any.
        segment_order: Stored segment order for the transfer path.
        segment_durations: Stored segment durations for the transfer path.
        segment_centers: Stored segment centers for the transfer path.
    """

    cluster_order: list | np.ndarray
    cluster_center_indices: list[int] | np.ndarray | None = None
    extreme_cluster_idx: list[int] | None = None
    segment_order: list | None = None
    segment_durations: list | None = None
    segment_centers: list | None = None


@dataclass(frozen=True)
class NormalizedData:
    """Carries everything needed for denormalization.

    Attributes:
        values: Normalized (unweighted) time series.
        scaler: Fitted on the original data, reusable for inverse_transform.
        normalized_mean: Mean before the scale_by_column_means division.
        scale_by_column_means: Whether scale_by_column_means was applied.
    """

    values: pd.DataFrame
    scaler: MinMaxScaler
    normalized_mean: pd.Series
    scale_by_column_means: bool


@dataclass(frozen=True)
class PeriodProfiles:
    """The 'candidates' matrix plus metadata for reconstruction.

    Attributes:
        column_index: Unstacked column structure.
        time_index: Datetime index (possibly extended).
        profiles_dataframe: Unstacked DataFrame.
        n_timesteps_per_period: Timesteps in one period.
        n_columns: Number of original columns.
        n_periods: Number of periods.
    """

    column_index: pd.MultiIndex
    time_index: pd.Index
    profiles_dataframe: pd.DataFrame
    n_timesteps_per_period: int
    n_columns: int
    n_periods: int


@dataclass(frozen=True)
class PreparedData:
    """Output of the prepare-data phase (`prepare_data`).

    Attributes:
        norm_data: Normalization state for later denormalization.
        period_profiles: The unstacked period profiles and metadata.
        candidates: Candidate period matrix (possibly weighted / augmented).
        representation_dict: Per-column representation overrides.
        n_feature_cols: Number of feature columns.
        original_column_order: Column order of the original input.
        original_data: Original input data (for rescale, bounds, reconstruct).
        weight_vector: Per-column weights baked into the candidates, if any.
        weighted_profiles_df: Weighted period profiles, if weights are active.
    """

    norm_data: NormalizedData
    period_profiles: PeriodProfiles
    candidates: np.ndarray
    representation_dict: dict[str, str]
    n_feature_cols: int
    original_column_order: list[str]
    original_data: pd.DataFrame
    weight_vector: np.ndarray | None = None
    weighted_profiles_df: pd.DataFrame | None = None


@dataclass(frozen=True)
class ClusteringOutput:
    """Output of the cluster and post-process phase (`cluster_and_postprocess`).

    Attributes:
        cluster_periods_list: The cluster representatives.
        cluster_order: Per-period cluster assignment.
        cluster_counts: Occurrence count per cluster.
        cluster_center_indices: Medoid period indices, if applicable.
        extreme_cluster_idx: Indices of the extreme clusters.
        extreme_periods: The preserved extreme periods (for transfer).
        clustering_duration: Wall-clock time spent clustering.
        rescale_deviations: Per-column residual deviations after rescaling.
    """

    cluster_periods_list: list[np.ndarray]
    cluster_order: np.ndarray
    cluster_counts: dict[int, float]
    cluster_center_indices: list[int] | None
    extreme_cluster_idx: list[int]
    extreme_periods: list[ExtremePeriod]
    clustering_duration: float
    rescale_deviations: dict[str, dict]


@dataclass(frozen=True)
class FormattedOutput:
    """Output of the format and reconstruct phase (`format_and_reconstruct`).

    Attributes:
        typical_periods: The formatted typical periods.
        reconstructed_data: Full-length reconstruction in original units.
        normalized_predicted: Full-length reconstruction in normalized units.
        segmented_df: Segmented typical periods, if segmentation ran.
        segment_center_indices: Segment center indices, if segmentation ran.
    """

    typical_periods: pd.DataFrame
    reconstructed_data: pd.DataFrame
    normalized_predicted: pd.DataFrame
    segmented_df: pd.DataFrame | None
    segment_center_indices: list | None


@dataclass(frozen=True)
class PipelineResult:
    """Output of the assemble phase (`assemble_result`).

    The single handoff from the pipeline to `tsam.api` / `tsam.config`, wrapped
    there as the user-facing `AggregationResult`.

    Attributes:
        typical_periods: Denormalized, MultiIndex (cluster, timestep).
        cluster_counts: Occurrence count per cluster.
        n_timesteps_per_period: Timesteps in one period.
        time_index: Datetime index of the original series.
        original_data: Original input data.
        clustering_duration: Wall-clock time spent clustering.
        rescale_deviations: Per-column residual deviations after rescaling.
        segmented_df: Segmented normalized typical periods, if segmentation ran.
        reconstructed_data: Full-length reconstruction in original units.
        clustering_result: The reusable clustering assignments for transfer.
    """

    typical_periods: pd.DataFrame
    cluster_counts: dict[int, float]
    n_timesteps_per_period: int
    time_index: pd.Index
    original_data: pd.DataFrame
    clustering_duration: float
    rescale_deviations: dict[str, dict]
    segmented_df: pd.DataFrame | None
    reconstructed_data: pd.DataFrame
    _norm_values: pd.DataFrame
    _normalized_predicted: pd.DataFrame
    clustering_result: ClusteringResult

    @cached_property
    def accuracy_indicators(self) -> pd.DataFrame:
        """Reconstruction accuracy metrics per column, computed on first access."""
        from tsam.pipeline.accuracy import compute_accuracy

        return compute_accuracy(self._norm_values, self._normalized_predicted)
