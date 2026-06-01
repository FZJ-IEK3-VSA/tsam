"""Milestone dataclasses for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

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


@dataclass(frozen=True)
class PipelineConfig:
    """All non-data parameters for a pipeline run."""

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
    """Predefined assignments for transfer/apply (skip clustering)."""

    cluster_order: list | np.ndarray
    cluster_center_indices: list[int] | np.ndarray | None = None
    extreme_cluster_idx: list[int] | None = None
    segment_order: list | None = None
    segment_durations: list | None = None
    segment_centers: list | None = None


@dataclass(frozen=True)
class NormalizedData:
    """Carries everything needed for denormalization."""

    values: pd.DataFrame  # normalized (unweighted) time series
    scaler: MinMaxScaler  # fitted on original, reusable for inverse_transform
    normalized_mean: pd.Series  # mean before scale_by_column_means division
    scale_by_column_means: bool  # whether scale_by_column_means was applied


@dataclass(frozen=True)
class PeriodProfiles:
    """The 'candidates' matrix + metadata for reconstruction."""

    column_index: pd.MultiIndex  # unstacked column structure
    time_index: pd.Index  # datetime index (possibly extended)
    profiles_dataframe: pd.DataFrame  # unstacked DataFrame
    n_timesteps_per_period: int
    n_columns: int
    n_periods: int


@dataclass(frozen=True)
class PreparedData:
    """Output of the prepare-data phase (`prepare_data`)."""

    norm_data: NormalizedData
    period_profiles: PeriodProfiles
    candidates: np.ndarray
    representation_dict: dict[str, str]
    n_feature_cols: int
    original_column_order: list[str]
    original_data: (
        pd.DataFrame
    )  # original input data (for rescale, bounds, reconstruct)
    weight_vector: np.ndarray | None = None
    weighted_profiles_df: pd.DataFrame | None = None


@dataclass(frozen=True)
class ClusteringOutput:
    """Output of the cluster & post-process phase (`cluster_and_postprocess`)."""

    cluster_periods_list: list[np.ndarray]
    cluster_order: np.ndarray
    cluster_counts: dict[int, float]
    cluster_center_indices: list[int] | None
    extreme_cluster_idx: list[int]
    extreme_periods_info: dict[str, dict]
    clustering_duration: float
    rescale_deviations: dict[str, dict]


@dataclass(frozen=True)
class FormattedOutput:
    """Output of the format & reconstruct phase (`format_and_reconstruct`)."""

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
    """

    typical_periods: pd.DataFrame  # denormalized, MultiIndex (cluster, timestep)
    cluster_counts: dict[int, float]
    n_timesteps_per_period: int
    time_index: pd.Index
    original_data: pd.DataFrame
    clustering_duration: float
    rescale_deviations: dict[str, dict]
    segmented_df: pd.DataFrame | None  # segmentedNormalizedTypicalPeriods
    reconstructed_data: pd.DataFrame
    _norm_values: pd.DataFrame
    _normalized_predicted: pd.DataFrame
    clustering_result: ClusteringResult

    @cached_property
    def accuracy_indicators(self) -> pd.DataFrame:
        from tsam.pipeline.accuracy import compute_accuracy

        return compute_accuracy(self._norm_values, self._normalized_predicted)
