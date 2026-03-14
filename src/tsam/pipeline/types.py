"""Milestone dataclasses for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    from tsam.config import ClusteringResult


@dataclass(frozen=True)
class PredefParams:
    """Predefined assignments for transfer/apply (skip clustering)."""

    cluster_order: list | np.ndarray
    cluster_center_indices: list | np.ndarray | None = None
    extreme_cluster_idx: list | None = None
    segment_order: list | None = None
    segment_durations: list | None = None
    segment_centers: list | None = None


@dataclass(frozen=True)
class NormalizedData:
    """Carries everything needed for denormalization."""

    values: pd.DataFrame  # normalized (unweighted) time series
    scaler: MinMaxScaler  # fitted on original, reusable for inverse_transform
    normalized_mean: pd.Series  # mean before normalize_column_means division
    original_data: pd.DataFrame  # for bounds check + rescale upper bound
    normalize_column_means: (
        bool  # whether normalize_column_means normalization was applied
    )


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
    """Output of the data preparation phase (steps 1-3)."""

    norm_data: NormalizedData
    period_profiles: PeriodProfiles
    candidates: np.ndarray
    weighted_candidates: np.ndarray | None
    representation_dict: dict[str, str]
    n_feature_cols: int
    original_column_order: list[str]
    has_period_sums: bool = False


@dataclass(frozen=True)
class ClusteringOutput:
    """Output of the clustering + post-processing phase (steps 4-9)."""

    cluster_periods_list: list
    cluster_order: list | np.ndarray
    cluster_counts: dict[int, float]
    cluster_center_indices: list | None
    extreme_cluster_idx: list[int]
    extreme_periods_info: dict
    clustering_duration: float
    rescale_deviations: dict


@dataclass(frozen=True)
class FormattedOutput:
    """Output of the formatting + reconstruction phase (steps 10-14)."""

    typical_periods: pd.DataFrame
    reconstructed_data: pd.DataFrame
    accuracy_df: pd.DataFrame
    segmented_df: pd.DataFrame | None
    segment_center_indices: list | None


@dataclass(frozen=True)
class PipelineResult:
    """Single handoff from pipeline to api.py / config.py."""

    typical_periods: pd.DataFrame  # denormalized, MultiIndex (cluster, timestep)
    cluster_counts: dict[int, float]
    n_timesteps_per_period: int
    time_index: pd.Index
    original_data: pd.DataFrame
    clustering_duration: float
    rescale_deviations: dict
    segmented_df: pd.DataFrame | None  # segmentedNormalizedTypicalPeriods
    reconstructed_data: pd.DataFrame
    accuracy_indicators: pd.DataFrame
    clustering_result: ClusteringResult
