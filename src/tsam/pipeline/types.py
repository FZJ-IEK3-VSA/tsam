"""Milestone dataclasses for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

    values: pd.DataFrame  # normalized + weighted time series
    scaler: MinMaxScaler  # fitted on original, reusable for inverse_transform
    normalized_mean: pd.Series  # mean before sameMean division
    original_data: pd.DataFrame  # for bounds check + rescale upper bound


@dataclass(frozen=True)
class PeriodProfiles:
    """The 'candidates' matrix + metadata for reconstruction."""

    profiles: np.ndarray  # (n_periods, n_cols * n_timesteps)
    column_index: pd.MultiIndex  # unstacked column structure
    time_index: pd.Index  # datetime index (possibly extended)
    profiles_dataframe: (
        pd.DataFrame
    )  # unstacked DataFrame (needed by extremes, rescale)
    n_timesteps_per_period: int
    n_columns: int
    n_periods: int


@dataclass(frozen=True)
class PipelineResult:
    """Single handoff from pipeline to api.py / config.py."""

    typical_periods: pd.DataFrame  # denormalized, MultiIndex (cluster, timestep)
    normalized_typical_periods: pd.DataFrame  # for reconstruction
    cluster_assignments: np.ndarray
    cluster_weights: dict[int, Any]
    cluster_center_indices: list[int] | None
    extreme_cluster_indices: list[int]
    extreme_periods_info: dict
    time_index: pd.Index
    n_timesteps_per_period: int
    original_data: pd.DataFrame
    normalized_data: NormalizedData
    period_profiles: PeriodProfiles
    clustering_duration: float
    rescale_deviations: dict
    segmented_df: pd.DataFrame | None  # segmentedNormalizedTypicalPeriods
    predicted_segmented_df: pd.DataFrame | None
    segment_center_indices: list | None
    normalize_column_means: bool
    weights: dict[str, float] | None
    round_decimals: int | None
    reconstructed_data: pd.DataFrame
    accuracy_indicators: pd.DataFrame
    clustering_result: ClusteringResult
