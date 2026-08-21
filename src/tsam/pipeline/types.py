"""Milestone dataclasses for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
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

    Located by `locate` and kept by `detect_extreme_periods`, one per extreme
    that survives the redundancy check. Mutable, because which cluster the
    period ends up in is only settled once the integration method has run.

    Attributes:
        column: Data column whose extreme this period holds.
        kind: Which extreme it is — the column's single highest (``"max"``) or
            lowest (``"min"``) value, or its highest (``"mean_max"``) or lowest
            (``"mean_min"``) period average.
        step_no: Index of the period in the period-profile matrix.
        profile: The period's profile, in the same (weighted) space as the
            cluster centers it joins.
        new_cluster_no: Cluster created for this period by ``"append"`` or
            ``"new_cluster"``; ``None`` under ``"replace"``, which splices the
            period into an existing cluster instead of creating one.
    """

    column: str | tuple
    kind: ExtremeKind
    step_no: int
    profile: np.ndarray
    new_cluster_no: int | None = None

    @classmethod
    def locate(
        cls,
        profiles_df: pd.DataFrame,
        column: str | tuple,
        kind: ExtremeKind,
    ) -> ExtremePeriod:
        """Find the period holding one extreme of one column, and capture it.

        The four kinds are the four questions tsam asks of a column: which
        period holds its single highest (``"max"``) or lowest (``"min"``)
        value, and which has the highest (``"mean_max"``) or lowest
        (``"mean_min"``) average. Pure — it reads *profiles_df* and nothing
        else, so it is the same answer before and after clustering.

        Args:
            profiles_df: Period profiles, in whatever space the caller works in
                (weighted, when weights are active).
            column: Column to search.
            kind: Which of the column's four extremes to look for.

        Returns:
            The located period. Whether it is worth *keeping* is the caller's
            call — see `detect_extreme_periods`.
        """
        # One column's block: the stubs cannot see that indexing a
        # (column, TimeStep) MultiIndex by level 0 yields a frame, not a series.
        values = cast("pd.DataFrame", profiles_df[column])
        if kind == "max":
            step_no = values.max(axis=1).idxmax()
        elif kind == "min":
            step_no = values.min(axis=1).idxmin()
        elif kind == "mean_max":
            step_no = values.mean(axis=1).idxmax()
        else:  # mean_min
            step_no = values.mean(axis=1).idxmin()
        return cls(
            column=column,
            kind=kind,
            # Period profiles are indexed by period number, so the label
            # `idxmax`/`idxmin` returns is the row index.
            step_no=int(step_no),
            profile=np.asarray(profiles_df.loc[step_no, :].values),
        )


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
        representation_dict: Per-column representation overrides for the
            **cluster** representation.
        n_feature_cols: Number of feature columns.
        original_column_order: Column order of the original input.
        original_data: Original input data (for rescale, bounds, reconstruct).
        weight_vector: Per-column weights baked into the candidates, if any.
        weighted_profiles_df: Weighted period profiles, if weights are active.
        segment_representation_dict: Per-column representation overrides for the
            **segment** representation. Separate from `representation_dict`
            because the two stages are configured independently —
            `ClusterConfig.representation` and `SegmentConfig.representation`
            may each be a `MinMaxMean` naming different columns.
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
    segment_representation_dict: dict[str, str] | None = None

    @property
    def attribute_columns(self) -> list[str]:
        """Column names in the order of the candidates' attribute blocks.

        Each candidate row holds one contiguous block of timesteps per column,
        in this order, so stages that address attributes positionally use this
        list to map a column name onto its block.
        """
        return list(
            self.period_profiles.profiles_dataframe.columns.get_level_values(0).unique()
        )


@dataclass(frozen=True)
class ClusterAssignment:
    """Output of the clustering phase (`cluster_candidates`).

    The raw grouping, before any of the optional adjustments. The
    representatives are still in weighted space, since the extreme detection
    that follows has to run there.

    Attributes:
        cluster_periods_list: The cluster representatives, still weighted.
        cluster_order: Per-period cluster assignment.
        cluster_center_indices: Medoid period indices, if applicable.
        clustering_duration: Wall-clock time spent clustering.
    """

    cluster_periods_list: list[np.ndarray]
    cluster_order: np.ndarray
    cluster_center_indices: list[int] | None
    clustering_duration: float


@dataclass(frozen=True)
class RefinedRepresentatives:
    """Output of the refine phase (`refine_representatives`).

    The final set of typical periods, still normalized: everything that can
    change *which* periods are represented, or *what* they contain, has already
    happened. What follows only expresses them in the user's units.

    Attributes:
        normalized_typical_periods: Representatives as a ``(PeriodNum,
            TimeStep)`` MultiIndex frame, unweighted and still normalized.
        cluster_order: Per-period cluster assignment, including any extremes.
        cluster_counts: Occurrence count per cluster.
        cluster_center_indices: Medoid period indices, if applicable.
        extreme_cluster_idx: Indices of the extreme clusters.
        extreme_periods: The preserved extreme periods (for transfer).
        rescale_deviations: Per-column residual deviations after rescaling.
        segmented_df: Segmented typical periods, if segmentation ran.
        predicted_segmented_df: Segment profiles expanded back to full period
            length, if segmentation ran.
        segment_center_indices: Segment center indices, if segmentation ran.
        clustering_duration: Wall-clock time spent clustering.
    """

    normalized_typical_periods: pd.DataFrame
    cluster_order: np.ndarray
    cluster_counts: dict[int, float]
    cluster_center_indices: list[int] | None
    extreme_cluster_idx: list[int]
    extreme_periods: list[ExtremePeriod]
    rescale_deviations: dict[str, dict]
    segmented_df: pd.DataFrame | None
    predicted_segmented_df: pd.DataFrame | None
    segment_center_indices: list | None
    clustering_duration: float


@dataclass(frozen=True)
class PipelineResult:
    """Output of the final phase (`build_result`).

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
