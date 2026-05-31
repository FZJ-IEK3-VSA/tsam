"""Pipeline orchestrator — the four-phase aggregation flow.

`run_pipeline` threads the data through four phases, each a function with an
explicit input and output: `prepare_data`, `cluster_and_postprocess`,
`format_and_reconstruct`, `assemble_result`. The individual stages they call
live in the sibling modules (normalize, periods, clustering, extremes, rescale,
segment, accuracy).

These four phase functions are public so their docstrings are discoverable in
the API reference, but they are deliberately not exported from
``tsam.pipeline`` — only `run_pipeline` is. They are orchestration glue, not a
stable API to call directly.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsam.options import options
from tsam.pipeline.accuracy import reconstruct
from tsam.pipeline.clustering import (
    cluster_periods,
    cluster_sorted_periods,
    use_predefined_assignments,
)
from tsam.pipeline.extremes import add_extreme_periods
from tsam.pipeline.normalize import denormalize, normalize
from tsam.pipeline.periods import add_period_sum_features, unstack_to_periods
from tsam.pipeline.rescale import rescale_representatives
from tsam.pipeline.segment import segment_typical_periods
from tsam.pipeline.types import (
    ClusteringOutput,
    FormattedOutput,
    PipelineConfig,
    PipelineResult,
    PredefParams,  # noqa: F401 (re-exported)
    PreparedData,
)

if TYPE_CHECKING:
    from tsam.config import (
        Distribution,
        MinMaxMean,
    )

# Only the orchestration entry points are public API of this module. Setting
# __all__ keeps the imported stage functions (normalize, cluster_periods, …)
# from being re-documented here — they have their own API-reference pages.
__all__ = [
    "assemble_result",
    "cluster_and_postprocess",
    "format_and_reconstruct",
    "prepare_data",
    "run_pipeline",
]


def _count_occurrences(cluster_order: np.ndarray) -> dict[int, float]:
    """Count how many original periods each cluster represents.

    Returns float values because the partial-period adjustment can produce
    fractional counts downstream.
    """
    nums, counts = np.unique(cluster_order, return_counts=True)
    return {int(num): float(counts[ii]) for ii, num in enumerate(nums)}


def _representatives_to_dataframe(
    cluster_periods_list: list[np.ndarray],
    column_index: pd.MultiIndex,
) -> pd.DataFrame:
    """Reshape flat cluster period vectors into a MultiIndex DataFrame.

    Converts a list of 1-D arrays (one per cluster) into a DataFrame
    indexed by (PeriodNum, TimeStep) with the original column names.
    """
    df = (
        pd.concat(
            [pd.Series(s, index=column_index) for s in cluster_periods_list],
            axis=1,
        )
        .unstack("TimeStep")
        .T
    )
    assert isinstance(df, pd.DataFrame)
    return df


def _warn_if_out_of_bounds(
    typical_periods: pd.DataFrame,
    original_data: pd.DataFrame,
    tolerance: float,
) -> None:
    """Warn if aggregated values exceed original data bounds."""
    exceeds_max = typical_periods.max(axis=0) > original_data.max(axis=0)
    if exceeds_max.any():
        diff = typical_periods.max(axis=0) - original_data.max(axis=0)
        exceeding_diff = diff[exceeds_max]
        if exceeding_diff.max() > tolerance:
            warnings.warn(
                "At least one maximal value of the "
                + "aggregated time series exceeds the maximal value "
                + "the input time series for: "
                + f"{exceeding_diff.to_dict()}"
                + ". To silence the warning set the 'numerical_tolerance' to a higher value."
            )
    below_min = typical_periods.min(axis=0) < original_data.min(axis=0)
    if below_min.any():
        diff = original_data.min(axis=0) - typical_periods.min(axis=0)
        exceeding_diff = diff[below_min]
        if exceeding_diff.max() > tolerance:
            warnings.warn(
                "Something went wrong... At least one minimal value of the "
                + "aggregated time series exceeds the minimal value "
                + "the input time series for: "
                + f"{exceeding_diff.to_dict()}"
                + ". To silence the warning set the 'numerical_tolerance' to a higher value."
            )


def _apply_weights_df(
    df: pd.DataFrame, weights: dict[str, float] | None
) -> pd.DataFrame:
    """Multiply DataFrame columns by weights for segmentation.

    Segmentation boundaries are determined in weighted space so that
    high-weight columns have more influence on where segments fall.
    """
    if not weights:
        return df
    out = df.copy()
    for col, w in weights.items():
        if col in out.columns:
            out[col] *= w
    return pd.DataFrame(out)


def _remove_weights_df(
    df: pd.DataFrame, weights: dict[str, float] | None
) -> pd.DataFrame:
    """Divide out weights after segmentation to restore unweighted values."""
    if not weights:
        return df
    out = df.copy()
    for col, w in weights.items():
        if col in out.columns:
            out[col] /= w
    return pd.DataFrame(out)


def _build_weight_vector(
    columns: pd.Index,
    weights: dict[str, float] | None,
) -> np.ndarray | None:
    """Build a weight array aligned to *columns*, defaulting unlisted columns to 1.0.

    Returns ``None`` if all weights are 1.0 (no weighting needed).
    """
    if not weights:
        return None
    result: list[float] = []
    any_non_unit = False
    for col in columns:
        w = weights.get(col, 1.0)
        if w < options.min_weight:
            warnings.warn(
                f'weight of "{col}" set to the minimal tolerable weighting',
                stacklevel=2,
            )
            w = options.min_weight
        if w != 1.0:
            any_non_unit = True
        result.append(w)
    return np.array(result) if any_non_unit else None


def _build_representation_dict(
    columns: pd.Index,
    cluster_representation: str | Distribution | MinMaxMean | None,
) -> dict[str, str]:
    """Build the representation dict (mean/min/max per column) from config."""
    from tsam.config import MinMaxMean

    representation_dict: dict[str, str] = dict.fromkeys(columns, "mean")
    if isinstance(cluster_representation, MinMaxMean):
        for col in cluster_representation.max_columns:
            if col in representation_dict:
                representation_dict[col] = "max"
        for col in cluster_representation.min_columns:
            if col in representation_dict:
                representation_dict[col] = "min"
    return representation_dict


def prepare_data(
    data: pd.DataFrame,
    cfg: PipelineConfig,
) -> PreparedData:
    """Phase 1 — Prepare data: turn raw input into clustering candidates.

    Returns a [`PreparedData`][tsam.pipeline.types.PreparedData] carrying
    everything the later phases need. In order:

    - **Normalize** ([`normalize`][tsam.pipeline.normalize.normalize]) — scale
      every column to ``[0, 1]``.
    - **Unstack to periods**
      ([`unstack_to_periods`][tsam.pipeline.periods.unstack_to_periods]) —
      reshape the flat series into a ``(period × timestep-feature)`` matrix.
    - **Apply weights** *(optional, the ``weights`` argument)* — bake a
      per-column weight vector into a copy of the candidates by a vectorized
      multiply, so weights influence clustering distance only. A weighted
      profile DataFrame is kept for the extremes and segmentation stages.
    - **Add period-sum features** *(optional,
      [`add_period_sum_features`][tsam.pipeline.periods.add_period_sum_features])*
      — append per-period column sums as extra distance-only features.

    See Also
    --------
    cluster_and_postprocess : The phase that consumes these candidates.
    """
    cluster = cfg.cluster
    cluster_representation = cluster.get_representation()
    representation_dict = _build_representation_dict(
        data.columns, cluster_representation
    )
    original_column_order = list(data.columns)
    original_data = data.copy()

    # Normalize
    norm_data = normalize(data, cluster.scale_by_column_means)

    # Unstack to periods
    period_profiles = unstack_to_periods(norm_data.values, cfg.n_timesteps_per_period)
    candidates = period_profiles.profiles_dataframe.values

    # Apply weights directly to candidates
    weight_vector = _build_weight_vector(norm_data.values.columns, cluster.weights)
    weighted_profiles_df: pd.DataFrame | None = None
    if weight_vector is not None:
        weight_tile = np.repeat(weight_vector, period_profiles.n_timesteps_per_period)
        candidates = candidates * weight_tile
        # Keep a weighted DataFrame for extremes/segmentation (need column labels).
        wpdf = period_profiles.profiles_dataframe.copy()
        for col_name, w in zip(
            wpdf.columns.get_level_values(0).unique(),
            weight_vector,
        ):
            wpdf[col_name] *= w
        weighted_profiles_df = wpdf

    # Add period sum features if requested
    # Period sums are extra columns appended for clustering distance only;
    # they must NOT reach representations() which expects original columns.
    n_feature_cols = candidates.shape[1]
    if cluster.include_period_sums:
        candidates, _n_extra = add_period_sum_features(
            period_profiles.profiles_dataframe, candidates
        )

    return PreparedData(
        norm_data=norm_data,
        period_profiles=period_profiles,
        candidates=candidates,
        representation_dict=representation_dict,
        n_feature_cols=n_feature_cols,
        original_column_order=original_column_order,
        original_data=original_data,
        weight_vector=weight_vector,
        weighted_profiles_df=weighted_profiles_df,
    )


def cluster_and_postprocess(
    prepared: PreparedData,
    cfg: PipelineConfig,
    data_length: int,
) -> ClusteringOutput:
    """Phase 2 — Cluster & post-process: group periods and finalize centers.

    Returns a [`ClusteringOutput`][tsam.pipeline.types.ClusteringOutput] with
    the final representatives, the per-period assignment, and the occurrence
    counts. In order:

    - **Cluster** ([`cluster_periods`][tsam.pipeline.clustering.cluster_periods],
      or [`cluster_sorted_periods`][tsam.pipeline.clustering.cluster_sorted_periods]
      for duration curves, or
      [`use_predefined_assignments`][tsam.pipeline.clustering.use_predefined_assignments]
      on the transfer path) — group periods and pick a representative for each.
    - **Add extremes** *(optional,
      [`add_extreme_periods`][tsam.pipeline.extremes.add_extreme_periods])* —
      inject extreme-value periods so peaks/troughs survive.
    - **Trim · unweight · count** — strip the period-sum features, divide
      weights back out of every representative (so downstream stages see
      unweighted data), count how many original periods fall in each cluster,
      and correct the padded last period's weight.
    - **Rescale** *(optional,
      [`rescale_representatives`][tsam.pipeline.rescale.rescale_representatives])*
      — scale non-extreme centers so their occurrence-weighted means match the
      original totals.

    See Also
    --------
    prepare_data : The phase that produces the candidates clustered here.
    format_and_reconstruct : The phase that consumes these representatives.
    """
    cluster = cfg.cluster
    cluster_representation = cluster.get_representation()
    candidates = prepared.candidates
    period_profiles = prepared.period_profiles

    # Cluster
    clustering_duration = 0.0
    cluster_center_indices: list[int] | None = None

    if cfg.predef is not None:
        cluster_centers, cluster_center_indices, cluster_order = (
            use_predefined_assignments(
                candidates,
                cfg.predef,
                cluster_representation,
                prepared.representation_dict,
                cfg.n_timesteps_per_period,
            )
        )
    else:
        t_start = time.time()
        # When period-sum features are appended, representations must run
        # on the non-augmented prefix so period-sum columns don't leak in.
        rep_candidates: np.ndarray | None = None
        if candidates.shape[1] != prepared.n_feature_cols:
            rep_candidates = candidates[:, : prepared.n_feature_cols]

        if not cluster.use_duration_curves:
            cluster_centers, cluster_center_indices, cluster_order = cluster_periods(
                candidates,
                cfg.n_clusters,
                cluster,
                prepared.representation_dict,
                cfg.n_timesteps_per_period,
                representation_candidates=rep_candidates,
            )
        else:
            cluster_centers, cluster_center_indices, cluster_order = (
                cluster_sorted_periods(
                    candidates,
                    period_profiles.n_columns,
                    cfg.n_clusters,
                    cluster,
                    prepared.representation_dict,
                    cfg.n_timesteps_per_period,
                )
            )
        clustering_duration = time.time() - t_start

    # Ensure cluster_order is always np.ndarray
    cluster_order = np.asarray(cluster_order)

    # Trim eval features from representatives (still weighted)
    cluster_periods_list: list[np.ndarray] = [
        center[: prepared.n_feature_cols] for center in cluster_centers
    ]

    # Add extreme periods if configured
    # Extremes run in weighted space (matching develop): weighted profiles
    # determine which period is extreme, and extracted profiles carry weights.
    # Unweighting happens after, so all centers are treated uniformly.
    extreme_periods_info: dict[str, dict] = {}
    extreme_cluster_idx: list[int] = []

    if cfg.extremes is not None:
        profiles_for_extremes = (
            prepared.weighted_profiles_df
            if prepared.weighted_profiles_df is not None
            else period_profiles.profiles_dataframe
        )
        (
            cluster_periods_list,
            cluster_order,
            extreme_cluster_idx,
            extreme_periods_info,
        ) = add_extreme_periods(
            profiles_for_extremes,
            cluster_periods_list,
            cluster_order,
            cfg.extremes,
        )
        cluster_order = np.asarray(cluster_order)
    else:
        if cfg.predef is not None and cfg.predef.extreme_cluster_idx is not None:
            extreme_cluster_idx = list(cfg.predef.extreme_cluster_idx)

    # Unweight all representatives (regular + extreme) — remove weights
    # before downstream steps (rescale, denorm) which expect unweighted data.
    if prepared.weight_vector is not None:
        inv_tile = np.repeat(1.0 / prepared.weight_vector, cfg.n_timesteps_per_period)
        cluster_periods_list = [center * inv_tile for center in cluster_periods_list]

    # Compute cluster counts
    cluster_counts = _count_occurrences(cluster_order)

    # Rescale if requested
    rescale_deviations: dict[str, dict] = {}
    rescale_exclude = cfg.rescale_exclude_columns or []
    if cfg.rescale_cluster_periods:
        cluster_periods_list, rescale_deviations = rescale_representatives(  # type: ignore[assignment]
            cluster_periods_list,
            cluster_counts,
            extreme_cluster_idx,
            period_profiles.profiles_dataframe,
            prepared.original_data,
            prepared.norm_data.scale_by_column_means,
            cfg.n_timesteps_per_period,
            rescale_exclude,
        )
        cluster_periods_list = list(cluster_periods_list)

    # Adjust for partial periods
    if data_length % cfg.n_timesteps_per_period != 0:
        last_cluster = int(cluster_order[-1])
        cluster_counts[last_cluster] -= (
            1
            - float(data_length % cfg.n_timesteps_per_period)
            / cfg.n_timesteps_per_period
        )

    return ClusteringOutput(
        cluster_periods_list=cluster_periods_list,
        cluster_order=cluster_order,
        cluster_counts=cluster_counts,
        cluster_center_indices=cluster_center_indices,
        extreme_cluster_idx=extreme_cluster_idx,
        extreme_periods_info=extreme_periods_info,
        clustering_duration=clustering_duration,
        rescale_deviations=rescale_deviations,
    )


def format_and_reconstruct(
    prepared: PreparedData,
    clustered: ClusteringOutput,
    cfg: PipelineConfig,
) -> FormattedOutput:
    """Phase 3 — Format & reconstruct: shape outputs and measure accuracy.

    Returns a [`FormattedOutput`][tsam.pipeline.types.FormattedOutput]. In order:

    - **Format representatives** — reshape the flat center vectors into a
      ``(PeriodNum, TimeStep)`` MultiIndex DataFrame.
    - **Segment** *(optional,
      [`segment_typical_periods`][tsam.pipeline.segment.segment_typical_periods])*
      — merge adjacent timesteps within each period into fewer segments. Runs
      in weighted space; weights are removed from the outputs afterwards.
    - **Denormalize** ([`denormalize`][tsam.pipeline.normalize.denormalize]) —
      convert the representatives back to the user's original units.
    - **Reconstruct + accuracy**
      ([`reconstruct`][tsam.pipeline.accuracy.reconstruct],
      [`compute_accuracy`][tsam.pipeline.accuracy.compute_accuracy]) — after a
      bounds check warns about out-of-range values, expand the typical periods
      back to a full-length series; accuracy is computed lazily on the result.

    See Also
    --------
    cluster_and_postprocess : The phase that produces the representatives.
    assemble_result : The phase that packs these outputs into the result.
    """
    norm_data = prepared.norm_data
    period_profiles = prepared.period_profiles

    # Format representatives to MultiIndex DataFrame
    normalized_typical_periods = _representatives_to_dataframe(
        clustered.cluster_periods_list, period_profiles.column_index
    )

    # Segmentation if configured
    segmented_df = None
    predicted_segmented_df = None
    segment_center_indices = None

    if cfg.segments is not None:
        # Segmentation runs in weighted space so that high-weight columns
        # have more influence on segment boundaries. Weights are removed
        # from the output before denormalization.
        weights = cfg.cluster.weights
        segmentation_input = _apply_weights_df(normalized_typical_periods, weights)
        segmented_df, predicted_segmented_df, segment_center_indices = (
            segment_typical_periods(
                segmentation_input,
                cfg.n_timesteps_per_period,
                cfg.segments,
                prepared.representation_dict,
                cfg.predef,
            )
        )
        segmented_df = _remove_weights_df(segmented_df, weights)
        predicted_segmented_df = _remove_weights_df(predicted_segmented_df, weights)
        segmented_normalized = segmented_df.reset_index(level=3, drop=True)
        denorm_source = segmented_normalized
        reconstruct_source = predicted_segmented_df
    else:
        denorm_source = normalized_typical_periods
        reconstruct_source = normalized_typical_periods

    # Denormalize -> typical_periods
    typical_periods = denormalize(denorm_source, norm_data)
    if cfg.round_decimals is not None:
        typical_periods = typical_periods.round(decimals=cfg.round_decimals)

    # Bounds check + warnings
    _warn_if_out_of_bounds(
        typical_periods, prepared.original_data, cfg.numerical_tolerance
    )

    # Reconstruct + compute accuracy
    reconstructed_data, normalized_predicted = reconstruct(
        reconstruct_source,
        clustered.cluster_order,
        period_profiles,
        norm_data,
        prepared.original_data,
    )
    if cfg.round_decimals is not None:
        reconstructed_data = reconstructed_data.round(decimals=cfg.round_decimals)

    # Restore original column order
    typical_periods = typical_periods[prepared.original_column_order]
    reconstructed_data = reconstructed_data[prepared.original_column_order]

    return FormattedOutput(
        typical_periods=typical_periods,
        reconstructed_data=reconstructed_data,
        normalized_predicted=normalized_predicted,
        segmented_df=segmented_df,
        segment_center_indices=segment_center_indices,
    )


def assemble_result(
    prepared: PreparedData,
    clustered: ClusteringOutput,
    formatted: FormattedOutput,
    cfg: PipelineConfig,
) -> PipelineResult:
    """Phase 4 — Assemble: pack everything into the pipeline result.

    Builds a serializable, transferable
    [`ClusteringResult`][tsam.config.ClusteringResult] from the cluster order,
    center indices, extremes, and segmentation, then packs it with the typical
    periods, counts, reconstructed series, and metadata into a
    [`PipelineResult`][tsam.pipeline.types.PipelineResult] — the single handoff
    to `tsam.api`, which wraps it as the user-facing
    [`AggregationResult`][tsam.result.AggregationResult].

    See Also
    --------
    format_and_reconstruct : The phase that produces the outputs packed here.
    """
    from tsam.config import ClusteringResult as _ClusteringResult

    original_data_out = prepared.original_data[prepared.original_column_order]

    input_time_index = (
        original_data_out.index
        if isinstance(original_data_out.index, pd.DatetimeIndex)
        else None
    )

    clustering_result = _ClusteringResult.from_pipeline(
        cluster_center_indices=clustered.cluster_center_indices,
        extreme_periods_info=clustered.extreme_periods_info,
        extremes_config=cfg.extremes,
        cluster_order=clustered.cluster_order,
        segmented_df=formatted.segmented_df,
        segment_center_indices=formatted.segment_center_indices,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        temporal_resolution=cfg.temporal_resolution,
        original_data=original_data_out,
        cluster_config=cfg.cluster,
        segment_config=cfg.segments,
        rescale_cluster_periods=cfg.rescale_cluster_periods,
        rescale_exclude_columns=cfg.rescale_exclude_columns or [],
        extreme_cluster_idx=clustered.extreme_cluster_idx,
        time_index=input_time_index,
    )

    return PipelineResult(
        typical_periods=formatted.typical_periods,
        cluster_counts=clustered.cluster_counts,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        time_index=prepared.period_profiles.time_index,
        original_data=original_data_out,
        clustering_duration=clustered.clustering_duration,
        rescale_deviations=clustered.rescale_deviations,
        segmented_df=formatted.segmented_df,
        reconstructed_data=formatted.reconstructed_data,
        _norm_values=prepared.norm_data.values,
        _normalized_predicted=formatted.normalized_predicted,
        clustering_result=clustering_result,
    )


def run_pipeline(
    data: pd.DataFrame,
    cfg: PipelineConfig,
) -> PipelineResult:
    """Run the full aggregation pipeline in four phases.

    The single orchestration entry point behind both `tsam.aggregate` and
    ``ClusteringResult.apply()``. It threads the data through four phases, each
    a pure function with an explicit input and output:

    1. [`prepare_data`][tsam.pipeline.orchestrator.prepare_data] — normalize,
       unstack, weight, augment.
    2. [`cluster_and_postprocess`][tsam.pipeline.orchestrator.cluster_and_postprocess]
       — cluster, add extremes, count, rescale.
    3. [`format_and_reconstruct`][tsam.pipeline.orchestrator.format_and_reconstruct]
       — format, segment, denormalize, reconstruct.
    4. [`assemble_result`][tsam.pipeline.orchestrator.assemble_result] — build
       the `ClusteringResult` and `PipelineResult`.

    Replaces the v3 ``create_typical_periods()`` + ``predict_original_data()`` +
    ``accuracy_indicators()`` trio.

    Parameters
    ----------
    data
        Input time series with a datetime index, one column per attribute.
    cfg
        Fully resolved `PipelineConfig` (clustering, extremes, segmentation,
        rescaling, and predefined-assignment settings).

    Returns
    -------
    PipelineResult
        The internal result handed to `tsam.api` for wrapping as an
        `AggregationResult`.
    """
    prepared = prepare_data(data, cfg)

    clustered = cluster_and_postprocess(
        prepared,
        cfg,
        data_length=len(data),
    )

    formatted = format_and_reconstruct(
        prepared,
        clustered,
        cfg,
    )

    return assemble_result(
        prepared,
        clustered,
        formatted,
        cfg,
    )
