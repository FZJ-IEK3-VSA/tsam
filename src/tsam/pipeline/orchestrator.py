"""Pipeline orchestrator — the four-phase aggregation flow.

`run_pipeline` threads the data through four phases, each a function with an
explicit input and output: `prepare_data`, `cluster_candidates`,
`refine_representatives`, `build_result`. The individual stages they call live
in the sibling modules (normalize, periods, clustering, extremes, rescale,
segmentation, accuracy).

The phases are drawn along what a caller can switch off: preparing and
clustering always happen, every optional adjustment lives in
`refine_representatives`, and `build_result` only expresses the outcome.
"""

from __future__ import annotations

import time
import warnings
from typing import cast

import numpy as np
import pandas as pd

from tsam.config import Distribution, MinMaxMean
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
from tsam.pipeline.segmentation import segment_typical_periods
from tsam.pipeline.types import (
    ClusterAssignment,
    ExtremePeriod,
    PipelineConfig,
    PipelineResult,
    PredefParams,  # noqa: F401 (re-exported)
    PreparedData,
    RefinedRepresentatives,
)

# Only the orchestration entry points are public API of this module. Setting
# __all__ keeps the imported stage functions (normalize, cluster_periods, …)
# from being re-documented here — they have their own API-reference pages.
__all__ = [
    "build_result",
    "cluster_candidates",
    "prepare_data",
    "refine_representatives",
    "run_pipeline",
]


def _count_occurrences(cluster_order: np.ndarray, n_clusters: int) -> dict[int, float]:
    """Count how many original periods each cluster represents.

    The result is total over the cluster ids: every id in ``range(n_clusters)``
    gets an entry, and a cluster that ended up representing no periods is
    counted 0 rather than left out. Cluster ids are positions downstream —
    representatives are looked up by id — so no consumer should have to tell
    "absent" apart from "zero". The counts are floats because the partial-period
    adjustment can produce fractional values.

    Args:
        cluster_order: Per-period cluster assignment.
        n_clusters: Size of the id space, i.e. the number of representatives.
            Every label in *cluster_order* must fall inside it.

    Returns:
        Occurrence count per cluster id, one entry per id in
        ``range(n_clusters)``.

    Raises:
        ValueError: If *cluster_order* holds a label outside the id space.
            Labels and representative positions have then gone out of step, and
            counting them here would silently drop a cluster's periods.
    """
    nums, counts = np.unique(cluster_order, return_counts=True)
    if nums.size and (nums[0] < 0 or nums[-1] >= n_clusters):
        outside = [int(num) for num in nums if num < 0 or num >= n_clusters]
        raise ValueError(
            f"cluster_order holds labels outside range(0, {n_clusters}): "
            f"{outside}. Cluster ids index the representatives, so a label "
            f"without a representative cannot be counted."
        )
    occurrences = dict.fromkeys(range(n_clusters), 0.0)
    occurrences.update({int(num): float(counts[ii]) for ii, num in enumerate(nums)})
    return occurrences


def _drop_empty_clusters(
    cluster_periods_list: list[np.ndarray],
    cluster_order: np.ndarray,
    extreme_cluster_idx: list[int],
    cluster_center_indices: list[int] | None,
) -> tuple[list[np.ndarray], np.ndarray, list[int], list[int] | None]:
    """Remove clusters that ended up representing no periods, and close the gap.

    A hole in the label space can open at either end of this phase.
    ``method="new_cluster"`` can absorb every member of a regular cluster; on
    the transfer path a stored assignment can arrive with a hole already in it.
    Cluster ids are positions downstream — representatives are looked up by id,
    and `representations` returns one center per *observed* label — so a hole
    misaligns every cluster above it. Nothing is lost by dropping it: a cluster
    with no periods contributes nothing to the reconstruction, though the run
    then returns fewer typical periods than ``n_clusters`` asked for.

    Clustering itself is not a source of holes here: `assign_clusters` already
    hands back a dense label space, so what this sees is what extremes did to
    it.

    Returns:
        The inputs with empty clusters removed and every id renumbered, or the
        inputs unchanged when no cluster is empty.
    """
    n_clusters = len(cluster_periods_list)
    kept_ids = sorted({int(label) for label in np.asarray(cluster_order).ravel()})
    if kept_ids and (kept_ids[0] < 0 or kept_ids[-1] >= n_clusters):
        outside = [label for label in kept_ids if label < 0 or label >= n_clusters]
        raise ValueError(
            f"cluster_order holds labels outside range(0, {n_clusters}): "
            f"{outside}. Every label names a representative by position, so a "
            f"label without one can be neither kept nor dropped."
        )
    if kept_ids == list(range(n_clusters)):
        return (
            cluster_periods_list,
            cluster_order,
            extreme_cluster_idx,
            cluster_center_indices,
        )

    renumbered = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}

    return (
        [cluster_periods_list[old_id] for old_id in kept_ids],
        np.array(
            [renumbered[int(label)] for label in np.asarray(cluster_order).ravel()],
            dtype=int,
        ),
        [renumbered[old_id] for old_id in extreme_cluster_idx],
        # Center indices cover only the clusters clustering produced; the
        # extremes get theirs appended later and always hold their own period.
        None
        if cluster_center_indices is None
        else [
            cluster_center_indices[old_id]
            for old_id in kept_ids
            if old_id < len(cluster_center_indices)
        ],
    )


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
    return cast("pd.DataFrame", pd.DataFrame(out))


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
    return cast("pd.DataFrame", pd.DataFrame(out))


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
    """Build the representation dict (mean/min/max per column) from config.

    Columns the configuration does not mention keep their ``"mean"`` default.
    A `MinMaxMean` cannot list the same column as both min and max — that is
    rejected when the configuration is built — so the two loops below can never
    disagree about a column.

    Raises:
        ValueError: If the configuration names a column that is not in the data.
    """
    representation_dict: dict[str, str] = dict.fromkeys(columns, "mean")
    if isinstance(cluster_representation, MinMaxMean):
        unknown = [
            col
            for col in (
                *cluster_representation.max_columns,
                *cluster_representation.min_columns,
            )
            if col not in representation_dict
        ]
        if unknown:
            raise ValueError(
                f"MinMaxMean names columns {unknown} that are not in the data "
                f"{list(columns)}."
            )
        for col in cluster_representation.max_columns:
            representation_dict[col] = "max"
        for col in cluster_representation.min_columns:
            representation_dict[col] = "min"
    return representation_dict


def _resolve_reference_attribute_idx(
    columns: list,
    cluster_representation: str | Distribution | MinMaxMean | None,
) -> int | None:
    """Translate ``Distribution.reference_attribute`` into a block index.

    A candidate row is laid out as one contiguous block of timesteps per
    attribute, and the duration representation addresses those blocks by
    position. The reference attribute is configured by name, so it has to be
    turned into the position of its block; doing it here keeps that layout
    detail out of the algorithm layer.

    Args:
        columns: Column names in the order of the candidates' attribute blocks.
        cluster_representation: The resolved representation. Only a
            `Distribution` can name a reference attribute.

    Returns:
        The position of the reference attribute's block, or ``None`` if there
        is no reference attribute to translate.

    Raises:
        ValueError: If the named reference attribute is not a data column.
    """
    if not isinstance(cluster_representation, Distribution):
        return None

    reference = cluster_representation.reference_attribute
    if reference is None:
        return None

    if reference not in columns:
        raise ValueError(
            f"reference_attribute {reference!r} is not one of the data columns "
            f"{columns}."
        )
    return columns.index(reference)


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
      reshape the flat series into a ``(period x timestep-feature)`` matrix.
    - **Apply weights** *(optional, the ``weights`` argument)* — bake a
      per-column weight vector into a copy of the candidates by a vectorized
      multiply, so weights influence clustering distance only. A weighted
      profile DataFrame is kept for the extremes and segmentation stages.
    - **Add period-sum features** *(optional,
      [`add_period_sum_features`][tsam.pipeline.periods.add_period_sum_features])*
      — append per-period column sums as extra distance-only features.

    Note:
        The candidates produced here are consumed by ``cluster_candidates``.
    """
    cluster = cfg.cluster
    cluster_representation = cluster.get_representation()
    representation_dict = _build_representation_dict(
        data.columns, cluster_representation
    )
    # Segmentation is configured independently of clustering, so it gets its
    # own dict rather than borrowing the cluster stage's.
    segment_representation_dict = (
        _build_representation_dict(data.columns, cfg.segments.representation)
        if cfg.segments is not None
        else None
    )
    original_column_order = list(data.columns)
    original_data = data.copy()

    # Normalize
    norm_data = normalize(data, cluster.scale_by_column_means)

    # Unstack to periods
    period_profiles = unstack_to_periods(norm_data.values, cfg.n_timesteps_per_period)
    candidates = period_profiles.profiles_dataframe.values

    # Apply weights directly to candidates
    weight_vector = _build_weight_vector(norm_data.values.columns, cfg.weights)
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
    # They are summed from the *weighted* profiles so that a column's sum
    # feature carries the same weight as its timestep features — summing the
    # unweighted profiles would make a column's period sum count for relatively
    # less the higher its weight.
    n_feature_cols = candidates.shape[1]
    if cluster.include_period_sums:
        candidates = add_period_sum_features(
            weighted_profiles_df
            if weighted_profiles_df is not None
            else period_profiles.profiles_dataframe,
            candidates,
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
        segment_representation_dict=segment_representation_dict,
    )


def cluster_candidates(
    prepared: PreparedData,
    cfg: PipelineConfig,
) -> ClusterAssignment:
    """Phase 2 — Cluster: group the candidate periods.

    Returns a [`ClusterAssignment`][tsam.pipeline.types.ClusterAssignment]:
    which cluster each period belongs to, and one representative per cluster.
    This phase does nothing optional — it is the one step every aggregation
    takes. One of three paths runs:

    - [`cluster_periods`][tsam.pipeline.clustering.cluster_periods] — the
      default: group by temporal shape and apply the configured representation.
    - [`cluster_sorted_periods`][tsam.pipeline.clustering.cluster_sorted_periods]
      — when ``ClusterConfig.use_duration_curves`` is set: group by value
      distribution instead.
    - [`use_predefined_assignments`][tsam.pipeline.clustering.use_predefined_assignments]
      — the transfer path, replaying a stored clustering on new data.

    Any period-sum features appended for the clustering distance are trimmed
    off the representatives afterwards. The representatives stay in weighted
    space: extreme detection in the next phase needs them that way, and it
    unweights everything in one go once the set of representatives is final.

    Note:
        The candidates come from ``prepare_data``; the representatives are
        adjusted by ``refine_representatives``.
    """
    cluster = cfg.cluster
    cluster_representation = cluster.get_representation()
    candidates = prepared.candidates
    period_profiles = prepared.period_profiles
    reference_attribute_idx = _resolve_reference_attribute_idx(
        prepared.attribute_columns, cluster_representation
    )

    t_start = time.time()

    if cfg.predef is not None:
        cluster_centers, cluster_center_indices, cluster_order = (
            use_predefined_assignments(
                candidates,
                cfg.predef,
                cluster,
                prepared.representation_dict,
                cfg.n_timesteps_per_period,
                reference_attribute_idx=reference_attribute_idx,
            )
        )
    else:
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
                reference_attribute_idx=reference_attribute_idx,
            )
        else:
            cluster_centers, cluster_center_indices, cluster_order = (
                cluster_sorted_periods(
                    # Never the augmented matrix: this path reshapes its input
                    # per column, and the period-sum block is not made of
                    # timesteps, so including it shifts every column's block.
                    rep_candidates if rep_candidates is not None else candidates,
                    period_profiles.n_columns,
                    cfg.n_timesteps_per_period,
                    cfg.n_clusters,
                    cluster,
                )
            )

    clustering_duration = time.time() - t_start

    # Ensure cluster_order is always np.ndarray
    cluster_order = np.asarray(cluster_order)

    # Trim eval features from representatives (still weighted)
    cluster_periods_list: list[np.ndarray] = [
        center[: prepared.n_feature_cols] for center in cluster_centers
    ]

    # Clustering does not promise to fill every cluster it was asked for: with
    # fewer distinct period shapes than clusters, some methods leave one empty.
    # `assign_clusters` drops those so ids stay usable as positions, which makes
    # the shortfall invisible unless it is said out loud.
    if cfg.predef is None and len(cluster_periods_list) < cfg.n_clusters:
        warnings.warn(
            f"Clustering produced {len(cluster_periods_list)} non-empty clusters "
            f"for the requested n_clusters={cfg.n_clusters}: "
            f"'{cluster.method}' found fewer distinct period shapes than "
            f"clusters, and empty clusters are dropped rather than kept as "
            f"typical periods representing no periods. The aggregation is valid "
            f"but returns fewer typical periods than requested."
        )

    return ClusterAssignment(
        cluster_periods_list=cluster_periods_list,
        cluster_order=cluster_order,
        cluster_center_indices=cluster_center_indices,
        clustering_duration=clustering_duration,
    )


def refine_representatives(
    prepared: PreparedData,
    assignment: ClusterAssignment,
    cfg: PipelineConfig,
    data_length: int,
) -> RefinedRepresentatives:
    """Phase 3 — Refine: apply the optional adjustments to the representatives.

    Returns a
    [`RefinedRepresentatives`][tsam.pipeline.types.RefinedRepresentatives] —
    the final typical periods, still normalized. Every stage that can change
    *which* periods are represented or *what* they contain lives here, and each
    one is optional; with no extremes, no rescaling and no segmentation this
    phase only unweights and counts. In order:

    - **Add extremes** *(optional,
      [`add_extreme_periods`][tsam.pipeline.extremes.add_extreme_periods])* —
      inject extreme-value periods so peaks/troughs survive the averaging.
      Runs before unweighting, in the weighted space the previous phase left
      the representatives in.
    - **Unweight · count** — divide the weights back out of every
      representative, count how many original periods each cluster stands for,
      and reduce the last count if the final period was padded.
    - **Rescale** *(optional,
      [`rescale_representatives`][tsam.pipeline.rescale.rescale_representatives])*
      — scale non-extreme centers so their occurrence-weighted means match the
      original totals.
    - **Segment** *(optional,
      [`segment_typical_periods`][tsam.pipeline.segmentation.segment_typical_periods])*
      — merge adjacent timesteps within each period into fewer segments. Runs
      in weighted space again, so that high-weight columns have more say in
      where the segment boundaries fall; the weights come back out afterwards.

    Note:
        The representatives come from ``cluster_candidates``; ``build_result``
        turns the refined set into the user-facing result.
    """
    period_profiles = prepared.period_profiles
    cluster_periods_list = assignment.cluster_periods_list
    cluster_order = assignment.cluster_order
    cluster_center_indices = assignment.cluster_center_indices

    # Add extreme periods if configured
    # Extremes run in weighted space (matching develop): weighted profiles
    # determine which period is extreme, and extracted profiles carry weights.
    # Unweighting happens after, so all centers are treated uniformly.
    extreme_periods: list[ExtremePeriod] = []
    extreme_cluster_idx: list[int] = []

    if cfg.extremes is not None:
        profiles_for_extremes = (
            prepared.weighted_profiles_df
            if prepared.weighted_profiles_df is not None
            else period_profiles.profiles_dataframe
        )
        (
            cluster_periods_list,
            extended_order,
            extreme_cluster_idx,
            extreme_periods,
        ) = add_extreme_periods(
            profiles_for_extremes,
            cluster_periods_list,
            cluster_order,
            cfg.extremes,
        )
        cluster_order = np.asarray(extended_order)
    else:
        if cfg.predef is not None and cfg.predef.extreme_cluster_idx is not None:
            extreme_cluster_idx = list(cfg.predef.extreme_cluster_idx)

    # Unconditional: extremes are one way to empty a cluster, not the only one,
    # and every id below indexes into cluster_periods_list.
    (
        cluster_periods_list,
        cluster_order,
        extreme_cluster_idx,
        cluster_center_indices,
    ) = _drop_empty_clusters(
        cluster_periods_list,
        cluster_order,
        extreme_cluster_idx,
        cluster_center_indices,
    )

    # Unweight all representatives (regular + extreme) — remove weights
    # before downstream steps (rescale, denorm) which expect unweighted data.
    if prepared.weight_vector is not None:
        inv_tile = np.repeat(1.0 / prepared.weight_vector, cfg.n_timesteps_per_period)
        cluster_periods_list = [center * inv_tile for center in cluster_periods_list]

    # Compute cluster counts
    cluster_counts = _count_occurrences(cluster_order, len(cluster_periods_list))

    # Rescale if requested
    rescale_deviations: dict[str, dict] = {}
    rescale_exclude = cfg.rescale_exclude_columns or []
    if cfg.rescale_cluster_periods:
        cluster_periods_list, rescale_deviations = rescale_representatives(  # type: ignore[assignment]
            cluster_periods=cluster_periods_list,
            cluster_period_no_occur=cluster_counts,
            extreme_cluster_idx=extreme_cluster_idx,
            profiles_df=period_profiles.profiles_dataframe,
            original_data=prepared.original_data,
            normalize_column_means=prepared.norm_data.scale_by_column_means,
            n_timesteps_per_period=cfg.n_timesteps_per_period,
            exclude_columns=rescale_exclude,
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

    # Format representatives to MultiIndex DataFrame
    normalized_typical_periods = _representatives_to_dataframe(
        cluster_periods_list, period_profiles.column_index
    )

    # Segmentation if configured
    segmented_df = None
    predicted_segmented_df = None
    segment_center_indices = None

    if cfg.segments is not None:
        # Segmentation runs in weighted space so that high-weight columns
        # have more influence on segment boundaries. Weights are removed
        # from the output before denormalization.
        weights = cfg.weights
        segmentation_input = _apply_weights_df(normalized_typical_periods, weights)
        segmented_df, predicted_segmented_df, segment_center_indices = (
            segment_typical_periods(
                segmentation_input,
                cfg.n_timesteps_per_period,
                cfg.segments,
                prepared.segment_representation_dict,
                cfg.predef,
            )
        )
        segmented_df = _remove_weights_df(segmented_df, weights)
        predicted_segmented_df = _remove_weights_df(predicted_segmented_df, weights)

    return RefinedRepresentatives(
        normalized_typical_periods=normalized_typical_periods,
        cluster_order=cluster_order,
        cluster_counts=cluster_counts,
        cluster_center_indices=cluster_center_indices,
        extreme_cluster_idx=extreme_cluster_idx,
        extreme_periods=extreme_periods,
        rescale_deviations=rescale_deviations,
        segmented_df=segmented_df,
        predicted_segmented_df=predicted_segmented_df,
        segment_center_indices=segment_center_indices,
        clustering_duration=assignment.clustering_duration,
    )


def build_result(
    prepared: PreparedData,
    refined: RefinedRepresentatives,
    cfg: PipelineConfig,
) -> PipelineResult:
    """Phase 4 — Build the result: express the representatives and pack them up.

    Returns the [`PipelineResult`][tsam.pipeline.types.PipelineResult], the
    single handoff to `tsam.api`, which wraps it as the user-facing
    [`AggregationResult`][tsam.result.AggregationResult]. Nothing here changes
    the aggregation any more; it only expresses the refined representatives and
    packages them. In order:

    - **Denormalize** ([`denormalize`][tsam.pipeline.normalize.denormalize]) —
      convert the representatives back to the user's original units.
    - **Reconstruct + accuracy**
      ([`reconstruct`][tsam.pipeline.accuracy.reconstruct],
      [`compute_accuracy`][tsam.pipeline.accuracy.compute_accuracy]) — after a
      bounds check warns about out-of-range values, expand the typical periods
      back to a full-length series; accuracy is computed lazily on the result.
    - **Assemble** — build the serializable, transferable
      [`ClusteringResult`][tsam.result.ClusteringResult] from the cluster order,
      center indices, extremes and segmentation, and pack it together with the
      typical periods, counts, reconstructed series and metadata.

    Note:
        The representatives come from ``refine_representatives``.
    """
    from tsam.result import ClusteringResult as _ClusteringResult

    norm_data = prepared.norm_data
    period_profiles = prepared.period_profiles

    if refined.segmented_df is not None and refined.predicted_segmented_df is not None:
        denorm_source = refined.segmented_df.reset_index(level=3, drop=True)
        reconstruct_source = refined.predicted_segmented_df
    else:
        denorm_source = refined.normalized_typical_periods
        reconstruct_source = refined.normalized_typical_periods

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
        refined.cluster_order,
        period_profiles,
        norm_data,
        prepared.original_data,
    )
    if cfg.round_decimals is not None:
        reconstructed_data = reconstructed_data.round(decimals=cfg.round_decimals)

    # Restore original column order
    typical_periods = typical_periods[prepared.original_column_order]
    reconstructed_data = reconstructed_data[prepared.original_column_order]

    original_data_out = prepared.original_data[prepared.original_column_order]

    input_time_index = (
        original_data_out.index
        if isinstance(original_data_out.index, pd.DatetimeIndex)
        else None
    )

    clustering_result = _ClusteringResult.from_pipeline(
        cluster_center_indices=refined.cluster_center_indices,
        extreme_periods=refined.extreme_periods,
        extremes_config=cfg.extremes,
        cluster_order=refined.cluster_order,
        segmented_df=refined.segmented_df,
        segment_center_indices=refined.segment_center_indices,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        temporal_resolution=cfg.temporal_resolution,
        original_data=original_data_out,
        cluster_config=cfg.cluster,
        segment_config=cfg.segments,
        rescale_cluster_periods=cfg.rescale_cluster_periods,
        rescale_exclude_columns=cfg.rescale_exclude_columns or [],
        extreme_cluster_idx=refined.extreme_cluster_idx,
        weights=cfg.weights,
        time_index=input_time_index,
    )

    return PipelineResult(
        typical_periods=typical_periods,
        cluster_counts=refined.cluster_counts,
        n_timesteps_per_period=cfg.n_timesteps_per_period,
        time_index=prepared.period_profiles.time_index,
        original_data=original_data_out,
        clustering_duration=refined.clustering_duration,
        rescale_deviations=refined.rescale_deviations,
        segmented_df=refined.segmented_df,
        reconstructed_data=reconstructed_data,
        _norm_values=prepared.norm_data.values,
        _normalized_predicted=normalized_predicted,
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
    2. [`cluster_candidates`][tsam.pipeline.orchestrator.cluster_candidates] —
       group the periods and pick a representative for each.
    3. [`refine_representatives`][tsam.pipeline.orchestrator.refine_representatives]
       — the optional adjustments: extremes, rescaling, segmentation.
    4. [`build_result`][tsam.pipeline.orchestrator.build_result] — denormalize,
       reconstruct, and pack the `ClusteringResult` and `PipelineResult`.

    Phases 2 and 3 are split along what is mandatory and what is not: phase 2
    is the one step every aggregation takes, phase 4 only expresses what phase
    3 settled on. Everything a caller can switch on or off sits in phase 3.

    Replaces the v3 ``create_typical_periods()`` + ``predict_original_data()`` +
    ``accuracy_indicators()`` trio.

    Args:
        data: Input time series with a datetime index, one column per attribute.
        cfg: Fully resolved `PipelineConfig` (clustering, extremes, segmentation,
            rescaling, and predefined-assignment settings).

    Returns:
        The internal result handed to `tsam.api` for wrapping as an
        `AggregationResult`.
    """
    prepared = prepare_data(data, cfg)

    assignment = cluster_candidates(prepared, cfg)

    refined = refine_representatives(
        prepared,
        assignment,
        cfg,
        data_length=len(data),
    )

    return build_result(prepared, refined, cfg)
