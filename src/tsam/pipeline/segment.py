"""Thin wrapper around tsam.utils.segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsam.utils.segmentation import segmentation

if TYPE_CHECKING:
    import pandas as pd

    from tsam.config import SegmentConfig
    from tsam.pipeline.types import PredefParams


def segment_typical_periods(
    normalized_typical_periods: pd.DataFrame,
    n_timesteps_per_period: int,
    segments: SegmentConfig,
    representation_dict: dict | None,
    predef: PredefParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Merge adjacent timesteps within each typical period into fewer segments.

    Optional stage, configured by `SegmentConfig`.

    **Problem.** After clustering, each typical period still has the full
    temporal resolution (e.g. 24 hourly timesteps). Some optimization models
    need fewer timesteps.

    **Solution.** Within each typical period, adjacent timesteps with similar
    values are merged into segments of variable duration. With ``n_segments=8``
    each 24-hour period collapses to 8 segments. The same constrained
    agglomerative clustering machinery as the main clustering step is used;
    ``SegmentConfig.representation`` controls how segment values are computed
    (typically ``"mean"``).

    **Weights.** The ``normalized_typical_periods`` arriving here are
    unweighted; the orchestrator multiplies column weights into a copy before
    calling this function so high-weight columns influence segment boundaries,
    then strips the weights from the outputs before denormalization. Two
    DataFrames come back: a segmented one (for denormalization) and a predicted
    one (for reconstruction).

    Parameters
    ----------
    normalized_typical_periods
        The formatted typical periods to segment (weighted by the caller).
    n_timesteps_per_period
        Timesteps per period before segmentation.
    segments
        Segmentation configuration: ``n_segments`` and ``representation``.
    representation_dict
        Per-column representation overrides.
    predef
        Predefined segment structure, supplied on the transfer path.

    Returns
    -------
    tuple
        ``(segmented_df, predicted_segmented_df, segment_center_indices)``.

    See Also
    --------
    reconstruct : Expands the segmented periods back to a full series.
    denormalize : Converts the segmented periods to original units.
    """
    return segmentation(  # type: ignore[no-any-return]
        normalized_typical_periods,
        segments.n_segments,
        n_timesteps_per_period,
        representation_method=segments.representation,
        representation_dict=representation_dict,
        predef_segment_order=predef.segment_order if predef is not None else None,
        predef_segment_durations=predef.segment_durations
        if predef is not None
        else None,
        predef_segment_centers=predef.segment_centers if predef is not None else None,
    )
