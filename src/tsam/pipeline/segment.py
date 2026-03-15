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
    """Segment typical periods into fewer timesteps.

    Returns (segmented_df, predicted_segmented_df, segment_center_indices).
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
