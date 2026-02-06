"""Thin wrapper around tsam.utils.segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsam.utils.segmentation import segmentation

if TYPE_CHECKING:
    import pandas as pd


def segment_typical_periods(
    normalized_typical_periods: pd.DataFrame,
    n_segments: int,
    n_timesteps_per_period: int,
    representation_method: str | None,
    representation_dict: dict | None,
    predef_segment_order: list | None = None,
    predef_segment_durations: list | None = None,
    predef_segment_centers: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Segment typical periods into fewer timesteps.

    Replicates monolith lines 1253-1270.

    Returns (segmented_df, predicted_segmented_df, segment_center_indices).
    """
    return segmentation(  # type: ignore[no-any-return]
        normalized_typical_periods,
        n_segments,
        n_timesteps_per_period,
        representation_method=representation_method,
        representation_dict=representation_dict,
        distribution_period_wise=True,
        predef_segment_order=predef_segment_order,
        predef_segment_durations=predef_segment_durations,
        predef_segment_centers=predef_segment_centers,
    )
