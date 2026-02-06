"""Thin wrapper around representations.representations()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsam.representations import representations

if TYPE_CHECKING:
    import numpy as np


def compute_representatives(
    candidates: np.ndarray,
    assignments: list | np.ndarray,
    representation_method: str | None,
    representation_dict: dict | None,
    distribution_period_wise: bool,
    n_timesteps_per_period: int,
) -> tuple[list, list | None]:
    """Compute representative periods from cluster assignments.

    Returns (cluster_centers, cluster_center_indices).
    """
    centers, center_indices = representations(
        candidates,
        assignments,
        default="medoid",
        representationMethod=representation_method,
        representationDict=representation_dict,
        distributionPeriodWise=distribution_period_wise,
        timeStepsPerPeriod=n_timesteps_per_period,
    )
    return centers, center_indices
