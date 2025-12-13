"""Configuration classes for tsam aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Type aliases for clarity
ClusterMethod = Literal[
    "averaging",
    "kmeans",
    "kmedoids",
    "kmedoids_exact",
    "kmaxoids",
    "hierarchical",
    "contiguous",
]

RepresentationMethod = Literal[
    "mean",
    "medoid",
    "maxoid",
    "duration",
    "distribution",
    "distribution_minmax",
    "minmax_mean",
]

ExtremeMethod = Literal[
    "append",
    "replace",
    "new_cluster",
]

Solver = Literal["highs", "cbc", "gurobi", "cplex"]


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for the clustering algorithm.

    Parameters
    ----------
    method : str, default "hierarchical"
        Clustering algorithm to use:
        - "averaging": Sequential averaging of periods
        - "kmeans": K-means clustering (fast, uses centroids)
        - "kmedoids": K-medoids using heuristic (uses actual periods)
        - "kmedoids_exact": K-medoids using MILP optimization (slower, optimal)
        - "kmaxoids": K-maxoids (selects most dissimilar periods)
        - "hierarchical": Agglomerative hierarchical clustering
        - "contiguous": Hierarchical with temporal contiguity constraint

    representation : str, optional
        How to represent cluster centers:
        - "mean": Centroid (average of cluster members)
        - "medoid": Actual period closest to centroid
        - "maxoid": Actual period most dissimilar to others
        - "distribution": Preserve value distribution (duration curve)
        - "distribution_minmax": Distribution + preserve min/max values
        - "minmax_mean": Combine min/max/mean per timestep

        Default depends on method:
        - "mean" for averaging, kmeans
        - "medoid" for kmedoids, hierarchical, contiguous
        - "maxoid" for kmaxoids

    weights : dict[str, float], optional
        Per-column weights for clustering distance calculation.
        Higher weight = more influence on clustering.
        Example: {"demand": 2.0, "solar": 1.0}

    normalize_means : bool, default False
        Normalize all columns to the same mean before clustering.
        Useful when columns have very different scales.

    use_duration_curves : bool, default False
        Sort values within each period before clustering.
        Matches periods by their value distribution rather than timing.

    include_period_sums : bool, default False
        Include period totals as additional features for clustering.
        Helps preserve total energy/load values.

    solver : str, default "highs"
        MILP solver for kmedoids_exact method.
        Options: "highs" (default, open source), "cbc", "gurobi", "cplex"
    """

    method: ClusterMethod = "hierarchical"
    representation: RepresentationMethod | None = None
    weights: dict[str, float] | None = None
    normalize_means: bool = False
    use_duration_curves: bool = False
    include_period_sums: bool = False
    solver: Solver = "highs"

    def get_representation(self) -> RepresentationMethod:
        """Get the representation method, using default if not specified."""
        if self.representation is not None:
            return self.representation

        # Default representation based on clustering method
        defaults: dict[ClusterMethod, RepresentationMethod] = {
            "averaging": "mean",
            "kmeans": "mean",
            "kmedoids": "medoid",
            "kmedoids_exact": "medoid",
            "kmaxoids": "maxoid",
            "hierarchical": "medoid",
            "contiguous": "medoid",
        }
        return defaults.get(self.method, "mean")


@dataclass(frozen=True)
class SegmentConfig:
    """Configuration for temporal segmentation within periods.

    Segmentation reduces the temporal resolution within each typical period,
    grouping consecutive timesteps into segments.

    Parameters
    ----------
    n_segments : int
        Number of segments per period.
        Must be less than or equal to the number of timesteps per period.
        Example: period_hours=24 with hourly data has 24 timesteps,
        so n_segments could be 1-24.

    representation : str, default "mean"
        How to represent each segment:
        - "mean": Average value of timesteps in segment
        - "medoid": Actual timestep closest to segment mean
        - "distribution": Preserve distribution within segment
    """

    n_segments: int
    representation: RepresentationMethod = "mean"

    def __post_init__(self) -> None:
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be positive, got {self.n_segments}")


@dataclass(frozen=True)
class ExtremeConfig:
    """Configuration for preserving extreme periods.

    Extreme periods contain critical peak values that must be preserved
    in the aggregated representation (e.g., peak demand for capacity sizing).

    Parameters
    ----------
    method : str, default "append"
        How to handle extreme periods:
        - "append": Add extreme periods as additional cluster centers
        - "replace": Replace the nearest cluster center with the extreme
        - "new_cluster": Add as new cluster and reassign affected periods

    max_timesteps : list[str], optional
        Column names where the timestep with maximum value should be preserved.
        The entire period containing that timestep becomes an extreme period.
        Example: ["electricity_demand"] to preserve peak demand hour.

    min_timesteps : list[str], optional
        Column names where the timestep with minimum value should be preserved.
        Example: ["temperature"] to preserve coldest hour.

    max_periods : list[str], optional
        Column names where the period with maximum sum should be preserved.
        Example: ["solar_generation"] to preserve highest solar day.

    min_periods : list[str], optional
        Column names where the period with minimum sum should be preserved.
        Example: ["wind_generation"] to preserve lowest wind day.
    """

    method: ExtremeMethod = "append"
    max_timesteps: list[str] = field(default_factory=list)
    min_timesteps: list[str] = field(default_factory=list)
    max_periods: list[str] = field(default_factory=list)
    min_periods: list[str] = field(default_factory=list)

    def has_extremes(self) -> bool:
        """Check if any extreme periods are configured."""
        return bool(
            self.max_timesteps
            or self.min_timesteps
            or self.max_periods
            or self.min_periods
        )


# Mapping from new API names to old API names
METHOD_MAPPING: dict[ClusterMethod, str] = {
    "averaging": "averaging",
    "kmeans": "k_means",
    "kmedoids": "k_medoids",
    "kmedoids_exact": "k_medoids",  # Same, but uses exact solver
    "kmaxoids": "k_maxoids",
    "hierarchical": "hierarchical",
    "contiguous": "adjacent_periods",
}

REPRESENTATION_MAPPING: dict[RepresentationMethod, str] = {
    "mean": "meanRepresentation",
    "medoid": "medoidRepresentation",
    "maxoid": "maxoidRepresentation",
    "duration": "durationRepresentation",
    "distribution": "distributionRepresentation",
    "distribution_minmax": "distributionAndMinMaxRepresentation",
    "minmax_mean": "minmaxmeanRepresentation",
}

EXTREME_METHOD_MAPPING: dict[ExtremeMethod, str] = {
    "append": "append",
    "replace": "replace_cluster_center",
    "new_cluster": "new_cluster_center",
}
