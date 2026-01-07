"""Configuration classes for tsam aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

# Type aliases for clarity
ClusterMethod = Literal[
    "averaging",
    "kmeans",
    "kmedoids",
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
        - "kmedoids": K-medoids using MILP optimization (uses actual periods)
        - "kmaxoids": K-maxoids (selects most dissimilar periods)
        - "hierarchical": Agglomerative hierarchical clustering
        - "contiguous": Hierarchical with temporal contiguity constraint

    representation : str, optional
        How to represent cluster centers:
        - "mean": Centroid (average of cluster members)
        - "medoid": Actual period closest to centroid
        - "maxoid": Actual period most dissimilar to others
        - "duration": Preserve value distribution (duration curve)
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
        MILP solver for kmedoids method.
        Options: "highs" (default, open source), "cbc", "gurobi", "cplex"

    predef_cluster_order : array-like, optional
        Predefined cluster assignments for each period.
        Use this to apply cluster assignments from one aggregation to another.
        Example: Use wind-only clustering order for multi-variable aggregation.

    predef_cluster_centers : array-like, optional
        Predefined cluster center indices.
        When combined with predef_cluster_order, uses the exact same
        representative periods instead of recalculating them.
    """

    method: ClusterMethod = "hierarchical"
    representation: RepresentationMethod | None = None
    weights: dict[str, float] | None = None
    normalize_means: bool = False
    use_duration_curves: bool = False
    include_period_sums: bool = False
    solver: Solver = "highs"
    predef_cluster_order: tuple[int, ...] | None = None
    predef_cluster_centers: tuple[int, ...] | None = None

    def get_representation(self) -> RepresentationMethod:
        """Get the representation method, using default if not specified."""
        if self.representation is not None:
            return self.representation

        # Default representation based on clustering method
        defaults: dict[ClusterMethod, RepresentationMethod] = {
            "averaging": "mean",
            "kmeans": "mean",
            "kmedoids": "medoid",
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

    predef_segment_order : tuple[tuple[int, ...], ...], optional
        Predefined segment assignments per timestep, per typical period.
        Use this to transfer segment assignments from one aggregation to another.
        Outer tuple has one entry per typical period (length = n_periods).
        Inner tuple has one entry per timestep (length = n_timesteps_per_period),
        containing the segment index (0 to n_segments-1) for that timestep.

    predef_segment_durations : tuple[tuple[int, ...], ...], optional
        Predefined durations (in timesteps) per segment, per typical period.
        Required when predef_segment_order is specified.
        Outer tuple has one entry per typical period.
        Inner tuple has one entry per segment, containing the number of
        timesteps in that segment.

    predef_segment_centers : tuple[tuple[int, ...], ...], optional
        Predefined center indices per segment, per typical period.
        When combined with predef_segment_order, uses the exact same
        segment representatives instead of recalculating them.
        Outer tuple has one entry per typical period.
        Inner tuple has one entry per segment, containing the original
        timestep index (within the period) used as the segment center.
    """

    n_segments: int
    representation: RepresentationMethod = "mean"
    predef_segment_order: tuple[tuple[int, ...], ...] | None = None
    predef_segment_durations: tuple[tuple[int, ...], ...] | None = None
    predef_segment_centers: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be positive, got {self.n_segments}")
        # Note: Upper bound validation (n_segments <= timesteps_per_period)
        # is performed in api.aggregate() when period_hours is known.

        # Validate predefined segment parameters
        if self.predef_segment_order is not None:
            if self.predef_segment_durations is None:
                raise ValueError(
                    "predef_segment_durations must be provided when "
                    "predef_segment_order is specified"
                )
            if len(self.predef_segment_order) != len(self.predef_segment_durations):
                raise ValueError(
                    f"predef_segment_order ({len(self.predef_segment_order)} periods) "
                    f"and predef_segment_durations ({len(self.predef_segment_durations)} periods) "
                    "must have the same number of periods"
                )
        elif self.predef_segment_durations is not None:
            raise ValueError(
                "predef_segment_order must be provided when "
                "predef_segment_durations is specified"
            )

        if self.predef_segment_centers is not None:
            if self.predef_segment_order is None:
                raise ValueError(
                    "predef_segment_order must be provided when "
                    "predef_segment_centers is specified"
                )


@dataclass(frozen=True)
class PredefinedConfig:
    """Predefined assignments for transferring results between aggregations.

    Use this to apply clustering and segmentation results from one aggregation
    to another dataset. Get this from `result.predefined` or create manually.

    Parameters
    ----------
    cluster_order : tuple[int, ...]
        Cluster assignments for each original period.
        Length equals the number of original periods in the data.

    cluster_centers : tuple[int, ...], optional
        Indices of original periods used as cluster centers.
        If not provided, centers will be recalculated.

    segment_order : tuple[tuple[int, ...], ...], optional
        Segment assignments per timestep, per typical period.
        Only needed if transferring segmentation results.

    segment_durations : tuple[tuple[int, ...], ...], optional
        Duration (in timesteps) per segment, per typical period.
        Required if segment_order is provided.

    Examples
    --------
    >>> # From a previous result
    >>> predefined = result.predefined

    >>> # Save to file
    >>> import json
    >>> with open("predefined.json", "w") as f:
    ...     json.dump(predefined.to_dict(), f)

    >>> # Load from file
    >>> with open("predefined.json") as f:
    ...     data = json.load(f)
    >>> predefined = PredefinedConfig.from_dict(data)

    >>> # Apply to new data
    >>> result2 = tsam.aggregate(new_data, n_periods=8, predefined=predefined)
    """

    cluster_order: tuple[int, ...]
    cluster_centers: tuple[int, ...] | None = None
    segment_order: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.segment_order is not None and self.segment_durations is None:
            raise ValueError(
                "segment_durations must be provided when segment_order is specified"
            )
        if self.segment_durations is not None and self.segment_order is None:
            raise ValueError(
                "segment_order must be provided when segment_durations is specified"
            )

    def __repr__(self) -> str:
        n_original_periods = len(self.cluster_order)
        n_typical_periods = len(set(self.cluster_order))
        has_centers = self.cluster_centers is not None
        has_segments = self.segment_order is not None

        lines = [
            "PredefinedConfig(",
            f"  n_original_periods={n_original_periods},",
            f"  n_typical_periods={n_typical_periods},",
            f"  has_cluster_centers={has_centers},",
        ]

        if has_segments:
            n_segments = len(self.segment_durations[0]) if self.segment_durations else 0
            n_timesteps = len(self.segment_order[0]) if self.segment_order else 0
            lines.append(f"  n_segments={n_segments},")
            lines.append(f"  n_timesteps_per_period={n_timesteps},")

        lines.append(")")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a readable DataFrame.

        Returns a DataFrame with one row per original period showing
        cluster assignments. If segments are defined, includes segment
        information for each typical period.

        Returns
        -------
        pd.DataFrame
            DataFrame with cluster_order indexed by original period.
            If segments are present, includes additional columns.
        """
        # Base DataFrame with cluster assignments
        df = pd.DataFrame(
            {"cluster": list(self.cluster_order)},
            index=pd.RangeIndex(len(self.cluster_order), name="original_period"),
        )

        if self.cluster_centers is not None:
            # Add a column showing which periods are cluster centers
            center_set = set(self.cluster_centers)
            df["is_center"] = [i in center_set for i in range(len(self.cluster_order))]

        return df

    def segment_dataframe(self) -> pd.DataFrame | None:
        """Get segment structure as a readable DataFrame.

        Returns a DataFrame showing segment durations per typical period.
        Returns None if no segmentation is defined.

        Returns
        -------
        pd.DataFrame | None
            DataFrame with typical periods as rows and segments as columns,
            values are segment durations in timesteps.
        """
        if self.segment_durations is None:
            return None

        n_periods = len(self.segment_durations)
        n_segments = len(self.segment_durations[0])

        return pd.DataFrame(
            list(self.segment_durations),
            index=pd.RangeIndex(n_periods, name="typical_period"),
            columns=pd.RangeIndex(n_segments, name="segment"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"cluster_order": list(self.cluster_order)}
        if self.cluster_centers is not None:
            result["cluster_centers"] = list(self.cluster_centers)
        if self.segment_order is not None:
            result["segment_order"] = [list(s) for s in self.segment_order]
        if self.segment_durations is not None:
            result["segment_durations"] = [list(s) for s in self.segment_durations]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> PredefinedConfig:
        """Create from dictionary (e.g., loaded from JSON)."""
        kwargs = {"cluster_order": tuple(data["cluster_order"])}
        if "cluster_centers" in data:
            kwargs["cluster_centers"] = tuple(data["cluster_centers"])
        if "segment_order" in data:
            kwargs["segment_order"] = tuple(tuple(s) for s in data["segment_order"])
        if "segment_durations" in data:
            kwargs["segment_durations"] = tuple(
                tuple(s) for s in data["segment_durations"]
            )
        return cls(**kwargs)


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

    max_value : list[str], optional
        Column names where the maximum value should be preserved.
        The entire period containing that single extreme value becomes an extreme period.
        Example: ["electricity_demand"] to preserve peak demand hour.

    min_value : list[str], optional
        Column names where the minimum value should be preserved.
        Example: ["temperature"] to preserve coldest hour.

    max_period : list[str], optional
        Column names where the period with maximum total should be preserved.
        Example: ["solar_generation"] to preserve highest solar day.

    min_period : list[str], optional
        Column names where the period with minimum total should be preserved.
        Example: ["wind_generation"] to preserve lowest wind day.
    """

    method: ExtremeMethod = "append"
    max_value: list[str] = field(default_factory=list)
    min_value: list[str] = field(default_factory=list)
    max_period: list[str] = field(default_factory=list)
    min_period: list[str] = field(default_factory=list)

    def has_extremes(self) -> bool:
        """Check if any extreme periods are configured."""
        return bool(
            self.max_value or self.min_value or self.max_period or self.min_period
        )


# Mapping from new API names to old API names
METHOD_MAPPING: dict[ClusterMethod, str] = {
    "averaging": "averaging",
    "kmeans": "k_means",
    "kmedoids": "k_medoids",
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
