"""Configuration classes for tsam aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

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
    "distribution",
    "distribution_minmax",
    "minmax_mean",
]

ExtremeMethod = Literal[
    "append",
    "replace",
    "new_cluster",
]

# Runtime-checkable tuple of the valid extreme-period methods, kept in sync with
# the ``ExtremeMethod`` type alias above (single source of truth).
EXTREME_METHODS: tuple[str, ...] = get_args(ExtremeMethod)

Solver = Literal["highs", "cbc", "gurobi", "cplex"]


@dataclass(frozen=True)
class Distribution:
    """Representation that preserves the value distribution (duration curve).

    Parameters
    ----------
    scope : "cluster" or "global", default "cluster"
        "cluster": preserve each cluster's distribution separately
        "global": preserve the overall time series distribution
    preserve_minmax : bool, default False
        If True, also preserves min/max values per timestep
        (equivalent to old "distribution_minmax").
    """

    scope: Literal["cluster", "global"] = "cluster"
    preserve_minmax: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"type": "distribution"}
        if self.scope != "cluster":
            result["scope"] = self.scope
        if self.preserve_minmax:
            result["preserve_minmax"] = self.preserve_minmax
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Distribution:
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            scope=data.get("scope", "cluster"),
            preserve_minmax=data.get("preserve_minmax", False),
        )


@dataclass(frozen=True)
class MinMaxMean:
    """Representation combining min, max, and mean per column.

    Columns not listed in max_columns or min_columns default to mean.

    Parameters
    ----------
    max_columns : list[str]
        Columns represented by their maximum value across cluster members.
    min_columns : list[str]
        Columns represented by their minimum value across cluster members.
    """

    max_columns: list[str] = field(default_factory=list)
    min_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"type": "minmax_mean"}
        if self.max_columns:
            result["max_columns"] = self.max_columns
        if self.min_columns:
            result["min_columns"] = self.min_columns
        return result

    @classmethod
    def from_dict(cls, data: dict) -> MinMaxMean:
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            max_columns=data.get("max_columns", []),
            min_columns=data.get("min_columns", []),
        )


# Union type for representation (strings remain valid for backward compat)
Representation = RepresentationMethod | Distribution | MinMaxMean


def representation_to_dict(rep: Representation) -> str | dict[str, Any]:
    """Serialize a representation value to a JSON-compatible format."""
    if isinstance(rep, (Distribution, MinMaxMean)):
        return rep.to_dict()
    return rep


def representation_from_dict(data: str | dict) -> Representation:
    """Deserialize a representation value from a JSON-compatible format."""
    if isinstance(data, str):
        return data  # type: ignore[return-value]
    # It's a dict with a "type" key
    rep_type = data.get("type")
    if rep_type == "distribution":
        return Distribution.from_dict(data)
    if rep_type == "minmax_mean":
        return MinMaxMean.from_dict(data)
    raise ValueError(f"Unknown representation type: {rep_type!r}")


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

    representation : str, Distribution, or MinMaxMean, optional
        How to represent cluster centers. Accepts either a string shortcut
        or a typed representation object for additional options:

        String shortcuts:
        - "mean": Centroid (average of cluster members)
        - "medoid": Actual period closest to centroid
        - "maxoid": Actual period most dissimilar to others
        - "distribution": Preserve value distribution (duration curve)
        - "distribution_minmax": Distribution + preserve min/max values
        - "minmax_mean": Combine min/max/mean per timestep

        Typed objects (for additional options):
        - ``Distribution(scope="cluster"|"global", preserve_minmax=False)``:
          Preserve value distribution. ``scope`` controls whether each
          cluster's distribution is preserved separately ("cluster") or
          the overall time series distribution ("global").
        - ``MinMaxMean(max_columns=[...], min_columns=[...])``:
          Combine min/max/mean per column. Columns not listed default to mean.

        Default depends on method:
        - "mean" for averaging, kmeans
        - "medoid" for kmedoids, hierarchical, contiguous
        - "maxoid" for kmaxoids

    scale_by_column_means : bool, default False
        Divide each column by its mean after MinMax normalization, so all
        columns have equal mean before clustering.
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
    """

    method: ClusterMethod
    representation: Representation | None
    scale_by_column_means: bool
    use_duration_curves: bool
    include_period_sums: bool
    solver: Solver

    __slots__ = (
        "include_period_sums",
        "method",
        "representation",
        "scale_by_column_means",
        "solver",
        "use_duration_curves",
    )

    def __init__(
        self,
        method: ClusterMethod = "hierarchical",
        representation: Representation | None = None,
        scale_by_column_means: bool = False,
        use_duration_curves: bool = False,
        include_period_sums: bool = False,
        solver: Solver = "highs",
    ) -> None:
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "scale_by_column_means", scale_by_column_means)
        object.__setattr__(self, "use_duration_curves", use_duration_curves)
        object.__setattr__(self, "include_period_sums", include_period_sums)
        object.__setattr__(self, "solver", solver)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ClusterConfig is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ClusterConfig is immutable")

    def __getstate__(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterConfig):
            return NotImplemented
        return all(getattr(self, s) == getattr(other, s) for s in self.__slots__)

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, s) for s in self.__slots__))

    def __repr__(self) -> str:
        parts = ", ".join(f"{s}={getattr(self, s)!r}" for s in self.__slots__)
        return f"ClusterConfig({parts})"

    def get_representation(self) -> Representation:
        """Get the representation, using default if not specified."""
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"method": self.method}
        if self.representation is not None:
            result["representation"] = representation_to_dict(self.representation)
        if self.scale_by_column_means:
            result["scale_by_column_means"] = self.scale_by_column_means
        if self.use_duration_curves:
            result["use_duration_curves"] = self.use_duration_curves
        if self.include_period_sums:
            result["include_period_sums"] = self.include_period_sums
        if self.solver != "highs":
            result["solver"] = self.solver
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ClusterConfig:
        """Create from dictionary (e.g., loaded from JSON)."""
        rep_data = data.get("representation")
        representation = (
            representation_from_dict(rep_data) if rep_data is not None else None
        )
        return cls(
            method=data.get("method", "hierarchical"),
            representation=representation,
            scale_by_column_means=data.get("scale_by_column_means", False),
            use_duration_curves=data.get("use_duration_curves", False),
            include_period_sums=data.get("include_period_sums", False),
            solver=data.get("solver", "highs"),
        )


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
        Example: period_duration=24 with hourly data has 24 timesteps,
        so n_segments could be 1-24.

    representation : str, Distribution, or MinMaxMean, default "mean"
        How to represent each segment:
        - "mean": Average value of timesteps in segment
        - "medoid": Actual timestep closest to segment mean
        - "distribution": Preserve distribution within segment
        - ``Distribution(...)``: Distribution with additional options
        - ``MinMaxMean(...)``: Per-column min/max/mean
    """

    n_segments: int
    representation: Representation = "mean"

    def __post_init__(self) -> None:
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be positive, got {self.n_segments}")
        # Note: Upper bound validation (n_segments <= timesteps_per_period)
        # is performed in api.aggregate() when period_duration is known.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"n_segments": self.n_segments}
        if self.representation != "mean":
            result["representation"] = representation_to_dict(self.representation)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> SegmentConfig:
        """Create from dictionary (e.g., loaded from JSON)."""
        rep_data = data.get("representation", "mean")
        return cls(
            n_segments=data["n_segments"],
            representation=representation_from_dict(rep_data),
        )


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

    def __post_init__(self) -> None:
        if self.method not in EXTREME_METHODS:
            raise ValueError(
                f"Unknown extreme period method {self.method!r}. "
                f"Valid options: {list(EXTREME_METHODS)}"
            )

    def has_extremes(self) -> bool:
        """Check if any extreme periods are configured."""
        return bool(
            self.max_value or self.min_value or self.max_period or self.min_period
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {}
        if self.method != "append":
            result["method"] = self.method
        if self.max_value:
            result["max_value"] = self.max_value
        if self.min_value:
            result["min_value"] = self.min_value
        if self.max_period:
            result["max_period"] = self.max_period
        if self.min_period:
            result["min_period"] = self.min_period
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ExtremeConfig:
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            method=data.get("method", "append"),
            max_value=data.get("max_value", []),
            min_value=data.get("min_value", []),
            max_period=data.get("max_period", []),
            min_period=data.get("min_period", []),
        )
