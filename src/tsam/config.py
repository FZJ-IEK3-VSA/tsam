"""Configuration classes for tsam aggregation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args

# Type aliases for clarity
ClusterMethodName = Literal[
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

# Representation each clustering method falls back to when
# ``ClusterConfig.representation`` is not set: mean-based methods are
# represented by the cluster mean, medoid-based methods by the medoid, and
# kmaxoids by the maxoid. This is the single source of these defaults — resolve
# them through ``ClusterConfig.get_representation`` rather than repeating them.
DEFAULT_REPRESENTATION: dict[ClusterMethodName, RepresentationMethod] = {
    "averaging": "mean",
    "kmeans": "mean",
    "kmedoids": "medoid",
    "kmaxoids": "maxoid",
    "hierarchical": "medoid",
    "contiguous": "medoid",
}

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
class KMedoids:
    """Exact k-medoids clustering configuration.

    Selects real periods as cluster centers by solving a MILP. Pass it to
    ``ClusterConfig(method=KMedoids(...))`` to tune the solver. The bare string
    ``method="kmedoids"`` is equivalent to ``KMedoids()`` with defaults.

    Args:
        solver: MILP solver. Options: "highs" (open source, default), "cbc",
            "gurobi", "cplex".
        options: Solver options forwarded verbatim to the solver. Option names
            are solver-specific — e.g. ``{"time_limit": 300}`` to bound HiGHS
            runtime in seconds, ``{"mip_rel_gap": 0.01}`` for the optimality gap,
            or ``{"threads": 4}``. Empty by default (the solver's own defaults
            apply, i.e. no enforced time limit).
    """

    solver: Solver = "highs"
    # Excluded from __hash__ (dicts are unhashable) but kept in __eq__ so two
    # configs differing only in options still compare unequal.
    options: dict[str, Any] = field(default_factory=dict, hash=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict tagged with ``type``."""
        result: dict[str, Any] = {"type": "kmedoids"}
        if self.solver != "highs":
            result["solver"] = self.solver
        if self.options:
            result["options"] = dict(self.options)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KMedoids:
        """Rebuild from a dict of solver options."""
        return cls(
            solver=data.get("solver", "highs"),
            options=data.get("options", {}),
        )


@dataclass(frozen=True)
class Distribution:
    """Representation that preserves the value distribution (duration curve).

    Args:
        scope: "local" preserves each group's own distribution separately. The
            group is a cluster of periods for a cluster representation, or a
            segment of timesteps for a segment representation. "global"
            preserves only the distribution of the enclosing whole (the full
            time series for a cluster representation, a single period for a
            segment representation). "cluster" is accepted as a deprecated alias
            for "local" (the old name, which was misleading for segment
            representations where the group is a segment, not a cluster).
        preserve_minmax: If True, also preserves min/max values per timestep
            (equivalent to old "distribution_minmax").
        reference_attribute: Name of a data column. If given, a single temporal
            ordering derived from this attribute is applied to all attributes,
            preserving their concurrency (co-incidence in time) with the
            reference attribute while each attribute still fits its own
            distribution. Only valid with ``scope="local"``. Equivalent to
            ``concurrency="reference"``.
        concurrency: Strategy used to derive the synthetic time axis, one of
            ``"independent"``, ``"reference"``, ``"medoid"``, ``"consensus"``,
            ``"assignment"``. All strategies preserve every attribute's marginal
            distribution and differ only in how the attributes' concurrency is
            preserved:

            - ``"independent"`` (default): each attribute ordered by its own mean
              profile — best marginal fit, concurrency across attributes lost.
            - ``"reference"``: single ordering from ``reference_attribute``.
            - ``"medoid"``: each attribute ordered by the cluster medoid's ranks,
              reproducing a real period's joint co-occurrence.
            - ``"consensus"``: single ordering from the first principal component
              of all attributes' mean profiles.
            - ``"assignment"``: optimal single ordering minimising total
              deviation from the cluster mean profile across all attributes.

            Only valid with ``scope="local"``. If None, resolves to
            ``"reference"`` when ``reference_attribute`` is given, otherwise
            ``"independent"``.

    Note:
        When used as a segment representation (``SegmentConfig.representation``),
        each segment is a single value per attribute, which constrains two
        options: ``preserve_minmax`` only takes effect with ``scope="global"``
        (a single value cannot carry both min and max, so ``scope="local"``
        keeps the integral-preserving mean), and ``concurrency`` /
        ``reference_attribute`` are not supported (there is no within-period time
        axis left to order — set them on the cluster representation, where
        ordering runs before segmentation).
    """

    # "cluster" is a deprecated alias for "local"; normalized in __post_init__.
    scope: Literal["local", "global", "cluster"] = "local"
    preserve_minmax: bool = False
    reference_attribute: str | None = None
    concurrency: (
        Literal["independent", "reference", "medoid", "consensus", "assignment"] | None
    ) = None

    def __post_init__(self) -> None:
        if self.scope == "cluster":
            warnings.warn(
                'Distribution scope="cluster" is deprecated; use scope="local" '
                "instead (the two are equivalent). The name was renamed because "
                '"cluster" is misleading for segment representations, where the '
                "group being preserved is a segment, not a cluster.",
                FutureWarning,
                stacklevel=2,
            )
            # frozen dataclass: normalize the deprecated alias in place.
            object.__setattr__(self, "scope", "local")
        if self.reference_attribute is not None and self.scope != "local":
            raise ValueError(
                "reference_attribute is only supported with scope='local'."
            )
        if (
            self.concurrency is not None
            and self.concurrency != "independent"
            and self.scope != "local"
        ):
            raise ValueError("concurrency is only supported with scope='local'.")
        if self.concurrency == "reference" and self.reference_attribute is None:
            raise ValueError(
                "concurrency='reference' requires reference_attribute to be set."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"type": "distribution"}
        if self.scope != "local":
            result["scope"] = self.scope
        if self.preserve_minmax:
            result["preserve_minmax"] = self.preserve_minmax
        if self.reference_attribute is not None:
            result["reference_attribute"] = self.reference_attribute
        if self.concurrency is not None:
            result["concurrency"] = self.concurrency
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Distribution:
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            scope=data.get("scope", "local"),
            preserve_minmax=data.get("preserve_minmax", False),
            reference_attribute=data.get("reference_attribute"),
            concurrency=data.get("concurrency"),
        )


@dataclass(frozen=True)
class MinMaxMean:
    """Representation combining min, max, and mean per column.

    Columns not listed in max_columns or min_columns default to mean.

    Args:
        max_columns: Columns represented by their maximum value across cluster
            members.
        min_columns: Columns represented by their minimum value across cluster
            members.
    """

    max_columns: list[str] = field(default_factory=list)
    min_columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        both = [col for col in self.max_columns if col in self.min_columns]
        if both:
            raise ValueError(
                f"Columns {both} are listed in both max_columns and min_columns. "
                "A column is represented by one value per timestep, so it cannot "
                "keep its maximum and its minimum at the same time — list it in "
                "one of the two."
            )

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

# The clustering method: a string name (backward compat) or a typed object
# carrying that method's options (currently only ``KMedoids``; more may follow).
ClusterMethod = ClusterMethodName | KMedoids


def method_to_dict(method: ClusterMethod) -> str | dict[str, Any]:
    """Serialize a clustering method to a JSON-compatible format."""
    if isinstance(method, KMedoids):
        return method.to_dict()
    return method


def method_from_dict(data: str | dict) -> ClusterMethod:
    """Deserialize a clustering method from a JSON-compatible format."""
    if isinstance(data, str):
        return data  # type: ignore[return-value]
    method_type = data.get("type")
    if method_type == "kmedoids":
        return KMedoids.from_dict(data)
    raise ValueError(f"Unknown clustering method type: {method_type!r}")


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

    Args:
        method: Clustering algorithm to use. Accepts a string shortcut or, for
            k-medoids, a ``KMedoids`` config object to tune the solver.
            - "averaging": Sequential averaging of periods
            - "kmeans": K-means clustering (fast, uses centroids)
            - "kmedoids": K-medoids using MILP optimization (uses actual
              periods); pass ``KMedoids(solver=..., timelimit=...)`` to
              configure the solver
            - "kmaxoids": K-maxoids (selects most dissimilar periods)
            - "hierarchical": Agglomerative hierarchical clustering
            - "contiguous": Hierarchical with temporal contiguity constraint
        representation: How to represent cluster centers. Accepts either a
            string shortcut or a typed representation object for additional
            options.

            String shortcuts:
            - "mean": Centroid (average of cluster members)
            - "medoid": Actual period closest to centroid
            - "maxoid": Actual period most dissimilar to others
            - "distribution": Preserve value distribution (duration curve)
            - "distribution_minmax": Distribution + preserve min/max values
            - "minmax_mean": Combine min/max/mean per timestep

            Typed objects (for additional options):
            - ``Distribution(scope="local"|"global", preserve_minmax=False)``:
              Preserve value distribution. ``scope`` controls whether each
              cluster's distribution is preserved separately ("local") or
              only the overall time series distribution ("global"). ``"cluster"``
              is a deprecated alias for ``"local"``.
            - ``MinMaxMean(max_columns=[...], min_columns=[...])``:
              Combine min/max/mean per column. Columns not listed default to mean.

            Default depends on method:
            - "mean" for averaging, kmeans
            - "medoid" for kmedoids, hierarchical, contiguous
            - "maxoid" for kmaxoids
        scale_by_column_means: Divide each column by its mean after MinMax
            normalization, so all columns have equal mean before clustering.
            Useful when columns have very different scales.
        use_duration_curves: Sort values within each period before clustering.
            Matches periods by their value distribution rather than timing.
        include_period_sums: Include period totals as additional features for
            clustering. Helps preserve total energy/load values.
        solver: Deprecated. Use ``method=KMedoids(solver=...)`` instead. When
            set, forwarded to the k-medoids config.
    """

    method: ClusterMethod
    representation: Representation | None
    scale_by_column_means: bool
    use_duration_curves: bool
    include_period_sums: bool

    __slots__ = (
        "include_period_sums",
        "method",
        "representation",
        "scale_by_column_means",
        "use_duration_curves",
    )

    def __init__(
        self,
        method: ClusterMethod = "hierarchical",
        representation: Representation | None = None,
        scale_by_column_means: bool = False,
        use_duration_curves: bool = False,
        include_period_sums: bool = False,
        solver: Solver | None = None,
    ) -> None:
        if solver is not None:
            warnings.warn(
                "ClusterConfig(solver=...) is deprecated; pass "
                "method=KMedoids(solver=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(method, KMedoids):
                method = replace(method, solver=solver)
            elif method == "kmedoids":
                method = KMedoids(solver=solver)
            # solver has no effect for non-kmedoids methods (it never did); ignored.

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "scale_by_column_means", scale_by_column_means)
        object.__setattr__(self, "use_duration_curves", use_duration_curves)
        object.__setattr__(self, "include_period_sums", include_period_sums)

        if representation is not None and use_duration_curves:
            warnings.warn(
                f"representation={representation!r} has no effect together with "
                "use_duration_curves=True. Periods clustered by their duration "
                "curve are always represented by the real period closest to the "
                "cluster's duration-curve centroid, since a synthesized "
                "representative would carry a temporal shape that no period in "
                "the cluster ever had. Drop one of the two settings.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def method_name(self) -> ClusterMethodName:
        """Canonical string name of the clustering method."""
        return "kmedoids" if isinstance(self.method, KMedoids) else self.method

    @property
    def solver(self) -> Solver:
        """Solver used by the k-medoids method (``"highs"`` for other methods)."""
        return self.method.solver if isinstance(self.method, KMedoids) else "highs"

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
        """Get the representation, using the method's default if not specified."""
        if self.representation is not None:
            return self.representation
        return DEFAULT_REPRESENTATION.get(self.method_name, "mean")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"method": method_to_dict(self.method)}
        if self.representation is not None:
            result["representation"] = representation_to_dict(self.representation)
        if self.scale_by_column_means:
            result["scale_by_column_means"] = self.scale_by_column_means
        if self.use_duration_curves:
            result["use_duration_curves"] = self.use_duration_curves
        if self.include_period_sums:
            result["include_period_sums"] = self.include_period_sums
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ClusterConfig:
        """Create from dictionary (e.g., loaded from JSON)."""
        rep_data = data.get("representation")
        representation = (
            representation_from_dict(rep_data) if rep_data is not None else None
        )
        method = method_from_dict(data.get("method", "hierarchical"))
        # Legacy flat form: {"method": "kmedoids", "solver": ...}
        if method == "kmedoids" and "solver" in data:
            method = KMedoids(solver=data["solver"])
        return cls(
            method=method,
            representation=representation,
            scale_by_column_means=data.get("scale_by_column_means", False),
            use_duration_curves=data.get("use_duration_curves", False),
            include_period_sums=data.get("include_period_sums", False),
        )


@dataclass(frozen=True)
class SegmentConfig:
    """Configuration for temporal segmentation within periods.

    Segmentation reduces the temporal resolution within each typical period,
    grouping consecutive timesteps into segments. The algorithm is fixed:
    constrained agglomerative clustering that may only merge *adjacent*
    timesteps. Unlike :class:`ClusterConfig`, there is therefore no ``method``
    choice — the only adjustable settings are the number of segments and how
    each segment is represented.

    Args:
        n_segments: Number of segments per period. Must be less than or equal to
            the number of timesteps per period. Example: period_duration=24 with
            hourly data has 24 timesteps, so n_segments could be 1-24.
        representation: How to represent each segment:
            - "mean": Average value of timesteps in segment
            - "medoid": Actual timestep closest to segment mean
            - "distribution": Preserve distribution within segment
            - ``Distribution(...)``: Distribution with additional options (see Note)
            - ``MinMaxMean(...)``: Per-column min/max/mean

    Note:
        A segment collapses each attribute to a **single value**, so the
        ``Distribution`` options behave differently than for a cluster
        representation:

        - ``scope="local"`` (the default) is equivalent to ``"mean"`` — each
          segment's single value is the mean of its timesteps. Only
          ``scope="global"`` produces a distinct result (it matches the whole
          period's value distribution rather than each segment's mean).
        - ``preserve_minmax`` only takes effect with ``scope="global"``; a
          single value cannot carry both the min and the max, so with
          ``scope="local"`` it is silently ignored and the integral-preserving
          mean is kept (a ``UserWarning`` is emitted).
    """

    n_segments: int
    representation: Representation = "mean"

    def __post_init__(self) -> None:
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be positive, got {self.n_segments}")
        # Note: Upper bound validation (n_segments <= timesteps_per_period)
        # is performed in api.aggregate() when period_duration is known.
        if (
            isinstance(self.representation, Distribution)
            and self.representation.preserve_minmax
            and self.representation.scope == "local"
        ):
            warnings.warn(
                'preserve_minmax has no effect on a scope="local" segment '
                "representation: each segment collapses to a single value per "
                "attribute, which cannot carry both the min and the max, so the "
                'integral-preserving mean is kept. Use scope="global" to preserve '
                "segment min/max.",
                UserWarning,
                stacklevel=2,
            )

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

    Args:
        method: How to handle extreme periods:
            - "append": Add extreme periods as additional cluster centers
            - "replace": Replace the nearest cluster center with the extreme
            - "new_cluster": Add as new cluster and reassign affected periods
        max_value: Column names where the maximum value should be preserved. The
            entire period containing that single extreme value becomes an extreme
            period. Example: ["electricity_demand"] to preserve peak demand hour.
        min_value: Column names where the minimum value should be preserved.
            Example: ["temperature"] to preserve coldest hour.
        max_period: Column names where the period with maximum total should be
            preserved. Example: ["solar_generation"] to preserve highest solar day.
        min_period: Column names where the period with minimum total should be
            preserved. Example: ["wind_generation"] to preserve lowest wind day.
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
