"""Configuration classes for tsam aggregation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from tsam.result import AggregationResult

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

    normalize_column_means : bool, default False
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
    """

    method: ClusterMethod = "hierarchical"
    representation: RepresentationMethod | None = None
    weights: dict[str, float] | None = None
    normalize_column_means: bool = False
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
            "kmaxoids": "maxoid",
            "hierarchical": "medoid",
            "contiguous": "medoid",
        }
        return defaults.get(self.method, "mean")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"method": self.method}
        if self.representation is not None:
            result["representation"] = self.representation
        if self.weights is not None:
            result["weights"] = self.weights
        if self.normalize_column_means:
            result["normalize_column_means"] = self.normalize_column_means
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
        return cls(
            method=data.get("method", "hierarchical"),
            representation=data.get("representation"),
            weights=data.get("weights"),
            normalize_column_means=data.get("normalize_column_means", False),
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
        # Note: Upper bound validation (n_segments <= timesteps_per_period)
        # is performed in api.aggregate() when period_duration is known.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"n_segments": self.n_segments}
        if self.representation != "mean":
            result["representation"] = self.representation
        return result

    @classmethod
    def from_dict(cls, data: dict) -> SegmentConfig:
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            n_segments=data["n_segments"],
            representation=data.get("representation", "mean"),
        )


@dataclass(frozen=True)
class ClusteringResult:
    """Clustering assignments that can be saved/loaded and applied to new data.

    This class bundles all clustering and segmentation assignments from an
    aggregation, enabling:
    - Simple IO via to_json()/from_json()
    - Applying the same clustering to different datasets via apply()
    - Preserving the parameters used to create the clustering

    Get this from `result.clustering` after running an aggregation.

    Transfer Fields (used by apply())
    ----------------------------------
    period_duration : float
        Length of each period in hours (e.g., 24 for daily periods).

    cluster_assignments : tuple[int, ...]
        Cluster assignments for each original period.
        Length equals the number of original periods in the data.

    n_timesteps_per_period : int
        Number of timesteps in each period. Used to validate that new data
        has compatible structure when calling apply().

    cluster_centers : tuple[int, ...], optional
        Indices of original periods used as cluster centers.
        If not provided, centers will be recalculated when applying.

    segment_assignments : tuple[tuple[int, ...], ...], optional
        Segment assignments per timestep, per typical period.
        Only present if segmentation was used.

    segment_durations : tuple[tuple[int, ...], ...], optional
        Duration (in timesteps) per segment, per typical period.
        Required if segment_assignments is present.

    segment_centers : tuple[tuple[int, ...], ...], optional
        Indices of timesteps used as segment centers, per typical period.
        Required for fully deterministic segment replication.

    preserve_column_means : bool, default True
        Whether to rescale typical periods to match original data means.

    rescale_exclude_columns : tuple[str, ...], optional
        Column names to exclude from rescaling. Useful for binary columns.

    representation : str, default "medoid"
        How to compute typical periods from cluster members.

    segment_representation : str, optional
        How to compute segment values. Only used if segmentation is present.

    temporal_resolution : float, optional
        Time resolution of input data in hours. If not provided, inferred.

    Reference Fields (for documentation, not used by apply())
    ---------------------------------------------------------
    cluster_config : ClusterConfig, optional
        Clustering configuration used to create this result.

    segment_config : SegmentConfig, optional
        Segmentation configuration used to create this result.

    extremes_config : ExtremeConfig, optional
        Extreme period configuration used to create this result.

    Examples
    --------
    >>> # Get clustering from a result
    >>> result = tsam.aggregate(df_wind, n_clusters=8)
    >>> clustering = result.clustering

    >>> # Save to file
    >>> clustering.to_json("clustering.json")

    >>> # Load from file
    >>> clustering = ClusteringResult.from_json("clustering.json")

    >>> # Apply to new data
    >>> result2 = clustering.apply(df_all)
    """

    # === Transfer fields (used by apply()) ===
    period_duration: float
    cluster_assignments: tuple[int, ...]
    n_timesteps_per_period: int
    cluster_centers: tuple[int, ...] | None = None
    segment_assignments: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None
    segment_centers: tuple[tuple[int, ...], ...] | None = None
    preserve_column_means: bool = True
    rescale_exclude_columns: tuple[str, ...] | None = None
    representation: RepresentationMethod = "medoid"
    segment_representation: RepresentationMethod | None = None
    temporal_resolution: float | None = None
    extreme_cluster_indices: tuple[int, ...] | None = None

    # === Reference fields (for documentation, not used by apply()) ===
    cluster_config: ClusterConfig | None = None
    segment_config: SegmentConfig | None = None
    extremes_config: ExtremeConfig | None = None

    def __post_init__(self) -> None:
        if self.segment_assignments is not None and self.segment_durations is None:
            raise ValueError(
                "segment_durations must be provided when segment_assignments is specified"
            )
        if self.segment_durations is not None and self.segment_assignments is None:
            raise ValueError(
                "segment_assignments must be provided when segment_durations is specified"
            )
        if self.segment_centers is not None and self.segment_assignments is None:
            raise ValueError(
                "segment_assignments must be provided when segment_centers is specified"
            )

    @property
    def n_clusters(self) -> int:
        """Number of clusters (typical periods)."""
        return len(set(self.cluster_assignments))

    @property
    def n_original_periods(self) -> int:
        """Number of original periods in the source data."""
        return len(self.cluster_assignments)

    @property
    def n_segments(self) -> int | None:
        """Number of segments per period, or None if no segmentation."""
        if self.segment_durations is None:
            return None
        return len(self.segment_durations[0])

    def __repr__(self) -> str:
        has_centers = self.cluster_centers is not None
        has_segments = self.segment_assignments is not None

        lines = [
            "ClusteringResult(",
            f"  period_duration={self.period_duration},",
            f"  n_original_periods={self.n_original_periods},",
            f"  n_clusters={self.n_clusters},",
            f"  has_cluster_centers={has_centers},",
        ]

        if has_segments:
            n_segments = len(self.segment_durations[0]) if self.segment_durations else 0
            n_timesteps = (
                len(self.segment_assignments[0]) if self.segment_assignments else 0
            )
            has_seg_centers = self.segment_centers is not None
            lines.append(f"  n_segments={n_segments},")
            lines.append(f"  n_timesteps_per_period={n_timesteps},")
            lines.append(f"  has_segment_centers={has_seg_centers},")

        lines.append(")")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a readable DataFrame.

        Returns a DataFrame with one row per original period showing
        cluster assignments.

        Returns
        -------
        pd.DataFrame
            DataFrame with cluster_assignments indexed by original period.
        """
        df = pd.DataFrame(
            {"cluster": list(self.cluster_assignments)},
            index=pd.RangeIndex(len(self.cluster_assignments), name="original_period"),
        )

        if self.cluster_centers is not None:
            center_set = set(self.cluster_centers)
            df["is_center"] = [
                i in center_set for i in range(len(self.cluster_assignments))
            ]

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

        n_clusters = len(self.segment_durations)
        n_segments = len(self.segment_durations[0])

        return pd.DataFrame(
            list(self.segment_durations),
            index=pd.RangeIndex(n_clusters, name="cluster"),
            columns=pd.RangeIndex(n_segments, name="segment"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Transfer fields (always included)
        result: dict[str, Any] = {
            "period_duration": self.period_duration,
            "cluster_assignments": list(self.cluster_assignments),
            "n_timesteps_per_period": self.n_timesteps_per_period,
            "preserve_column_means": self.preserve_column_means,
            "representation": self.representation,
        }
        if self.cluster_centers is not None:
            result["cluster_centers"] = list(self.cluster_centers)
        if self.segment_assignments is not None:
            result["segment_assignments"] = [list(s) for s in self.segment_assignments]
        if self.segment_durations is not None:
            result["segment_durations"] = [list(s) for s in self.segment_durations]
        if self.segment_centers is not None:
            result["segment_centers"] = [list(s) for s in self.segment_centers]
        if self.rescale_exclude_columns is not None:
            result["rescale_exclude_columns"] = list(self.rescale_exclude_columns)
        if self.segment_representation is not None:
            result["segment_representation"] = self.segment_representation
        if self.temporal_resolution is not None:
            result["temporal_resolution"] = self.temporal_resolution
        if self.extreme_cluster_indices is not None:
            result["extreme_cluster_indices"] = list(self.extreme_cluster_indices)
        # Reference fields (optional, for documentation)
        if self.cluster_config is not None:
            result["cluster_config"] = self.cluster_config.to_dict()
        if self.segment_config is not None:
            result["segment_config"] = self.segment_config.to_dict()
        if self.extremes_config is not None:
            result["extremes_config"] = self.extremes_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ClusteringResult:
        """Create from dictionary (e.g., loaded from JSON)."""
        # Transfer fields
        kwargs: dict[str, Any] = {
            "period_duration": data["period_duration"],
            "cluster_assignments": tuple(data["cluster_assignments"]),
            "n_timesteps_per_period": data["n_timesteps_per_period"],
            "preserve_column_means": data.get("preserve_column_means", True),
            "representation": data.get("representation", "medoid"),
        }
        if "cluster_centers" in data:
            kwargs["cluster_centers"] = tuple(data["cluster_centers"])
        if "segment_assignments" in data:
            kwargs["segment_assignments"] = tuple(
                tuple(s) for s in data["segment_assignments"]
            )
        if "segment_durations" in data:
            kwargs["segment_durations"] = tuple(
                tuple(s) for s in data["segment_durations"]
            )
        if "segment_centers" in data:
            kwargs["segment_centers"] = tuple(tuple(s) for s in data["segment_centers"])
        if "rescale_exclude_columns" in data:
            kwargs["rescale_exclude_columns"] = tuple(data["rescale_exclude_columns"])
        if "segment_representation" in data:
            kwargs["segment_representation"] = data["segment_representation"]
        if "temporal_resolution" in data:
            kwargs["temporal_resolution"] = data["temporal_resolution"]
        if "extreme_cluster_indices" in data:
            kwargs["extreme_cluster_indices"] = tuple(data["extreme_cluster_indices"])
        # Reference fields
        if "cluster_config" in data:
            kwargs["cluster_config"] = ClusterConfig.from_dict(data["cluster_config"])
        if "segment_config" in data:
            kwargs["segment_config"] = SegmentConfig.from_dict(data["segment_config"])
        if "extremes_config" in data:
            kwargs["extremes_config"] = ExtremeConfig.from_dict(data["extremes_config"])
        return cls(**kwargs)

    def to_json(self, path: str) -> None:
        """Save clustering result to a JSON file.

        Parameters
        ----------
        path : str
            File path to save to.

        Notes
        -----
        If the clustering used the 'replace' extreme method, a warning will be
        issued because the saved clustering cannot be perfectly reproduced when
        loaded and applied later. See :meth:`apply` for details.

        Examples
        --------
        >>> result.clustering.to_json("clustering.json")
        """
        import json

        # Warn if using replace extreme method (transfer is not exact)
        if (
            self.extremes_config is not None
            and self.extremes_config.method == "replace"
        ):
            warnings.warn(
                "Saving a clustering that used the 'replace' extreme method. "
                "The 'replace' method creates a hybrid cluster representation "
                "(some columns from the medoid, some from the extreme period) that "
                "cannot be perfectly reproduced when loaded and applied later. "
                "For exact transfer, use 'append' or 'new_cluster' extreme methods.",
                UserWarning,
                stacklevel=2,
            )

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> ClusteringResult:
        """Load clustering result from a JSON file.

        Parameters
        ----------
        path : str
            File path to load from.

        Returns
        -------
        ClusteringResult
            Loaded clustering result.

        Examples
        --------
        >>> clustering = ClusteringResult.from_json("clustering.json")
        >>> result = clustering.apply(new_data)
        """
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))

    def apply(
        self,
        data: pd.DataFrame,
        *,
        temporal_resolution: float | None = None,
        round_decimals: int | None = None,
        numerical_tolerance: float = 1e-13,
    ) -> AggregationResult:
        """Apply this clustering to new data.

        Uses the stored cluster assignments and transfer fields to aggregate
        a different dataset with the same clustering structure deterministically.

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data with a datetime index.
            Must have the same number of periods as the original data.

        temporal_resolution : float, optional
            Time resolution of input data in hours.
            If not provided, uses stored temporal_resolution or infers from data index.

        round_decimals : int, optional
            Round output values to this many decimal places.

        numerical_tolerance : float, default 1e-13
            Tolerance for numerical precision issues.

        Returns
        -------
        AggregationResult
            Aggregation result using this clustering.

        Notes
        -----
        **Extreme period transfer limitations:**

        The 'replace' extreme method creates a hybrid cluster representation where
        some columns use the medoid values and others use the extreme period values.
        This hybrid representation cannot be perfectly reproduced during transfer.
        When applying a clustering that used 'replace', a warning will be issued
        and the transferred result will use the medoid representation for all columns.

        For exact transfer with extreme periods, use 'append' or 'new_cluster'
        extreme methods instead.

        Examples
        --------
        >>> # Cluster on wind data, apply to full dataset
        >>> result_wind = tsam.aggregate(df_wind, n_clusters=8)
        >>> result_all = result_wind.clustering.apply(df_all)

        >>> # Load saved clustering and apply
        >>> clustering = ClusteringResult.from_json("clustering.json")
        >>> result = clustering.apply(df)
        """
        # Import here to avoid circular imports
        from tsam.api import _build_old_params
        from tsam.exceptions import LegacyAPIWarning
        from tsam.result import AccuracyMetrics, AggregationResult
        from tsam.timeseriesaggregation import TimeSeriesAggregation

        # Warn if using replace extreme method (transfer is not exact)
        if (
            self.extremes_config is not None
            and self.extremes_config.method == "replace"
        ):
            warnings.warn(
                "The 'replace' extreme method creates a hybrid cluster representation "
                "(some columns from the medoid, some from the extreme period) that cannot "
                "be perfectly reproduced during transfer. The transferred result will use "
                "the medoid representation for all columns instead of the hybrid values. "
                "For exact transfer, use 'append' or 'new_cluster' extreme methods.",
                UserWarning,
                stacklevel=2,
            )

        # Use stored temporal_resolution if not provided
        effective_temporal_resolution = (
            temporal_resolution
            if temporal_resolution is not None
            else self.temporal_resolution
        )

        # Validate n_timesteps_per_period matches data
        # Infer timestep duration from data if not provided
        if effective_temporal_resolution is None:
            if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
                inferred = (data.index[1] - data.index[0]).total_seconds() / 3600
            else:
                inferred = 1.0  # Default to hourly
        else:
            inferred = effective_temporal_resolution

        inferred_timesteps = int(self.period_duration / inferred)
        if inferred_timesteps != self.n_timesteps_per_period:
            raise ValueError(
                f"Data has {inferred_timesteps} timesteps per period "
                f"(period_duration={self.period_duration}h, timestep={inferred}h), "
                f"but clustering expects {self.n_timesteps_per_period} timesteps per period"
            )

        # Validate number of periods matches
        n_periods_in_data = len(data) // self.n_timesteps_per_period
        if n_periods_in_data != self.n_original_periods:
            raise ValueError(
                f"Data has {n_periods_in_data} periods, "
                f"but clustering expects {self.n_original_periods} periods"
            )

        # Build minimal ClusterConfig with just the representation.
        # We intentionally ignore stored cluster_config.weights since:
        # 1. Weights were only used to compute the original assignments
        # 2. Assignments are now fixed, so weights are irrelevant
        # 3. New data may have different columns than the original
        cluster = ClusterConfig(representation=self.representation)

        # Use stored segment config if available, otherwise build from transfer fields
        segments: SegmentConfig | None = None
        n_segments: int | None = None
        if self.segment_assignments is not None and self.segment_durations is not None:
            n_segments = len(self.segment_durations[0])
            segments = self.segment_config or SegmentConfig(
                n_segments=n_segments,
                representation=self.segment_representation or "mean",
            )

        # Build old API parameters, passing predefined values directly
        # Note: Don't pass extremes config - extreme clusters are handled via
        # extreme_cluster_indices and representations are computed from
        # the periods assigned to those clusters in cluster_assignments
        old_params = _build_old_params(
            data=data,
            n_clusters=self.n_clusters,
            period_duration=self.period_duration,
            temporal_resolution=effective_temporal_resolution,
            cluster=cluster,
            segments=segments,
            extremes=None,
            preserve_column_means=self.preserve_column_means,
            rescale_exclude_columns=list(self.rescale_exclude_columns)
            if self.rescale_exclude_columns
            else None,
            round_decimals=round_decimals,
            numerical_tolerance=numerical_tolerance,
            # Predefined values from this ClusteringResult
            predef_cluster_assignments=self.cluster_assignments,
            predef_cluster_centers=self.cluster_centers,
            predef_extreme_cluster_indices=self.extreme_cluster_indices,
            predef_segment_assignments=self.segment_assignments,
            predef_segment_durations=self.segment_durations,
            predef_segment_centers=self.segment_centers,
        )

        # Run aggregation using old implementation (suppress deprecation warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LegacyAPIWarning)
            agg = TimeSeriesAggregation(**old_params)
            cluster_representatives = agg.createTypicalPeriods()

        # Rename index levels for consistency with new API terminology
        cluster_representatives = cluster_representatives.rename_axis(
            index={"PeriodNum": "cluster", "TimeStep": "timestep"}
        )

        # Build accuracy metrics
        accuracy_df = agg.accuracyIndicators()

        # Build rescale deviations DataFrame
        rescale_deviations_dict = getattr(agg, "_rescaleDeviations", {})
        if rescale_deviations_dict:
            rescale_deviations = pd.DataFrame.from_dict(
                rescale_deviations_dict, orient="index"
            )
            rescale_deviations.index.name = "column"
        else:
            rescale_deviations = pd.DataFrame(
                columns=["deviation_pct", "converged", "iterations"]
            )

        accuracy = AccuracyMetrics(
            rmse=accuracy_df["RMSE"],
            mae=accuracy_df["MAE"],
            rmse_duration=accuracy_df["RMSE_duration"],
            rescale_deviations=rescale_deviations,
        )

        # Build ClusteringResult - preserve stored values
        from tsam.api import _build_clustering_result

        clustering_result = _build_clustering_result(
            agg=agg,
            n_segments=n_segments,
            cluster_config=cluster,
            segment_config=segments,
            extremes_config=self.extremes_config,
            preserve_column_means=self.preserve_column_means,
            rescale_exclude_columns=list(self.rescale_exclude_columns)
            if self.rescale_exclude_columns
            else None,
            temporal_resolution=effective_temporal_resolution,
        )

        # Build result object
        return AggregationResult(
            cluster_representatives=cluster_representatives,
            cluster_weights=dict(agg.clusterPeriodNoOccur),
            n_timesteps_per_period=agg.timeStepsPerPeriod,
            segment_durations=self.segment_durations,
            accuracy=accuracy,
            clustering_duration=getattr(agg, "clusteringDuration", 0.0),
            clustering=clustering_result,
            is_transferred=True,
            _aggregation=agg,
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

    include_in_count : bool, default False
        Whether extreme periods count toward n_clusters.
        - False (default): Extremes are added on top of n_clusters
          (e.g., n_clusters=10 + 2 extremes = 12 final clusters)
        - True: Extremes are included in n_clusters
          (e.g., n_clusters=10 with 2 extremes = 8 from clustering + 2 extremes)
        Only valid with "append" or "new_cluster" methods (not "replace").
    """

    method: ExtremeMethod = "append"
    max_value: list[str] = field(default_factory=list)
    min_value: list[str] = field(default_factory=list)
    max_period: list[str] = field(default_factory=list)
    min_period: list[str] = field(default_factory=list)
    include_in_count: bool = False

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
        if self.include_in_count:
            result["include_in_count"] = self.include_in_count
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
            include_in_count=data.get("include_in_count", False),
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
    "distribution": "distributionRepresentation",
    "distribution_minmax": "distributionAndMinMaxRepresentation",
    "minmax_mean": "minmaxmeanRepresentation",
}

EXTREME_METHOD_MAPPING: dict[ExtremeMethod, str] = {
    "append": "append",
    "replace": "replace_cluster_center",
    "new_cluster": "new_cluster_center",
}
