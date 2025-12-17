# tsam API Redesign Proposal (v3.0)

## Design Principles

1. **Explicit over implicit** - No hidden defaults based on other parameters
2. **Fail fast** - Raise errors immediately for invalid input, never auto-correct
3. **Grouped parameters** - Related options in config dataclasses
4. **Descriptive names** - snake_case, clear purpose from name alone
5. **Sensible defaults** - Works well with minimal configuration

---

## Proposed API

### Basic Usage (Simple Case)

```python
import tsam

# Minimal - just data and number of periods
result = tsam.aggregate(df, n_periods=8)

# Access results
result.typical_periods      # DataFrame with typical periods
result.cluster_assignments  # Which cluster each original period belongs to
result.accuracy            # RMSE and other metrics
```

### Standard Usage (Common Case)

```python
import tsam

result = tsam.aggregate(
    df,
    n_periods=8,
    period_hours=24,           # Default: 24
    method="hierarchical",      # Default: "hierarchical"
    representation="medoid",    # Default: based on method
)
```

### Advanced Usage (Full Control)

```python
from tsam import aggregate, ClusterConfig, SegmentConfig, ExtremeConfig

result = aggregate(
    df,
    n_periods=8,
    period_hours=24,

    # Clustering configuration
    cluster=ClusterConfig(
        method="hierarchical",      # "kmeans", "kmedoids", "hierarchical", "averaging"
        representation="medoid",    # "mean", "medoid", "distribution"
        weights={"solar": 2.0, "wind": 1.0},
        normalize_means=True,
    ),

    # Optional: Temporal segmentation within periods
    segments=SegmentConfig(
        n_segments=12,
        representation="mean",
    ),

    # Optional: Preserve extreme periods
    extremes=ExtremeConfig(
        method="append",            # "append", "replace", "new_cluster"
        max_timesteps=["demand"],   # Columns where max timestep matters
        min_timesteps=["temperature"],
        max_periods=["solar"],      # Columns where max period sum matters
    ),
)
```

---

## Config Classes

### ClusterConfig

```python
@dataclass
class ClusterConfig:
    """Configuration for the clustering algorithm."""

    method: str = "hierarchical"
    # Options: "averaging", "kmeans", "kmedoids", "kmedoids_exact",
    #          "hierarchical", "contiguous"

    representation: str | None = None
    # Options: "mean", "medoid", "distribution", "distribution_minmax"
    # Default: "mean" for kmeans/averaging, "medoid" for others

    weights: dict[str, float] | None = None
    # Per-column weights for clustering distance calculation

    normalize_means: bool = False
    # Normalize all columns to same mean before clustering

    use_duration_curves: bool = False
    # Sort values within periods before clustering (duration curve matching)

    include_period_sums: bool = False
    # Include period totals as features for clustering

    solver: str = "highs"
    # MILP solver for exact k-medoids: "highs", "cbc", "gurobi", "cplex"
```

### SegmentConfig

```python
@dataclass
class SegmentConfig:
    """Configuration for temporal segmentation within periods."""

    n_segments: int = 12
    # Number of segments per period (must be <= timesteps per period)

    representation: str = "mean"
    # How to represent each segment: "mean", "medoid", "distribution"
```

### ExtremeConfig

```python
@dataclass
class ExtremeConfig:
    """Configuration for preserving extreme periods."""

    method: str = "append"
    # How to handle extreme periods:
    # - "append": Add as additional cluster centers
    # - "replace": Replace nearest cluster center
    # - "new_cluster": Add as new cluster, reassign periods

    max_timesteps: list[str] | None = None
    # Columns where the timestep with maximum value should be preserved

    min_timesteps: list[str] | None = None
    # Columns where the timestep with minimum value should be preserved

    max_periods: list[str] | None = None
    # Columns where the period with maximum sum should be preserved

    min_periods: list[str] | None = None
    # Columns where the period with minimum sum should be preserved
```

---

## Result Object

```python
@dataclass
class AggregationResult:
    """Result of time series aggregation."""

    # Primary outputs
    typical_periods: pd.DataFrame
    # The aggregated typical periods with MultiIndex (period, timestep)

    cluster_assignments: np.ndarray
    # Which cluster each original period belongs to (length = n_original_periods)

    cluster_weights: dict[int, int]
    # How many original periods each cluster represents

    # Metadata
    n_periods: int
    n_timesteps_per_period: int
    n_segments: int | None

    # For analysis
    accuracy: AccuracyMetrics
    # RMSE, MAE, duration curve RMSE per column

    cluster_center_indices: np.ndarray
    # Indices of original periods used as cluster centers

    # Methods
    def reconstruct(self) -> pd.DataFrame:
        """Reconstruct the original time series from typical periods."""

    def to_optimization_input(self) -> dict:
        """Export in format suitable for optimization models."""
```

---

## Parameter Renaming

| Old Name | New Name | Reason |
|----------|----------|--------|
| `timeSeries` | `data` | Simpler, standard |
| `noTypicalPeriods` | `n_periods` | Standard naming |
| `hoursPerPeriod` | `period_hours` | Clearer |
| `clusterMethod` | `method` or `cluster.method` | Grouped |
| `representationMethod` | `representation` | Shorter |
| `noSegments` | `segments.n_segments` | Grouped |
| `sortValues` | `cluster.use_duration_curves` | Descriptive |
| `sameMean` | `cluster.normalize_means` | Descriptive |
| `addPeakMax` | `extremes.max_timesteps` | Clear meaning |
| `addMeanMax` | `extremes.max_periods` | Clear meaning |
| `weightDict` | `cluster.weights` | Grouped |
| `rescaleClusterPeriods` | `rescale` | Simpler |
| `extremePeriodMethod` | `extremes.method` | Grouped |
| `distributionPeriodWise` | Part of representation config | Grouped |

---

## Method Options Renaming

### Clustering Methods

| Old | New | Notes |
|-----|-----|-------|
| `"averaging"` | `"averaging"` | Keep |
| `"k_means"` | `"kmeans"` | Remove underscore |
| `"k_medoids"` | `"kmedoids"` | Remove underscore |
| `"k_maxoids"` | `"kmaxoids"` | Remove underscore |
| `"hierarchical"` | `"hierarchical"` | Keep |
| `"adjacent_periods"` | `"contiguous"` | Clearer name |

### Representation Methods

| Old | New | Notes |
|-----|-----|-------|
| `"meanRepresentation"` | `"mean"` | Simplified |
| `"medoidRepresentation"` | `"medoid"` | Simplified |
| `"maxoidRepresentation"` | `"maxoid"` | Simplified |
| `"durationRepresentation"` | `"distribution"` | Clearer |
| `"distributionRepresentation"` | `"distribution"` | Merged |
| `"distributionAndMinMaxRepresentation"` | `"distribution_minmax"` | Shortened |
| `"minmaxmeanRepresentation"` | `"minmax_mean"` | Shortened |

---

## Validation Behavior

### Old (Silent Auto-Correction)
```python
# Old: silently caps to 24
TimeSeriesAggregation(df, noSegments=100, hoursPerPeriod=24)
```

### New (Fail Fast)
```python
# New: raises ValueError immediately
tsam.aggregate(df, segments=SegmentConfig(n_segments=100))
# ValueError: n_segments (100) cannot exceed timesteps per period (24)
```

---

## Presets for Common Use Cases

```python
# Quick aggregation (fast, less accurate)
result = tsam.aggregate(df, n_periods=8, preset="fast")
# Uses: kmeans, mean representation, no segmentation

# Balanced (default)
result = tsam.aggregate(df, n_periods=8, preset="balanced")
# Uses: hierarchical, medoid representation

# Accurate (slower, preserves distributions)
result = tsam.aggregate(df, n_periods=8, preset="accurate")
# Uses: hierarchical, distribution_minmax, with extremes

# Energy system optimization
result = tsam.aggregate(df, n_periods=8, preset="energy_system")
# Uses: hierarchical, distribution_minmax, extreme preservation, rescaling
```

---

## Migration Example

### Old API
```python
from tsam import TimeSeriesAggregation

agg = TimeSeriesAggregation(
    timeSeries=df,
    noTypicalPeriods=8,
    hoursPerPeriod=24,
    clusterMethod='hierarchical',
    representationMethod='distributionAndMinMaxRepresentation',
    segmentation=True,
    noSegments=12,
    extremePeriodMethod='append',
    addPeakMax=['demand'],
    weightDict={'solar': 2.0},
    rescaleClusterPeriods=True,
)
typical = agg.createTypicalPeriods()
```

### New API
```python
import tsam
from tsam import ClusterConfig, SegmentConfig, ExtremeConfig

result = tsam.aggregate(
    df,
    n_periods=8,
    period_hours=24,
    cluster=ClusterConfig(
        method="hierarchical",
        representation="distribution_minmax",
        weights={"solar": 2.0},
    ),
    segments=SegmentConfig(n_segments=12),
    extremes=ExtremeConfig(
        method="append",
        max_timesteps=["demand"],
    ),
    rescale=True,
)
typical = result.typical_periods
```

---

## Questions for Discussion

1. **Function vs Class**: Should `aggregate()` be a function or should we keep a class?
   - Function: Simpler, stateless, result object has all data
   - Class: Can access intermediate results, more familiar to current users

2. **Config classes vs kwargs**: Should we use dataclasses or just nested dicts?
   - Dataclasses: Type hints, IDE autocomplete, validation
   - Dicts: Simpler, more flexible, JSON-serializable

3. **Preset names**: What presets would be most useful?
   - "fast", "balanced", "accurate"?
   - Domain-specific: "energy_system", "weather", "load_profile"?

4. **Legacy support**: Should we keep the old class with deprecation warnings?
   - Helps migration but adds maintenance burden
