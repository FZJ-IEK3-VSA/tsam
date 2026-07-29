# Migrating from v3 to v4 { #migration-v3-to-v4 }

tsam v4 is a pipeline rewrite of the internals. **The legacy class-based
`TimeSeriesAggregation` API has been removed** — use the `tsam.aggregate()`
function instead. If you are coming from v2, start with the
[v2 to v3 guide](v2-to-v3.md), which maps every old parameter to its
current equivalent, then return here.

There are also behavioral changes that may affect your results.

## Weight semantics

`weights` (top-level parameter to `aggregate()`) now affects
**only** the clustering distance calculation. Previously, weights were baked
into normalized data, which forced rescaling, reconstruction, and accuracy
computation to compensate. In v4, all those steps operate on unweighted data.

**What changes:**

- Cluster *assignments* are identical (the weighted distance matrix is
  mathematically equivalent).
- With medoid or maxoid representation and non-uniform weights, the selected
  representative may differ because the medoid is now chosen in the unweighted
  output space.
- Across all golden regression tests, the **only** affected configuration is
  `hierarchical_weighted` — everything else is bit-identical.

**Action required:** If you use non-uniform `weights` with medoid or maxoid
representation, verify that your downstream results are acceptable. For most
users, this change is invisible.

## Column order (new API only)

`cluster_representatives`, `reconstructed`, and `original` now return
columns in the same order as the input DataFrame. Previously (v3), columns
were alphabetically sorted.

**Action required:** If your code indexes columns by position (e.g.,
`df.iloc[:, 0]`), verify that the order matches your expectation. To keep the
old behavior, sort the columns yourself: `result.cluster_representatives.sort_index(axis=1)`.

## Resolution defaults for non-datetime indices

When the input DataFrame does **not** have a `DatetimeIndex` and no
`temporal_resolution` is supplied, `aggregate()` now defaults to an hourly
resolution (1 hour per timestep). Previously the legacy API raised a
`ValueError` ("`'resolution' argument has to be nonnegative float or int or the
given timeseries needs a datetime index`").

**Action required:** If you rely on a non-datetime index without passing
`temporal_resolution`, pass it explicitly (e.g. `temporal_resolution='15min'`)
to be sure the timestep length matches your data instead of defaulting to 1h.

## Integral and min/max preservation

Two internal corrections that could push values outside the input envelope or
shift the integral (per-attribute sum) were replaced by a bounded *water-fill*
that redistributes values within the `[min, max]` envelope instead of flattening
them against the cap. This is two independent changes with different scopes:

1. **Min/max representation** (`distribution_minmax`, i.e.
   `Distribution(preserve_minmax=True)`). By code path, this affects **only**
   representations that preserve min/max — `mean`, `medoid`, `maxoid`,
   `minmax_mean`, and plain `distribution` never execute this code. Changes:
    - The integral is now preserved when pinning the per-cluster min/max
      (previously it could drift by a few percent, especially without rescaling).
    - Single-value segments (`distribution_minmax` with `n_segments`) keep the
      segment **mean** rather than being pushed to the segment maximum, since one
      value cannot carry both the minimum and the maximum. Because a segment is a
      single value, `Distribution(scope="local")` as a *segment* representation is
      equivalent to `"mean"`, and `preserve_minmax` only takes effect with
      `scope="global"` — `SegmentConfig` now emits a `UserWarning` if you set
      `preserve_minmax=True` with `scope="local"`.
2. **Cluster-period rescaling.** Because `preserve_column_means` **defaults to
   `True`**, this step runs in almost every aggregation, on the shared path for
   **every** representation — it is not an opt-in corner case. It is identical to
   the old behavior *unless* a representative value would have been clipped
   against the `[0, scale_ub]` envelope; where that clipping occurred, values are
   now redistributed instead of flattened, so the integral is better preserved.
   So "I only use the defaults" does not by itself mean your results are
   unchanged — it depends on whether any representative hit the envelope.

**What changes:**

- Results differ for `distribution_minmax` (new min/max algorithm) and for any
  configuration where rescaling clipped values against the envelope. The largest
  improvements are for `distribution_minmax` combined with segmentation.
- The differences are usually small relative to the values but can reach ~20% at
  an individual peak cell, and are **not** always accompanied by the
  "maximal value … exceeds" warning.
- On the packaged example datasets, the common `mean` / `medoid` / `kmeans`
  paths are bit-identical to v3; the observed changes are confined to
  `distribution_minmax`, `maxoid`/`kmaxoids`, and rescale-heavy configurations
  (contiguous, extremes). Note this is an observation on the example data, not a
  guarantee — the rescaling change above can in principle affect any
  representation on other data.

**Action required:** If you use `distribution_minmax`, `maxoid`/`kmaxoids`, or
rely on exact aggregated values with rescaling enabled, regenerate any pinned
references. Aggregate metrics (integral, min/max envelope) are preserved or
improved.

## Cluster and segment representations are resolved independently

`ClusterConfig.representation` and `SegmentConfig.representation` are two
separate settings, but v3 resolved them into a **single** per-column
`representationDict`, and the segment representation was applied last. Setting a
`MinMaxMean` on `SegmentConfig` therefore silently overwrote the cluster
representation's per-column assignment:

```python
tsam.aggregate(
    data,
    n_clusters=8,
    cluster=ClusterConfig(representation=MinMaxMean(max_columns=["GHI"])),
    segments=SegmentConfig(n_segments=6, representation=MinMaxMean(min_columns=["Load"])),
)
# v3: BOTH stages used {Load: min}; the cluster's {GHI: max} was discarded.
# v4: the cluster stage uses {GHI: max}, the segment stage uses {Load: min}.
```

Each stage now builds its own dict from its own configuration.

**What changes:**

- Configurations that set a `MinMaxMean` (or a `Distribution` with per-column
  effects) on **both** `ClusterConfig` and `SegmentConfig` now honour both.
  Previously the cluster one was dropped, so cluster representatives change.
- Configurations that set it on only one of the two are unaffected — this is the
  overwhelming majority, and every golden regression case is bit-identical
  except the one added to cover this.

**Action required:** If you set a per-column representation on both
`ClusterConfig` and `SegmentConfig`, your cluster representatives were
previously computed with the *segment* column assignment. They are now computed
with the one you asked for. Regenerate any pinned references.

## Representative selection is deterministic on ties

`medoid` and `maxoid` pick a cluster member by the extreme of a summed-distance
array. Ties in that array are common rather than exotic — every member of a
two-member group is equidistant from the others — and v3 resolved them with a
bare `argmin`/`argmax`, so the winner was decided by the order the squared
differences happened to be accumulated in. It could change with the column
order of the input, the BLAS build, or the array layout: the same code on the
same data could pick a different representative on a different machine.

Distances are now rounded before comparison, so noise-level differences count
as ties and the earliest member wins, identically everywhere.

**What changes:** results move only where the previous choice was undefined. On
the packaged example data, 25 of 464 medoid selections sit on a tie and two
were close enough to flip; one golden regression case changed as a result.

**Action required:** none, unless you pinned a value that happened to fall on a
tie. If you did, it was not reproducible across machines to begin with.

## Period-sum features (`include_period_sums`)

Two corrections, both affecting only configurations that switch this on:

- **With `weights`.** The appended per-column sums are computed from the
  *weighted* profiles again, so a column's sum feature carries the same weight
  as its timestep features. In between they were summed unweighted and appended
  to weighted candidates, which meant the higher a column's weight, the *less*
  its period sum counted relative to its own timesteps.
- **With `use_duration_curves`.** The sum features no longer reach the
  duration-curve path. That path reshapes its input into one block of timesteps
  per column, and the appended block is not made of timesteps, so including it
  shifted every column's block and sorted across column boundaries. Period sums
  do not influence duration-curve clustering, as in v3.

**Action required:** if you use `include_period_sums` together with `weights` or
`use_duration_curves`, regenerate any pinned references.

## Duration-curve clustering (`use_duration_curves`)

`ClusteringResult.cluster_centers` now identifies the periods the returned
centers were actually taken from. Previously the indices came from a different
criterion than the centers — and were `None` entirely for representations that
are computed rather than selected — so replaying a stored clustering could
produce different typical periods than the original run.

**Action required:** none for a fresh aggregation; the typical periods are
unchanged. Transfers of a duration-curve clustering now reproduce the original
run, where before they silently did not.

## Transferring a clustering (`ClusteringResult.apply()`)

- **`ClusterConfig` is replayed in full.** `apply()` rebuilt a minimal
  configuration from the representation alone, so settings that shape the data
  rather than the assignment — `scale_by_column_means` above all — reverted to
  their defaults and the transferred result came back silently rescaled.
- **A padded partial last period is accepted.** The period count is rounded up
  to match how the pipeline counts, so a clustering built from a series that
  does not fill whole periods can be applied at all. Previously it raised
  regardless of the data given to it, including its own input.
- **Inexact transfers warn.** `apply()` and `to_json()` now warn for both
  configurations that cannot be replayed exactly: `extremes="replace"`, and
  `extremes="append"`/`"new_cluster"` combined with a representation that is
  computed rather than selected. The v3 advice to "use `append` or
  `new_cluster` for exact transfer" was only true for the selected
  representations (`medoid`, `maxoid`).

**Action required:** if you transfer a clustering built with
`scale_by_column_means=True`, the result changes — it is now the one you
configured.

## Configurations that now raise or warn

Three previously silent cases are reported:

| Configuration | v3 | v4 |
|---|---|---|
| A column in both `MinMaxMean.max_columns` and `min_columns` | resolved by whichever loop ran last | `ValueError` |
| `MinMaxMean` naming a column that is not in the data | silently ignored | `ValueError` |
| `ClusterConfig.representation` with `use_duration_curves=True` | silently ignored | `UserWarning` |

A series whose length is not a whole number of periods also emits a
`UserWarning` now. This is still valid input — the last period is padded, its
occurrence count reduced accordingly, and the padding dropped on reconstruction
— but the padding is no longer invisible.

**Action required:** a typo in a `MinMaxMean` column name used to be a no-op and
is now an error. If you relied on that, fix the name.

## Removed deprecated APIs

The v3 deprecation shims have been **removed** in v4:

| Removed | Use instead |
|---------|-------------|
| `AggregationResult.cluster_weights` | `AggregationResult.cluster_counts` |
| `ClusterConfig(normalize_column_means=...)` | `ClusterConfig(scale_by_column_means=...)` |
| `ClusterConfig(weights=...)` | top-level `aggregate(..., weights={...})` |
| Verbose representation names (`"meanRepresentation"`, `"distributionRepresentation"`, …) | short names (`"mean"`, `"distribution"`, …) — see [representation values](v2-to-v3.md#representation-method-values) |
| `LegacyAPIWarning` | — (no longer needed; the legacy API is gone) |
| `tsam.weights.MIN_WEIGHT` | `tsam.options.min_weight` |

In v4, per-column `weights` are a **top-level input to `aggregate()`**, not part
of `ClusterConfig` — they are an aggregation parameter, not clustering
configuration. Passing `weights=` to `ClusterConfig` now raises `TypeError`.

## Newly deprecated (alias kept for one release)

- `result.plot.cluster_weights()` is renamed to `result.plot.cluster_counts()`,
  matching the `AggregationResult.cluster_counts` attribute. The old name still
  works but emits a `FutureWarning` and will be removed in a future release.
- `Distribution(scope="cluster")` is renamed to `Distribution(scope="local")`.
  `"cluster"` was misleading for **segment** representations, where the group
  whose distribution is preserved is a segment, not a cluster; `"local"` is
  stage-neutral (each group's own distribution) versus `"global"` (the enclosing
  whole's). The old value still works — it is normalized to `"local"` and emits a
  `FutureWarning` — and is behaviorally identical.

## Internal changes (no action required)

- The pipeline has been decomposed into stateless functions in
  `src/tsam/pipeline/`. `tsam.aggregate()` delegates to `run_pipeline()`.
- All internal identifiers have been renamed from camelCase to snake_case.
