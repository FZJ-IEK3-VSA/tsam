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

## Internal changes (no action required)

- The pipeline has been decomposed into stateless functions in
  `src/tsam/pipeline/`. `tsam.aggregate()` delegates to `run_pipeline()`.
- All internal identifiers have been renamed from camelCase to snake_case.
