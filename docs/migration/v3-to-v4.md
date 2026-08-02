# Migrating from v3 to v4 { #migration-v3-to-v4 }

tsam v4 is a pipeline rewrite of the internals. **The legacy class-based
`TimeSeriesAggregation` API has been removed** — use the `tsam.aggregate()`
function instead. If you are coming from v2, start with the
[v2 to v3 guide](v2-to-v3.md), which maps every old parameter to its
current equivalent, then return here.

Beyond the API removal there are behavioural changes. Most need nothing from
you — these are the ones that do:

| If you… | See |
|---|---|
| set a per-column representation on **both** `ClusterConfig` and `SegmentConfig` | [Cluster and segment representations](#cluster-and-segment-representations-are-resolved-independently) — v3 ignored the cluster one |
| index result columns by position | [Column order](#column-order-new-api-only) |
| pass a non-datetime index without `temporal_resolution` | [Resolution defaults](#resolution-defaults-for-non-datetime-indices) |
| upgrade from **3.4.1 or earlier** and use `distribution_minmax` / `maxoid` | [Integral and min/max preservation](#integral-and-minmax-preservation) |
| transfer a clustering built with `scale_by_column_means` | [Transferring a clustering](#transferring-a-clustering-clusteringresultapply) |
| rely on a `MinMaxMean` column name that does not exist | [Configurations that now raise or warn](#configurations-that-now-raise-or-warn) |

Weight handling and representative tie-breaking also changed, but neither
alters results for anyone coming from 3.4.2.

## Cluster and segment representations are resolved independently

!!! warning "v3 silently ignored a setting you passed"

    If you set a per-column representation (`MinMaxMean`) on **both**
    `ClusterConfig` and `SegmentConfig`, v3 discarded the cluster one entirely.
    Not partially, not merged — ignored. Your cluster representatives were built
    from the *segment* column assignment, whatever you asked for.

    Measured on v3.4.2: `cluster=MinMaxMean(max_columns=["GHI"])` produces
    **byte-identical** output to `cluster=MinMaxMean(min_columns=["Load"])` when
    the segment representation is `min_columns=["Load"]` — the cluster setting
    makes no difference at all. On v4 the two differ by 601.5.

The two are separate settings, but v3 resolved them into a **single** per-column
`representationDict` and applied the segment one last:

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

- Configurations setting a per-column representation on **both** stages now
  honour both. Cluster representatives change — to the ones you configured.
- Configurations setting it on only one stage are unaffected. That is the
  overwhelming majority: every golden regression case is bit-identical except
  the one added to cover this.

**Action required:** If you set a per-column representation on both
`ClusterConfig` and `SegmentConfig`, treat your v3 results for that
configuration as wrong rather than merely different — they were computed with a
column assignment you did not ask for. Regenerate any pinned references, and
re-check any conclusion that rested on them.

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

!!! info "Already released in 3.4.2 — no change if you are upgrading from there"

    This landed on the v3 line as
    [#373](https://github.com/FZJ-IEK3-VSA/tsam/issues/373), released in
    **3.4.2**. Measured against 3.4.2, v4 is identical here: `distribution_minmax`
    agrees to within `1e-14`, with segmentation and with rescaling disabled alike.

    It is a change only if you are coming from **3.4.1 or earlier**, where it
    arrived as a one-line bug-fix entry rather than a flagged behaviour change.

**What changes, coming from 3.4.1 or earlier:**

- Results differ for `distribution_minmax` (new min/max algorithm) and for any
  configuration where rescaling clipped values against the envelope. The largest
  improvements are for `distribution_minmax` combined with segmentation.
- The differences are usually small relative to the values but can reach ~20% at
  an individual peak cell, and are **not** always accompanied by the
  "maximal value … exceeds" warning.
- The common `mean` / `medoid` / `kmeans` paths are bit-identical; the observed
  changes are confined to `distribution_minmax`, `maxoid`/`kmaxoids`, and
  rescale-heavy configurations. That is an observation on the example data, not
  a guarantee — the rescaling change can in principle affect any representation
  on other data.

**Action required:** None if you are on 3.4.2. From 3.4.1 or earlier, if you use
`distribution_minmax`, `maxoid`/`kmaxoids`, or rely on exact aggregated values
with rescaling enabled, regenerate any pinned references. Aggregate metrics
(integral, min/max envelope) are preserved or improved.

## Representative selection is deterministic on ties

`medoid` and `maxoid` picked a cluster member with a bare `argmin`/`argmax` over
summed distances. Ties are common — the two members of a two-member group are
always equidistant — and were decided by floating-point noise, so the same data
could give a different representative on a different machine. Distances are now
rounded before comparison and the earliest member wins.

**Action required:** none, unless you pinned a value that fell on a tie — in
which case it was never reproducible anyway.

## Transferring a clustering (`ClusteringResult.apply()`)

- **The full `ClusterConfig` is replayed.** Only the representation was, so
  `scale_by_column_means` (v3: `normalize_column_means`) reverted to its default
  and the result came back silently rescaled.
- **A padded partial last period is accepted.** Such a clustering previously
  raised for any data, including its own input.
- **`cluster_centers` is correct for `use_duration_curves`.** v3 recorded
  indices from a different criterion than the centers, so a replay could produce
  different typical periods. `aggregate()` itself is unaffected — only the
  recorded indices, and therefore the transfer.
- **Inexact transfers warn.** Both `extremes="replace"` and
  `extremes="append"`/`"new_cluster"` with a *computed* representation
  (`mean`, `distribution`, …) now warn. v3's advice to use `append` or
  `new_cluster` for exact transfer held only for `medoid`/`maxoid`.

**Action required:** if you transfer a clustering built with
`scale_by_column_means=True`, the result changes — to the one you configured.

## Configurations that now raise or warn

| Configuration | v3 | v4 |
|---|---|---|
| Column in both `MinMaxMean.max_columns` and `min_columns` | silently `max` | `ValueError` |
| `MinMaxMean` naming a column not in the data | silently ignored | `ValueError` |
| `ClusterConfig.representation` with `use_duration_curves=True` | silently ignored | `UserWarning` |
| Series length not a whole number of periods | padded silently | padded, with a `UserWarning` |

**Action required:** a typo in a `MinMaxMean` column name used to be a no-op and
is now an error.

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

## Weight semantics — restructured, not changed

`weights` is applied in one place now. v3 multiplied it into the normalized
series before unstacking, so every downstream step saw weighted data and had to
divide it back out again — rescaling, reconstruction and accuracy each carried
their own compensation. v4 applies it to the clustering candidates only, and
removes it once, after the set of representatives is final.

**The same things are weighted either way:** the clustering distance, medoid and
maxoid selection, extreme-period detection, and segment boundaries. And the same
things are unweighted: rescaling, denormalization, the returned values, and the
accuracy metrics. Only the location of the division moved.

**What changes: nothing observable.** Every configuration in the golden matrix
matches v3.4.2 to within `2e-13`, medoid and maxoid under non-uniform weights
included. For `hierarchical_weighted` the agreement is total — same cluster
assignments, same reconstruction, and the same selected medoid periods
(`245, 78, 64, 17, 320, 201, 319, 263`). That is structural rather than lucky:
both versions select the representative from weighted candidates, so the choice
cannot diverge.

**Action required:** None.

## Internal changes (no action required)

- The pipeline has been decomposed into stateless functions in
  `src/tsam/pipeline/`. `tsam.aggregate()` delegates to `run_pipeline()`.
- All internal identifiers have been renamed from camelCase to snake_case.
