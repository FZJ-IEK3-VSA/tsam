# Migration Guide { #migration-guide }

Upgrading ETHOS.TSAM is split into two guides, one per major step. Pick the one
that matches the version you are coming from.

| You are on | Read | What it covers |
|---|---|---|
| **v3** | [Migrating from v3 to v4](migration/v3-to-v4.md) | The pipeline rewrite: removed deprecation shims, and the behavioral changes (weight semantics, column order, resolution defaults) that can move your results. |
| **v2** | [Migrating from v2 to v3](migration/v2-to-v3.md), then [v3 to v4](migration/v3-to-v4.md) | The move off the class-based `TimeSeriesAggregation` API: a mapping for every old parameter, result access, tuning, and plotting. |

!!! note "Coming from v2? You need both."

    The class-based `TimeSeriesAggregation` API was deprecated in v3 and
    **removed in v4**. The v2 to v3 guide carries the full parameter mapping you
    need to move off it; the v3 to v4 guide then covers what changed underneath.
