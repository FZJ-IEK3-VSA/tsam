# 0001 — V4 pipeline replaces `create_typical_periods`

- **Status:** *to be filled in* (Proposed / Accepted)
- **Date:** *YYYY-MM-DD — date of the decision, not the date of writing*
- **Deciders:** *@handles of people who agreed to this*

> **Note to the author (Julian):** this is a stub. The structure is in place; the *why* of the v4 rewrite is something only you (and the people who lived through it) can fill in faithfully. Replace the `TODO` blocks below with the actual rationale. Once filled in and merged, the ADR is append-only — don't edit it later.

## Context

<!-- TODO: describe the situation BEFORE the v4 pipeline existed. Suggested points to cover:
- What did `TimeSeriesAggregation.create_typical_periods()` look like?
- What problems did it have? (size, testability, hidden state, hard to extend, ...)
- What forced the rewrite? (a specific feature request? a bug class? maintenance burden? onboarding pain?)
- Were there alternatives considered short of a rewrite? (incremental refactor, parameter-object pattern, ...)
-->

`TODO`

## Decision

The aggregation flow is implemented as a sequence of **pure functions** organised under [`src/tsam/pipeline/`](https://github.com/FZJ-IEK3-VSA/tsam/tree/develop/src/tsam/pipeline), orchestrated by `run_pipeline()` in `pipeline/__init__.py`. Configuration is passed in as immutable dataclasses (`PipelineConfig` composed of `ClusterConfig`, `ExtremeConfig`, `SegmentConfig`). The user-facing entry point is `tsam.aggregate()` in [`src/tsam/api.py`](https://github.com/FZJ-IEK3-VSA/tsam/blob/develop/src/tsam/api.py), which builds the config, calls the pipeline, and wraps the output as an `AggregationResult`.

The legacy `TimeSeriesAggregation` class is preserved and **also delegates to `run_pipeline()`**, so both APIs share a single engine.

<!-- TODO: confirm or correct the above. Anything I got wrong? Anything important left out? -->

## Consequences

**Positive**

<!-- TODO: what got better? Suggested points:
- testability of individual stages
- ability to add/reorder stages
- single source of truth for behaviour (no drift between legacy and new)
- clearer separation between config, orchestration, and algorithms
- ...
-->

`TODO`

**Negative / costs**

<!-- TODO: what got harder or got locked in? Suggested points:
- maintenance cost of two public APIs
- DataFrame-passing overhead vs. shared state
- bigger surface area to document
- migration burden on users
- ...
-->

`TODO`

**Locked-in choices**

<!-- TODO: what is now hard to change? Suggested points:
- the four-phase orchestration boundary (_prepare / _cluster / _format / _assemble)
- the pure-function contract between stages
- the dataclass-based config shape
- ...
-->

`TODO`

## References

- [Components diagram](../architecture/components.md) — shows the resulting structure.
- [Pipeline Data Flow](../architecture/pipeline-dataflow.md) — shows what flows between stages.
- [Pipeline Guide](../pipeline_guide.md) — step-by-step user-facing walk-through.
- *TODO: link the PR(s) that introduced the v4 pipeline.*
- *TODO: link the migration guide entry, if there is one.*
