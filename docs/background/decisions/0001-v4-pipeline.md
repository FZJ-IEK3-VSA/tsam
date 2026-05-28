# 0001 — V4 pipeline replaces `create_typical_periods`

- **Released in:** [v4.0.0](../../changelog.md)

## Context

The primary driver for rewriting the aggregation flow was to make tsam understandable to new developers. Before v4, essentially all aggregation logic lived inside `TimeSeriesAggregation.create_typical_periods()` — a single large method that combined configuration parsing, orchestration, clustering, extreme-period selection, and result assembly in one place. Hidden instance state flowed between conceptually separate steps, which made it hard to reason about any one step in isolation.

As a result, onboarding new contributors was slow: understanding what the method did required reading it start to finish, and any change risked unintended side effects elsewhere. Testing individual stages was also impractical — unit tests had to exercise the full method, making failures hard to diagnose and edge cases hard to cover.

The same coupling that hurt readability also blocked extensibility. Adding a new clustering stage or a different representation method required touching the monolithic method and threading new state through it, raising the risk of regressions.

The rewrite aimed to address three problems. First, improve understandability for new developers by making the pipeline's structure explicit. Second, improve maintainability and testability by separating each stage into a function with a clear input and output. Finally, enable easier extension by making stages composable rather than embedded in shared state.

## Decision

The aggregation flow is implemented as a sequence of **pure functions** organised under [`src/tsam/pipeline/`](https://github.com/FZJ-IEK3-VSA/tsam/tree/develop/src/tsam/pipeline), orchestrated by `run_pipeline()` in `pipeline/__init__.py`. Configuration is passed in as immutable dataclasses (`PipelineConfig` composed of `ClusterConfig`, `ExtremeConfig`, `SegmentConfig`). The user-facing entry point is `tsam.aggregate()` in [`src/tsam/api.py`](https://github.com/FZJ-IEK3-VSA/tsam/blob/develop/src/tsam/api.py), which builds the config, calls the pipeline, and wraps the output as an `AggregationResult`.

