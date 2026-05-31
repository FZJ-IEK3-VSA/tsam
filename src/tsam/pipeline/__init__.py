"""Pipeline package — pure-function rewrite of create_typical_periods.

Orchestration lives in [`tsam.pipeline.orchestrator`][]; the stage transforms
live in their own modules (normalize, periods, clustering, extremes, rescale,
segment, accuracy). The orchestration entry points are re-exported here for
convenience but are intentionally *not* re-exported from the top-level `tsam`
package — they are internal glue, not public API.
"""

from __future__ import annotations

from tsam.pipeline.orchestrator import (
    _build_weight_vector,  # noqa: F401 — re-exported for test/test_weight_decoupling.py
    assemble_result,
    cluster_and_postprocess,
    format_and_reconstruct,
    prepare_data,
    run_pipeline,
)

__all__ = [
    "assemble_result",
    "cluster_and_postprocess",
    "format_and_reconstruct",
    "prepare_data",
    "run_pipeline",
]
