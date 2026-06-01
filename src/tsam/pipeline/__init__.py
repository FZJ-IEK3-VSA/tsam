"""Pure-function aggregation pipeline.

Orchestration lives in [`tsam.pipeline.orchestrator`][]; the stage transforms
live in their own modules (normalize, periods, clustering, extremes, rescale,
segmentation, accuracy).
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
