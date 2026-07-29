"""Pure-function aggregation pipeline.

Orchestration lives in [`tsam.pipeline.orchestrator`][]; the stage transforms
live in their own modules (normalize, periods, clustering, extremes, rescale,
segmentation, accuracy).
"""

from __future__ import annotations

from tsam.pipeline.orchestrator import (
    _build_weight_vector,  # noqa: F401 — re-exported for test/test_weight_decoupling.py
    build_result,
    cluster_candidates,
    prepare_data,
    refine_representatives,
    run_pipeline,
)

__all__ = [
    "build_result",
    "cluster_candidates",
    "prepare_data",
    "refine_representatives",
    "run_pipeline",
]
