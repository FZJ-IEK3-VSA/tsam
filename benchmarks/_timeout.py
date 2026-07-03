"""Hard wall-clock timeout for a single ``aggregate`` call.

Runs the aggregation in a spawned subprocess and terminates it if it overruns —
the only reliable way to bound methods (e.g. the exact-MILP ``kmedoids``) whose
heavy work happens in C extensions that ignore Python-level signals. Used by the
method-comparison grid to drop cases that would otherwise explode the runtime;
dropped cases simply go missing from the results (NaN in the figure).
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path


def _worker(
    timesteps: int,
    columns: int,
    method: str,
    period_hours: int,
    resolution_hours: float,
    clusters: int,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    from _data import make_data

    import tsam
    from tsam import ClusterConfig

    np.random.seed(42)  # determinism for kmeans / kmaxoids
    data = make_data(timesteps, columns, resolution_hours=resolution_hours)
    result = tsam.aggregate(
        data,
        n_clusters=clusters,
        period_duration=period_hours,
        temporal_resolution=resolution_hours,
        cluster=ClusterConfig(method=method),
    )
    _ = result.reconstructed


def completes_within(
    timeout_s: float,
    timesteps: int,
    columns: int,
    method: str,
    period_hours: int,
    resolution_hours: float,
    clusters: int,
) -> bool:
    """Return True iff one aggregation finishes within ``timeout_s`` seconds."""
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_worker,
        args=(timesteps, columns, method, period_hours, resolution_hours, clusters),
    )
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    return proc.exitcode == 0
