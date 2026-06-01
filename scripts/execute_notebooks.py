"""Execute the docs notebooks in parallel and write outputs in place.

The mkdocs build is configured with ``execute: false`` so that mkdocs-jupyter
only renders pre-executed notebooks. This script is the producer step: it
discovers ``docs/notebooks/*.ipynb``, runs each in its own kernel via
``nbclient``, and writes the executed notebook back in place.

Each notebook gets its own kernel subprocess, so a ThreadPoolExecutor is
sufficient — the GIL is not the bottleneck. Notebooks run in their own
directory (``cwd = notebook.parent``) so relative paths inside cells behave
the same as in Jupyter Lab.

Usage:
    python scripts/execute_notebooks.py                 # default: CPU-1 workers
    python scripts/execute_notebooks.py --workers 4
    python scripts/execute_notebooks.py --exclude tuning.ipynb
    python scripts/execute_notebooks.py --timeout 1200

Exit code is non-zero if any notebook fails. Cell errors abort that notebook
but do not stop the rest of the batch — the summary at the end lists failures.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Render Plotly figures as inline <div>s via the "notebook_connected" renderer:
# the figure flows with the page and auto-sizes, and Plotly.js is pulled once
# from the CDN. This relies on the docs NOT using mkdocs-material's instant
# navigation (navigation.instant is intentionally disabled in mkdocs.yml) —
# under instant nav the renderer's <script> would not re-run on in-page swaps.
# Set via env var so notebooks pick it up without per-notebook setup cells.
# Notebooks that set pio.renderers.default themselves will override this.
os.environ.setdefault("PLOTLY_RENDERER", "notebook_connected")


@dataclass
class Result:
    path: Path
    elapsed: float
    error: str | None


def execute_one(path: Path, timeout: int) -> Result:
    start = time.perf_counter()
    try:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        nbformat.write(nb, path)
        return Result(path, time.perf_counter() - start, None)
    except CellExecutionError as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else "CellExecutionError"
        return Result(path, time.perf_counter() - start, first_line)
    except Exception as exc:
        return Result(path, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path("docs/notebooks"),
        help="Directory containing notebooks (default: docs/notebooks)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="FILENAME",
        help="Notebook filename to skip (repeat for multiple)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.notebooks_dir.is_dir():
        print(f"error: {args.notebooks_dir} is not a directory", file=sys.stderr)
        return 2

    notebooks = sorted(
        p for p in args.notebooks_dir.glob("*.ipynb") if p.name not in args.exclude
    )
    if not notebooks:
        print(f"No notebooks to execute in {args.notebooks_dir}", file=sys.stderr)
        return 1

    print(
        f"Executing {len(notebooks)} notebooks with {args.workers} worker(s), "
        f"per-cell timeout {args.timeout}s",
        flush=True,
    )

    overall_start = time.perf_counter()
    results: list[Result] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_one, nb, args.timeout): nb for nb in notebooks}
        for fut in as_completed(futures):
            r = fut.result()
            tag = "FAIL" if r.error else "OK  "
            print(f"  [{tag}] {r.path.name:<40} {r.elapsed:6.1f}s", flush=True)
            results.append(r)

    total = time.perf_counter() - overall_start
    failures = [r for r in results if r.error]
    print(
        f"\nTotal: {total:.1f}s ({len(results) - len(failures)}/{len(results)} succeeded)"
    )
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r.path}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
