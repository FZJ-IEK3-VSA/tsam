"""Execute a single notebook and write outputs in place."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

os.environ.setdefault("PLOTLY_RENDERER", "notebook_connected")

nb_path = Path(sys.argv[1]).resolve()
cwd = str(nb_path.parent)

print(f"Executing: {nb_path}")
print(f"Working directory: {cwd}")

nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": cwd}},
)
try:
    client.execute()
    nbformat.write(nb, nb_path)
    print("SUCCESS")
    sys.exit(0)
except CellExecutionError as exc:
    print(f"CELL ERROR:\n{exc}")
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: {type(exc).__name__}: {exc}")
    sys.exit(1)
