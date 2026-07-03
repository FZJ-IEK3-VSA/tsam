"""Make the sibling helper modules (``_data``, ``_timeout``) importable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
