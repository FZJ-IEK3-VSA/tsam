#!/usr/bin/env python3
"""Generate environment.yml from pyproject.toml (single source of truth).

Usage:
    python scripts/sync_environment.py          # write environment.yml
    python scripts/sync_environment.py --check  # check if in sync, exit 1 if not
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
ENV_FILE = ROOT / "environment.yml"


def generate() -> str:
    with open(PYPROJECT, "rb") as f:
        cfg = tomllib.load(f)

    project = cfg["project"]
    deps = project["dependencies"]
    python_requires = project["requires-python"]
    develop_deps = project.get("optional-dependencies", {}).get("develop", [])

    # Split develop deps into categories
    test_deps = [
        d for d in develop_deps if d.startswith(("pytest", "codecov", "nbval"))
    ]
    doc_deps = [d for d in develop_deps if d.startswith(("sphinx", "nbsphinx"))]
    lint_deps = [
        d for d in develop_deps if d.startswith(("ruff", "mypy", "pre-commit"))
    ]

    # Convert python requires to conda format
    # ">=3.10,<3.15" -> ">=3.10,<3.15"
    python_spec = f"python{python_requires}"

    lines = [
        "# AUTO-GENERATED from pyproject.toml — do not edit manually.",
        "# Run: python scripts/sync_environment.py",
        "name: tsam_env",
        "channels:",
        "  - conda-forge",
        "dependencies:",
        f"  - {python_spec}",
        "  - pip",
        "  # Core dependencies",
    ]

    for dep in deps:
        # Normalize: pip uses ==, conda uses =, but >= and <= work in both
        lines.append(f"  - {dep}")

    # Optional deps commonly needed
    plot_deps = project.get("optional-dependencies", {}).get("plot", [])
    for dep in plot_deps:
        lines.append(f"  - {dep}")

    lines.append("  # Testing")
    for dep in test_deps:
        lines.append(f"  - {dep}")

    lines.append("  # Documentation")
    for dep in doc_deps:
        lines.append(f"  - {dep}")

    lines.append("  # Linting and formatting")
    for dep in lint_deps:
        lines.append(f"  - {dep}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    content = generate()

    if "--check" in sys.argv:
        if not ENV_FILE.exists():
            print(f"ERROR: {ENV_FILE} does not exist")
            sys.exit(1)
        current = ENV_FILE.read_text()
        if current != content:
            print(f"ERROR: {ENV_FILE} is out of sync with {PYPROJECT}")
            print("Run: python scripts/sync_environment.py")
            sys.exit(1)
        print("OK: environment.yml is in sync with pyproject.toml")
    else:
        ENV_FILE.write_text(content)
        print(f"Wrote {ENV_FILE}")


if __name__ == "__main__":
    main()
