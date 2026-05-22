"""Generate the component-architecture diagram from the tsam source tree.

This produces the Mermaid diagram shown on
docs/background/architecture/components.md, *semi-automatically*:

  * the **structure** — which modules exist and which imports which — is read
    straight from the code, so the diagram can never silently drift from
    reality;
  * the **grouping** is supplied by a one-line annotation in each module::

        __architecture_group__ = "Pipeline stages"

A module with no ``__architecture_group__`` is placed in an "Unannotated"
subgraph — so a newly added module always shows up and asks to be sorted,
rather than going silently missing. To deliberately keep a module out of the
diagram entirely, annotate it with ``None``::

        __architecture_group__ = None

Modules are read with the ``ast`` module — parsed, never imported — so this
script is safe to run without the package installed and has no import side
effects. (Only absolute ``tsam.*`` imports are followed; tsam uses absolute
imports throughout.)

Usage — run from the repository root::

    python scripts/gen_component_diagram.py              # print to stdout
    python scripts/gen_component_diagram.py -o out.md    # write to a file

The output is a ready-to-paste ```kroki-mermaid fenced block.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- configuration --------------------------------------------------------

PACKAGE_DIR = Path("src/tsam")
TOP_PACKAGE = "tsam"

# Files that are never part of the architecture picture.
SKIP_FILENAMES = {"_version.py"}

# Subgraph order in the rendered diagram. Groups not listed here are appended
# in first-seen order; "Unannotated" is always forced last.
GROUP_ORDER = [
    "Public API",
    "Config & Results",
    "Pipeline orchestrator",
    "Pipeline stages",
    "Clustering backends",
    "Auxiliary",
]
UNANNOTATED = "Unannotated"

# Sentinel: the __architecture_group__ variable is absent from a module.
_ABSENT = object()


@dataclass
class ModuleInfo:
    dotted: str  # e.g. "tsam.pipeline.clustering"
    is_package: bool  # True if the file was an __init__.py
    group: object  # str | None | _ABSENT
    imports: set[str] = field(default_factory=set)


# --- source scanning ------------------------------------------------------


def dotted_name(path: Path) -> tuple[str, bool]:
    """tsam-rooted dotted name + is_package flag.

    .../tsam/pipeline/clustering.py -> ("tsam.pipeline.clustering", False)
    .../tsam/pipeline/__init__.py   -> ("tsam.pipeline", True)
    """
    rel = path.relative_to(PACKAGE_DIR.parent).with_suffix("")
    parts = list(rel.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def read_architecture_group(tree: ast.Module) -> object:
    """Return the module's __architecture_group__ value, or _ABSENT."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__architecture_group__":
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    return _ABSENT
    return _ABSENT


def read_imports(tree: ast.Module) -> set[str]:
    """Dotted names of tsam modules imported by this module (absolute only)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_tsam(alias.name):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module or not _is_tsam(node.module):
                continue
            found.add(node.module)
            # `from tsam.pkg import name` may be importing a submodule `name`.
            for alias in node.names:
                found.add(f"{node.module}.{alias.name}")
    return found


def _is_tsam(name: str) -> bool:
    return name == TOP_PACKAGE or name.startswith(TOP_PACKAGE + ".")


def scan() -> dict[str, ModuleInfo]:
    """Parse every module under the package directory."""
    if not PACKAGE_DIR.is_dir():
        sys.exit(f"error: {PACKAGE_DIR} not found — run from the repo root")
    modules: dict[str, ModuleInfo] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path.name in SKIP_FILENAMES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dotted, is_package = dotted_name(path)
        modules[dotted] = ModuleInfo(
            dotted=dotted,
            is_package=is_package,
            group=read_architecture_group(tree),
            imports=read_imports(tree),
        )
    return modules


# --- mermaid emission -----------------------------------------------------


def _node_id(dotted: str) -> str:
    return dotted.replace(".", "_")


def _group_id(group: str) -> str:
    return "g_" + group.lower().replace(" ", "_").replace("&", "and")


def _node_label(info: ModuleInfo) -> str:
    """Readable label — the path relative to the package. A package is shown
    as 'pkg/' to avoid '__init__.py' (whose underscores render as markdown)."""
    inside = info.dotted[len(TOP_PACKAGE) + 1 :].replace(".", "/")
    if not inside:  # the top-level package itself
        return f"{TOP_PACKAGE}/"
    return f"{inside}/" if info.is_package else f"{inside}.py"


def build_diagram(modules: dict[str, ModuleInfo]) -> str:
    # Modules included in the diagram (group is not explicitly None).
    included = {m: info for m, info in modules.items() if info.group is not None}

    # Assign each included module to a subgraph.
    members: dict[str, list[str]] = {}
    for name, info in included.items():
        group = UNANNOTATED if info.group is _ABSENT else str(info.group)
        members.setdefault(group, []).append(name)

    # Order subgraphs: known groups first, then any extras, Unannotated last.
    ordered = [g for g in GROUP_ORDER if g in members]
    ordered += sorted(g for g in members if g not in GROUP_ORDER and g != UNANNOTATED)
    if UNANNOTATED in members:
        ordered.append(UNANNOTATED)

    # Edges: only between included modules, no self-loops.
    edges: set[tuple[str, str]] = set()
    for name, info in included.items():
        for target in info.imports:
            if target in included and target != name:
                edges.add((name, target))

    lines = [
        "```kroki-mermaid",
        "%%{init: {'flowchart': {'padding': 18}}}%%",
        "graph LR",
    ]
    for group in ordered:
        lines.append(f'    subgraph {_group_id(group)}["{group}"]')
        for name in sorted(members[group]):
            lines.append(f'        {_node_id(name)}["{_node_label(included[name])}"]')
        lines.append("    end")
        lines.append("")
    for src, dst in sorted(edges):
        lines.append(f"    {_node_id(src)} --> {_node_id(dst)}")
    lines.append("")
    # Theme-neutral cluster styling — Kroki cannot theme subgraph clusters
    # (see docs/background/architecture/components.md).
    for group in ordered:
        lines.append(f"    style {_group_id(group)} fill:none,stroke:#8a8a8a")
    lines.append("```")
    return "\n".join(lines) + "\n"


# --- entry point ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the diagram to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    modules = scan()
    diagram = build_diagram(modules)

    if args.output:
        args.output.write_text(diagram, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(diagram)

    # Report unannotated modules on stderr so they are noticed.
    unannotated = sorted(
        m for m, info in modules.items() if info.group is _ABSENT
    )
    if unannotated:
        print(
            f"\n{len(unannotated)} module(s) without __architecture_group__:",
            file=sys.stderr,
        )
        for name in unannotated:
            print(f"  - {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
