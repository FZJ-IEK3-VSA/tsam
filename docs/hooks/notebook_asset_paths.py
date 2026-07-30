"""mkdocs hook: re-resolve relative paths inside notebook pages.

A relative path in a notebook is correct in Jupyter/VSCode and wrong on the
built site, because the two resolve it against different directories:

* mkdocs-jupyter renders a notebook by monkey-patching ``page.render`` to assign
  raw nbconvert HTML (``mkdocs_jupyter/plugin.py``). That HTML never passes
  through MkDocs' markdown pipeline, so the ``relpath`` treeprocessor — which
  rewrites ``![](../../x.svg)`` and ``[](../notation.md)`` on every ``.md`` page
  — never sees it. The path reaches the browser verbatim.
* With ``use_directory_urls`` (the default), ``docs/a/b/nb.ipynb`` is served at
  ``/a/b/nb/`` — one level deeper than the notebook's own directory.

So a verbatim path is off by exactly one level and 404s. This hits both kinds of
relative reference a notebook can make:

* ``<img src>`` — ``docs/explanation/how-aggregation-works/00_overview.ipynb``
  referencing ``../../assets/architecture/x.svg`` resolves to
  ``/explanation/assets/architecture/x.svg`` instead of
  ``/assets/architecture/x.svg``.
* ``<a href>`` — the same notebook linking ``01_preprocessing.ipynb`` resolves
  to ``/explanation/how-aggregation-works/00_overview/01_preprocessing.ipynb``
  instead of ``/explanation/how-aggregation-works/01_preprocessing/``. Links
  additionally need the source extension swapped for the target page's URL,
  which ``File.url`` supplies.

Rewriting the source to ``../../../`` would fix the site and break the local
preview. Instead, resolve each notebook path the way MkDocs resolves a markdown
one — against the notebook's source directory, then relative to the page URL —
so a single ``../../`` source path works in both places.

Unresolvable targets are logged as warnings, which makes ``mkdocs build
--strict`` fail on a broken notebook image or link. Strict mode cannot otherwise
see these: to MkDocs the ``<img>`` and ``<a>`` are opaque nbconvert output.
"""

from __future__ import annotations

import logging
import posixpath
import re

from mkdocs.utils import get_relative_url

logger = logging.getLogger("mkdocs.hooks.notebook_asset_paths")

# src="..." of any <img>, href="..." of any <a>. nbconvert emits plain,
# unquoted-attribute-free tags.
_IMG_SRC = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')
_A_HREF = re.compile(r'(<a\b[^>]*?\bhref=")([^"]+)(")')

# Paths the browser resolves without our help: absolute URLs, protocol-relative
# URLs, site-root paths, and the data: URIs nbconvert uses for cell outputs
# (every plot in these notebooks is one, so this prefix carries the volume).
# A bare "#..." is a same-page anchor, including the anchor-link nbconvert adds
# to every heading.
_NON_RELATIVE = ("http://", "https://", "//", "/", "data:", "#", "mailto:")


def _resolve(src: str, *, kind: str, page, files) -> str | None:
    """Resolve a notebook-relative path to a page-URL-relative one."""
    src_dir = posixpath.dirname(page.file.src_uri)
    target = posixpath.normpath(posixpath.join(src_dir, src))

    file = files.get_file_from_path(target)
    if file is None:
        logger.warning(
            "%s '%s' in '%s' points at '%s', which is not in the docs "
            "directory. Write the path as it stands in the source tree "
            "(e.g. '../other.ipynb'); this hook maps it to the built URL.",
            kind,
            src,
            page.file.src_uri,
            target,
        )
        return None

    return get_relative_url(file.url, page.url)


def on_page_content(html: str, *, page, config, files) -> str:
    """Rewrite relative <img src> / <a href> on notebook pages to page-relative."""
    if not page.file.src_uri.endswith(".ipynb"):
        return html

    def _rewrite_img(match: re.Match[str]) -> str:
        prefix, src, suffix = match.groups()
        if src.startswith(_NON_RELATIVE):
            return match.group(0)

        resolved = _resolve(src, kind="Image", page=page, files=files)
        return match.group(0) if resolved is None else prefix + resolved + suffix

    def _rewrite_link(match: re.Match[str]) -> str:
        prefix, href, suffix = match.groups()
        if href.startswith(_NON_RELATIVE):
            return match.group(0)

        # Keep any fragment: a link may target a heading on the other page.
        path, sep, fragment = href.partition("#")
        resolved = _resolve(path, kind="Link", page=page, files=files)
        return (
            match.group(0)
            if resolved is None
            else prefix + resolved + sep + fragment + suffix
        )

    if "<img" in html:
        html = _IMG_SRC.sub(_rewrite_img, html)
    if "<a" in html:
        html = _A_HREF.sub(_rewrite_link, html)
    return html
