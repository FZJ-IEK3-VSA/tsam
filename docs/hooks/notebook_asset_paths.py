"""mkdocs hook: re-resolve relative image paths inside notebook pages.

A relative image path in a notebook is correct in Jupyter/VSCode and wrong on the
built site, because the two resolve it against different directories:

* mkdocs-jupyter renders a notebook by monkey-patching ``page.render`` to assign
  raw nbconvert HTML (``mkdocs_jupyter/plugin.py``). That HTML never passes
  through MkDocs' markdown pipeline, so the ``relpath`` treeprocessor — which
  rewrites ``![](../../x.svg)`` on every ``.md`` page — never sees it. The path
  reaches the browser verbatim.
* With ``use_directory_urls`` (the default), ``docs/a/b/nb.ipynb`` is served at
  ``/a/b/nb/`` — one level deeper than the notebook's own directory.

So a verbatim path is off by exactly one level and 404s. ``docs/explanation/
how-it-works/00_overview.ipynb`` referencing ``../../assets/architecture/x.svg``
resolves to ``/explanation/assets/architecture/x.svg`` instead of
``/assets/architecture/x.svg``.

Rewriting the source to ``../../../`` would fix the site and break the local
preview. Instead, resolve each notebook image the way MkDocs resolves a markdown
one — against the notebook's source directory, then relative to the page URL — so
a single ``../../`` source path works in both places.

Unresolvable targets are logged as warnings, which makes ``mkdocs build
--strict`` fail on a broken notebook image. Strict mode cannot otherwise see
these: to MkDocs the ``<img>`` is opaque nbconvert output.
"""

from __future__ import annotations

import logging
import posixpath
import re

from mkdocs.utils import get_relative_url

logger = logging.getLogger("mkdocs.hooks.notebook_asset_paths")

# src="..." of any <img>. nbconvert emits plain, unquoted-attribute-free tags.
_IMG_SRC = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')

# Paths the browser resolves without our help: absolute URLs, protocol-relative
# URLs, site-root paths, and the data: URIs nbconvert uses for cell outputs
# (every plot in these notebooks is one, so this prefix carries the volume).
_NON_RELATIVE = ("http://", "https://", "//", "/", "data:", "#")


def on_page_content(html: str, *, page, config, files) -> str:
    """Rewrite relative <img src> on notebook pages to be page-URL-relative."""
    if not page.file.src_uri.endswith(".ipynb") or "<img" not in html:
        return html

    src_dir = posixpath.dirname(page.file.src_uri)

    def _rewrite(match: re.Match[str]) -> str:
        prefix, src, suffix = match.groups()
        if src.startswith(_NON_RELATIVE):
            return match.group(0)

        target = posixpath.normpath(posixpath.join(src_dir, src))
        asset = files.get_file_from_path(target)
        if asset is None:
            logger.warning(
                "Image '%s' in '%s' points at '%s', which is not in the docs "
                "directory.",
                src,
                page.file.src_uri,
                target,
            )
            return match.group(0)

        return prefix + get_relative_url(asset.url, page.url) + suffix

    return _IMG_SRC.sub(_rewrite, html)
