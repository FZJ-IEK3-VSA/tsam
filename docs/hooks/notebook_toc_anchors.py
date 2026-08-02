"""mkdocs hook: point a notebook page's table of contents at the ids that
nbconvert actually emitted.

mkdocs-jupyter builds the right-hand TOC separately from the page body
(``get_nb_toc`` in ``mkdocs_jupyter/plugin.py``): it converts the notebook to
markdown and runs python-markdown's ``toc`` extension over the result, while the
body comes from nbconvert. The two disagree about any heading containing inline
code — python-markdown slugifies the heading *after* its stashed ``<code>`` span
has been removed, nbconvert keeps the code text:

    ## What `mean` clips   ->  body id "what-mean-clips", TOC link "#what-clips"

Every such TOC entry is a dead link. The body ids are the side to keep: they are
what heading permalinks and hand-written cross-page anchors point at. So repair
the TOC instead, pairing its entries with the body's headings in document order
— both are generated from the same headings in the same sequence.

The pairing is only applied when the two agree on how many headings there are
*and* the entry's words are all present in the heading it was matched with;
anything else is logged and left alone, so a layout this hook did not anticipate
degrades to today's behaviour rather than silently rewriting the wrong anchor.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("mkdocs.hooks.notebook_toc_anchors")

_HEADING = re.compile(r'<h([1-6])\b[^>]*?\bid="([^"]+)"[^>]*>(.*?)</h\1>', re.DOTALL)
_ANY_ID = re.compile(r'\bid="([^"]+)"')
_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[0-9a-z]+")

# nbconvert appends this to every heading; it is chrome, not part of the title.
_ANCHOR_LINK = re.compile(r'<a\b[^>]*\bclass="anchor-link"[^>]*>.*?</a>', re.DOTALL)


def _words(html: str) -> set[str]:
    """The lowercase word set of a heading, ignoring markup and punctuation."""
    return set(_WORD.findall(_TAG.sub(" ", _ANCHOR_LINK.sub("", html)).lower()))


def _flatten(items):
    for item in items:
        yield item
        yield from _flatten(item.children)


def _toc_depth(config) -> int:
    plugin = config.plugins.get("mkdocs-jupyter")
    return plugin.config["toc_depth"] if plugin else 6


def on_page_content(html: str, *, page, config, files) -> str:
    """Rewrite dead TOC anchors on notebook pages to the body's heading ids."""
    if not page.file.src_uri.endswith(".ipynb"):
        return html

    entries = list(_flatten(page.toc))
    if not entries:
        return html

    depth = _toc_depth(config)
    headings = [
        (match.group(2), match.group(3))
        for match in _HEADING.finditer(html)
        if int(match.group(1)) <= depth
    ]

    if len(headings) != len(entries):
        logger.warning(
            "'%s' has %d table-of-contents entries but %d headings at depth "
            "<= %d; leaving its anchors alone.",
            page.file.src_uri,
            len(entries),
            len(headings),
            depth,
        )
        return html

    present = set(_ANY_ID.findall(html))
    for entry, (heading_id, heading_html) in zip(entries, headings):
        if entry.id in present:
            continue  # already lands somewhere; not ours to second-guess
        if not _words(entry.title) <= _words(heading_html):
            logger.warning(
                "Table-of-contents entry '%s' in '%s' points at the missing "
                "anchor '#%s' and does not match heading '#%s'; leaving it "
                "alone.",
                entry.title,
                page.file.src_uri,
                entry.id,
                heading_id,
            )
            continue
        entry.id = heading_id

    return html
