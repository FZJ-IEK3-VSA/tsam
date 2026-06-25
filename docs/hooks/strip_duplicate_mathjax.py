"""mkdocs hook: remove the duplicate MathJax v2 setup that mkdocs-jupyter /
plotly bake into notebook pages.

Notebook pages end up with THREE MathJax setups fighting over ``window.MathJax``:

1. nbconvert injects a MathJax **v2** ``<script type="text/x-mathjax-config">``
   block (``MathJax.Hub.Config`` / ``init_mathjax``).
2. Plotly's connected bundle injects a MathJax **v2.7.5** loader
   (``cdnjs .../mathjax/2.x/MathJax.js?config=TeX-AMS-MML_SVG``) for LaTeX in
   figure labels.
3. The site itself loads MathJax **v3** (``javascripts/mathjax.js`` +
   ``tex-mml-chtml.js`` from ``extra_javascript``).

v2 and v3 both claim ``window.MathJax`` and clobber each other, which leaves
display math (``$$ ... $$``) unrendered. Our figures contain no LaTeX, so the
plotly/nbconvert MathJax v2 is dead weight — strip it and let the site's
MathJax v3 (configured in ``javascripts/mathjax.js``) own all typesetting.
"""

from __future__ import annotations

import re

# nbconvert's MathJax v2 config block (init_mathjax / MathJax.Hub.Config).
_NBCONVERT_MJ_CONFIG = re.compile(
    r"<script type=\"text/x-mathjax-config\">.*?</script>",
    re.DOTALL,
)

# Plotly's MathJax v2.x loader (any cdnjs mathjax 2.x build).
_PLOTLY_MJ_LOADER = re.compile(
    r"<script src=\"https://cdnjs\.cloudflare\.com/ajax/libs/mathjax/2[^\"]*\"></script>"
)


def on_post_page(output: str, *, page, config) -> str:
    """Strip the conflicting MathJax v2 scripts from each rendered page."""
    if "mathjax/2" not in output and "x-mathjax-config" not in output:
        return output
    output = _NBCONVERT_MJ_CONFIG.sub("", output)
    output = _PLOTLY_MJ_LOADER.sub("", output)
    return output
