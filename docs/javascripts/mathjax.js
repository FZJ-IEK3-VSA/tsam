window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // Process the whole page EXCEPT code text and cell outputs. The original
    // ".*|" pattern ignored every element (and re-ignored the class-less
    // <p>/<li> holding notebook math), so notebook formulas never rendered —
    // only arithmatex-wrapped .md math did, since that sits inside a
    // class-bearing <span class="arithmatex">.
    //
    // NOTE: do NOT ignore `jp-InputArea` — mkdocs-jupyter renders *markdown*
    // cells inside `jp-InputArea jp-Cell-inputArea` too, so ignoring it hides
    // all notebook math. Ignore only the Pygments code (`highlight`/
    // `highlight-ipynb`) and cell outputs (`jp-OutputArea`); that leaves stray
    // `$` in code/output alone while still typesetting markdown-cell math.
    ignoreHtmlClass: "highlight|jp-OutputArea",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset on instant navigation (needed for navigation.instant).
// Guard against MathJax not being fully loaded yet — instant-nav can fire
// before MathJax.startup is populated, which used to throw a TypeError on
// every page swap.
document$.subscribe(function () {
  if (
    typeof MathJax === "undefined" ||
    !MathJax.startup ||
    !MathJax.startup.output ||
    !MathJax.typesetClear ||
    !MathJax.typesetPromise
  ) {
    return;
  }
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
