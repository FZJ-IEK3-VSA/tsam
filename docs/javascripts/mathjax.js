window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
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
