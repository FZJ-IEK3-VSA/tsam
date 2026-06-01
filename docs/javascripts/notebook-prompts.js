// Strip "In " / "Out" prefixes from notebook prompts (keep only [N]:),
// and make the whole input/output code cell clickable to copy.
//
// Scoped to notebook cells only — Material's built-in content.code.copy
// already provides a copy button on regular markdown code fences, so we
// deliberately do NOT attach here to avoid two competing copy mechanisms.
//
// Subscribes to Material's `document$` instant-nav lifecycle when available
// so both behaviours re-attach on every page transition.

function stripPromptPrefixes() {
  document.querySelectorAll('.jp-InputPrompt, .jp-OutputPrompt').forEach((el) => {
    if (el.dataset.fixed) return;
    const m = el.textContent.match(/^\s*(?:In|Out)\s*(\[\s*\d*\s*\]:?)\s*$/);
    if (m) {
      el.textContent = m[1];
      el.dataset.fixed = '1';
    }
  });
}

function attachClickToCopy() {
  const blocks = document.querySelectorAll('.jp-CodeCell .highlight-ipynb');
  blocks.forEach((block) => {
    if (block.dataset.copyAttached) return;
    block.dataset.copyAttached = '1';
    block.addEventListener('click', (event) => {
      // Don't fire if the user is selecting text (or just clicked a link inside)
      if (window.getSelection().toString().length > 0) return;
      if (event.target.closest('a, button')) return;

      const codeEl = block.querySelector('pre');
      if (!codeEl) return;
      const text = codeEl.innerText;

      navigator.clipboard.writeText(text).then(() => {
        block.classList.add('copied');
        setTimeout(() => block.classList.remove('copied'), 700);
      }).catch(() => {});
    });
  });
}

function init() {
  stripPromptPrefixes();
  attachClickToCopy();
}

if (typeof document$ !== 'undefined') {
  document$.subscribe(init);
} else {
  document.addEventListener('DOMContentLoaded', init);
}
