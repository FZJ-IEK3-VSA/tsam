// Inject "Notebook" + "Data" download links and a "GitHub" link at the top of
// each notebook page.
//
// The .ipynb source is copied next to index.html by mkdocs-jupyter's
// `include_source: true`, so the notebook URL is just <slug>.ipynb relative
// to the page. The example dataset lives at docs/notebooks/testdata.csv,
// which mkdocs copies to site/notebooks/testdata.csv — one level above each
// notebook's own page directory, hence `../testdata.csv`. Every notebook
// loads it via `read_csv("testdata.csv")`, so downloading both files into the
// same folder yields a runnable notebook.
//
// The GitHub link points at the same notebook in the repo, where the data
// files sit alongside it — handy for users who'd rather clone or browse than
// download piecemeal. The repo base is read from Material's header source
// link so it survives an org/repo rename; only the branch is hard-coded
// (matches edit_uri: develop).
//
// Subscribes to Material's `document$` instant-nav lifecycle so the links
// re-attach on every page transition.

const GITHUB_BRANCH = 'develop';
const REPO_FALLBACK = 'https://github.com/FZJ-IEK3-VSA/tsam';

function makeLink(href, label, title, opts = {}) {
  const link = document.createElement('a');
  link.className = 'notebook-download';
  link.href = href;
  link.title = title;
  if (opts.download) link.setAttribute('download', opts.download);
  if (opts.external) {
    link.target = '_blank';
    link.rel = 'noopener';
  }
  const icon = opts.icon || '↓';
  link.innerHTML = `<span class="notebook-download__icon">${icon}</span> ${label}`;
  return link;
}

function repoBase() {
  const source = document.querySelector('.md-header a.md-source, a.md-source');
  return (source ? source.href : REPO_FALLBACK).replace(/\/$/, '');
}

function injectNotebookDownloads() {
  const wrapper = document.querySelector('.jupyter-wrapper');
  if (!wrapper) return;
  const parent = wrapper.parentNode;
  if (!parent || parent.querySelector(':scope > .notebook-downloads')) return;

  const path = window.location.pathname.replace(/\/$/, '');
  const slug = path.split('/').pop();
  if (!slug) return;

  const group = document.createElement('div');
  group.className = 'notebook-downloads';
  group.appendChild(
    makeLink(`${slug}.ipynb`, 'Notebook', 'Download this notebook (.ipynb)', {
      download: `${slug}.ipynb`,
    })
  );
  group.appendChild(
    makeLink('../testdata.csv', 'Data', 'Download the example dataset (testdata.csv)', {
      download: 'testdata.csv',
    })
  );
  group.appendChild(
    makeLink(
      `${repoBase()}/blob/${GITHUB_BRANCH}/docs/notebooks/${slug}.ipynb`,
      'GitHub',
      'View this notebook and its data on GitHub',
      { external: true, icon: '↗' }
    )
  );

  // Insert as a sibling right before the notebook, alongside Material's
  // floated edit/view action buttons. As a `float: right` element later in
  // source order, it stacks to the LEFT of those icons on the same line.
  parent.insertBefore(group, wrapper);
}

if (typeof document$ !== 'undefined') {
  document$.subscribe(injectNotebookDownloads);
} else {
  document.addEventListener('DOMContentLoaded', injectNotebookDownloads);
}
