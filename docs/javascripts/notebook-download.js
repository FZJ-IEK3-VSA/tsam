// Inject "Notebook" + "Data" download links and a "GitHub" link at the top of
// each notebook page.
//
// The .ipynb source is copied next to index.html by mkdocs-jupyter's
// `include_source: true`, so the notebook URL is just <slug>.ipynb relative
// to the page.
//
// Everything else is resolved against the site root rather than a fixed number
// of `../` hops, because notebook pages sit at three different depths
// (/tutorials/x/, /explanation/how-it-works/x/, /explanation/how-it-works/
// 02_clustering/x/). Material publishes the relative path back to the site root
// in its `__config` script; that value is build-time and prefix-agnostic, so it
// works both on the bare site and under ReadTheDocs' /en/<version>/ mount.
//
// The docs tree mirrors the URL tree exactly (docs/tutorials/quickstart.ipynb
// -> /tutorials/quickstart/), so the GitHub link is derived from the page path
// instead of being hard-coded. The repo base is read from Material's header
// source link so it survives an org/repo rename; only the branch is hard-coded
// (matches edit_uri: develop).
//
// Subscribes to Material's `document$` instant-nav lifecycle so the links
// re-attach on every page transition.

const GITHUB_BRANCH = 'develop';
const REPO_FALLBACK = 'https://github.com/FZJ-IEK3-VSA/tsam';
const DATASET = 'testdata.csv';

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

// URL of the site root, from Material's build-time `base` (e.g. "../../..").
function siteRoot() {
  const el = document.getElementById('__config');
  if (!el) return null;
  try {
    const base = JSON.parse(el.textContent).base;
    if (typeof base !== 'string') return null;
    return new URL(base.endsWith('/') ? base : `${base}/`, window.location.href);
  } catch {
    return null;
  }
}

// This page's path relative to the site root, e.g. "tutorials/quickstart".
function docPath(root) {
  const here = window.location.pathname;
  if (!here.startsWith(root.pathname)) return null;
  return here.slice(root.pathname.length).replace(/\/+$/, '');
}

function injectNotebookDownloads() {
  const wrapper = document.querySelector('.jupyter-wrapper');
  if (!wrapper) return;
  const parent = wrapper.parentNode;
  if (!parent || parent.querySelector(':scope > .notebook-downloads')) return;

  const path = window.location.pathname.replace(/\/$/, '');
  const slug = path.split('/').pop();
  if (!slug) return;

  const root = siteRoot();
  const rel = root ? docPath(root) : null;

  const group = document.createElement('div');
  group.className = 'notebook-downloads';
  group.appendChild(
    makeLink(`${slug}.ipynb`, 'Notebook', 'Download this notebook (.ipynb)', {
      download: `${slug}.ipynb`,
    })
  );
  if (root) {
    group.appendChild(
      makeLink(
        new URL(`data/${DATASET}`, root).href,
        'Data',
        `Download the example dataset (${DATASET})`,
        { download: DATASET }
      )
    );
  }
  if (rel) {
    group.appendChild(
      makeLink(
        `${repoBase()}/blob/${GITHUB_BRANCH}/docs/${rel}.ipynb`,
        'GitHub',
        'View this notebook and its data on GitHub',
        { external: true, icon: '↗' }
      )
    );
  }

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
