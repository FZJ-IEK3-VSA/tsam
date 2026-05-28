/**
 * diagram-zoom-init.js
 *
 * Click an architecture diagram (any <img> whose src contains
 * "assets/architecture/" under .md-typeset) to open it in a centered modal
 * overlay. The image is pan/zoomable inside the modal via panzoom
 * (mouse-wheel zoom, drag to pan). Esc or click outside closes.
 *
 * The path-based predicate keeps the handler off the landing-page logos,
 * notebook output images, and plotly figures — only the architecture
 * SVGs match. Keep the CSS selector in extra.css in sync if this changes.
 *
 * Uses Material for MkDocs' document$ observable to re-bind after instant
 * navigation page swaps. A data-zoom-bound guard on document.body prevents
 * duplicate handlers.
 */

(function () {
  "use strict";

  var _dialog = null;
  var _panzoomInstance = null;
  var _styleInjected = false;

  function injectStyles() {
    if (_styleInjected) return;
    _styleInjected = true;
    var style = document.createElement("style");
    style.textContent = [
      "dialog.diagram-zoom-modal {",
      "  border: none;",
      "  background: transparent;",
      "  padding: 0;",
      "  width: 90vw;",
      "  height: 90vh;",
      "  max-width: 90vw;",
      "  max-height: 90vh;",
      "  overflow: hidden;",
      "  border-radius: 6px;",
      "}",
      "dialog.diagram-zoom-modal[open] {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: center;",
      "}",
      "dialog.diagram-zoom-modal::backdrop {",
      "  background: rgba(0, 0, 0, 0.72);",
      "}",
      "dialog.diagram-zoom-modal .dzm-inner {",
      "  width: 100%;",
      "  height: 100%;",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: center;",
      "  overflow: hidden;",
      "}",
      "dialog.diagram-zoom-modal img {",
      "  max-width: 100%;",
      "  max-height: 100%;",
      "  object-fit: contain;",
      "  cursor: grab;",
      "  user-select: none;",
      "  -webkit-user-drag: none;",
      "}",
      "dialog.diagram-zoom-modal img.pz-dragging {",
      "  cursor: grabbing;",
      "}",
    ].join("\n");
    document.head.appendChild(style);
  }

  function getOrCreateDialog() {
    if (_dialog) return _dialog;
    _dialog = document.createElement("dialog");
    _dialog.className = "diagram-zoom-modal";

    var inner = document.createElement("div");
    inner.className = "dzm-inner";
    _dialog.appendChild(inner);

    // Close on backdrop click (click lands directly on <dialog>, not on inner)
    _dialog.addEventListener("click", function (e) {
      if (e.target === _dialog) {
        _dialog.close();
      }
    });

    // Clean up on close (Esc or backdrop click)
    _dialog.addEventListener("close", function () {
      if (_panzoomInstance) {
        _panzoomInstance.dispose();
        _panzoomInstance = null;
      }
      inner.innerHTML = "";
    });

    document.body.appendChild(_dialog);
    return _dialog;
  }

  function openDiagram(originalImg) {
    injectStyles();
    var dialog = getOrCreateDialog();
    var inner = dialog.querySelector(".dzm-inner");

    // Clone the image so we don't move the original out of the page
    var clone = new Image();
    clone.src = originalImg.src;
    clone.alt = originalImg.alt || "Diagram";
    // Remove the #only-light / #only-dark anchors from src for the modal
    // (they are src fragment hints for Material; the clone is always visible)
    clone.src = originalImg.src.replace(/#only-(light|dark)$/, "");

    inner.appendChild(clone);
    dialog.showModal();

    // Apply panzoom after the dialog is visible so layout is stable
    if (typeof panzoom === "function") {
      _panzoomInstance = panzoom(clone, {
        bounds: true,
        boundsPadding: 0.1,
        maxZoom: 10,
        minZoom: 0.5,
      });
    }
  }

  function isArchitectureDiagram(img) {
    if (!img.closest(".md-typeset")) return false;
    var src = img.getAttribute("src") || "";
    return src.indexOf("assets/architecture/") !== -1;
  }

  function bindClickHandler() {
    if (document.body.dataset.zoomBound) return;
    document.body.dataset.zoomBound = "1";

    document.addEventListener("click", function (e) {
      var img = e.target && e.target.closest && e.target.closest("img");
      if (!img || !isArchitectureDiagram(img)) return;
      e.preventDefault();
      e.stopPropagation();
      openDiagram(img);
    }, true /* capture phase */);
  }

  // Hook into Material's instant-navigation observable
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(function () {
      bindClickHandler();
    });
  } else {
    // Fallback for non-Material or static builds
    document.addEventListener("DOMContentLoaded", function () {
      bindClickHandler();
    });
  }
})();
