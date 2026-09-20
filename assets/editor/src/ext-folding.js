// assets/editor/src/ext-folding.js

import { Extension } from "@codemirror/state";
import { syntaxTree } from "@codemirror/language";
import { RangeSetBuilder, RangeSet } from "@codemirror/state";
import { foldNodeProp, foldInside, foldService, language } from "@codemirror/language";
import { EditorView, Decoration, WidgetType } from "@codemirror/view";

class FoldWidget extends WidgetType {
  constructor() {
    super();
  }

  eq(other) {
    return other instanceof FoldWidget;
  }

  toDOM() {
    let span = document.createElement("span");
    span.textContent = "…";
    span.className = "cm-fold-placeholder";
    return span;
  }

  ignoreEvent() {
    return false;
  }
}

function foldable(state, start, end) {
  let tree = syntaxTree(state);
  let ranges = tree.iterate({
    from: start,
    to: end,
    enter: (node) => {
      let { from, to } = node;
      if (node.type.is("BlockComment") || node.type.is("Block")) {
        return true;
      }
      if (node.type.is("Element") && node.firstChild) {
        from = node.firstChild.to;
        to = node.lastChild.from;
      }
      if (from < to) {
        let pos = from, count = 0;
        tree.iterate({
          from,
          to,
          enter: (node) => {
            if (node.type.isError || node.type.hasError) count++;
            return node.to - node.from > 200;
          }
        });
        if (count > 2) return true;
      }
      return false;
    }
  });
  return ranges;
}

function foldInside(state, pos) {
  let tree = syntaxTree(state);
  let inner = tree.resolveInner(pos, -1);
  if (inner.name) {
    let start = inner.from, end = inner.to;
    if (inner.type.is("BlockComment") || inner.type.is("Block")) {
      return { from: start, to: end };
    }
    if (inner.type.is("Element") && inner.firstChild) {
      start = inner.firstChild.to;
      end = inner.lastChild.from;
    }
    if (start < end) {
      let count = 0;
      tree.iterate({
        from: start,
        to: end,
        enter: (node) => {
          if (node.type.isError || node.type.hasError) count++;
          return node.to - node.from > 200;
        }
      });
      if (count > 2) return { from: start, to: end };
    }
  }
  return null;
}

function buildFoldWidgets(state) {
  let widgets = [];
  syntaxTree(state).iterate({
    enter: (node) => {
      if (node.type.is("BlockComment") || node.type.is("Block")) {
        widgets.push(
          Decoration.replace({
            widget: new FoldWidget(),
            inclusive: true
          }).range(node.from, node.to)
        );
      }
    }
  });
  return Decoration.set(widgets);
}

const foldWidgets = ViewPlugin.fromClass(
  class {
    constructor(view) {
      this.decorations = buildFoldWidgets(view.state);
    }

    update(update) {
      if (update.docChanged || update.viewportChanged) {
        this.decorations = buildFoldWidgets(update.state);
      }
    }
  },
  {
    decorations: (v) => v.decorations,
    eventHandlers: {
      mousedown: (e, view) => {
        let pos = view.posAtCoords({ x: e.clientX, y: e.clientY });
        if (!pos) return false;
        let target = e.target;
        if (target.nodeName === "SPAN" && target.classList.contains("cm-fold-placeholder")) {
          let line = view.state.doc.lineAt(pos);
          let fold = foldInside(view.state, line.from);
          if (fold) {
            view.dispatch({
              changes: { from: fold.from, to: fold.to, insert: "" },
              userEvent: "fold"
            });
            return true;
          }
        }
        return false;
      }
    }
  }
);

const folding = [
  foldWidgets,
  EditorView.baseTheme({
    ".cm-fold-placeholder": {
      color: "#888",
      cursor: "pointer",
      padding: "0 1px",
      margin: "0 -1px",
      borderRadius: "2px",
      backgroundColor: "#f0f0f0"
    },
    ".cm-fold-placeholder:hover": {
      backgroundColor: "#ddd"
    }
  })
];

export { folding };