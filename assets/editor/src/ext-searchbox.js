import { EditorView } from "@codemirror/view";
import { search, openSearchPanel, SearchQuery } from "@codemirror/search";
import { Extension } from "@codemirror/state";

export const searchBox = (): Extension => {
  const searchBox = EditorView.domEventHandlers({
    "Ctrl-f": (event, view) => {
      event.preventDefault();
      openSearchPanel(view);
    },
  });

  const multiFileSearch = EditorView.domEventHandlers({
    "Ctrl-Shift-f": (event, view) => {
      event.preventDefault();
      // Implement multi-file search functionality here
      // This could involve opening a dialog, sending a request to a server, etc.
      console.log("Multi-file search triggered");
    },
  });

  return [search(), searchBox, multiFileSearch];
};