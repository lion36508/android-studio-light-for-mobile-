"""Regression checks for real workspace tabs and editor recreation state."""

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EDITOR = ROOT / "src/main/java/com/bikash/mobileide/EditorActivity.java"


class EditorWorkspaceTabsStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = EDITOR.read_text(encoding="utf-8")

    def test_demo_tabs_are_not_hard_coded(self):
        self.assertNotIn("buildLangTab(", self.source)
        self.assertNotIn("Main.kt", self.source)
        self.assertNotIn("script.py", self.source)
        self.assertIn("showWorkspaceFilePicker()", self.source)
        self.assertIn("collectWorkspaceFiles(root, root, files, 0)", self.source)

    def test_workspace_scan_is_bounded_and_skips_symlinks(self):
        self.assertIn("files.size() >= 200", self.source)
        self.assertIn("depth > 24", self.source)
        self.assertIn("if (!canonical.equals(child.getAbsoluteFile())) continue;", self.source)
        self.assertIn('".gradle".equals(name)', self.source)

    def test_tabs_switch_save_and_close_workspace_files(self):
        self.assertIn("switchToWorkspaceFile(item.fileName, item.language)", self.source)
        self.assertIn("if (workspaceFileLoaded && !saveCurrentFile(false)) return;", self.source)
        self.assertIn("closeWorkspaceTab(item.fileName)", self.source)
        self.assertIn("Keep at least one file tab open.", self.source)

    def test_activity_restores_current_file_language_and_unsaved_text(self):
        self.assertIn("protected void onSaveInstanceState(Bundle outState)", self.source)
        for key in ("STATE_FILE", "STATE_LANGUAGE", "STATE_CONTENT", "STATE_DIRTY"):
            self.assertIn(key, self.source)
        self.assertIn('savedInstanceState.getString(STATE_CONTENT, loaded.content)', self.source)
        self.assertIn('outState.putBoolean(STATE_DIRTY, isDirty)', self.source)

    def test_activity_restores_cursor_scroll_and_open_tabs(self):
        for key in (
            "STATE_SELECTION_START",
            "STATE_SELECTION_END",
            "STATE_SCROLL_Y",
            "STATE_SCROLL_X",
            "STATE_TAB_PATHS",
            "STATE_TAB_LANGUAGES",
        ):
            self.assertIn(key, self.source)
        self.assertIn("codeEditor.setSelection", self.source)
        self.assertIn("editorScroll.scrollTo", self.source)
        self.assertIn("editorHorizontalScroll.scrollTo", self.source)
        self.assertIn("outState.putStringArray(STATE_TAB_PATHS, paths)", self.source)

    def test_run_snapshots_source_identity_before_starting_worker(self):
        self.assertIn("if (runInProgress)", self.source)
        self.assertIn("final String code = codeEditor.getText().toString();", self.source)
        self.assertIn("final String language = currentLanguage;", self.source)
        self.assertIn("final String fileName = currentFileName;", self.source)
        self.assertIn("EngineCore.executeCode(\n                    getApplicationContext(), language, fileName, code", self.source)
        self.assertIn('runButton.setEnabled(false)', self.source)
        self.assertIn('runButton.setEnabled(true)', self.source)

    def test_console_does_not_claim_live_streaming_or_successful_execution(self):
        self.assertIn('consoleTitle.setText("RUN STATUS")', self.source)
        self.assertNotIn("LIVE TERMINAL OUTPUT", self.source)
        self.assertIn("Execution is disabled until an isolated sandbox is available", self.source)

    def test_dirty_indicator_updates_without_rebuilding_tab_row(self):
        self.assertIn('activeTabLabel.setText("● " + new File(currentFileName).getName())', self.source)


if __name__ == "__main__":
    unittest.main()
