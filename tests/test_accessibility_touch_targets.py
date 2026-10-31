"""Accessibility contract checks for dashboard and editor controls."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"
EDITOR = ROOT / "src/main/java/com/bikash/mobileide/EditorActivity.java"


class AccessibilityTouchTargetTests(unittest.TestCase):
    def test_dashboard_icon_only_controls_have_spoken_names(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn('menuBtn.setContentDescription("Open navigation menu")', source)
        self.assertIn('searchBtn.setContentDescription("Search projects")', source)
        self.assertIn('moreBtn.setContentDescription("More OxDroid Studio options")', source)
        self.assertIn('menuDots.setContentDescription("More options for " + item.name)', source)

    def test_dashboard_touch_targets_are_at_least_48dp(self):
        source = MAIN.read_text(encoding="utf-8")
        for snippet in (
            "menuBtn.setMinimumHeight(dp(48))",
            "searchBtn.setMinimumHeight(dp(48))",
            "moreBtn.setMinimumHeight(dp(48))",
            "menuDots.setMinimumHeight(dp(48))",
            "tab.setMinimumHeight(dp(48))",
            "touchZone.setMinimumHeight(dp(48))",
            "tv.setMinimumHeight(dp(48))",
        ):
            self.assertIn(snippet, source)

    def test_dashboard_action_and_project_cards_have_single_clear_label(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn('card.setContentDescription((title + ". " + sub).replace("\\n", " "))', source)
        self.assertIn("View.IMPORTANT_FOR_ACCESSIBILITY_NO", source)
        self.assertIn('card.setContentDescription("Open project " + item.name + ", " + item.subtitle)', source)

    def test_editor_primary_controls_are_named_and_large_enough(self):
        source = EDITOR.read_text(encoding="utf-8")
        for snippet in (
            'backBtn.setContentDescription("Return to workspace")',
            'runBtn.setContentDescription("Run current source;',
            'buildBtn.setContentDescription("Build current project;',
            'saveBtn.setContentDescription("Save current file")',
            'closeConsoleBtn.setContentDescription("Close terminal output panel")',
            "backBtn.setMinimumHeight(dp(48))",
            "runBtn.setMinimumHeight(dp(48))",
            "buildBtn.setMinimumHeight(dp(48))",
            "saveBtn.setMinimumHeight(dp(48))",
            "closeConsoleBtn.setMinimumHeight(dp(48))",
        ):
            self.assertIn(snippet, source)

    def test_editor_symbols_and_tabs_are_announceable(self):
        source = EDITOR.read_text(encoding="utf-8")
        self.assertIn("symBtn.setContentDescription", source)
        self.assertIn('tab.setContentDescription("Open " + file + " in " + lang + " editor")', source)
        self.assertIn('codeEditor.setContentDescription("Source code editor")', source)


if __name__ == "__main__":
    unittest.main()
