import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"


class ProfessionalMinimalDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_dashboard_uses_a_two_column_action_grid(self):
        self.assertIn("LinearLayout primaryActionsRow", self.source)
        self.assertIn("LinearLayout secondaryActionsRow", self.source)
        for label in ("New project", "Open project", "Git", "Samples"):
            self.assertIn('"' + label + '"', self.source)

    def test_existing_quick_actions_keep_their_original_handlers(self):
        for handler in (
            "showNewProjectDialog();",
            "showOpenProjectDialog();",
            "showCloneGitDialog();",
            "showSamplesDialog();",
        ):
            self.assertIn(handler, self.source)

    def test_live_project_search_and_open_actions_remain_connected(self):
        self.assertIn("searchInput.addTextChangedListener", self.source)
        self.assertIn("refreshProjectList(s.toString())", self.source)
        self.assertIn("openEditorWithProject(item)", self.source)
        self.assertIn("showProjectItemMenu(item)", self.source)

    def test_build_environment_card_reports_real_build_tool_status(self):
        self.assertIn("private View buildBuildStatusCard()", self.source)
        self.assertIn('state.setText("NOT READY")', self.source)
        self.assertIn("EngineCore.buildToolStatus(MainActivity.this)", self.source)
        self.assertIn("VIEW TOOLCHAIN DETAILS", self.source)

    def test_project_rows_use_monograms_and_accessible_menu_targets(self):
        self.assertIn("String projectMark =", self.source)
        self.assertIn('menuDots.setMinWidth(dp(40))', self.source)
        self.assertIn('menuDots.setMinHeight(dp(40))', self.source)
        self.assertIn("card.setContentDescription", self.source)

    def test_bottom_navigation_preserves_existing_destinations(self):
        for destination in ("Home", "Projects", "Editor", "Settings"):
            self.assertIn('"' + destination + '"', self.source)
        self.assertIn('buildNavTab("⌂", "Home"', self.source)
        self.assertIn('buildNavTab("▦", "Projects"', self.source)


if __name__ == "__main__":
    unittest.main()
