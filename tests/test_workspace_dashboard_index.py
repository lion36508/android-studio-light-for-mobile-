"""Regression checks for workspace indexing, empty state, and search."""

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"


class WorkspaceDashboardIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAIN.read_text(encoding="utf-8")

    def test_new_install_starts_with_an_empty_catalog(self):
        on_create = self.source.split("protected void onCreate", 1)[1].split(
            "private void initDefaultProjects", 1
        )[0]
        self.assertIn("projectList.clear()", on_create)
        self.assertIn("saveProjectCatalog()", on_create)
        self.assertNotIn("initDefaultProjects()", on_create)

    def test_samples_remain_a_separate_explicit_action(self):
        self.assertIn("showSamplesDialog()", self.source)
        self.assertIn("Recent project list cleared. Workspace files and Samples were kept.", self.source)
        self.assertIn("Clear Recent Projects List", self.source)

    def test_refresh_and_search_rescan_nested_workspace_files(self):
        self.assertIn(
            "collectWorkspaceSourceFiles(workspaceRoot, workspaceRoot, workspaceFiles, 0)",
            self.source,
        )
        self.assertIn(
            'String searchable = (item.name + " " + item.subtitle + " " + item.fileName)',
            self.source,
        )
        self.assertIn("refreshProjectList(s.toString())", self.source)
        self.assertIn("refreshProjectList(\"")", self.source)

    def test_workspace_scan_is_bounded_and_rejects_redirected_paths(self):
        self.assertIn("depth > 24", self.source)
        self.assertIn("files.size() >= 200", self.source)
        self.assertIn("if (!canonical.equals(child.getAbsoluteFile())) continue;", self.source)
        self.assertIn('".gradle".equals(name)', self.source)
        self.assertIn('"build".equals(name)', self.source)

    def test_open_project_dialog_lists_nested_files_not_directories(self):
        method = self.source.split("private void showOpenProjectDialog()", 1)[1].split(
            "// DIALOG 3: GIT CLONE", 1
        )[0]
        self.assertIn("collectWorkspaceSourceFiles(root, root, workspaceFiles, 0)", method)
        self.assertNotIn("File[] files = ws.listFiles()", method)
        self.assertIn('setTitle("Open project or workspace file")', method)
        self.assertIn("openEditorWithProject(choices.get(which))", method)

    def test_empty_workspace_has_real_empty_state_and_safe_actions(self):
        self.assertIn("No projects yet. Create a project or open a starter from Samples.", self.source)
        self.assertIn('setTitle("Workspace is empty")', self.source)
        self.assertIn('setPositiveButton("Create project"', self.source)
        self.assertIn('setNegativeButton("Samples"', self.source)

    def test_removing_a_catalog_item_never_deletes_its_workspace_file(self):
        self.assertIn("if (!projectList.contains(item))", self.source)
        self.assertIn("projectList.remove(item)", self.source)
        self.assertIn("Workspace files were kept.", self.source)
        self.assertNotIn("deleteWorkspaceTree", self.source)

    def test_empty_catalog_does_not_break_dashboard_hero_action(self):
        self.assertIn("if (projectList.isEmpty())", self.source)
        self.assertIn("showNewProjectDialog()", self.source)


if __name__ == "__main__":
    unittest.main()
