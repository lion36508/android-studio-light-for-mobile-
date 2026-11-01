"""Regression checks for generated Android project templates."""

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "src/main/java/com/bikash/mobileide/ProjectScaffold.java"


class ProjectScaffoldTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCAFFOLD.read_text(encoding="utf-8")

    def test_generated_projects_use_api_36(self):
        self.assertIn("compileSdk 36", self.source)
        self.assertIn("targetSdk 36", self.source)
        self.assertNotIn("compileSdk 34", self.source)
        self.assertNotIn("targetSdk 34", self.source)

    def test_generated_gradle_plugin_supports_api_36(self):
        self.assertIn("com.android.application' version '8.9.2'", self.source)

    def test_generated_manifest_uses_gradle_namespace_not_package_attribute(self):
        self.assertIn('xmlns:android=\\"http://schemas.android.com/apk/res/android', self.source)
        self.assertNotIn('package=\\"', self.source)
        self.assertIn('namespace ', self.source)

    def test_scaffold_remains_atomic_on_partial_write_failure(self):
        self.assertIn("if (!EngineCore.saveFile(context, generatedFile[0], generatedFile[1]))", self.source)
        self.assertIn("EngineCore.deleteWorkspaceTree(context, root)", self.source)


if __name__ == "__main__":
    unittest.main()
