"""Contract tests for safe, all-or-nothing project scaffolding."""

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "src/main/java/com/bikash/mobileide/ProjectScaffold.java"
ENGINE = ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java"


class ProjectScaffoldContractTests(unittest.TestCase):
    def test_existing_project_directory_is_never_overwritten(self):
        source = SCAFFOLD.read_text(encoding="utf-8")
        self.assertIn("EngineCore.workspaceEntryExists(context, root)", source)
        self.assertIn("// Never replace an existing project", source)

    def test_every_generated_file_must_be_saved_before_success_is_returned(self):
        source = SCAFFOLD.read_text(encoding="utf-8")
        self.assertIn("String[][] generatedFiles", source)
        self.assertIn(
            "if (!EngineCore.saveFile(context, generatedFile[0], generatedFile[1]))",
            source,
        )
        self.assertIn("for (String[] generatedFile : generatedFiles)", source)
        loop_end = source.index("return javaPath;")
        save_check = source.index(
            "if (!EngineCore.saveFile(context, generatedFile[0], generatedFile[1]))"
        )
        self.assertLess(save_check, loop_end)

    def test_failed_scaffold_rolls_back_its_new_project_directory(self):
        source = SCAFFOLD.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("EngineCore.deleteWorkspaceTree(context, root)"), 2)
        self.assertIn("return null;", source)

    def test_workspace_existence_check_fails_closed_for_invalid_paths(self):
        source = ENGINE.read_text(encoding="utf-8")
        start = source.index("public static synchronized boolean workspaceEntryExists")
        end = source.index("public static synchronized boolean deleteWorkspaceTree", start)
        method = source[start:end]
        self.assertIn("resolveWorkspaceEntry(context, relativePath).exists()", method)
        self.assertIn("catch (Exception error)", method)
        self.assertIn("return true;", method)

    def test_workspace_rollback_is_canonical_path_bounded(self):
        source = ENGINE.read_text(encoding="utf-8")
        start = source.index("private static boolean deleteWorkspaceEntry")
        end = source.index("public static synchronized boolean saveFile", start)
        method = source[start:end]
        self.assertIn("entry.getCanonicalFile()", method)
        self.assertIn('root.getPath() + File.separator', method)
        self.assertIn("if (!canonical.equals(absolute)) return absolute.delete();", method)
        self.assertIn("normalized.endsWith(\"/\")", source)


if __name__ == "__main__":
    unittest.main()
