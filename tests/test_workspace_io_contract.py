"""Regression checks for atomic workspace writes and truthful read failures."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java"
EDITOR = ROOT / "src/main/java/com/bikash/mobileide/EditorActivity.java"
MAIN = ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"


class WorkspaceIoContractTests(unittest.TestCase):
    def test_writes_use_android_atomic_file_backup_and_finish(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("new AtomicFile(file)", source)
        self.assertIn("atomicFile.startWrite()", source)
        self.assertIn("atomicFile.finishWrite(output)", source)
        self.assertIn("atomicFile.failWrite(output)", source)
        self.assertNotIn("new FileOutputStream(file)", source)

    def test_reads_loop_until_eof_instead_of_ignoring_a_single_read_count(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("while ((count = input.read(buffer)) != -1)", source)
        self.assertIn("bytes.write(buffer, 0, count)", source)
        self.assertNotIn("fis.read(data)", source)

    def test_read_result_distinguishes_success_from_io_error(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("class WorkspaceReadResult", source)
        self.assertIn("public final boolean success", source)
        self.assertIn("public final String error", source)
        self.assertIn("WorkspaceReadResult.failure", source)
        self.assertIn("if (!result.success) throw new IllegalStateException(result.error)", source)

    def test_starter_content_is_created_only_after_file_not_found(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("catch (FileNotFoundException missing)", source)
        self.assertIn("saveFile(context, fileName, starter)", source)
        self.assertIn("Unable to read workspace file:", source)

    def test_editor_never_shows_starter_code_for_unreadable_file(self):
        source = EDITOR.read_text(encoding="utf-8")
        self.assertIn("EngineCore.loadWorkspaceFile", source)
        self.assertIn("if (loaded.success)", source)
        self.assertIn("codeEditor.setEnabled(false)", source)
        self.assertIn("Run blocked: the source file could not be loaded safely.", source)
        self.assertIn("Build blocked: the source file could not be loaded safely.", source)
        self.assertIn("Save blocked: the source file was not loaded safely.", source)

    def test_dashboard_surfaces_project_file_load_errors(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn("EngineCore.loadWorkspaceFile", source)
        self.assertIn("if (!loaded.success)", source)
        self.assertIn("loaded.error", source)
        self.assertNotIn("EngineCore.loadFile", source)


if __name__ == "__main__":
    unittest.main()
