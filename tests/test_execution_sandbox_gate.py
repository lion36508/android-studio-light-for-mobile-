import ast
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ExecutionSandboxGateTest(unittest.TestCase):
    def test_daemon_has_no_process_execution_imports_or_calls(self):
        source = (ROOT / "ox_industrial_daemon.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("subprocess", imports)
        self.assertNotIn("os", imports)
        self.assertIn("self._send_json(501, UNAVAILABLE)", source)
        self.assertIn('"status": "unsupported"', source)

    def test_android_runtime_does_not_fall_back_to_unsandboxed_execution(self):
        source = (ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java").read_text(encoding="utf-8")
        start = source.index("public static ExecutionResult executeCodeResult(")
        end = source.index("public static String executeCode(", start)
        method = source[start:end]
        self.assertIn("saveFile(context, fileName, code)", method)
        self.assertIn("was NOT executed", method)
        for forbidden in ("tryIndustrialDaemon(", "runShellCommand(", "executeRealPython(", "executeRealCpp(", "executeRealJvm("):
            self.assertNotIn(forbidden, method)
        self.assertIn("UNAVAILABLE", method)
        self.assertIn("was NOT executed", method)
        self.assertIn("elapsedMillis(startedAt)", method)

    def test_execution_result_has_typed_fields_and_truthful_status(self):
        source = (ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java").read_text(encoding="utf-8")
        for declaration in (
            "public final String status;",
            "public final String backend;",
            "public final String stdout;",
            "public final String stderr;",
            "public final Integer exitCode;",
            "public final long durationMillis;",
            "public final boolean timedOut;",
            "public final boolean cancelled;",
            "public final String launchError;",
        ):
            self.assertIn(declaration, source)
        self.assertIn("public String toDisplayText()", source)
        self.assertIn("no process executed", source)

    def test_disabled_state_is_explained_to_the_user(self):
        daemon = (ROOT / "ox_industrial_daemon.py").read_text(encoding="utf-8")
        engine = (ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java").read_text(encoding="utf-8")
        self.assertIn("isolated sandbox", daemon)
        self.assertIn("isolated execution sandbox", engine)


if __name__ == "__main__":
    unittest.main()
