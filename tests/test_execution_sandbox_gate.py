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
        match = re.search(
            r"public static String executeCode\\(Context context, String language, String fileName, String code\\) \\{(.*?)\\n    \\}",
            source,
            re.S,
        )
        self.assertIsNotNone(match)
        method = match.group(1)
        self.assertIn("saveFile(context, fileName, code)", method)
        self.assertIn("was NOT executed", method)
        for forbidden in ("tryIndustrialDaemon(", "runShellCommand(", "executeRealPython(", "executeRealCpp(", "executeRealJvm("):
            self.assertNotIn(forbidden, method)

    def test_disabled_state_is_explained_to_the_user(self):
        daemon = (ROOT / "ox_industrial_daemon.py").read_text(encoding="utf-8")
        engine = (ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java").read_text(encoding="utf-8")
        self.assertIn("isolated sandbox", daemon)
        self.assertIn("isolated execution sandbox", engine)


if __name__ == "__main__":
    unittest.main()
