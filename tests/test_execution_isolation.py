"""Regression tests for OxDroid's fail-closed code execution policy."""
import ast
import importlib.util
import pathlib
import threading
import unittest
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = ROOT / "ox_industrial_daemon.py"
ENGINE_PATH = ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java"

spec = importlib.util.spec_from_file_location("oxdroid_daemon", DAEMON_PATH)
daemon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daemon)


def java_method_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise AssertionError("Unclosed Java method body")


class ExecutionIsolationTests(unittest.TestCase):
    def test_daemon_rejects_execution_over_http(self):
        server = daemon.ReusableTCPServer(
            ("127.0.0.1", 0), daemon.IndustrialCompilerHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:{}/execute".format(server.server_address[1]),
                data=b'{"language":"python","filename":"probe.py","code":"print(1)"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=3)
            self.assertEqual(raised.exception.code, 501)
            payload = raised.exception.read().decode("utf-8")
            self.assertIn('"status": "unsupported"', payload)
            self.assertIn("Your code was not executed", payload)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_daemon_does_not_import_process_execution_modules(self):
        tree = ast.parse(DAEMON_PATH.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertNotIn("subprocess", imported_roots)
        self.assertNotIn("os", imported_roots)
        self.assertNotIn("tempfile", imported_roots)

    def test_android_execution_entry_point_fails_closed(self):
        source = ENGINE_PATH.read_text(encoding="utf-8")
        body = java_method_body(
            source,
            "public static String executeCode(Context context, String language, String fileName, String code)",
        )
        self.assertIn("saveFile(context, fileName, code)", body)
        self.assertIn("NOT executed", body)
        self.assertIn("isolated execution sandbox is not available", body)
        self.assertNotIn("runShellCommand(", body)
        self.assertNotIn("executeRealPython(", body)
        self.assertNotIn("executeRealCpp(", body)
        self.assertNotIn("executeRealJvm(", body)
        self.assertNotIn("tryIndustrialDaemon(", body)


if __name__ == "__main__":
    unittest.main()
