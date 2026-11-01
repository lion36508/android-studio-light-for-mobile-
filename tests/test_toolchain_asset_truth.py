"""Ensure placeholder compiler assets are not mistaken for working toolchains."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ToolchainAssetTruthTests(unittest.TestCase):
    def test_placeholder_compiler_files_are_not_shipped(self):
        placeholders = (
            ROOT / "assets/bin/python",
            ROOT / "assets/bin/cpp_engine",
            ROOT / "assets/lib/kotlin-compiler.jar",
        )
        present = [str(path.relative_to(ROOT)) for path in placeholders if path.exists()]
        self.assertEqual([], present, "Placeholder compiler files must not be bundled.")

    def test_application_reports_on_device_build_tools_as_unavailable(self):
        source = (
            ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java"
        ).read_text(encoding="utf-8")
        start = source.index("public static String buildToolStatus")
        end = source.index("private static final int MAX_DAEMON_REQUEST_BYTES", start)
        status = source[start:end]
        self.assertIn("AAPT2/AAPT: unavailable", status)
        self.assertIn("D8: unavailable", status)
        self.assertIn("ApkSigner: unavailable", status)
        self.assertIn("BUILD NOT AVAILABLE", status)

    def test_code_execution_stays_disabled_until_sandbox_is_ready(self):
        source = (
            ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java"
        ).read_text(encoding="utf-8")
        start = source.index("public static String executeCode")
        end = source.index("// =========================================================================", start)
        body = source[start:end]
        self.assertIn("NOT executed", body)
        self.assertIn("isolated execution sandbox is not available", body)
        self.assertNotIn("executeRealPython(", body)
        self.assertNotIn("executeRealCpp(", body)


if __name__ == "__main__":
    unittest.main()
