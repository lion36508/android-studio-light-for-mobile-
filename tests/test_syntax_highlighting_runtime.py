import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/main/java/com/bikash/mobileide/SourceSyntaxHighlighter.java"
HARNESS = ROOT / "tests/java/com/bikash/mobileide/SourceSyntaxHighlighterHarness.java"


class SourceSyntaxHighlighterRuntimeTests(unittest.TestCase):
    def test_string_and_comment_tokens_do_not_overlap(self):
        javac = shutil.which("javac")
        java = shutil.which("java")
        if not javac or not java:
            self.skipTest("JDK is unavailable; Android APK CI runs this behavioral harness.")
        with tempfile.TemporaryDirectory(prefix="oxdroid-syntax-") as output_dir:
            compiled = subprocess.run(
                [javac, "-encoding", "UTF-8", "-d", output_dir, str(SOURCE), str(HARNESS)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + "\\n" + compiled.stderr)
            executed = subprocess.run(
                [java, "-cp", output_dir, "com.bikash.mobileide.SourceSyntaxHighlighterHarness"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + "\\n" + executed.stderr)
            self.assertIn("all checks passed", executed.stdout)


if __name__ == "__main__":
    unittest.main()
