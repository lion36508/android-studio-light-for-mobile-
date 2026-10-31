import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/main/java/com/bikash/mobileide/ProjectName.java"
HARNESS = ROOT / "tests/java/com/bikash/mobileide/ProjectNameHarness.java"


class ProjectNameSafetyTests(unittest.TestCase):
    def test_project_name_runtime_contract(self):
        javac = shutil.which("javac")
        java = shutil.which("java")
        self.assertIsNotNone(javac, "JDK compiler is required for project-name runtime tests.")
        self.assertIsNotNone(java, "JDK runtime is required for project-name runtime tests.")

        with tempfile.TemporaryDirectory(prefix="oxdroid-project-name-") as output_dir:
            compiled = subprocess.run(
                [javac, "-encoding", "UTF-8", "-d", output_dir, str(SOURCE), str(HARNESS)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + "\\n" + compiled.stderr)

            executed = subprocess.run(
                [java, "-cp", output_dir, "com.bikash.mobileide.ProjectNameHarness"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + "\\n" + executed.stderr)
            self.assertIn("all checks passed", executed.stdout)


if __name__ == "__main__":
    unittest.main()
