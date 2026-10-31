import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_SOURCE = ROOT / "src/main/java/com/bikash/mobileide/CommandRunner.java"
HARNESS_SOURCE = ROOT / "tests/java/com/bikash/mobileide/CommandRunnerHarness.java"


class CommandRunnerRuntimeTests(unittest.TestCase):
    def test_java_runtime_harness(self):
        javac = shutil.which("javac")
        java = shutil.which("java")
        self.assertIsNotNone(javac, "JDK compiler is required for process-runner runtime tests.")
        self.assertIsNotNone(java, "JDK runtime is required for process-runner runtime tests.")

        with tempfile.TemporaryDirectory(prefix="oxdroid-command-runner-") as output_dir:
            compile_result = subprocess.run(
                [
                    javac,
                    "-encoding",
                    "UTF-8",
                    "-d",
                    output_dir,
                    str(RUNNER_SOURCE),
                    str(HARNESS_SOURCE),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                "CommandRunner JVM compilation failed:\\n{}\\n{}".format(
                    compile_result.stdout, compile_result.stderr
                ),
            )

            run_result = subprocess.run(
                [
                    java,
                    "-cp",
                    output_dir,
                    "com.bikash.mobileide.CommandRunnerHarness",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                run_result.returncode,
                0,
                "CommandRunner runtime checks failed:\\n{}\\n{}".format(
                    run_result.stdout, run_result.stderr
                ),
            )
            self.assertIn("all checks passed", run_result.stdout)


if __name__ == "__main__":
    unittest.main()
