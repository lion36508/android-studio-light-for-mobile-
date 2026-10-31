import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class BoundedProcessRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = (ROOT / "src/main/java/com/bikash/mobileide/CommandRunner.java").read_text(encoding="utf-8")
        cls.linux_engine = (ROOT / "src/main/java/com/bikash/mobileide/OxLinuxEngine.java").read_text(encoding="utf-8")
        cls.engine_core = (ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java").read_text(encoding="utf-8")
        cls.main_activity = (ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java").read_text(encoding="utf-8")
        cls.workflow = (ROOT / ".github/workflows/code_quality_guard.yml").read_text(encoding="utf-8")

    def test_stdout_and_stderr_are_drained_by_separate_reader_threads(self):
        self.assertIn("oxdroid-process-stdout", self.runner)
        self.assertIn("oxdroid-process-stderr", self.runner)
        self.assertIn("startReader(stdoutStream", self.runner)
        self.assertIn("startReader(stderrStream", self.runner)
        self.assertIn("redirectErrorStream(false)", self.runner)

    def test_deadline_is_monotonic_and_output_is_bounded(self):
        self.assertIn("System.nanoTime()", self.runner)
        self.assertIn("waitForWithinDeadline(process, safeTimeout)", self.runner)
        self.assertIn("DEFAULT_MAX_OUTPUT_CHARS = 64 * 1024", self.runner)
        self.assertIn("text.delete(0, overflow)", self.runner)
        self.assertIn("outputTruncated", self.runner)

    def test_timeout_and_interruption_attempt_process_tree_cleanup(self):
        self.assertIn("process.destroy()", self.runner)
        self.assertIn("destroyDescendantsIfSupported(process)", self.runner)
        self.assertIn('Class.forName("java.lang.ProcessHandle")', self.runner)
        self.assertIn("forceDestroy(process)", self.runner)
        self.assertIn("cancelled", self.runner)

    def test_structured_result_preserves_exit_code_and_diagnostics(self):
        for field in (
            "stdout",
            "stderr",
            "exitCode",
            "timedOut",
            "cancelled",
            "durationMillis",
            "launchError",
        ):
            self.assertIn("public final " + ("int " if field == "exitCode" else "long " if field == "durationMillis" else "boolean " if field in ("timedOut", "cancelled") else "String " if field in ("stdout", "stderr", "launchError") else ""), self.runner)
        self.assertIn("Process exited with code ", self.runner)
        self.assertIn("[timeout]", self.runner)

    def test_both_existing_shell_entry_points_use_the_shared_runner(self):
        self.assertIn("CommandRunner.run(builder, timeoutMillis, maxOutputChars)", self.linux_engine)
        self.assertIn("public CommandRunner.Result execResult(", self.linux_engine)
        self.assertIn("public static CommandRunner.Result runShellCommandResult(", self.engine_core)
        self.assertIn("CommandRunner.run(builder, timeoutMillis, maxOutputChars)", self.engine_core)
        self.assertNotIn("p.waitFor()", self.engine_core)

    def test_runtime_status_no_longer_launches_shell_from_ui_thread(self):
        self.assertIn("Build.SUPPORTED_ABIS", self.main_activity)
        self.assertNotIn('EngineCore.runShellCommand("getprop ro.product.cpu.abi")', self.main_activity)

    def test_quality_workflow_sets_up_jdk_for_behavioral_tests(self):
        self.assertIn("actions/setup-java@", self.workflow)
        self.assertIn("java-version: '17'", self.workflow)


if __name__ == "__main__":
    unittest.main()
