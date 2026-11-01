"""Validate immutable action pins and automated update configuration."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class WorkflowIntegrityTests(unittest.TestCase):
    def test_every_external_action_is_pinned_to_a_full_commit_sha(self):
        uses_lines = []
        for path in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("uses:"):
                    uses_lines.append((path.name, line_number, stripped.split("uses:", 1)[1].strip().split()[0]))
        self.assertTrue(uses_lines, "Expected workflow action references.")
        for filename, line_number, reference in uses_lines:
            self.assertRegex(
                reference,
                r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$",
                "{}:{} action must use a full immutable SHA: {}".format(filename, line_number, reference),
            )

    def test_dependabot_updates_github_actions_weekly(self):
        path = ROOT / ".github" / "dependabot.yml"
        config = path.read_text(encoding="utf-8")
        self.assertIn('package-ecosystem: "github-actions"', config)
        self.assertIn('directory: "/"', config)
        self.assertIn('interval: "weekly"', config)

    def test_android_emulator_boot_waits_are_bounded_and_diagnostic(self):
        workflow = (WORKFLOWS / "android-apk.yml").read_text(encoding="utf-8")
        start = workflow.index("- name: Boot Android 16 emulator")
        end = workflow.index("- name: Install and smoke-test", start)
        boot_step = workflow[start:end]
        self.assertIn("timeout 300s adb wait-for-device", boot_step)
        self.assertIn("-accel off", boot_step)
        self.assertIn("boot_deadline=$((SECONDS + 300))", boot_step)
        self.assertIn("timeout 5s adb shell getprop sys.boot_completed", boot_step)
        self.assertIn('tail -n 160 "$EMULATOR_LOG"', boot_step)
        self.assertIn("did not finish booting before the 300-second deadline", boot_step)

    def test_android_emulator_binary_is_added_to_path(self):
        workflow = (WORKFLOWS / "android-apk.yml").read_text(encoding="utf-8")
        start = workflow.index("- name: Wire the Android toolchain")
        end = workflow.index("- name: Create temporary CI-only signing key", start)
        tool_step = workflow[start:end]
        self.assertIn('EMULATOR_BIN="$SDK_ROOT/emulator"', tool_step)
        self.assertIn('[[ ! -x "$EMULATOR_BIN/emulator" ]]', tool_step)
        self.assertIn('echo "$EMULATOR_BIN" >> "$GITHUB_PATH"', tool_step)

    def test_android_emulator_steps_are_gated_when_runner_lacks_kvm(self):
        workflow = (WORKFLOWS / "android-apk.yml").read_text(encoding="utf-8")
        self.assertIn("id: emulator_accel", workflow)
        self.assertIn("[[ -c /dev/kvm && -r /dev/kvm && -w /dev/kvm ]]", workflow)
        self.assertIn('emulator -accel-check > "$ACCEL_LOG" 2>&1', workflow)
        self.assertIn('echo "available=false" >> "$GITHUB_OUTPUT"', workflow)
        self.assertIn("Android 16 device smoke test skipped", workflow)
        for step in (
            "Create Android 16 emulator",
            "Boot Android 16 emulator",
            "Install and smoke-test dashboard and editor",
        ):
            start = workflow.index("- name: " + step)
            self.assertIn(
                "if: steps.emulator_accel.outputs.available == 'true'",
                workflow[start:start + 180],
            )

    def test_quality_workflow_runs_real_checks(self):
        workflow = (WORKFLOWS / "code_quality_guard.yml").read_text(encoding="utf-8")
        self.assertIn("bash -n", workflow)
        self.assertIn("ast.parse", workflow)
        self.assertIn("python3 -m unittest discover -s tests -v", workflow)
        self.assertIn("git diff --check", workflow)
        self.assertNotIn("exit 0", workflow)


if __name__ == "__main__":
    unittest.main()
