"""Tests for the opt-in, controllable OxDroid background guard."""
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ox_background_guard.sh"


class BackgroundGuardTests(unittest.TestCase):
    def run_guard(self, command, env, timeout=10):
        return subprocess.run(
            ["bash", str(SCRIPT), command],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

    def test_guard_does_not_start_without_an_explicit_command(self):
        with tempfile.TemporaryDirectory() as temp:
            env = os.environ.copy()
            env["OX_GUARD_STATE_DIR"] = str(pathlib.Path(temp) / "state")
            result = self.run_guard("help", env)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("does not start automatically", result.stdout)
            self.assertFalse((pathlib.Path(env["OX_GUARD_STATE_DIR"]) / "guard.pid").exists())

    def test_run_once_detects_an_unexpected_keystore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "project"
            state = pathlib.Path(temp) / "state"
            root.mkdir()
            (root / "unexpected.jks").write_bytes(b"placeholder for test only")
            env = os.environ.copy()
            env["OX_GUARD_PROJECT_ROOT"] = str(root)
            env["OX_GUARD_STATE_DIR"] = str(state)
            result = self.run_guard("run-once", env)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("unexpected.jks", result.stdout)

    def test_start_status_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "project"
            state = pathlib.Path(temp) / "state"
            root.mkdir()
            env = os.environ.copy()
            env["OX_GUARD_PROJECT_ROOT"] = str(root)
            env["OX_GUARD_STATE_DIR"] = str(state)
            env["OX_GUARD_INTERVAL_SECONDS"] = "10"

            started = self.run_guard("start", env)
            self.assertEqual(started.returncode, 0, started.stdout)
            status = self.run_guard("status", env)
            self.assertIn("RUNNING", status.stdout)

            stopped = self.run_guard("stop", env)
            self.assertEqual(stopped.returncode, 0, stopped.stdout)
            status_after = self.run_guard("status", env)
            self.assertIn("STOPPED", status_after.stdout)


if __name__ == "__main__":
    unittest.main()
