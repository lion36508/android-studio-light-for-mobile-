"""Regression tests for destructive build-output path validation."""
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "build.sh"


class BuildPathSafetyTests(unittest.TestCase):
    def run_build(self, env):
        return subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )

    def base_env(self, temp):
        env = os.environ.copy()
        temp_path = pathlib.Path(temp)
        env["ANDROID_JAR"] = str(ROOT / "android.jar")
        env["BUILD_DIR"] = str(temp_path / "build")
        env["GEN_DIR"] = str(temp_path / "generated")
        env["OUT_DIR"] = str(temp_path / "output")
        return env

    def assert_rejected_before_toolchain(self, env, expected):
        result = self.run_build(env)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(expected, result.stdout)
        self.assertNotIn("required build tool", result.stdout)

    def test_rejects_filesystem_root_as_build_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            env["BUILD_DIR"] = "/"
            self.assert_rejected_before_toolchain(env, "filesystem root")

    def test_rejects_temporary_directory_root_as_build_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            env["BUILD_DIR"] = "/tmp"
            self.assert_rejected_before_toolchain(env, "temporary-directory root")

    def test_rejects_repository_root_as_build_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            env["BUILD_DIR"] = str(ROOT)
            self.assert_rejected_before_toolchain(env, "repository root")

    def test_resolves_symlinks_before_protecting_source_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            source_link = pathlib.Path(temp) / "source-link"
            source_link.symlink_to(ROOT / "src", target_is_directory=True)
            env["BUILD_DIR"] = str(source_link)
            self.assert_rejected_before_toolchain(env, "protected source/project directory")

    def test_rejects_source_directory_as_generation_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            env["GEN_DIR"] = str(ROOT / "src")
            self.assert_rejected_before_toolchain(env, "protected source/project directory")
            self.assertTrue((ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java").is_file())

    def test_rejects_overlapping_build_and_generation_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            env["BUILD_DIR"] = str(pathlib.Path(temp) / "build")
            env["GEN_DIR"] = str(pathlib.Path(temp) / "build" / "generated")
            self.assert_rejected_before_toolchain(env, "must be separate, non-overlapping")

    def test_rejects_apk_name_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            env["APK_NAME"] = "../../README.md"
            self.assert_rejected_before_toolchain(env, "APK_NAME must be a simple file name")

    def test_valid_nested_output_paths_pass_path_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.base_env(temp)
            env["BUILD_DIR"] = str(pathlib.Path(temp) / "new" / "build")
            env["GEN_DIR"] = str(pathlib.Path(temp) / "new" / "generated")
            env["OUT_DIR"] = str(pathlib.Path(temp) / "new" / "output")
            env["ANDROID_JAR"] = str(pathlib.Path(temp) / "missing-android.jar")
            result = self.run_build(env)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("android.jar not found", result.stdout)
            self.assertNotIn("Unsafe", result.stdout)
            self.assertFalse((pathlib.Path(temp) / "new").exists())


if __name__ == "__main__":
    unittest.main()
