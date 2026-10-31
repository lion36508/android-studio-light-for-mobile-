"""Regression checks preventing signing material from re-entering Git."""
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SigningSecretPolicyTests(unittest.TestCase):
    def test_no_signing_material_or_env_file_is_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        )
        tracked = [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
        sensitive = []
        for path in tracked:
            lower = path.lower()
            name = pathlib.PurePosixPath(lower).name
            if lower.endswith((".jks", ".keystore")) or name == "debug.keystore":
                sensitive.append(path)
            elif name == ".env" or (name.startswith(".env.") and name != ".env.example"):
                sensitive.append(path)
        self.assertEqual([], sensitive, "Signing secrets must never be tracked.")

    def test_gitignore_excludes_local_signing_material(self):
        rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for rule in ("*.jks", "*.keystore", ".env", ".env.*"):
            self.assertIn(rule, rules)

    def test_ci_signing_key_is_ephemeral_and_random(self):
        workflow = (ROOT / ".github/workflows/android-apk.yml").read_text(encoding="utf-8")
        self.assertIn('KEYSTORE="$RUNNER_TEMP/oxdroid-ci-debug.jks"', workflow)
        self.assertIn('openssl rand -hex 24', workflow)
        self.assertIn('::add-mask::$KEYSTORE_PASS', workflow)
        self.assertNotIn("ide_keystore.jks", workflow)

    def test_release_workflow_does_not_reuse_a_repository_keystore(self):
        workflow = (ROOT / ".github/workflows/publish-release.yml").read_text(encoding="utf-8")
        self.assertIn('KEYSTORE="$RUNNER_TEMP/oxdroid-release-ci.jks"', workflow)
        self.assertIn('openssl rand -hex 24', workflow)
        self.assertIn('::add-mask::$KEYSTORE_PASS', workflow)
        self.assertNotIn("ide_keystore.jks", workflow)

    def test_background_guard_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(ROOT / "ox_background_guard.sh")], check=True)

    def test_background_guard_never_whitelists_a_keystore(self):
        guard = (ROOT / "ox_background_guard.sh").read_text(encoding="utf-8")
        self.assertNotIn("WHITELIST_KEYSTORE", guard)
        self.assertNotIn("ide_keystore.jks", guard)
        self.assertIn("-name '*.keystore'", guard)
        self.assertIn("-name '*.jks'", guard)
        self.assertIn('WHITELIST_APK="${OX_GUARD_WHITELIST_APK:-./bin/OxDroid-Mobile-IDE.apk}"', guard)


if __name__ == "__main__":
    unittest.main()
