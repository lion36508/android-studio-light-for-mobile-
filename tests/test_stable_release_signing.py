"""Ensure release publishing requires a stable, protected signing identity."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/publish-release.yml"
BUILD_SCRIPT = ROOT / "build.sh"
DOCS = ROOT / "docs/BUILDING.md"


class StableReleaseSigningTests(unittest.TestCase):
    def test_release_workflow_requires_all_stable_signing_secrets(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        for secret in (
            "OXDROID_RELEASE_KEYSTORE_BASE64",
            "OXDROID_RELEASE_KEYSTORE_PASSWORD",
            "OXDROID_RELEASE_KEY_ALIAS",
            "OXDROID_RELEASE_KEY_PASSWORD",
        ):
            self.assertIn(secret, content)
        self.assertIn("Required stable release signing secret is missing", content)
        self.assertIn("keytool -list", content)

    def test_release_workflow_does_not_create_a_temporary_key(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("keytool -genkeypair", content)
        self.assertNotIn("openssl rand", content)
        self.assertIn('KEYSTORE="$RUNNER_TEMP/oxdroid-release.jks"', content)
        self.assertIn("chmod 600", content)

    def test_release_build_uses_android_16_toolchain(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("platforms;android-36", content)
        self.assertIn("build-tools;36.0.0", content)
        self.assertIn('ANDROID_JAR=$ANDROID_PLATFORM_JAR', content)

    def test_build_script_supports_key_alias_and_distinct_key_password(self):
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('KEY_PASS="${KEY_PASS:-$KEYSTORE_PASS}"', content)
        self.assertIn('APKSIGNER_ALIAS_ARGS+=(--ks-key-alias "$KEY_ALIAS")', content)
        self.assertIn("--key-pass env:KEY_PASS", content)

    def test_release_signing_instructions_warn_against_the_old_development_key(self):
        content = DOCS.read_text(encoding="utf-8")
        self.assertIn("OXDROID_RELEASE_KEYSTORE_BASE64", content)
        self.assertIn("never reuse the previously exposed development key", content)
        self.assertIn("disposable key", content)


if __name__ == "__main__":
    unittest.main()
