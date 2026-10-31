"""Static regression checks for the Android 16 / API 36 migration."""
import pathlib
import unittest
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"


class Android16TargetTests(unittest.TestCase):
    def test_manifest_targets_api_36_and_preserves_minimum_api(self):
        manifest = ET.parse(ROOT / "AndroidManifest.xml").getroot()
        uses_sdk = manifest.find("uses-sdk")
        self.assertIsNotNone(uses_sdk)
        self.assertEqual(uses_sdk.get(ANDROID_NS + "minSdkVersion"), "24")
        self.assertEqual(uses_sdk.get(ANDROID_NS + "targetSdkVersion"), "36")

    def test_ci_compiles_against_platform_36_and_build_tools_36(self):
        workflow = (ROOT / ".github/workflows/android-apk.yml").read_text(encoding="utf-8")
        self.assertIn("platforms;android-36", workflow)
        self.assertIn("build-tools;36.0.0", workflow)
        self.assertIn('ANDROID_PLATFORM_JAR="$SDK_ROOT/platforms/android-36/android.jar"', workflow)
        self.assertIn('echo "ANDROID_JAR=$ANDROID_PLATFORM_JAR" >> "$GITHUB_ENV"', workflow)

    def test_local_build_defaults_to_api_36_when_sdk_root_is_set(self):
        script = (ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_ANDROID_JAR="$SDK_ROOT/platforms/android-36/android.jar"', script)
        self.assertIn("Android SDK Platform 36 is required", script)

    def test_dashboard_and_editor_apply_system_bar_insets(self):
        main = (ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java").read_text(encoding="utf-8")
        editor = (ROOT / "src/main/java/com/bikash/mobileide/EditorActivity.java").read_text(encoding="utf-8")
        self.assertIn("SystemBarInsets.apply(root)", main)
        self.assertIn("SystemBarInsets.apply(root)", editor)

    def test_insets_helper_only_applies_on_android_15_and_newer(self):
        helper = (ROOT / "src/main/java/com/bikash/mobileide/SystemBarInsets.java").read_text(encoding="utf-8")
        self.assertIn("Build.VERSION.SDK_INT < 35", helper)
        self.assertIn("getSystemWindowInsetTop()", helper)
        self.assertIn("getSystemWindowInsetBottom()", helper)
        self.assertIn("requestApplyInsets()", helper)

    def test_network_security_policy_stays_enabled(self):
        manifest = ET.parse(ROOT / "AndroidManifest.xml").getroot()
        app = manifest.find("application")
        self.assertEqual(app.get(ANDROID_NS + "usesCleartextTraffic"), "false")
        self.assertEqual(app.get(ANDROID_NS + "networkSecurityConfig"), "@xml/network_security_config")


if __name__ == "__main__":
    unittest.main()
