import pathlib
import unittest
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"


class NetworkSecurityConfigTest(unittest.TestCase):
    def test_manifest_disables_global_cleartext_and_uses_network_config(self):
        manifest = ET.parse(ROOT / "AndroidManifest.xml").getroot()
        application = manifest.find("application")
        self.assertIsNotNone(application)
        self.assertEqual(application.get(ANDROID_NS + "usesCleartextTraffic"), "false")
        self.assertEqual(
            application.get(ANDROID_NS + "networkSecurityConfig"),
            "@xml/network_security_config",
        )

    def test_only_localhost_is_allowed_cleartext(self):
        config = ET.parse(ROOT / "res/xml/network_security_config.xml").getroot()
        base = config.find("base-config")
        self.assertIsNotNone(base)
        self.assertEqual(base.get("cleartextTrafficPermitted"), "false")

        allowed = config.findall("domain-config")
        self.assertEqual(len(allowed), 1)
        self.assertEqual(allowed[0].get("cleartextTrafficPermitted"), "true")
        domains = [node.text for node in allowed[0].findall("domain")]
        self.assertEqual(domains, ["localhost"])

    def test_daemon_client_uses_the_allowed_localhost_name(self):
        source = (ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java").read_text()
        self.assertIn('new URL("http://localhost:49152/execute")', source)
        self.assertNotIn('new URL("http://127.0.0.1:49152/execute")', source)


if __name__ == "__main__":
    unittest.main()
