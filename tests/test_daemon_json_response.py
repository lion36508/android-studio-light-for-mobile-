import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/main/java/com/bikash/mobileide/EngineCore.java"


class DaemonJsonResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ENGINE.read_text(encoding="utf-8")

    def test_requests_are_serialized_with_android_json_api(self):
        self.assertIn("import org.json.JSONObject;", self.source)
        self.assertIn("JSONObject request = new JSONObject();", self.source)
        self.assertIn('request.put("language"', self.source)
        self.assertIn('request.put("filename"', self.source)
        self.assertIn('request.put("code"', self.source)
        self.assertIn("MAX_DAEMON_REQUEST_BYTES = 64 * 1024", self.source)

    def test_response_parser_checks_http_and_application_status(self):
        self.assertIn("connection.getResponseCode()", self.source)
        self.assertIn("connection.getErrorStream()", self.source)
        self.assertIn("new JSONObject(responseBody)", self.source)
        self.assertIn('response.opt("status")', self.source)
        self.assertIn('response.opt("output")', self.source)
        self.assertIn('"unsupported".equalsIgnoreCase(status)', self.source)
        self.assertIn('"error".equalsIgnoreCase(status)', self.source)
        self.assertIn('"success".equalsIgnoreCase(status)', self.source)

    def test_response_body_is_bounded_and_connection_is_released(self):
        self.assertIn("MAX_DAEMON_RESPONSE_CHARS = 256 * 1024", self.source)
        self.assertIn("readDaemonResponse(responseStream)", self.source)
        self.assertIn("connection.disconnect()", self.source)
        self.assertIn('try (OutputStream output = connection.getOutputStream())', self.source)

    def test_malformed_json_and_transport_failures_are_visible(self):
        self.assertIn("Daemon response rejected: invalid JSON", self.source)
        self.assertIn("Daemon communication failed:", self.source)
        self.assertIn("Daemon returned HTTP ", self.source)

    def test_public_execution_entry_point_stays_disabled(self):
        match = re.search(
            r"public static String executeCode\(Context context, String language, String fileName, String code\) \{(.*?)\n    \}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(match)
        method = match.group(1)
        self.assertIn("was NOT executed", method)
        self.assertNotIn("tryIndustrialDaemon(", method)


if __name__ == "__main__":
    unittest.main()
