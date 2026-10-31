import http.client
import http.server
import threading
import unittest
from email.message import Message

from ox_industrial_daemon import (
    IndustrialCompilerHandler,
    MAX_CONCURRENT_REQUESTS,
    MAX_REQUEST_BYTES,
    ReusableTCPServer,
    validate_request_headers,
)


def make_headers(*pairs):
    headers = Message()
    for name, value in pairs:
        headers.add_header(name, value)
    return headers


class RequestHeaderValidationTest(unittest.TestCase):
    def test_requires_content_length(self):
        self.assertEqual(
            validate_request_headers(make_headers()),
            (411, "Content-Length is required."),
        )

    def test_rejects_non_decimal_content_length(self):
        self.assertEqual(
            validate_request_headers(make_headers(("Content-Length", "1x")))[0],
            400,
        )

    def test_rejects_duplicate_content_length(self):
        headers = make_headers(
            ("Content-Length", "2"),
            ("Content-Length", "2"),
        )
        self.assertEqual(validate_request_headers(headers)[0], 400)

    def test_rejects_transfer_encoding(self):
        headers = make_headers(
            ("Content-Length", "0"),
            ("Transfer-Encoding", "chunked"),
        )
        self.assertEqual(validate_request_headers(headers)[0], 400)

    def test_accepts_body_at_limit_and_rejects_oversized_body(self):
        self.assertIsNone(
            validate_request_headers(make_headers(("Content-Length", str(MAX_REQUEST_BYTES))))
        )
        self.assertEqual(
            validate_request_headers(
                make_headers(("Content-Length", str(MAX_REQUEST_BYTES + 1)))
            )[0],
            413,
        )


class DaemonHttpIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.server = ReusableTCPServer(
            ("127.0.0.1", 0), IndustrialCompilerHandler
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, headers=None, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request("POST", "/execute", body=body, headers=headers or {})
        response = connection.getresponse()
        status = response.status
        payload = response.read()
        connection.close()
        return status, payload

    def test_oversized_request_is_rejected_before_execution(self):
        status, payload = self.request(
            headers={"Content-Length": str(MAX_REQUEST_BYTES + 1)}
        )
        self.assertEqual(status, 413)
        self.assertIn(b'"status": "error"', payload)

    def test_valid_request_still_fails_closed(self):
        status, payload = self.request(body=b"{}")
        self.assertEqual(status, 501)
        self.assertIn(b'"status": "unsupported"', payload)
        self.assertIn(b"Your code was not executed", payload)


class BoundedConcurrencyTest(unittest.TestCase):
    def test_server_rejects_requests_above_concurrency_limit(self):
        barrier = threading.Barrier(MAX_CONCURRENT_REQUESTS)

        class BlockingHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                barrier.wait(timeout=4)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = ReusableTCPServer(("127.0.0.1", 0), BlockingHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        results = []
        result_lock = threading.Lock()

        def make_request():
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                connection.request("GET", "/")
                response = connection.getresponse()
                response.read()
                with result_lock:
                    results.append(response.status)
            except Exception:
                with result_lock:
                    results.append("error")
            finally:
                connection.close()

        clients = [
            threading.Thread(target=make_request)
            for _ in range(MAX_CONCURRENT_REQUESTS + 1)
        ]
        try:
            for client in clients:
                client.start()
            for client in clients:
                client.join(timeout=6)
            self.assertTrue(all(not client.is_alive() for client in clients))
            self.assertEqual(results.count(200), MAX_CONCURRENT_REQUESTS)
            self.assertEqual(results.count(503), 1)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
