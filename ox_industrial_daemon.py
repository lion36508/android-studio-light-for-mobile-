#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OxDroid local execution endpoint.

Code execution is intentionally unavailable until a real OS-enforced sandbox
is implemented and verified. A temporary directory and subprocess timeout do
not isolate untrusted code from the Android app's user identity or data.
"""
import http.server
import json
import socketserver

PORT = 49152
UNAVAILABLE = {
    "status": "unsupported",
    "output": (
        "Execution disabled: OxDroid does not yet provide an isolated sandbox. "
        "Your code was not executed. Save and edit remain available."
    ),
}


class IndustrialCompilerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = json.dumps(UNAVAILABLE).encode("utf-8")
        self.send_response(501)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print("OxDroid execution endpoint disabled: isolated sandbox unavailable.")
    with ReusableTCPServer(("127.0.0.1", PORT), IndustrialCompilerHandler) as httpd:
        httpd.serve_forever()
