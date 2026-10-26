# SPDX-License-Identifier: AGPL-3.0-only
"""HTTP proxy PostgREST → /rest/v1/ pour le cluster banc (port 56100)."""
from __future__ import annotations

import argparse
import secrets
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ProxyHandler(BaseHTTPRequestHandler):
    service_key: str = ""
    upstream_port: int = 56101

    def log_message(self, *_args) -> None:
        pass

    def forward(self) -> None:
        if not secrets.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + self.service_key
        ):
            self.send_error(401)
            return
        if not self.path.startswith("/rest/v1/"):
            self.send_error(404)
            return
        upstream = HTTPConnection("127.0.0.1", self.upstream_port, timeout=120)
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() in {"content-type", "prefer", "accept", "range", "range-unit"}
            }
            upstream.request(self.command, self.path[len("/rest/v1") :], body=body, headers=headers)
            response = upstream.getresponse()
            payload = response.read()
            self.send_response(response.status)
            for name in ("Content-Type", "Content-Range", "Preference-Applied"):
                if response.getheader(name):
                    self.send_header(name, response.getheader(name))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            upstream.close()

    do_GET = do_POST = do_PATCH = forward


def main() -> int:
    parser = argparse.ArgumentParser(description="Proxy banc PostgREST /rest/v1/")
    parser.add_argument("--port", type=int, default=56100)
    parser.add_argument("--upstream", type=int, default=56101)
    parser.add_argument("--service-key", required=True)
    args = parser.parse_args()
    ProxyHandler.service_key = args.service_key
    ProxyHandler.upstream_port = args.upstream
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
