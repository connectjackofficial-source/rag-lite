"""Read-only HTTP retrieval endpoint for a SearchIndex.

A tiny stdlib server (no Flask, no FastAPI): GET /?q=... returns the
top-k hits as JSON. Useful for wiring rag-lite into scripts and demo
UIs without adding dependencies.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .store import SearchIndex


class _Handler(BaseHTTPRequestHandler):
    index: SearchIndex = None
    limit: int = 5

    def do_GET(self):  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self._json(404, {"error": "not found"})
            return
        qs = parse_qs(parsed.query)
        q = qs.get("q", [""])[0]
        if not q.strip():
            self._json(400, {"error": "empty query"})
            return
        hits = _Handler.index.query(q, limit=self.limit)
        self._json(200, {"hits": hits, "count": len(hits)})

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the console quiet
        pass


def serve(index: SearchIndex, host: str = "127.0.0.1", port: int = 8765,
          limit: int = 5, blocking: bool = True):
    """Serve *index* at http://host:port. Returns when Ctrl-C (or, with
    blocking=False, immediately after starting a daemon thread)."""
    _Handler.index = index
    _Handler.limit = limit
    httpd = HTTPServer((host, port), _Handler)
    if not blocking:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
