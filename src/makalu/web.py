"""A read-only static file server for the report site.

It serves files and nothing else: no API, no jobs, no writes. Reports are
built by `report` on the command line; this only hands out what is on disk.

It binds 127.0.0.1 by default. The public path is nginx on pisces-01
(`/makalu/` on :7700) -> the autossh reverse tunnel -> this port (9776), and
nginx strips the `/makalu/` prefix, which is why every link in the site is
relative.
"""

from __future__ import annotations

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from logging import getLogger
from pathlib import Path

LOG = getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9776


class SiteHandler(SimpleHTTPRequestHandler):
    # The site is rebuilt in place, so make browsers ask again rather than show
    # yesterday's numbers. Vendored libraries are versioned by filename and can be cached.
    def end_headers(self):
        if "/vendor/" in self.path:
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404, "Not found")
        return None

    def log_message(self, fmt, *args):
        LOG.info(
            "%s %s", self.headers.get("X-Real-IP") or self.client_address[0], fmt % args
        )


def serve(site: Path, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    site.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(SiteHandler, directory=str(site))
    httpd = ThreadingHTTPServer((host, port), handler)
    LOG.info(f"Serving {site} on http://{host}:{port}/")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
