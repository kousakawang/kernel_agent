"""Tiny CUDA workload used to compare direct and service capture windows."""

from __future__ import annotations

import argparse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_LEFT: Any = None
_RIGHT: Any = None


def high_level() -> Any:
    import torch

    return torch.add(_LEFT, _RIGHT)


def _initialize() -> None:
    global _LEFT, _RIGHT
    import torch

    _LEFT = torch.randn(256 * 1024, device="cuda", dtype=torch.float32)
    _RIGHT = torch.randn_like(_LEFT)


def _direct() -> None:
    import torch

    _initialize()
    torch.add(_LEFT, _RIGHT)
    torch.cuda.synchronize()
    result = high_level()
    torch.cuda.synchronize()
    print(float(result[0]))


def _serve(port: int) -> None:
    import torch

    _initialize()
    # These target calls deliberately happen before readiness. Service-mode
    # collection and KID recording must exclude all of them.
    for _ in range(3):
        high_level()
    torch.cuda.synchronize()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ready")
                return
            if self.path == "/run":
                result = high_level()
                torch.cuda.synchronize()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(str(float(result[0])).encode("ascii"))
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _client(port: int) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/run", timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected response status: {response.status}")
        print(response.read().decode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("direct", "server", "client"))
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "direct":
        _direct()
    elif args.mode == "server":
        _serve(args.port)
    else:
        _client(args.port)


if __name__ == "__main__":
    main()
