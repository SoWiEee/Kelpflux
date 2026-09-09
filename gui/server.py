"""Serve the dashboard and a small, clearly synthetic jobs endpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent


def mock_payload() -> dict[str, object]:
    now = datetime.now(timezone.utc)
    now_epoch = now.timestamp()
    return {
        "updated_at": now.isoformat(),
        "jobs": [
            {
                "job_id": "8421",
                "state": "RUNNING",
                "node": "slurm-worker-gpu-rtx4070-0",
                "gpu": {"type": "rtx4070", "index": 0},
                "mps": {"requested": 25, "allocated": 25},
                "submit_ts": now_epoch - 240,
                "resource_usage": None,
            },
            {
                "job_id": "8422",
                "state": "PENDING",
                "mps": {"requested": 50},
                "submit_ts": now_epoch - 95,
                "resource_usage": None,
            },
            {
                "job_id": "8423",
                "state": "RUNNING",
                "node": "slurm-worker-gpu-rtx3080-0",
                "gpu": {"type": "rtx3080", "index": 0},
                "mps": {"requested": 12, "allocated": 12},
                "submit_ts": now_epoch - 680,
                "resource_usage": None,
            },
        ],
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path.rstrip("/") == "/api/jobs":
            body = json.dumps(mock_payload()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[gui] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Kelpflux GUI with mock /api/jobs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Kelpflux GUI listening at http://{args.host}:{args.port}/")
    print("Synthetic data endpoint: /api/jobs")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
