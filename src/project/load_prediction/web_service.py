"""Loopback HTTP adapter, deliberately using standard library and static assets."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from .common import json_value
from .job_service import JobService
from .training_service import TrainingService


def create_server(*, port=8765, output_dir=None):
    service = TrainingService(output_dir)
    jobs = JobService(service)
    assets = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200):
            data = json.dumps(json_value(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == "/api/v1/protocol":
                    return self.respond(service.protocol())
                if route == "/api/v1/info":
                    return self.respond(dict(output_dir=str(service.output_dir), protocol_version=service.protocol()["version"]))
                if route == "/api/v1/models":
                    return self.respond(service.models())
                if route == "/api/v1/jobs":
                    return self.respond(jobs.list())
                if route == "/api/v1/experiments":
                    return self.respond(service.experiments())
                if route.startswith("/api/v1/jobs/"):
                    return self.respond(jobs.get(route.rsplit("/", 1)[-1]))
                filenames = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "application/javascript"),
                             "/style.css": ("style.css", "text/css")}
                if route not in filenames:
                    return self.respond({"error": "not_found"}, 404)
                filename, mime = filenames[route]
                data = (assets / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)

        def do_POST(self):
            route = urlparse(self.path).path
            # No CORS. Reject a foreign browser origin before accepting mutations.
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                return self.respond({"error": "origin_mismatch"}, 403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("JSON请求为空或过大")
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                if route == "/api/v1/discover":
                    return self.respond(service.discover(**data))
                if route == "/api/v1/inspect":
                    return self.respond(service.inspect(data["manifest"], base_validation=data.get("base_validation")))
                if route == "/api/v1/prepare":
                    return self.respond(service.prepare(**data))
                if route == "/api/v1/prepare-feed":
                    return self.respond(service.prepare_feed(**data))
                if route == "/api/v1/preview":
                    return self.respond(service.preview(**data))
                if route == "/api/v1/jobs":
                    return self.respond(jobs.create(data), 201)
                if route.startswith("/api/v1/jobs/") and route.endswith("/cancel"):
                    return self.respond(jobs.cancel(route.split("/")[-2]))
                if route == "/api/v1/predict":
                    return self.respond(service.predict(data))
                if route == "/api/v1/evaluate":
                    return self.respond(service.evaluate(data))
                if route == "/api/v1/calibrate":
                    return self.respond(service.calibrate(data))
                if route == "/api/v1/experiments":
                    return self.respond(jobs.create({**data, "mode": "experiment"}), 201) if data.get("execute") else self.respond(service.experiment(data))
                return self.respond({"error": "not_found"}, 404)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception as exc:
                self.respond({"error": str(exc)}, 500)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except BaseException:
        jobs.shutdown()
        raise
    server.service, server.jobs = service, jobs
    return server


def main():
    parser = argparse.ArgumentParser(description="AFC2.0本地负载预测网页")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    server = create_server(port=args.port, output_dir=args.output_dir)
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.jobs.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
