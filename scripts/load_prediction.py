"""Unified local CLI; all decisions are delegated to the public services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project.load_prediction.common import json_value
from project.load_prediction.training_service import TrainingService


def main():
    parser = argparse.ArgumentParser(description="AFC2.0负载预测统一入口")
    parser.add_argument("--output-dir")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("protocol", "models", "experiments", "web"):
        p = sub.add_parser(action)
        if action == "web":
            p.add_argument("--port", type=int, default=8765)
    for action in ("discover", "inspect", "prepare", "prepare-feed", "refresh-load-route", "preview", "train", "predict", "evaluate", "calibrate", "experiment"):
        p = sub.add_parser(action)
        p.add_argument("request", help="请求JSON文件；参数与HTTP公共接口一致")
    args = parser.parse_args()
    service = TrainingService(args.output_dir)
    if args.action == "web":
        from project.load_prediction.web_service import create_server
        server = create_server(port=args.port, output_dir=args.output_dir)
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.jobs.shutdown()
            server.server_close()
        return
    request = json.loads(Path(args.request).read_text(encoding="utf-8")) if hasattr(args, "request") else None
    if args.action == "protocol":
        result = service.protocol()
    elif args.action == "models":
        result = service.models()
    elif args.action == "experiments":
        result = service.experiments()
    elif args.action == "train" or args.action == "experiment" and request.get("execute"):
        # The CLI waits while its durable background process performs training.
        from project.load_prediction.job_service import JobService, TERMINAL
        import time
        if args.action == "experiment":
            request["mode"] = "experiment"
        jobs = JobService(service)
        state = jobs.create(request)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        try:
            while True:
                state = jobs.get(state["job_id"])
                if state["status"] in TERMINAL:
                    break
                time.sleep(.5)
            result = state
        except KeyboardInterrupt:
            jobs.cancel(state["job_id"])
            raise
        finally:
            jobs.shutdown()
    elif args.action in ("discover", "inspect", "prepare", "prepare-feed", "refresh-load-route", "preview"):
        result = getattr(service, args.action.replace("-", "_"))(**request)
    else:
        result = getattr(service, args.action)(request)
    print(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
    if isinstance(result, dict) and result.get("status") in ("failed", "interrupted", "cancelled"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
