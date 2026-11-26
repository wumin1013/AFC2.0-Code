"""Derive 450 development load CSVs; never launch training or modify parent evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-service-output", required=True)
    parser.add_argument("--parent-snapshot-id", required=True)
    parser.add_argument("--historical-route-source", required=True)
    parser.add_argument("--output-service", required=True, type=Path)
    parser.add_argument("--artifact-relocations", type=Path,
                        help="Existing JSON object with relocation receipt path and sha256")
    parser.add_argument("--endpoint-policy", choices=["terminal_uniform_ulp_v1"],
                        help="Explicit v2 normalization of proven one-ULP-short terminal source cells")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    request = dict(parent_service_output=args.parent_service_output, parent_snapshot_id=args.parent_snapshot_id,
        historical_route_source=args.historical_route_source,
        artifact_relocations=json.loads(args.artifact_relocations.read_text(encoding="utf-8"))
                             if args.artifact_relocations else None,
        endpoint_policy=args.endpoint_policy, execute=args.execute)
    if not args.execute:
        print(json.dumps(dict(status="planned_not_executed", output_service=str(args.output_service.resolve()),
                              training_started=False, training_released=False, request=request), ensure_ascii=False))
        return 0
    if args.output_service.exists():
        raise FileExistsError("候选服务输出目录已存在，拒绝覆盖")
    from spindle_power.load_prediction.training_service import TrainingService
    def progress(event):
        if event["completed"] % 25 == 0 or event["completed"] == event["total"]:
            print(json.dumps(event, ensure_ascii=False), flush=True)
    result = TrainingService(args.output_service).derive_support_fill(**request, progress=progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
