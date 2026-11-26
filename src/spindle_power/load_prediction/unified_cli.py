"""Model-host command interface. Not part of the interval tool executable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import save_json
from .unified import CONTEXT_DEFAULT, UnifiedSpindleModel, predict_files


def main(argv=None):
    parser = argparse.ArgumentParser(description="AFC统一主轴功率模型")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--cutting", required=True)
    build.add_argument("--base", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--evidence", action="append", default=[])
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--output", required=True)
    predict = commands.add_parser("predict")
    for field in ("model", "nc", "process", "context", "output"):
        predict.add_argument("--" + field, required=True)
    predict.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    context = commands.add_parser("write-default-context")
    context.add_argument("--output", required=True)
    update = commands.add_parser("update-base")
    for field in ("model", "base", "output"):
        update.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    source_paths = {Path(getattr(args, key)).resolve() for key in ("cutting", "base", "model", "nc", "process", "context")
                    if getattr(args, key, None)}
    source_paths.update(Path(path).resolve() for path in getattr(args, "evidence", []))
    targets = [Path(args.output)] if args.command != "predict" else [Path(args.output) / name for name in ("prediction.csv", "prediction.json")]
    if args.command == "build":
        targets.append(Path(args.output).with_suffix(".manifest.json"))
    if any(path.resolve() in source_paths for path in targets):
        parser.error("输出不得覆盖来源模型、输入或证据文件，请指定独立输出路径")
    directory = Path(args.output).parent if args.command != "predict" else Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    if args.command == "write-default-context":
        save_json(args.output, CONTEXT_DEFAULT)
        return 0
    if args.command == "build":
        model = UnifiedSpindleModel.build(args.cutting, args.base, runtime_dir=directory, evidence_paths=args.evidence)
        receipt = model.save(args.output)
        save_json(Path(args.output).with_suffix(".manifest.json"), {**model.manifest, "artifact": receipt})
    elif args.command == "inspect":
        model = UnifiedSpindleModel.load(args.model, runtime_dir=directory)
        save_json(args.output, {"package_version": model.version, "manifest": model.manifest})
        receipt = {"version": model.version}
    elif args.command == "update-base":
        model = UnifiedSpindleModel.load(args.model, runtime_dir=directory)
        receipt = model.with_base(args.base, runtime_dir=directory).save(args.output)
    else:
        model = UnifiedSpindleModel.load(args.model, runtime_dir=directory, device=args.device)
        receipt = predict_files(model, nc_path=args.nc, process_path=args.process, context_path=args.context,
                                output_dir=args.output)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
