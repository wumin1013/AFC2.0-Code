"""Independent actual-feed preparation, training and inference command."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Set this before NumPy or PyTorch initializes the Windows numerical runtime.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if value.get("schema") not in ("afc.feed_manifest.1", "afc.dataset_manifest.1"):
        raise ValueError("清单 schema 应为 afc.feed_manifest.1 或 afc.dataset_manifest.1")
    return value, path.parent


def input_path(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def prepare_case(record, base):
    from spindle_power.load_prediction.feed import MachineConfig
    from spindle_power.load_prediction.feed_samples import prepare_feed_features

    required = {"nc", "machine", "program_id", "feed_override", "machining_mode", "machine_settings_version"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"案例缺少字段：{sorted(missing)}")
    return prepare_feed_features(
        input_path(base, record["nc"]), MachineConfig.load(input_path(base, record["machine"])),
        program_id=record["program_id"], feed_override=record["feed_override"],
        machining_mode=record["machining_mode"], machine_settings_version=record["machine_settings_version"],
        spacing_mm=record.get("spacing_mm", .5))


def main(argv=None):
    parser = argparse.ArgumentParser(description="实际进给 BiLSTM（独立入口）")
    parser.add_argument("action", choices=("check", "train", "predict"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="训练执行开关；完整90轮审核通过后使用")
    parser.add_argument("--model", type=Path, help="预测使用的已训练模型")
    parser.add_argument("--snapshot", help="共享服务已冻结的进给样本快照，正式训练使用九折开发和全量重训")
    parser.add_argument("--service-output", type=Path, default=ROOT / "data" / "paper1" / "service")
    parser.add_argument("--diagnostic-program-holdout", action="store_true", help="显式旧程序留出诊断；不代表正式九折协议")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "paper1" / "actual_feed")
    args = parser.parse_args(argv)
    if args.execute and args.action != "train":
        parser.error("--execute 仅用于 train")
    if args.action == "predict" and args.model is None:
        parser.error("predict 需要 --model")

    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction.feed_model import FeedModelConfig, load_feed_model, predict_actual_feed

    manifest, base = read_manifest(args.manifest)
    records = manifest.get("records", [])
    if not records:
        raise ValueError("清单 records 为空")
    if args.action == "predict":
        if len(records) != 1:
            raise ValueError("预测清单只填写一个新案例")
        # No measurement file is opened on this path, even if listed.
        features = prepare_case(records[0], base)
        result = predict_actual_feed(load_feed_model(args.model), features)
        args.output.mkdir(parents=True, exist_ok=True)
        result.frame.to_csv(args.output / "predicted_feed.csv", index=False, encoding="utf-8")
        save_json(args.output / "prediction.json", result.metadata)
        print(f"预测完成：{args.output / 'predicted_feed.csv'}")
        return 0

    if not args.diagnostic_program_holdout:
        from spindle_power.load_prediction.dataset_manifest import inspect_manifest
        report = inspect_manifest(args.manifest)
        report.update(action="dry_run", trained=False, ready_to_train=report["feed_eligible"])
        if not args.snapshot:
            save_json(args.output / "preparation.json", report)
            if args.action == "train" and args.execute:
                if not report["feed_eligible"]:
                    raise ValueError("实际进给训练尚未就绪：" + "；".join(x["message"] for x in report["blockers"] if x["scope"] in ("feed", "common")))
                raise ValueError("正式训练需要共享服务冻结的 --snapshot；不能直接使用会变化的文件清单")
            print(f"已检查，未训练：{args.output / 'preparation.json'}")
            return 0
        from spindle_power.load_prediction.training_service import TrainingService
        service = TrainingService(args.service_output)
        request = {"snapshot_id": args.snapshot, "mode": "feed", "candidates": manifest.get("candidates")}
        if args.action != "train" or not args.execute:
            save_json(args.output / "preparation.json", service.preview(args.snapshot, request["candidates"], mode="feed"))
            print("共享九折任务已预览，未训练")
            return 0
        import time
        from spindle_power.load_prediction.job_service import JobService, TERMINAL
        jobs = JobService(service)
        state = jobs.create(request)
        last = None
        try:
            while state["status"] not in TERMINAL:
                state = jobs.get(state["job_id"])
                event = state.get("progress")
                if event != last:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    last = event
                time.sleep(.2)
            save_json(args.output / "training.json", state)
            if state["status"] != "succeeded":
                raise ValueError(state.get("error", f"训练任务状态：{state['status']}"))
            print(json.dumps(state["result"], ensure_ascii=False))
        except KeyboardInterrupt:
            jobs.cancel(state["job_id"])
            raise
        finally:
            jobs.shutdown()
        return 0

    from spindle_power.load_prediction.feed_samples import prepare_feed_samples
    from spindle_power.load_prediction.feed_training import FeedTrainingConfig, train_feed
    from spindle_power.load_prediction.measurement import read_measurement

    samples = []
    for record in records:
        if not record.get("measurement"):
            raise ValueError("训练检查缺少 measurement；新案例请使用 predict")
        features = prepare_case(record, base)
        measurement = read_measurement(input_path(base, record["measurement"]),
                                       sample_period_s=record.get("sample_period_s"))
        samples.append(prepare_feed_samples(
            features, measurement, record.get("metadata", {}),
            line_map={int(k): v for k, v in record.get("line_map", {}).items()} or None,
            allow_position_derivative=record.get("metadata", {}).get("position_derivative_verified") is True))
    result = train_feed(
        samples, assignment=manifest.get("assignment", {}),
        config=FeedTrainingConfig(**manifest.get("training", {})),
        model_config=FeedModelConfig(**manifest.get("model", {})),
        execute=args.action == "train" and args.execute,
        progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
    if args.action == "train" and args.execute:
        result.save(args.output / "feed_model.pt")
        save_json(args.output / "training.json", result.metadata)
        print(f"训练完成：{args.output / 'feed_model.pt'}")
    else:
        save_json(args.output / "preparation.json", result)
        print(f"已检查，未训练：{args.output / 'preparation.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
