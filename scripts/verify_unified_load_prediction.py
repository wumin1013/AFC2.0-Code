"""Execute 90-round synthetic acceptance; never consume experiment measurements."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

for key, value in {"PYTHONUTF8": "1", "MKL_THREADING_LAYER": "SEQUENTIAL",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
    os.environ.setdefault(key, value)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, round_spec
from spindle_power.load_prediction.feed_samples import FEED_FEATURE_COLUMNS, FEED_SAMPLE_SCHEMA, FeedSample
from spindle_power.load_prediction.samples import SampleSet
from spindle_power.load_prediction.training_service import TrainingService, balanced_metrics


SMALL_CANDIDATE = {
    "id": "synthetic_1epoch_only",
    "feed_model": {"hidden_size": 4, "num_layers": 1, "window_points": 8, "context_points": 0},
    "feed_training": {"epochs": 1, "batch_size": 128, "patience": 1, "seed": 731},
    "load_model": {"coefficient_widths": [4], "hidden": 8, "attention_heads": 2,
                   "attention_layers": 1, "dilations": [1], "kernel_size": 3, "window_points": 8},
    "load_training": {"main_epochs": 1, "residual_epochs": 1, "patience": 1, "seed": 731},
}


def synthetic_records(rounds=range(1, 91), points=4):
    """Five different categories per round, exact protocol and physical units."""
    feeds, loads = [], []
    for round_id in rounds:
        spec = round_spec(round_id)
        for feature_index, feature in enumerate(PROGRAM_FEATURES):
            positions = np.arange(points, dtype=float) * .5
            programmed = spec["feed_program_mm_min"]
            actual = programmed * (.85 + .03 * np.sin(positions + feature_index))
            feed = pd.DataFrame(np.zeros((points, len(FEED_FEATURE_COLUMNS))), columns=FEED_FEATURE_COLUMNS)
            feed["x_mm"] = positions
            feed["end_x_mm"] = positions[-1]
            feed["motion_g1"] = 1.
            feed["direction_x"] = 1.
            feed["feed_program_mm_min"] = programmed
            feed["feed_override"] = 1.
            feed["block_length_mm"] = positions[-1]
            feed["segment_fraction"] = positions / positions[-1]
            feed["curvature_per_mm"] = .01 * feature_index
            feed["line"], feed["execution_instance"], feed["sequence_id"], feed["motion"] = 10, 0, 0, 1
            feed["path_mm"], feed["segment_mm"] = positions, positions
            feed["feed_actual_mm_min"], feed["valid_label"] = actual, True
            feed["spindle_cmd_rpm"] = spec["spindle_cmd_rpm"]
            feed["cutting_contact"] = True
            record_id = f"synthetic_R{round_id:02d}_F{feature_index}"
            meta = {**spec, "record_id": record_id, "feature": feature,
                    "program_slot": spec["program_slots"][feature], "program_id": f"synthetic_family_{feature_index}",
                    "program_version": f"synthetic_nc_{feature_index}", "schema": FEED_SAMPLE_SCHEMA,
                    "feature_columns": list(FEED_FEATURE_COLUMNS), "machine_version": "synthetic_machine",
                    "machine_settings_version": "synthetic_settings", "machining_mode": "synthetic_only",
                    "feed_override": 1., "spacing_mm": .5, "diameter_mm": 10., "teeth": 4,
                    "dataset_role": "development", "complete": True, "round_complete": True,
                    "alignment_reliable": True, "no_alarm": True, "tool_healthy": True,
                    "tool_session": "synthetic_tool", "actual_event_order": round_id * 5 + feature_index,
                    "measurement_content_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
                    "condition_descriptors": {"machine": {"category": "synthetic_mill"},
                                              "tool": {"material": "synthetic_carbide"}},
                    "provenance": "synthetic_acceptance_only_not_real_accuracy"}
            load = feed.copy()
            load["ap_mm"] = 1. + .1 * feature_index
            load["ae_mm"] = 2. + .2 * feature_index
            load["diameter_mm"], load["teeth"] = 10., 4.
            load["spindle_rpm"] = spec["spindle_cmd_rpm"]
            load["feed_input_mm_min"], load["feed_source"] = actual, "measured_actual_feed"
            load["power_cut_w"] = 30. + .04 * actual * load.ap_mm + 2 * np.cos(positions)
            for key in ("valid_main", "valid_residual", "aligned", "steady", "spindle_stable"):
                load[key] = True
            feeds.append(FeedSample(feed, deepcopy(meta)))
            loads.append(SampleSet(load, deepcopy(meta)))
    return feeds, loads


def check_protocol_interfaces(service, output):
    from spindle_power.load_prediction.web_service import create_server
    server = create_server(port=0, output_dir=service.output_dir)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/v1/protocol"
        with urlopen(url, timeout=10) as response:
            http = json.load(response)
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/load_prediction.py"),
                                    "--output-dir", str(service.output_dir), "protocol"],
                                   capture_output=True, text=True, encoding="utf-8", check=True, timeout=40,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        cli = json.loads(completed.stdout)
        assert cli == http == service.protocol()
        save_json(output / "protocol_http_cli.json", {"equal": True, "protocol": cli})
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.jobs.shutdown()
        server.server_close()


def execute_acceptance(output):
    import torch
    from spindle_power.load_prediction.feed_model import predict_actual_feed
    from spindle_power.load_prediction.models import predict_cutting_power
    from spindle_power.load_prediction.neural import state_version
    from spindle_power.load_prediction.training_service import _predicted_frame
    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_before = {p.name: file_hash(p) for p in (ROOT / "src/spindle_power/load_prediction").glob("*.py")}
    service = TrainingService(output / "service")
    feeds, loads = synthetic_records()
    assert len(feeds) == len(loads) == 450
    prepared = service.save_prepared(feeds, loads, source_snapshot={"synthetic": True}, synthetic=True)
    before = {p.name: file_hash(p) for p in Path(prepared["directory"]).glob("*.csv")}
    preview = service.preview(prepared["snapshot_id"], [SMALL_CANDIDATE])
    assert len(preview["phases"]) == 9
    events = []
    def progress(event):
        events.append(event)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        if "epoch" in event:
            print(json.dumps({k: event.get(k) for k in ("fold", "stage", "epoch", "train_loss")}, ensure_ascii=False), flush=True)
    started = time.monotonic()
    result = service.execute({"snapshot_id": prepared["snapshot_id"], "job_id": "synthetic_90round_450records",
                              "candidates": [SMALL_CANDIDATE]}, progress=progress)
    phases = [e for e in events if "epoch" in e]
    assert len(phases) == 30, len(phases)
    assert {stage: sum(e["stage"] == stage for e in phases) for stage in ("feed", "main", "residual")} == {"feed": 10, "main": 10, "residual": 10}
    assert {p.name: file_hash(p) for p in Path(prepared["directory"]).glob("*.csv")} == before
    assert not service.models()
    assert len(service.models(include_synthetic=True)) == 1
    try:
        service.load_model(result["model_id"])
        raise AssertionError("synthetic model leaked into formal models")
    except ValueError:
        pass
    feed, bundle, metadata = service.load_model(result["model_id"], allow_synthetic=True)
    assert not any(p.requires_grad for p in feed.network.parameters())
    assert not any(p.requires_grad for net in (bundle.main, bundle.residual) for p in net.parameters())
    assert bundle.main_version == state_version(bundle.main) == bundle.residual_main_version
    assert bundle.residual_version == state_version(bundle.residual)
    frame = _predicted_frame(loads[0], feeds[0], feed)
    predicted = predict_cutting_power(bundle, frame, context=loads[0].metadata)
    assert (np.diff(predicted["cutting_quantiles_w"], axis=1) >= 0).all()
    feed2, bundle2, _ = service.load_model(result["model_id"], allow_synthetic=True)
    repeated = predict_cutting_power(bundle2, _predicted_frame(loads[0], feeds[0], feed2), context=loads[0].metadata)
    np.testing.assert_array_equal(repeated["cutting_quantiles_w"], predicted["cutting_quantiles_w"])
    np.testing.assert_array_equal(predict_actual_feed(feed, feeds[0]).frame.feed_predicted_mm_min,
                                  predict_actual_feed(feed2, feeds[0]).frame.feed_predicted_mm_min)
    development = json.loads(Path(result["development"]).read_text(encoding="utf-8"))
    assert development["full_epochs"] == {"feed": 1, "main": 1, "residual": 1}
    assert not development["independent_test_used"]
    assert len(development["candidates"][0]["folds"]) == 9
    for fold in development["candidates"][0]["folds"]:
        assert len(fold["validation_rounds"]) == 10
        assert len(fold["without_memory"]["records"]) == 50
        assert all(r["feed_mode"] == "predicted" for r in fold["without_memory"]["records"])
        assert all(r["feed_mode"] == "measured" for r in fold["measured_feed_diagnostic"]["records"])
    check_protocol_interfaces(service, output)
    source_after = {p.name: file_hash(p) for p in (ROOT / "src/spindle_power/load_prediction").glob("*.py")}
    changed_sources = [name for name in sorted(source_before) if source_after.get(name) != source_before[name]]
    summary = dict(status="passed_synthetic_acceptance", completed_utc=datetime.now(timezone.utc).isoformat(),
                   seconds=time.monotonic() - started, rounds=90, records=450, network_stages=30,
                   phases_by_network={s: 10 for s in ("feed", "main", "residual")}, epoch_per_stage=1,
                   frozen=True, quantile_ordered=True, reload_identical=True, snapshot_csv_unchanged=True,
                   formal_models_empty=True, protocol_http_cli_equal=True, result=result,
                   metrics_interpretation="synthetic_software_acceptance_not_real_prediction_accuracy",
                   source_fingerprints=source_after, source_fingerprints_before=source_before,
                   source_changes_during_run=changed_sources)
    save_json(output / "verification_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="合成90轮450程序位置、30阶段软件验收，不读取真实实验CSV")
    parser.add_argument("--output-dir", required=True, help="必须为不存在的新验收目录")
    return execute_acceptance(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
