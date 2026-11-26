"""Audit frozen event confirmation against the production label interface.

No calibration, model promotion or cutting-label qualification is performed.
Outputs belong to a NEW run directory; original data and reports are read-only.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spindle_power.load_prediction.base_power import BasePowerModel, CalibrationConfig
from spindle_power.load_prediction.base_events import _filtered, _settling, predict_event
from spindle_power.load_prediction.base_qualification import BRIDGE_SCHEMA
from spindle_power.load_prediction.base_timeline import (AUDITED_INTERFACE_VERSION, AuditedEventTimeline,
                                                  predict_audited_measured, speed_timeline_fingerprint)
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.measurement import read_measurement


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def alignment_source_contract(source, development):
    tree = ast.parse(source)
    align = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "alignment")
    strings = [node.value for node in ast.walk(align) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    no_power = not any("power" in value.lower() for value in strings)
    devtree = ast.parse(development)
    calls = [node for node in ast.walk(devtree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "alignment"]
    disabled = bool(calls) and all(
        call.args and isinstance(call.args[0], ast.Call)
        and any(k.arg == "power" and isinstance(k.value, ast.Constant) and k.value.value is False
                for k in call.args[0].keywords) for call in calls)
    return no_power and disabled


def verify_nc_events(events, lines):
    failures = []
    for event in events:
        if event.get("direction") not in ("startup", "braking"):
            continue
        raw_line = int(event["raw_program_line"])
        command = lines[raw_line].replace(" ", "").upper()
        target = float(event["speed_rpm"])
        if event["direction"] == "startup":
            speed = re.search(r"S([0-9]+(?:\.[0-9]+)?)", command)
            good = "M03" in command and speed is not None and float(speed.group(1)) == target
        else:
            good = "M05" in command and target == 0
        dwell = re.search(r"G0?4X([0-9]+(?:\.[0-9]+)?)", lines[raw_line + 1].replace(" ", "").upper())
        good = good and dwell is not None and float(dwell.group(1)) == float(event["expected_dwell_s"])
        if not good:
            failures.append(dict(raw_program_line=raw_line, command=command, target_rpm=target))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--interface-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, interface, output = args.model_run.resolve(), args.interface_run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    freeze_path, confirmation_path = original / "freeze_manifest.json", original / "candidate_confirmation.json"
    freeze, confirmation = read(freeze_path), read(confirmation_path)
    model_path = original / freeze["model_path"]
    model = BasePowerModel.load(model_path)
    before_model = asdict(model)
    config = CalibrationConfig(**model.config)
    protected = read(interface / "input_integrity.json")["after"]
    actual_hashes = {path: file_hash(path) for path in protected}
    raw_unchanged = all(actual_hashes[path] == expected.lower() for path, expected in protected.items())
    core = ROOT / "src/spindle_power/load_prediction/base_events.py"
    frozen_core_hash = freeze["snapshot_sha256"]["load_prediction\\base_events.py"].lower()
    code_ok = all(file_hash(original / name) == expected.lower()
                  for name, expected in freeze["numerical_source_sha256"].items())
    started = datetime.fromisoformat(confirmation["started_utc"])
    frozen = datetime.fromisoformat(freeze["frozen_utc"])
    aligned_path = original / "aligned_events.json"
    aligns = read(aligned_path)
    no_power = alignment_source_contract((original / "analysis_core.py").read_text(encoding="utf-8"),
                                        (original / "develop.py").read_text(encoding="utf-8"))
    nc_path = next(Path(path) for path in protected if Path(path).name == "OM03S.NC")
    nc_lines = nc_path.read_text(encoding="utf-8").splitlines()
    all_rows, round_rows, old_score_differences = [], [], []
    all_matches, all_nc_good, all_cache_good, all_coverage, all_grid_explained = True, True, True, True, True
    report_saved = read(interface / "interface_validation.json")
    for number in range(1, 6):
        timeline_path = interface / f"round{number}_audited_timeline.json"
        timeline = AuditedEventTimeline(**read(timeline_path))
        raw = read_measurement(timeline.record_source)
        with np.load(original / f"cache/round{number}.npz") as cached:
            cache_ok = all(np.array_equal(first, second) for first, second in (
                (raw.time_s, cached["time"]), (raw.values["spindle_speed_rpm"], cached["speed"]),
                (raw.values["spindle_power_w"], cached["power"]), (raw.values["program_line_raw"], cached["line"])))
        all_cache_good &= cache_ok
        timeline_matches = (timeline.events == aligns[str(number)]
                            and timeline.provenance["audit_file_sha256"].lower() == file_hash(aligned_path)
                            and timeline.time_speed_sha256 == speed_timeline_fingerprint(raw.time_s, raw.values["spindle_speed_rpm"]))
        all_matches &= timeline_matches
        nc_failures = verify_nc_events(timeline.events, nc_lines)
        all_nc_good &= not nc_failures
        # Prediction reads only time, speed and the previously audited timeline.
        prediction = predict_audited_measured(raw.time_s, raw.values["spindle_speed_rpm"], model, timeline,
                                              record_source=timeline.record_source)
        all_coverage &= bool(prediction["valid"].all())
        old_scores = next((s for s in confirmation["scores"] if s["round"] == number), None)
        saved_new = next(s for s in report_saved["results"] if s["round"] == number)["score"]
        event_deltas = []
        for index, event in enumerate(timeline.events):
            if event.get("direction") not in ("startup", "braking"):
                continue
            origin = timeline.events[index - 1]["speed_rpm"]
            target, onset = event["speed_rpm"], event["command_time_s"]
            use = (raw.time_s >= onset) & (raw.time_s <= event["end_s"])
            relative_time = raw.time_s[use] - onset
            original_prediction = predict_event(target, model, initial_speed_rpm=origin)
            old_power = np.interp(relative_time, original_prediction["time_s"], original_prediction["power_w"])
            new_power = prediction["power_w"][use]
            max_delta = float(np.max(np.abs(old_power - new_power)))
            ss = original_prediction["characteristics"]["steady_power_w"]
            tolerance = max(config.settling_absolute_band_w, config.settling_relative_band * abs(ss))
            old_inside = abs(old_power - ss) <= tolerance
            new_inside = abs(new_power - ss) <= tolerance
            differences = old_inside != new_inside
            boundary_distance = float(np.max(abs(abs(old_power[differences] - ss) - tolerance))) if differences.any() else 0.
            peak_old = int(np.argmax(_filtered(old_power, .001, config)) if target else np.argmin(_filtered(old_power, .001, config)))
            peak_new = int(np.argmax(_filtered(new_power, .001, config)) if target else np.argmin(_filtered(new_power, .001, config)))
            settle_old = _settling(relative_time, old_power, ss, relative_time[peak_old], config)
            settle_new = _settling(relative_time, new_power, ss, relative_time[peak_new], config)
            settle_delta = abs(settle_old - settle_new) if settle_old is not None and settle_new is not None else None
            explained = (settle_delta is not None and settle_delta <= .001 + 1e-10 and max_delta <= 1e-8
                         and boundary_distance <= 1e-8 and peak_old == peak_new)
            all_grid_explained &= explained
            row = dict(round=number, event_index=index, direction=event["direction"], target_rpm=target,
                       samples=len(relative_time), max_curve_difference_w=max_delta,
                       original_settling_s=settle_old, production_settling_s=settle_new,
                       settling_difference_s=settle_delta, band_membership_difference_samples=int(differences.sum()),
                       maximum_band_boundary_distance_w=boundary_distance, numerical_difference_explained=explained)
            all_rows.append(row)
            event_deltas.append(max_delta)
            if old_scores:
                old = next(r for r in old_scores["events"] if r["index"] == index)
                new = next(r for r in saved_new["events"] if r["index"] == index)
                old_score_differences.append(dict(round=number, event_index=index,
                    **{key: abs(old[key] - new[key]) for key in
                       ("peak_relative_error", "peak_time_error_s", "settling_time_error_s", "steady_error_w", "curve_rmse_w")}))
        round_rows.append(dict(round=number, samples=len(raw.time_s), valid_samples=int(prediction["valid"].sum()),
            coverage_fraction=float(prediction["valid"].mean()), raw_matches_original_cache=cache_ok,
            timeline_matches_original_audit=timeline_matches, nc_event_failures=nc_failures,
            maximum_curve_difference_w=max(event_deltas), timeline_sha256=timeline.fingerprint(),
            raw_sha256=file_hash(timeline.record_source), formal_training_release=prediction["formal_training_release"]))
        print(f"round {number}: {len(raw.time_s)} points, coverage {prediction['valid'].mean():.6f}, max curve delta {max(event_deltas):.3g} W", flush=True)
    checks = dict(model_unchanged=asdict(model) == before_model and file_hash(model_path) == freeze["model_sha256"].lower(),
        core_unchanged=file_hash(core) == frozen_core_hash, thresholds_unchanged=model.config == freeze["score_config"],
        alignment_frozen_before_confirmation=bool(code_ok and frozen < started and all_matches and all_nc_good),
        alignment_uses_no_power=no_power, event_curves_equivalent=max(r["max_curve_difference_w"] for r in all_rows) <= 1e-8,
        settling_grid_difference_explained=all_grid_explained, full_record_valid=all_coverage,
        raw_inputs_unchanged=raw_unchanged and all_cache_good)
    input_manifest = read(original / "input_manifest.json")
    missing_historical_support = [row["absolute_path"] for row in input_manifest["files"]
                                  if not Path(row["absolute_path"]).is_file()]
    report = dict(schema=BRIDGE_SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        status="equivalent_in_audited_domain" if all(checks.values()) else "bridge_failed",
        model_sha256=file_hash(model_path), model_version=model.version,
        interface_version=AUDITED_INTERFACE_VERSION, model_status_unchanged=model.status,
        confirmation_sha256=file_hash(confirmation_path), freeze_sha256=file_hash(freeze_path),
        production_source_sha256={name: file_hash(ROOT / "src/spindle_power/load_prediction" / name)
                                  for name in ("base_events.py", "base_timeline.py")},
        audit_script_sha256=file_hash(__file__), checks=checks, rounds=round_rows,
        original_confirmation_summary=confirmation["summary"],
        original_confirmation_independence=confirmation["independence"],
        maximum_curve_difference_w=max(r["max_curve_difference_w"] for r in all_rows),
        maximum_settling_difference_s=max(r["settling_difference_s"] or 0. for r in all_rows),
        numerical_explanation="same frozen piecewise-linear template; direct evaluation versus 1ms interpolation differs only at floating-point band boundaries, at most one 1ms sample",
        missing_historical_support=missing_historical_support,
        support_scope="existing NC physical lines and exact S/dwell independently rechecked; missing historical helper files not reconstructed or claimed present",
        original_artifact_hashes=actual_hashes, new_independent_interface_confirmation=False,
        formal_training_release=False,
        remaining_requirements=["真实加工记录中的NC预先指定空载段核验", "机床及采样设置和功率通道兼容证据",
                                "逐记录稳定初态与完整审核时间线", "适用域报告通过后才签发新的哈希绑定资格"],
        environment=dict(python=sys.executable, python_version=sys.version, numpy=np.__version__))
    save_json(output / "bridge.json", report)
    save_json(output / "original_report_comparison.json", old_score_differences)
    with (output / "event_equivalence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    paragraphs = ["# 基础功率原独立确认至生产接口等价性审核", "",
        f"结论：{report['status']}。本报告不是新独立确认，也不是正式标签放行凭据。", "",
        f"五轮共{sum(r['samples'] for r in round_rows)}采样点、{len(all_rows)}启停事件；原模型与阈值未修改。",
        f"逐事件曲线最大差{report['maximum_curve_difference_w']:.12g} W；恢复时刻最大差{report['maximum_settling_difference_s']:.12g} s。",
        "恢复差来自浮点数在固定稳态带边界的比较，最多一个1 ms采样点；没有修改预测曲线或验收阈值。", "",
        "原第4/5轮独立事件确认仍属于原冻结模型。当前结果只证明限定事件域内生产实现与原确认预测的一致性。",
        "机床身份200C-2已由用户确认，但设置/采样兼容及真实加工初态仍需证据，不由机床名称推断。", "",
        "缺失的历史辅助输入："] + [f"- {value}" for value in missing_historical_support] + ["", "仍需完成："] + [f"- {value}" for value in report["remaining_requirements"]]
    (output / "audit.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], checks=checks, output=str(output)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
