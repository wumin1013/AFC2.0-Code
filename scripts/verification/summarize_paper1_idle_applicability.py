"""Summarize fixed pre-cut idle candidates without granting baseline qualification.

The NC interval and final 1000 raw samples are selected upstream without using
power. This audit reproduces that selection, preserves every development record,
and treats its statistics as diagnostics, not independent dynamic validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spindle_power.load_prediction.base_qualification import APPLICABILITY_SCHEMA
from spindle_power.load_prediction.base_timeline import AUDITED_INTERFACE_VERSION
from spindle_power.load_prediction.common import file_hash


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path))


def fixed_tail(trace, commanded_rpm, last_sample):
    """Use the original raw-index interval, never the last 1000 filtered points."""
    return ((trace["source_indices"] >= last_sample - 999)
            & (np.abs(trace["rpm"] - commanded_rpm) <= .01 * commanded_rpm))


def longest_contiguous(indices):
    if not len(indices):
        return 0
    boundaries = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1, len(indices)]
    return int(np.diff(boundaries).max())


def inspect_record(task, meta, source, numeric, model, model_hash, plan_hash):
    header = meta.get("raw_controller_header", [])
    row = dict(record_id=task["record_id"], round_id=task["round_id"], feature=task["feature"],
               spindle_cmd_rpm=task["spindle_cmd_rpm"],
               controller_version=header[7] if len(header) > 7 else "unknown",
               controller_build=header[6] if len(header) > 6 else "unknown",
               controller_firmware=header[8] if len(header) > 8 else "unknown",
               measurement_settings_version=meta.get("measurement_settings_version"),
               machine_settings_version=meta.get("machine_settings_version"),
               status="missing_alignment", diagnostic_only=True, formal_applicability=False)
    if source is None:
        return row
    issues = []
    if source.get("record_id") != task["record_id"]:
        issues.append("alignment_record_id_mismatch")
    if source.get("identity", {}).get("base_model_sha256") != model_hash:
        issues.append("base_model_hash_mismatch")
    if source.get("identity", {}).get("plan_sha256") != plan_hash:
        issues.append("plan_hash_mismatch")
    if not numeric or numeric.get("file_sha256") != source.get("measurement_sha256"):
        issues.append("numeric_audit_measurement_hash_mismatch")
    if file_hash(task["nc"]) != source.get("nc_sha256"):
        issues.append("nc_hash_mismatch")
    row.update(measurement_sha256=source.get("measurement_sha256"), nc_sha256=source.get("nc_sha256"),
               coordinate_semantics=source.get("coordinate_semantics"),
               alignment_candidate_p95_error_mm=source.get("selected_candidate", {}).get("p95_error_mm"),
               aligned_sample_fraction=source.get("aligned_sample_fraction"),
               upstream_status=source.get("status"),
               initial_state_verified=None, timeline_verified=None)
    idle = source.get("idle_candidate", {})
    if not idle.get("trace_path"):
        return {**row, "status": "no_idle_trace", "issues": ";".join(issues)}
    path = Path(idle["trace_path"])
    row.update(trace_path=str(path), trace_sha256=idle.get("trace_sha256"))
    if not path.is_file() or file_hash(path) != idle.get("trace_sha256"):
        return {**row, "status": "trace_integrity_failure", "issues": ";".join(issues + ["trace_hash_mismatch"])}
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    keys = ("source_indices", "time_s", "rpm", "power_w", "raw_line", "xyz")
    if any(key not in trace for key in keys):
        raise ValueError(f"{row['record_id']}: incomplete trace fields")
    indices = trace["source_indices"]
    if any(len(trace[key]) != len(indices) or not np.isfinite(trace[key]).all() for key in keys):
        raise ValueError(f"{row['record_id']}: unequal/nonfinite trace arrays")
    if not len(indices) or np.any(np.diff(indices) <= 0) or np.any(indices != np.floor(indices)):
        raise ValueError(f"{row['record_id']}: invalid source indices")
    if len(indices) != idle.get("samples") or indices[0] != idle.get("first_sample") or indices[-1] != idle.get("last_sample"):
        issues.append("trace_raw_interval_mismatch")
    sample_period = numeric.get("sample_period_s") if numeric else None
    dt = np.diff(trace["time_s"]) / np.diff(indices)
    row.update(idle_candidate_samples=len(indices), raw_sample_period_s=sample_period,
               trace_sample_period_s=float(np.median(dt)) if len(dt) else None,
               sample_period_is_1ms=bool(sample_period is not None and np.isclose(sample_period, .001, atol=1e-9, rtol=0)),
               source_time_step_consistent=bool(len(dt) and sample_period is not None and np.allclose(dt, sample_period, atol=1e-8, rtol=0)))
    rpm = task["spindle_cmd_rpm"]
    base = float(np.interp(rpm, model["speeds_rpm"], model["powers_w"]))
    selected = fixed_tail(trace, rpm, idle["last_sample"])
    raw_tail = indices >= idle["last_sample"] - 999
    row.update(stable_tail_samples=int(selected.sum()), fixed_tail_raw_samples=1000,
               available_raw_tail_samples=int(raw_tail.sum()),
               unfiltered_tail_rpm_min=float(trace["rpm"][raw_tail].min()),
               unfiltered_tail_rpm_median=float(np.median(trace["rpm"][raw_tail])),
               unfiltered_tail_rpm_max=float(trace["rpm"][raw_tail].max()),
               candidate_start_rpm=float(trace["rpm"][0]), candidate_end_rpm=float(trace["rpm"][-1]),
               stable_tail_fraction=float(selected.sum() / 1000),
               longest_contiguous_tail_samples=longest_contiguous(indices[selected]),
               frozen_steady_power_w=base,
               commanded_rpm_in_model_range=bool(model["speeds_rpm"][0] <= rpm <= model["speeds_rpm"][-1]))
    if int(selected.sum()) != idle.get("stable_tail_samples"):
        issues.append("tail_count_mismatch")
    if not selected.any():
        return {**row, "status": "empty_speed_stable_tail", "issues": ";".join(issues)}
    power, speed, time = (trace[key][selected] for key in ("power_w", "rpm", "time_s"))
    stats = dict(mean_rpm=float(speed.mean()), mean_power_w=float(power.mean()),
                 median_power_w=float(np.median(power)), bias_to_frozen_w=float(power.mean() - base),
                 rmse_to_frozen_w=float(np.sqrt(np.mean((power - base) ** 2))))
    for key, value in {**stats, "frozen_steady_power_w": base}.items():
        if idle.get(key) is None or not np.isclose(value, idle[key], rtol=1e-9, atol=1e-6):
            issues.append(f"upstream_{key}_mismatch")
    centered = time - time.mean()
    slope = float(np.dot(centered, power - power.mean()) / np.dot(centered, centered)) if np.dot(centered, centered) else None
    row.update(stats)
    row.update(median_bias_to_frozen_w=float(np.median(power) - base),
               power_std_w=float(power.std()), power_p95_minus_p05_w=float(np.quantile(power, .95) - np.quantile(power, .05)),
               power_linear_slope_w_s=slope, rpm_min=float(speed.min()), rpm_max=float(speed.max()),
               selected_duration_s=float(time[-1] - time[0]),
               status="diagnostic_reproduced" if not issues else "integrity_or_reproduction_failure",
               issues=";".join(issues))
    return row


def group_rows(rows, keys, comparator):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "diagnostic_reproduced":
            groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, members in sorted(groups.items()):
        bias = np.array([r["bias_to_frozen_w"] for r in members])
        rounds = defaultdict(list)
        for r in members:
            rounds[r["round_id"]].append(r["bias_to_frozen_w"])
        result.append(dict(zip(keys, key), records=len(members), rounds=len(rounds),
                           equal_record_mean_bias_w=float(bias.mean()), equal_record_median_bias_w=float(np.median(bias)),
                           min_record_bias_w=float(bias.min()), max_record_bias_w=float(bias.max()),
                           mean_absolute_record_bias_w=float(np.abs(bias).mean()),
                           records_abs_mean_bias_gt_comparator=int(np.sum(np.abs(bias) > comparator)),
                           records_abs_median_bias_gt_comparator=sum(abs(r["median_bias_to_frozen_w"]) > comparator for r in members),
                           equal_round_mean_bias_w=float(np.mean([np.mean(values) for values in rounds.values()])),
                           full_1000_sample_tails=sum(r["stable_tail_samples"] == 1000 for r in members)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("alignment-dir", "manifest", "plan", "numeric-audit-dir", "base-model", "freeze-manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output must be a new directory; preserve earlier diagnostic snapshots.")
    args.output.mkdir(parents=True)
    manifest, plan, model, freeze = (load(p) for p in (args.manifest, args.plan, args.base_model, args.freeze_manifest))
    comparator = freeze["score_config"]["maximum_steady_error_w"]
    tasks = [r for r in plan["records"] if r.get("dataset_role") == "development"]
    metadata = {r["record_id"]: r for r in manifest["records"] if r.get("dataset_role") == "development"}
    if len(tasks) != len({r["record_id"] for r in tasks}) or set(metadata) != {r["record_id"] for r in tasks}:
        raise ValueError("Development plan/manifest must contain the same unique record IDs.")
    model_hash, plan_hash = file_hash(args.base_model), file_hash(args.plan)
    if str(freeze.get("model_sha256", "")).lower() != model_hash:
        raise ValueError("Frozen comparison criteria do not belong to the supplied model.")
    # Freeze the path list once; newly arriving per-record files belong to a later run.
    files = {p.stem: p for p in (args.alignment_dir / "records").glob("*.json")}
    rows, snapshot = [], []
    for task in tasks:
        rid = task["record_id"]
        source_path = files.get(rid)
        source = load(source_path) if source_path else None
        numeric_path = args.numeric_audit_dir / "records" / (rid + ".json")
        numeric = load(numeric_path) if numeric_path.is_file() else None
        snapshot.append(dict(record_id=rid, alignment=artifact(source_path) if source_path else None,
                             numeric_audit=artifact(numeric_path) if numeric else None,
                             alignment_identity=source.get("identity") if source else None))
        row = inspect_record(task, metadata[rid], source, numeric, model, model_hash, plan_hash)
        if "bias_to_frozen_w" in row:
            row["abs_mean_bias_gt_original_steady_comparator"] = abs(row["bias_to_frozen_w"]) > comparator
            row["abs_median_bias_gt_original_steady_comparator"] = abs(row["median_bias_to_frozen_w"]) > comparator
        rows.append(row)
    grouped = group_rows(rows, ("spindle_cmd_rpm", "controller_version"), comparator)
    by_rpm = group_rows(rows, ("spindle_cmd_rpm",), comparator)
    valid = [r for r in rows if r["status"] == "diagnostic_reproduced"]
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    checks = [
        "确认候选区间内刀具尚未接触工件，并核对NC/实测行号映射；目前最早G1/G2/G3仅是几何边界。",
        "对照OM03S标定与加工采样的G432含义、量纲、零点/缩放和采样设置记录；控制器版本分组仅作线索，不能推定因果。",
        "核实机床运行设置及测量配置适用域，解释各转速下可重复的稳态偏差；不得以整段减均值或修改状态替代核验。",
        "逐记录核对已观察的稳定初态、实际启停时间线及程序/转速哈希，处理启动前已旋转、未完整覆盖或截断情形。",
        "现有资料能闭环时优先补齐上述证据；仅对现有记录不能解决的项补充最少独立验证，不默认重采五轮。",
    ]
    report = dict(schema=APPLICABILITY_SCHEMA, status="blocked", diagnostic_kind="nc_precut_fixed_tail_comparison",
                  generated_at=datetime.now(timezone.utc).isoformat(), model_sha256=model_hash,
                  model_version=model.get("version"), model_status_unchanged=model.get("status"),
                  interface_version=AUDITED_INTERFACE_VERSION, expected_records=len(tasks),
                  available_alignment_records=sum(r["alignment"] is not None for r in snapshot),
                  snapshot_complete=all(r["alignment"] is not None for r in snapshot), status_counts=counts,
                  diagnostic_records=len(valid), formal_training_release=False, eligible_for_load_training=False,
                  power_channel_identity_verified=None, machine_settings_compatible=None,
                  measurement_settings_compatible=None, baseline_on_machining_idle_verified=None,
                  nominal_selection="NC startup before first G1/G2/G3; raw indices >= last_candidate_index - 999; speed within 1%; power not used",
                  original_steady_error_comparator_w=comparator,
                  comparator_interpretation="Diagnostic comparison only; not a new acceptance rule or full dynamic validation.",
                  records_abs_mean_bias_gt_comparator=sum(r["abs_mean_bias_gt_original_steady_comparator"] for r in valid),
                  records_abs_median_bias_gt_comparator=sum(r["abs_median_bias_gt_original_steady_comparator"] for r in valid),
                  full_1000_sample_tails=sum(r["stable_tail_samples"] == 1000 for r in valid),
                  source_time_step_consistent_records=sum(r["source_time_step_consistent"] for r in valid),
                  diagnostic_boundary="All candidates retained; geometric alignment quality and unexpected RPM must be reviewed before interpreting a candidate as actual no-contact idle.",
                  group_weighting="equal record; equal round means supplied separately; no sample pooling or causal controller inference",
                  group_by_rpm=by_rpm, group_by_rpm_controller=grouped, remaining_checks=checks,
                  record_audits=[dict(record_id=r["record_id"], measurement_sha256=r.get("measurement_sha256"),
                                      initial_state_verified=None, timeline_verified=None, diagnostic_status=r["status"])
                                 for r in rows],
                  inputs={name: artifact(path) for name, path in dict(script=Path(__file__), manifest=args.manifest,
                      plan=args.plan, base_model=args.base_model, freeze_manifest=args.freeze_manifest).items()},
                  runtime=dict(python_executable=sys.executable, python_version=platform.python_version(),
                               numpy_version=np.__version__, platform=platform.platform()))
    save(args.output / "input_snapshot.json", snapshot)
    write_csv(args.output / "per_record.csv", rows)
    write_csv(args.output / "review_candidates.csv", [r for r in rows if r["status"] != "diagnostic_reproduced"
              or r.get("stable_tail_samples") != 1000 or r.get("abs_mean_bias_gt_original_steady_comparator")])
    write_csv(args.output / "grouped_rpm_controller.csv", grouped)
    write_csv(args.output / "grouped_rpm.csv", by_rpm)
    save(args.output / "applicability.json", report)
    lines = ["# 加工记录启动前空载适用性诊断", "", "状态：**blocked**。本报告没有放行基础功率标签或负载正式训练。", "",
             f"固定清单 {len(tasks)} 条；本次快照包含 {report['available_alignment_records']} 条对齐记录，{len(valid)} 条候选片段复算一致。",
             f"原独立事件确认的稳态误差限 {comparator:g} W 仅作为诊断比较：候选均值绝对偏差超过该值 {report['records_abs_mean_bias_gt_comparator']} 条，候选中位数超过该值 {report['records_abs_median_bias_gt_comparator']} 条。",
             f"完整含1000个速度范围内样本的尾段为 {report['full_1000_sample_tails']} 条；该事实不能证明初态稳定或无接触。", "",
             "|指令转速 rpm|控制器版本|记录/轮次|等记录均值偏差 W|中位偏差 W|均值偏差超比较值|", "|---:|---|---:|---:|---:|---:|"]
    for g in grouped:
        lines.append(f"|{g['spindle_cmd_rpm']:g}|{g['controller_version']}|{g['records']}/{g['rounds']}|{g['equal_record_mean_bias_w']:.3f}|{g['equal_record_median_bias_w']:.3f}|{g['records_abs_mean_bias_gt_comparator']}|")
    lines.extend(["", "偏差符号为实测功率减冻结模型稳态功率。各记录等权；控制器分组与采集批次、工艺等混杂，不作因果解释。", "",
                  "候选时间段由NC和实测转速固定选定，不使用功率筛选；最早切削运动不等同于已核实接触时刻。此结果仅检查稳态候选片段，不能替代动态峰值、恢复时间及逐记录初态/时间线验证。", "", "最少后续证据核查：", ""])
    lines.extend(f"{i}. {item}" for i, item in enumerate(checks, 1))
    lines.extend(["", "O1203未读取、未统计；原始数据和冻结模型保持只读。所有逐记录状态、缺失项、输入哈希保存在同目录JSON/CSV中。", ""])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "expected_records", "available_alignment_records", "diagnostic_records", "status_counts", "records_abs_mean_bias_gt_comparator", "records_abs_median_bias_gt_comparator")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
