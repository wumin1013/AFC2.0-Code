"""Read-only, resumable full-content audit of the registered R01--R90 data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.measurement import detect_encoding, iter_measurement, read_header, select_position_channels


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def scan(task):
    path = Path(task["measurement"])
    before = path.stat()
    result = {**task, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
              "file_sha256": file_hash(path), "formal_training_eligible": False}
    header = read_header(path)
    source, columns = select_position_channels(header["mapping"])
    canonical = sorted(header["mapping"])
    digests = {k: hashlib.sha256() for k in canonical}
    stats = {k: dict(finite=0, nonfinite=0, min=None, max=None, sum=0.) for k in canonical + ["feed_norm_mm_min"]}
    count, invalid, conflicts, issues = 0, 0, Counter(), []
    line_runs, last_line, start = [], None, 0
    noninteger, backwards, invalid_feed = 0, [], 0
    duplicate_blocks, pending = [], np.empty((0, len(canonical)))
    first, last, stationary = None, None, 0
    speed_counts = Counter()
    for chunk in iter_measurement(path, chunk_size=65536):
        n = len(chunk.valid)
        invalid += int((~chunk.valid).sum())
        conflicts.update(chunk.metadata["duplicate_conflicts"])
        issues.extend(chunk.metadata["issues"][:max(0, 30-len(issues))])
        values = dict(chunk.values)
        values["feed_norm_mm_min"] = np.sqrt(sum(values[f"actual_v{a}_mm_min"]**2 for a in "xyz"))
        invalid_feed += int((~np.isfinite(values["feed_norm_mm_min"])).sum())
        for key in stats:
            v = values[key]
            finite = np.isfinite(v)
            state = stats[key]
            state["finite"] += int(finite.sum())
            state["nonfinite"] += int((~finite).sum())
            if finite.any():
                good = v[finite]
                state["sum"] += float(good.sum())
                state["min"] = min(float(good.min()), state["min"]) if state["min"] is not None else float(good.min())
                state["max"] = max(float(good.max()), state["max"]) if state["max"] is not None else float(good.max())
            if key in digests:
                normalized = np.asarray(v, dtype="<f8").copy()
                normalized[normalized == 0] = 0.0
                normalized[np.isnan(normalized)] = np.nan
                digests[key].update(normalized.tobytes())
        matrix = np.column_stack([values[k] for k in canonical]).astype("<f8")
        pending = np.concatenate((pending, matrix))
        while len(pending) >= 4096:
            block, pending = pending[:4096], pending[4096:]
            # Constant idle values are not evidence of a copied machining trace.
            if np.isfinite(block).all() and np.count_nonzero(np.ptp(block, axis=0) > 1e-8) >= 3:
                duplicate_blocks.append(hashlib.sha256(block.tobytes()).hexdigest())
        line = values["program_line_raw"]
        finite_line = np.isfinite(line)
        noninteger += int((finite_line & (np.abs(line - np.round(line)) > 1e-8)).sum())
        boundaries = np.flatnonzero(np.r_[True, line[1:] != line[:-1], True])
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            current = float(line[a]) if np.isfinite(line[a]) else None
            if count + a == 0:
                last_line, start = current, 0
            elif current != last_line:
                line_runs.append(dict(line=last_line, first_sample=start, last_sample=int(count+a-1)))
                if current is not None and last_line is not None and current < last_line:
                    backwards.append(dict(sample=int(count+a), previous=last_line, current=current))
                start, last_line = int(count+a), current
        speed = values["spindle_speed_rpm"]
        expected = float(task["spindle_cmd_rpm"])
        speed_counts["within_1pct_command"] += int((np.abs(speed-expected) <= .01*expected).sum())
        speed_counts["stopped_abs_le_20"] += int((np.abs(speed) <= 20).sum())
        speed_counts["running_outside_1pct"] += int(((speed > 500) & (np.abs(speed-expected) > .01*expected)).sum())
        stationary += int(((values["feed_norm_mm_min"] <= 1) & (speed > 500)).sum())
        endpoint_keys = ["program_line_raw", "spindle_speed_rpm", "spindle_power_w", "feed_norm_mm_min"] + list(columns)
        if first is None:
            first = {k: float(values[k][0]) if np.isfinite(values[k][0]) else None for k in endpoint_keys}
        last = {k: float(values[k][-1]) if np.isfinite(values[k][-1]) else None for k in endpoint_keys}
        count += n
    if count:
        line_runs.append(dict(line=last_line, first_sample=start, last_sample=count-1))
    for state in stats.values():
        state["mean"] = state.pop("sum") / state["finite"] if state["finite"] else None
    payload = dict(samples=count, channels={k: v.hexdigest() for k, v in digests.items()})
    content_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    nc_path = Path(task["nc"])
    nc_lines = nc_path.read_text(encoding=detect_encoding(nc_path)).splitlines()
    m30 = [i+1 for i, line in enumerate(nc_lines) if re.search(r"\bM0?30\b", line)]
    after = path.stat()
    result.update(samples=count, sample_period_s=header["sample_period_s"], duration_s=count*header["sample_period_s"],
        scan_status="completed", invalid_samples=invalid, invalid_feed_samples=invalid_feed,
        duplicate_conflicts=dict(conflicts), duplicate_channels=header["duplicate_channels"], reader_issues=issues,
        channels=canonical, position_source=source, feed_label_source="norm_actual_xyz_velocity_mm_min",
        content_sha256=content_hash, content_hash_schema="canonical_channel_float64_sha256_v1",
        dynamic_block_sha256=duplicate_blocks, statistics=stats, first_sample=first, last_sample=last,
        line_runs=line_runs, line_noninteger_samples=noninteger, line_backwards=backwards,
        speed_counts=dict(speed_counts), stationary_running_samples=stationary,
        nc_sha256=file_hash(nc_path), nc_physical_lines=len(nc_lines), m30_physical_lines=m30,
        observed_end_marker_candidates=[{"physical_line": x, "raw_equal": any(r["line"]==x for r in line_runs),
                                        "raw_minus_one": any(r["line"]==x-1 for r in line_runs)} for x in m30],
        source_unchanged=before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns and result["file_sha256"]==file_hash(path),
        sampling_continuity="1ms reconstructed from SmplInfo1000; independent hardware timestamps unavailable",
        quality={"tool_healthy": True, "tool_health_source":"author_confirmation", "no_alarm":None,
                 "complete":None, "alignment_reliable":None, "actual_acquisition_time":None, "actual_tool_session":None},
        analysis_assumptions={"order":"round_number", "tool_group":"same_specification_pooled", "source":"explicit_author_instruction"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping_root = args.data_root / "负载预测实验G代码及工件模型"
    mapping = mapping_root / "90轮G代码映射.csv"
    categories = {"O":"外形", "C":"曲面", "M":"混合", "S":"斜面", "T":"台阶"}
    tasks = []
    with mapping.open(encoding=detect_encoding(mapping), newline="") as stream:
        for row in csv.DictReader(stream):
            rid, feature = int(row["轮次"].removeprefix("R")), categories[row["工况代码"][0]]
            paths = list((args.data_root / "实际加工采样数据" / f"第{rid}轮加工").glob(feature+"*.csv"))
            if len(paths) != 1:
                raise ValueError(f"R{rid} {feature}: expected one CSV, found {len(paths)}")
            tasks.append(dict(record_id=f"R{rid:02d}_{feature}", round_id=rid, feature=feature,
                dataset_role="development", measurement=str(paths[0]), nc=str(mapping_root/row["NC文件夹"]/row["NC文件"]),
                combination=row["组合"], parameter_group=row["参数组"], feed_program_mm_min=float(row["进给速度"]),
                spindle_cmd_rpm=float(row["主轴转速"])))
    if len(tasks)!=450 or len({t["record_id"] for t in tasks})!=450:
        raise ValueError("Expected exactly 450 unique registered records")
    reader_hash = file_hash(ROOT/"src/spindle_power/load_prediction/measurement.py")
    identity = dict(script_sha256=file_hash(__file__), reader_sha256=reader_hash, mapping_sha256=file_hash(mapping))
    save(args.output/"audit_plan.json", dict(**identity, python=sys.executable, workers=args.workers, chunk_size=65536,
        started_utc=datetime.now(timezone.utc).isoformat(), records=tasks))
    records, pending = [], []
    for task in tasks:
        cached = args.output/"records"/(task["record_id"]+".json")
        if cached.exists():
            prior=json.loads(cached.read_text(encoding="utf-8"))
            if prior.get("audit_identity")==identity and prior.get("file_sha256")==file_hash(task["measurement"]):
                records.append(prior)
                continue
        pending.append(task)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan, task): task for task in pending}
        for future in as_completed(futures):
            task=futures[future]
            try:
                result=future.result()
            except Exception as exc:
                result={**task,"scan_status":"failed","error":f"{type(exc).__name__}: {exc}"}
            result["audit_identity"]=identity
            save(args.output/"records"/(task["record_id"]+".json"),result)
            records.append(result)
            print(json.dumps(dict(completed=len(records),record_id=task["record_id"],status=result["scan_status"],samples=result.get("samples")),ensure_ascii=False),flush=True)
    records.sort(key=lambda r:(r["round_id"],r["feature"]))
    contents, blocks = defaultdict(list), defaultdict(set)
    for r in records:
        if r.get("content_sha256"):
            contents[r["content_sha256"]].append(r["record_id"])
        for block in r.get("dynamic_block_sha256",[]):
            blocks[block].add(r["record_id"])
    summary=dict(records=len(records),completed=sum(r["scan_status"]=="completed" for r in records),
        samples=sum(r.get("samples",0) for r in records), invalid_samples=sum(r.get("invalid_samples",0) for r in records),
        invalid_feed_samples=sum(r.get("invalid_feed_samples",0) for r in records),
        duplicate_channel_conflicts=sum(sum(r.get("duplicate_conflicts",{}).values()) for r in records),
        position_sources=dict(Counter(r.get("position_source") for r in records)),
        all_sources_unchanged=all(r.get("source_unchanged") is True for r in records),
        duplicate_records=[v for v in contents.values() if len(v)>1],
        matching_dynamic_blocks=[dict(sha256=k,records=sorted(v)) for k,v in blocks.items() if len(v)>1],
        formal_training_release=False)
    save(args.output/"audit_summary.json",dict(**identity,completed_utc=datetime.now(timezone.utc).isoformat(),summary=summary))
    save(args.output/"fixed_raw_inventory.json",dict(schema="afc.raw_audit_inventory.1",audit_identity=identity,records=records))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
