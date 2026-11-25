"""Verify completed calibration inputs and reject renamed measured copies."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .common import file_hash, json_value
from .samples import SampleSet


def measured_frame_digest(frame):
    columns = [c for c in ("line", "execution_instance", "segment_mm", "path_mm", "power_cut_w",
        "feed_actual_mm_min", "spindle_measured_rpm", "ap_mm", "ae_mm", "spindle_rpm", "diameter_mm", "teeth") if c in frame]
    if "power_cut_w" not in columns:
        raise ValueError("校准资料缺少实测切削功率")
    # Canonical numeric values ignore filenames, CSV rendering and record names.
    values = frame[columns].astype(float).round(10).replace([np.inf, -np.inf], np.nan)
    payload = {"columns": columns, "values": values.to_dict("list")}
    return hashlib.sha256(json.dumps(json_value(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_csv(path, expected):
    if not expected or not Path(path).is_file() or file_hash(path) != expected:
        raise ValueError("校准CSV文件与登记SHA-256不一致")
    return pd.read_csv(path)


def load_calibration_sample(record, *, synthetic=False):
    frame = checked_csv(record["prepared_csv"], record.get("prepared_csv_sha256"))
    metadata = dict(record["metadata"])
    digest = measured_frame_digest(frame)
    if metadata.get("prepared_measurement_sha256") not in (None, digest):
        raise ValueError("校准实测数值内容指纹不一致")
    verify_raw_source(metadata, synthetic=synthetic)
    metadata["prepared_measurement_sha256"] = digest
    return SampleSet(frame, metadata)


def verify_raw_source(metadata, *, synthetic=False):
    path = metadata.get("resolved_files", {}).get("measurement")
    if path:
        from .dataset_manifest import measurement_content_hash
        if file_hash(path) != metadata.get("file_sha256", {}).get("measurement"):
            raise ValueError("校准原始实测文件指纹不一致")
        if measurement_content_hash(path) != metadata.get("measurement_content_sha256"):
            raise ValueError("校准原始实测内容指纹不一致")
    elif not synthetic:
        raise ValueError("正式校准缺少可核验的原始实测来源")


def identity_tokens(metadata):
    result = set()
    for key in ("record_id", "measurement_content_sha256", "prepared_measurement_sha256"):
        if metadata.get(key):
            result.add((key, metadata[key]))
    for key in ("measurement_hash", "measurement_content_hash"):
        if metadata.get(key):
            result.add(("measurement_content_sha256", metadata[key]))
    return result


def assert_independent(groups, training_metadata=()):
    occupied = set().union(*(identity_tokens(m) for m in training_metadata))
    for name, records in groups:
        for metadata in records:
            tokens = identity_tokens(metadata)
            if not tokens or not metadata.get("record_id"):
                raise ValueError("校准记录缺少可核验身份")
            if occupied & tokens:
                raise ValueError(f"{name}与训练、拟合、验证资料复用了实测内容或记录")
            occupied.update(tokens)


def checked_interval_record(record, prediction_version, *, synthetic=False):
    if record.get("prediction_version") != prediction_version:
        raise ValueError("区间资料prediction_version与当前条件校准/记忆策略不一致")
    frame = checked_csv(record["aligned_measurement"], record.get("aligned_measurement_sha256"))
    target = np.asarray(record["target_w"], dtype=float)
    if "power_cut_w" not in frame or target.shape != (len(frame),) or not np.allclose(target, frame.power_cut_w, rtol=0, atol=1e-9):
        raise ValueError("区间目标与已核验实测CSV不一致")
    metadata = {**record.get("metadata", {}), "record_id": record.get("group_id"),
                "prepared_measurement_sha256": measured_frame_digest(frame)}
    verify_raw_source(metadata, synthetic=synthetic)
    source = record.get("prediction_directory")
    if source:
        source = Path(source)
        saved = json.loads((source / "prediction.json").read_text(encoding="utf-8"))
        predicted = checked_csv(source / "prediction.csv", saved.get("frame_sha256"))
        if saved.get("prediction_version") != prediction_version:
            raise ValueError("已保存预测版本与区间资料不一致")
        for field, column in (("lower_w", "q05_w"), ("upper_w", "q95_w"), ("median_w", "q50_w")):
            if field not in record or len(predicted) != len(target) or not np.allclose(record[field], predicted[column], rtol=0, atol=1e-8):
                raise ValueError("区间资料与已保存原始模型分位数不一致")
    elif not synthetic:
        raise ValueError("正式区间校准必须绑定已保存的加工前预测")
    return metadata
