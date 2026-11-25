"""Device-aware BiLSTM runtime; saved models load on CPU by default."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import checkpoint
from .feed_model import FeedModelConfig, sequence_windows
from .feed_samples import FEED_FEATURE_COLUMNS, FEED_SAMPLE_SCHEMA, FeedSample
from .sample_rules import SAMPLE_RULES_VERSION

CONTEXT_FIELDS = ("machine_version", "machine_settings_version", "feed_override", "machining_mode", "spacing_mm", "coordinate_mode")


def machine_context(metadata):
    result = {}
    for key in CONTEXT_FIELDS:
        value = metadata.get(key, "machine") if key == "coordinate_mode" else metadata.get(key)
        if value is None or value == "":
            raise ValueError(f"实际进给输入缺少 {key}")
        if key in ("feed_override", "spacing_mm"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} 必须为已知正有限数")
            value = float(value)
        elif not isinstance(value, str):
            raise ValueError(f"{key} 必须为非空版本/模式字符串")
        if key == "coordinate_mode" and value not in ("machine", "workpiece_cutting"):
            raise ValueError("实际进给坐标模式必须为machine或workpiece_cutting")
        result[key] = value
    return result


def feature_values(sample):
    if sample.metadata.get("schema") != FEED_SAMPLE_SCHEMA:
        raise ValueError("实际进给输入 schema 不匹配")
    if tuple(sample.metadata.get("feature_columns", ())) != tuple(FEED_FEATURE_COLUMNS):
        raise ValueError("实际进给特征顺序或特征 schema 不匹配")
    frame = sample.frame
    if frame.columns.duplicated().any() or not set(FEED_FEATURE_COLUMNS) <= set(frame):
        raise ValueError("实际进给输入存在缺失或重复特征列")
    if frame.empty or "sequence_id" not in frame or frame.sequence_id.isna().any():
        raise ValueError("实际进给输入必须包含非空、有效NC序列")
    values = frame.loc[:, list(FEED_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("实际进给NC特征包含非有限值")
    machine_context(sample.metadata)
    return values


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.lstm = nn.LSTM(input_size, config.hidden_size, num_layers=config.num_layers,
                            dropout=config.dropout if config.num_layers > 1 else 0.,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(config.hidden_size * 2, 1), nn.Softplus())

    def forward(self, features, lengths):
        if features.ndim != 3 or lengths.ndim != 1 or len(lengths) != len(features):
            raise ValueError("BiLSTM 需要批量可变长序列")
        if (lengths < 1).any() or (lengths > features.shape[1]).any():
            raise ValueError("BiLSTM 序列长度越界")
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        prediction = self.head(encoded).squeeze(-1)
        valid = torch.arange(features.shape[1], device=features.device)[None] < lengths.to(features.device)[:, None]
        return prediction.masked_fill(~valid, 0.)


def padded_batch(arrays):
    lengths = torch.tensor([len(array) for array in arrays], dtype=torch.long)
    output = torch.zeros((len(arrays), int(lengths.max()), len(FEED_FEATURE_COLUMNS)), dtype=torch.float32)
    for index, array in enumerate(arrays):
        output[index, :len(array)] = torch.as_tensor(array, dtype=torch.float32)
    return output, lengths


class FeedModel:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.network = BiLSTMRegressor(len(FEED_FEATURE_COLUMNS), config).cpu()
        self.trained = False
        self.normalization = None
        self.compatible_contexts = []
        self.model_version = None
        self.metadata = {}

    def _validate_normalization(self):
        norm = self.normalization
        if not isinstance(norm, dict) or set(norm) != {"mean", "scale", "target_scale_mm_min"}:
            raise ValueError("实际进给模型归一化字段不完整")
        for key in ("mean", "scale"):
            values = np.asarray(norm[key], dtype=float)
            if values.shape != (len(FEED_FEATURE_COLUMNS),) or not np.isfinite(values).all():
                raise ValueError("实际进给归一化维度或数值无效")
            if key == "scale" and np.any(values <= 0):
                raise ValueError("实际进给归一化尺度必须为正")
        target = norm["target_scale_mm_min"]
        if not isinstance(target, (int, float)) or not np.isfinite(target) or target <= 0:
            raise ValueError("实际进给输出尺度必须为正")

    def current_version(self):
        definition = dict(architecture=asdict(self.config), feature_schema=FEED_SAMPLE_SCHEMA,
                          sample_rules_version=SAMPLE_RULES_VERSION,
                          feature_columns=list(FEED_FEATURE_COLUMNS), normalization=self.normalization,
                          compatible_contexts=self.compatible_contexts)
        digest = hashlib.sha256(json.dumps(definition, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False, separators=(",", ":")).encode("utf-8"))
        for key, tensor in sorted(self.network.state_dict().items()):
            if not torch.isfinite(tensor).all():
                raise ValueError("实际进给模型权重包含非有限值")
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def assert_ready(self):
        if self.trained is not True:
            raise ValueError("实际进给模型尚未训练，禁止使用随机权重预测")
        self.config.validate()
        self._validate_normalization()
        if not self.compatible_contexts or self.metadata.get("training_status") != "trained":
            raise ValueError("实际进给模型缺少训练来源或兼容域")
        for context in self.compatible_contexts:
            machine_context(context)
        if not self.model_version or self.model_version != self.current_version():
            raise ValueError("实际进给权重、特征或归一化版本不一致")

    def save(self, path):
        if self.trained:
            self.assert_ready()
        elif self.normalization is not None or self.model_version is not None or self.compatible_contexts:
            raise ValueError("未训练模型不能包含已训练归一化或版本标志")
        payload = dict(schema=self.config.schema, architecture=asdict(self.config),
                       feature_schema=FEED_SAMPLE_SCHEMA, feature_columns=list(FEED_FEATURE_COLUMNS),
                       weights={key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()},
                       trained=self.trained,
                       normalization=self.normalization, compatible_contexts=self.compatible_contexts,
                       model_version=self.model_version, metadata=self.metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {"schema", "architecture", "feature_schema", "feature_columns", "weights", "trained",
                    "normalization", "compatible_contexts", "model_version", "metadata"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("实际进给模型包字段不完整或不受支持")
        if payload["schema"] != "afc.feed_model.2" or payload["feature_schema"] != FEED_SAMPLE_SCHEMA:
            raise ValueError("实际进给模型包 schema 不匹配；旧窗口规则模型必须重新训练")
        if tuple(payload["feature_columns"]) != tuple(FEED_FEATURE_COLUMNS):
            raise ValueError("实际进给模型特征顺序不匹配")
        if type(payload["trained"]) is not bool:
            raise ValueError("实际进给模型训练标志无效")
        model = cls(FeedModelConfig(**payload["architecture"]))
        model.network.load_state_dict(payload["weights"], strict=True)
        for key in ("trained", "normalization", "compatible_contexts", "model_version", "metadata"):
            setattr(model, key, payload[key])
        if model.trained:
            model.assert_ready()
            model.network.eval()
            for parameter in model.network.parameters():
                parameter.requires_grad_(False)
        elif model.normalization is not None or model.model_version is not None or model.compatible_contexts:
            raise ValueError("实际进给模型训练标志与内容不一致")
        else:
            model.current_version()  # Also reject non-finite untrained weights.
        return model


def predict_values(model, sample, *, cancel=None):
    """Internal evaluation shares inference windows; caller handles ready state."""
    values = feature_values(sample)
    if machine_context(sample.metadata) not in [machine_context(context) for context in model.compatible_contexts]:
        raise ValueError("输入机床、设置、倍率、加工模式、坐标模式或空间采样间距不在该进给模型兼容域内")
    norm = model.normalization
    values = (values - np.asarray(norm["mean"])) / np.asarray(norm["scale"])
    result = np.full(len(values), np.nan, dtype=float)
    model.network.eval()
    with torch.no_grad():
        for window in sequence_windows(sample.frame, model.config):
            checkpoint(cancel)
            x, lengths = padded_batch([values[window.start:window.stop]])
            x = x.to(next(model.network.parameters()).device)
            prediction = model.network(x, lengths)[0].cpu().numpy() * norm["target_scale_mm_min"]
            a, b = window.core_start - window.start, window.core_stop - window.start
            result[window.core_start:window.core_stop] = prediction[a:b]
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("实际进给模型产生无效预测")
    return result


def predict_actual_feed(model, features):
    model.assert_ready()
    output = features.frame.copy()
    output["feed_predicted_mm_min"] = predict_values(model, features)
    metadata = {**features.metadata, "source": "bilstm_actual_feed", "trained": True,
                "model_version": model.model_version, "feed_units": "mm/min",
                "prediction_kind": "actual_feed_before_machining", "time_mapping": "not_generated"}
    return FeedSample(output, metadata)
