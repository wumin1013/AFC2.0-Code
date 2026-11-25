"""Frozen-network four-parameter calibration and optional grouped intervals.

Calibration records are completed, reviewed measurements explicitly assigned
to ``condition_calibration``. They are never network training or test records.
The two positive coefficient multipliers are fitted on steady labels first;
path scale/bias are then fitted with that main trend held fixed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math

import numpy as np

from .common import save_json, version_of
from .condition import descriptors


@dataclass
class CalibrationResult:
    model_version: str
    ks_scale: float = 1.
    ke_scale: float = 1.
    residual_scale: float = 1.
    residual_bias_w: float = 0.
    target_condition: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    schema: str = "afc.low_dim_calibration.1"

    @property
    def version(self):
        return version_of(asdict(self))

    def validate_for(self, bundle, metadata=None):
        if self.schema != "afc.low_dim_calibration.1" or self.model_version != bundle.current_version():
            raise ValueError("低维校准与冻结模型版本不兼容")
        values = (self.ks_scale, self.ke_scale, self.residual_scale, self.residual_bias_w)
        if not all(math.isfinite(x) for x in values) or min(values[:3]) <= 0:
            raise ValueError("低维校准倍率必须为正有限值，偏置必须有限")
        if metadata is None and any(self.target_condition.values()):
            raise ValueError("应用低维校准必须提供目标物理工况以核对适用范围")
        if metadata is not None and self.target_condition and descriptors(metadata) != self.target_condition:
            raise ValueError("低维校准不适用于当前物理工况")

    def save(self, path):
        save_json(path, {**asdict(self), "version": self.version})

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        expected = payload.pop("version", None)
        result = cls(**payload)
        if result.schema != "afc.low_dim_calibration.1" or expected != result.version:
            raise ValueError("低维校准格式或校验值不匹配")
        return result


def _identifiability(matrices, limit):
    matrix = np.concatenate([values / np.sqrt(len(values) * len(matrices)) for values in matrices])
    scales = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(matrix).all() or np.any(scales <= 1e-12):
        raise ValueError("校准设计矩阵缺少可辨识激励")
    normalized = matrix / scales
    rank = int(np.linalg.matrix_rank(normalized))
    condition = float(np.linalg.cond(normalized))
    if rank < matrix.shape[1] or not np.isfinite(condition) or condition > limit:
        raise ValueError(f"校准矩阵不可辨识：rank={rank}, condition={condition:.6g}")
    return {"rank": rank, "condition_number": condition, "column_scales": scales.tolist()}


def _fit(matrices, targets, *, kind, delta, regularization, max_nfev):
    from scipy.optimize import minimize

    def objective(parameters):
        if kind == "main":
            coefficients = np.exp(parameters)
        else:
            coefficients = np.array([np.exp(parameters[0]), parameters[1] * delta])
        total = 0.
        for matrix, target in zip(matrices, targets):
            error = (matrix @ coefficients - target) / delta
            absolute = np.abs(error)
            total += float(np.mean(np.where(absolute <= 1., .5 * error ** 2, absolute - .5))) / len(matrices)
        return total + regularization * float(np.dot(parameters, parameters))

    bounds = [(-4., 4.), (-4., 4.) if kind == "main" else (-100., 100.)]
    result = minimize(objective, np.zeros(2), method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_nfev, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"低维校准优化未收敛：{result.message}")
    values = np.exp(result.x) if kind == "main" else np.array([np.exp(result.x[0]), result.x[1] * delta])
    return values, {"objective": float(result.fun), "iterations": int(result.nit), "log_parameters": result.x.tolist()}


def calibrate_model(bundle, records, *, max_condition_number=3000., regularization=.01,
                    huber_delta_w=20., min_records=2, max_nfev=200):
    """Return a separate calibration object; never mutate a model or registry.

records: SampleSet objects, or dictionaries with ``frame`` and ``metadata``.
Frame columns are the normal load-sample contract including power_cut_w,
valid_main and valid_residual. Source hashes/IDs must be unique records.
"""
    import torch
    from .neural import residual_windows, state_version
    from .samples import PHYSICS_COLUMNS, assert_feed_inputs, path_features
    bundle.assert_ready()
    if max_condition_number <= 1 or regularization < 0 or huber_delta_w <= 0 or min_records < 2 or max_nfev < 1:
        raise ValueError("低维校准参数无效")
    records = list(records)
    if len(records) < min_records:
        raise ValueError(f"低维校准至少需要{min_records}份独立记录")
    model_version = bundle.current_version()
    weights_before = (state_version(bundle.main), state_version(bundle.residual))
    seen, prepared, condition_reference = set(), [], None
    main_matrices, main_targets = [], []
    training_sources = set(bundle.metadata.get("training_record_hashes", []))
    for item in bundle.metadata.get("samples", []):
        training_sources.update(str(item[key]) for key in ("measurement_hash", "measurement_content_hash", "measurement_content_sha256", "measurement_source", "record_id") if item.get(key))
    for record in records:
        frame = getattr(record, "frame", None)
        metadata = getattr(record, "metadata", None)
        if frame is None:
            frame, metadata = record["frame"], record["metadata"]
        if metadata.get("dataset_role") not in ("target_adaptation", "condition_calibration", "calibration"):
            raise ValueError("低维校准需独立condition_calibration资料，不能使用开发/记忆验证/测试集")
        if any(metadata.get(key) is not True for key in ("tool_healthy", "no_alarm", "complete", "alignment_reliable")):
            raise ValueError("低维校准记录质量尚未显式通过")
        source = metadata.get("measurement_content_sha256", metadata.get("measurement_content_hash", metadata.get("measurement_hash")))
        if not source and metadata.get("provenance") == "synthetic_test_only":
            source = metadata.get("record_id")
        if not source or source in seen or source in training_sources:
            raise ValueError("低维校准需独立实测内容哈希，且不能重复或复用训练来源")
        seen.add(source)
        description = descriptors(metadata)
        if condition_reference is not None and description != condition_reference:
            raise ValueError("一次低维校准只能针对一组固定物理工况")
        condition_reference = description
        required = set(PHYSICS_COLUMNS + ["power_cut_w", "valid_main", "valid_residual"])
        if required - set(frame):
            raise ValueError("低维校准缺少物理量、标签或审核掩码")
        main_mask = frame.valid_main.to_numpy(dtype=bool)
        residual_mask = frame.valid_residual.to_numpy(dtype=bool)
        if not main_mask.any() or not residual_mask.any() or np.any(main_mask & ~residual_mask):
            raise ValueError("每个校准记录需合格稳态及切削样本")
        assert_feed_inputs(frame.loc[residual_mask], "measured")
        subset = frame.loc[residual_mask].copy().reset_index(drop=True)
        if not np.isfinite(subset[PHYSICS_COLUMNS + ["power_cut_w"]].to_numpy(float)).all():
            raise ValueError("低维校准有效样本含非有限值")
        selected = np.flatnonzero(residual_mask)
        identity = frame["sequence_id" if "sequence_id" in frame else "execution_instance"].to_numpy()[selected]
        subset["sequence_id"] = np.cumsum(np.r_[True, (np.diff(selected) != 1) | (identity[1:] != identity[:-1])])
        condition = bundle.condition_values(metadata, len(subset), allow_unknown=True)
        condition = torch.as_tensor(condition) if condition is not None else None
        with torch.no_grad():
            components = bundle.main.components(torch.tensor(subset[PHYSICS_COLUMNS].to_numpy(), dtype=torch.float32), condition)
        matrix = np.column_stack([components[key].numpy() for key in ("shear_w", "edge_w")])
        steady = main_mask[residual_mask]
        target = subset.power_cut_w.to_numpy(float)
        main_matrices.append(matrix[steady])
        main_targets.append(target[steady])
        prepared.append((subset, matrix, target, condition))
    main_rank = _identifiability(main_matrices, max_condition_number)
    main, main_fit = _fit(main_matrices, main_targets, kind="main", delta=huber_delta_w,
                          regularization=regularization, max_nfev=max_nfev)
    residual_matrices, residual_targets = [], []
    for frame, matrix, target, condition in prepared:
        main_values = matrix @ main
        norm = bundle.normalization
        features = path_features(frame, main_values, norm["main_mean_w"], norm["main_scale_w"], bundle.config.feed_reference_mm_min)
        residual = np.empty(len(frame))
        condition_bias = np.empty(len(frame))
        with torch.no_grad():
            for indices in residual_windows(frame, bundle.config.window_points):
                x = torch.tensor(features[indices], dtype=torch.float32)[None]
                cond = condition[indices][None] if condition is not None else None
                residual[indices] = bundle.residual(x, condition=cond)[0].numpy() * norm["residual_scale_w"]
                effects = bundle.residual.modulation_effects(x, cond)
                condition_bias[indices] = effects[0, :, 1].numpy() * norm["residual_scale_w"]
        residual_matrices.append(np.column_stack((residual - condition_bias, np.ones(len(frame)))))
        residual_targets.append(target - main_values - condition_bias)
    path_rank = _identifiability(residual_matrices, max_condition_number)
    path, path_fit = _fit(residual_matrices, residual_targets, kind="path", delta=huber_delta_w,
                          regularization=regularization, max_nfev=max_nfev)
    if weights_before != (state_version(bundle.main), state_version(bundle.residual)) or model_version != bundle.current_version():
        raise RuntimeError("低维校准意外改变网络状态")
    return CalibrationResult(model_version, float(main[0]), float(main[1]), float(path[0]), float(path[1]),
                             condition_reference, {"records": sorted(seen), "weighting": "equal_record_then_equal_valid_point",
                                                   "loss": "Huber", "huber_delta_w": huber_delta_w,
                                                   "unseen_descriptor_policy": "training_center_with_missing_mask_only_for_calibrated_target",
                                                   "main_identifiability": main_rank, "path_identifiability": path_rank,
                                                   "main_fit": main_fit, "path_fit": path_fit,
                                                   "order": ["frozen_network", "steady_coefficients", "fixed_main", "path_scale_bias"],
                                                   "weights_unchanged": True})


@dataclass
class IntervalCalibrator:
    model_version: str
    correction_w: float
    alpha: float = .1
    group_ids: list = field(default_factory=list)
    schema: str = "afc.group_interval_calibration.1"

    @property
    def version(self):
        return version_of(asdict(self))

    @classmethod
    def fit(cls, records, *, model_version, alpha=.1, excluded_group_ids=()):
        """Conformal group-max scores on held-out complete records.

Each record: group_id, lower_w, upper_w, target_w, and
dataset_role='interval_calibration'. Development/test groups are excluded.
Whole-record exchangeability is required for any coverage interpretation.
"""
        if not model_version or not 0 < alpha < 1:
            raise ValueError("区间校准模型版本或alpha无效")
        scores, ids, excluded = [], set(), set(excluded_group_ids)
        for record in records:
            key = record.get("group_id")
            if record.get("dataset_role") != "interval_calibration" or not key or key in ids or key in excluded:
                raise ValueError("区间校准必须使用独立完整分组，不能重复或复用开发/测试组")
            lower, upper, target = [np.asarray(record[name], dtype=float) for name in ("lower_w", "upper_w", "target_w")]
            if lower.ndim != 1 or not len(lower) or lower.shape != upper.shape or lower.shape != target.shape:
                raise ValueError("区间校准需要等长非空记录")
            if not np.isfinite([lower, upper, target]).all() or (lower > upper).any():
                raise ValueError("区间校准包含无效分位或标签")
            scores.append(float(np.maximum(lower - target, target - upper).max()))
            ids.add(key)
        rank = math.ceil((len(scores) + 1) * (1 - alpha))
        if not scores or rank > len(scores):
            raise ValueError("独立校准分组不足，不能形成有限的指定覆盖区间")
        correction = max(0., float(np.sort(scores)[rank - 1]))
        return cls(model_version, correction, alpha, sorted(ids))

    def apply(self, lower, upper, *, model_version):
        if self.schema != "afc.group_interval_calibration.1" or self.model_version != model_version:
            raise ValueError("区间校准与模型版本不匹配")
        if not np.isfinite(self.correction_w) or self.correction_w < 0:
            raise ValueError("区间校准扩展量无效")
        return np.asarray(lower) - self.correction_w, np.asarray(upper) + self.correction_w

    def save(self, path):
        save_json(path, {**asdict(self), "version": self.version})

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        expected = data.pop("version", None)
        result = cls(**data)
        if expected != result.version or result.schema != "afc.group_interval_calibration.1":
            raise ValueError("区间校准版本不匹配")
        return result
