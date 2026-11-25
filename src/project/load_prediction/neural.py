"""Optional PyTorch implementation. Import via models.py only."""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import ModelConfig
from .samples import PHYSICS_COLUMNS, path_features, path_feature_names, physical_quantities, LOAD_PROCESS_RULES_VERSION, PROCESS_MAPPING_RULES_VERSION
from .sample_rules import SAMPLE_RULES_VERSION
from .feed_samples import SPATIAL_ALIGNMENT_VERSION
from .metrics import quantile_loss
from .common import version_of, json_value


class ConditionEffects(nn.Module):
    """Independent physical branches; reference effects exactly (1,1,1,0)."""
    def __init__(self, config, outputs):
        super().__init__()
        self.spec, self.outputs = config.condition_spec, outputs
        self.branches = nn.ModuleDict()
        if self.spec is not None:
            self.register_buffer("reference", torch.tensor(self.spec["reference_encoded"], dtype=torch.float32))
            for name, branch in self.spec["branches"].items():
                network = nn.Sequential(nn.Linear(branch["width"], config.condition_width), nn.SiLU(),
                                        nn.Linear(config.condition_width, outputs))
                if not branch["active"]:
                    for parameter in network.parameters():
                        parameter.requires_grad_(False)
                self.branches[name] = network

    def forward(self, condition, shape, reference_tensor):
        result = reference_tensor.new_zeros((*shape, self.outputs))
        if self.spec is None:
            if condition is not None and condition.shape[-1] != 0:
                raise ValueError("模型没有训练工况编码，不能输入任意条件向量")
            return result
        if condition is None:
            condition = self.reference.expand(*shape, -1)
        if condition.shape == (*shape[:-1], self.spec["encoded_dim"]) and len(shape) > 1:
            condition = condition.unsqueeze(-2).expand(*shape, -1)
        if condition.shape != (*shape, self.spec["encoded_dim"]) or not torch.isfinite(condition).all():
            raise ValueError("工况编码维度不匹配或含非有限值")
        for name, branch in self.spec["branches"].items():
            if branch["active"]:
                start, width = branch["offset"], branch["width"]
                network = self.branches[name]
                selected = condition[..., start:start + width]
                reference = self.reference[start:start + width].expand_as(selected)
                result = result + network(selected) - network(reference)
        return result


class PhysicalTrend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        implicit = config.coefficient_condition == "implicit" and config.use_condition_effects and config.condition_spec is not None
        self.implicit_condition = implicit
        widths = [2 + (config.condition_spec["encoded_dim"] if implicit else 0), *config.coefficient_widths, 2]
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(nn.SiLU())
        self.coefficients = nn.Sequential(*layers) if config.physical_mode == "local" else nn.Linear(2, 2)
        if config.physical_mode == "fixed":
            with torch.no_grad():
                self.coefficients.weight.zero_()
            self.coefficients.weight.requires_grad_(False)
        from dataclasses import replace
        self.condition_effects = ConditionEffects(config if config.use_condition_effects and not implicit else replace(config, condition_spec=None), 2)
        if implicit:
            self.register_buffer("implicit_reference", torch.tensor(config.condition_spec["reference_encoded"], dtype=torch.float32))
            mask = torch.zeros(config.condition_spec["encoded_dim"])
            for branch in config.condition_spec["branches"].values():
                if branch["active"]:
                    mask[branch["offset"]:branch["offset"] + branch["width"]] = 1.
            self.register_buffer("implicit_active_mask", mask)

    def components(self, physics, condition=None):
        if physics.shape[-1] != 6:
            raise ValueError("物理层需要 ap,ae,F,S,D,z 六列")
        ap, ae, feed, speed, diameter, teeth = physics.unbind(-1)
        vc = math.pi * diameter * speed / 60000
        fz = feed / (teeth * speed).clamp_min(1e-9)
        features = [vc.unsqueeze(-1) / self.config.vc_reference_m_s,
                    fz.unsqueeze(-1) / self.config.fz_reference_mm]
        if self.implicit_condition:
            condition = self.implicit_reference.expand(*physics.shape[:-1], -1) if condition is None else condition
            if condition.shape != (*physics.shape[:-1], len(self.implicit_reference)) or not torch.isfinite(condition).all():
                raise ValueError("隐式工况输入形状或数值不匹配")
            features.append((condition - self.implicit_reference) * self.implicit_active_mask)
        positive = F.softplus(self.coefficients(torch.cat(features, dim=-1))) + 1e-6
        effects = self.condition_effects(condition if self.config.use_condition_effects and not self.implicit_condition else None, physics.shape[:-1], physics)
        factors = torch.exp(effects.clamp(-8., 8.))
        ks, ke = positive[..., 0] * 1000 * factors[..., 0], positive[..., 1] * 10 * factors[..., 1]
        contact = (ap > 0) & (ae > 0) & (feed > 0) & (speed > 0)
        # Endpoints are clipped for finite backward gradients; noncontact is zero.
        phi = torch.acos((1 - 2 * ae / diameter.clamp_min(1e-9)).clamp(-1 + 1e-7, 1 - 1e-7))
        edge = teeth * ap * phi / (2 * math.pi)
        shear = torch.where(contact, ks * ap * ae * feed / 60000, torch.zeros_like(feed))
        edge_power = torch.where(contact, ke * vc * edge, torch.zeros_like(feed))
        return {"shear_w": shear, "edge_w": edge_power, "ks": ks, "ke": ke,
                "ks_multiplier": factors[..., 0], "ke_multiplier": factors[..., 1], "power_w": shear + edge_power}

    def forward(self, physics, condition=None):
        return self.components(physics, condition)["power_w"]


class PathResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(len(path_feature_names(config.route)), config.hidden)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(config.hidden, config.hidden, config.kernel_size,
                      padding=d * (config.kernel_size // 2), dilation=d) for d in config.dilations])
        layer = nn.TransformerEncoderLayer(config.hidden, config.attention_heads,
                                           dim_feedforward=4 * config.hidden, dropout=0., batch_first=True)
        self.attention = nn.TransformerEncoder(layer, config.attention_layers, enable_nested_tensor=False)
        self.output = nn.Linear(config.hidden, 1)
        self.spread = nn.Linear(config.hidden, 2)
        from dataclasses import replace
        self.condition_effects = ConditionEffects(config if config.use_residual_modulation else replace(config, condition_spec=None), 2)
        if config.residual_output == "point":
            for parameter in self.spread.parameters():
                parameter.requires_grad_(False)

    def modulation_effects(self, features, condition=None):
        return self.condition_effects(condition if self.config.use_residual_modulation else None,
                                      features.shape[:2], features)

    def forward_quantiles(self, features, valid=None, condition=None):
        """N×L×registered_features -> N×L×3; condition is N×L×C or N×C."""
        if features.ndim != 3 or features.shape[-1] != self.projection.in_features:
            raise ValueError(f"路径网络输入为 batch × length × {self.projection.in_features}")
        if valid is None:
            valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if valid.shape != features.shape[:2] or not torch.all(valid.any(dim=1)):
            raise ValueError("每个窗口至少一个有效点，mask需匹配窗口")
        x = self.projection(features.masked_fill(~valid[..., None], 0))
        x = x.masked_fill(~valid[..., None], 0)
        for conv in self.convolutions:
            x = (x + F.gelu(conv(x.transpose(1, 2)).transpose(1, 2))).masked_fill(~valid[..., None], 0)
        position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        frequencies = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) * (-math.log(10000.) / x.shape[-1]))
        encoding = torch.zeros(x.shape[1:], device=x.device, dtype=x.dtype)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequencies), torch.cos(position * frequencies)
        x = self.attention(x + encoding, src_key_padding_mask=~valid)
        median = self.output(x).squeeze(-1)
        spread = F.softplus(self.spread(x)) + 1e-6 if self.config.residual_output == "quantiles" else x.new_zeros((*x.shape[:2], 2))
        values = torch.stack((median - spread[..., 0], median, median + spread[..., 1]), dim=-1)
        effects = self.modulation_effects(features, condition)
        gamma, bias = torch.exp(effects[..., 0].clamp(-8., 8.)), effects[..., 1]
        return (values * gamma[..., None] + bias[..., None]).masked_fill(~valid[..., None], 0)

    def forward(self, features, valid=None, condition=None):
        return self.forward_quantiles(features, valid, condition)[..., 1]


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def state_version(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


class ModelBundle:
    def __init__(self, config):
        self.config = config
        self.main = PhysicalTrend(config)
        self.residual = PathResidual(config)
        self.main_trained = False
        self.residual_trained = False
        self.main_version = None
        self.residual_version = None
        self.residual_main_version = None
        self.normalization = None
        self.metadata = {}

    @property
    def device(self):
        return next(self.main.parameters()).device

    def to(self, device):
        from .load_device import configure_load_device
        resolved, _ = configure_load_device(str(device))
        self.main.to(resolved)
        self.residual.to(resolved)
        return self

    def input_definition(self):
        definition = {"physics": PHYSICS_COLUMNS, "path": path_feature_names(self.config.route),
                      "quantiles": list(self.config.quantiles)}
        if self.execution_window_policy() != "legacy_v1":
            definition["residual_window_policy"] = self.execution_window_policy()
        return definition

    def execution_window_policy(self):
        policy = self.metadata.get("config", {}).get("residual_window_policy", "legacy_v1")
        if policy not in ("legacy_v1", "prospective_v1"):
            raise ValueError("模型残差窗口策略无效")
        if policy == "prospective_v1" and self.config.route != "nominal_programmed_residual_v1":
            raise ValueError("加工前残差窗口仅支持名义编程进给路线")
        return policy

    def execution_batch_windows(self):
        count = self.metadata.get("config", {}).get("residual_batch_windows", 1)
        if type(count) is not int or not 1 <= count <= 1024:
            raise ValueError("模型路径批量执行配置无效")
        return count

    def execution_batch_strategy(self):
        strategy = self.metadata.get("config", {}).get("residual_batch_strategy", "equal_length")
        if strategy not in ("equal_length", "padded"):
            raise ValueError("模型路径批量执行策略无效")
        return strategy

    def condition_values(self, metadata, count=None, *, allow_unknown=False):
        """Physical metadata -> C or count×C array; no condition schema -> None."""
        if self.config.condition_spec is None:
            return None
        from .condition import encode_conditions
        value = encode_conditions([metadata], self.config.condition_spec, allow_unknown=allow_unknown)[0]
        return value if count is None else np.repeat(value[None], count, axis=0)

    def current_version(self):
        configuration = asdict(self.config)
        # Existing afc.load.3 packages predate the explicit route field.
        if self.config.route == "predicted_feed_v1":
            configuration.pop("route")
        execution = {}
        if self.execution_window_policy() != "legacy_v1":
            execution["residual_window_policy"] = self.execution_window_policy()
        if self.execution_batch_strategy() == "padded":
            execution["residual_execution"] = dict(padded_batch_windows=self.execution_batch_windows())
        elif self.execution_batch_windows() != 1:
            execution["residual_execution"] = dict(equal_length_batch_windows=self.execution_batch_windows())
        input_rules = dict(sample=SAMPLE_RULES_VERSION, process=PROCESS_MAPPING_RULES_VERSION,
                           geometry=LOAD_PROCESS_RULES_VERSION, alignment=SPATIAL_ALIGNMENT_VERSION)
        samples = self.metadata.get("samples", [])
        if any(str(row.get("process_mapping_rules_version", "")).startswith(
                "afc.data_supported_process_coordinates.") for row in samples):
            from .process_coordinates import PROCESS_COORDINATE_RULES_VERSION
            if any(row.get("process_mapping_rules_version") != PROCESS_COORDINATE_RULES_VERSION
                   or row.get("position_policy") != "command_only_v1" for row in samples):
                raise ValueError("模型训练样本混合或缺少统一指令位置与工艺坐标规则")
            input_rules.update(process=PROCESS_COORDINATE_RULES_VERSION, position_policy="command_only_v1")
        return version_of({"config": configuration, "normalization": self.normalization, **execution,
                           "input_rules": input_rules,
                           "main": state_version(self.main), "residual": state_version(self.residual),
                           "provenance": {key: self.metadata.get(key) for key in
                             ("protocol_version", "snapshot_id", "feed_model_version", "base_version", "sample_rule_version")}})

    def assert_ready(self):
        self.config.validate()
        self.execution_window_policy()
        if not self.main_trained or not self.residual_trained or self.normalization is None:
            raise ValueError("模型未训练，不能生成正式预测")
        if self.main_version != self.residual_main_version or self.main_version != state_version(self.main) or self.residual_version != state_version(self.residual):
            raise ValueError("主趋势/残差权重或标签版本不匹配，需重新生成标签并训练")
        for name in ("main_mean_w", "main_scale_w", "residual_scale_w"):
            value = self.normalization.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or (name != "main_mean_w" and value <= 0):
                raise ValueError("负载模型训练归一化缺失或无效")
        if any(not torch.isfinite(value).all() for module in (self.main, self.residual) for value in module.state_dict().values()):
            raise ValueError("负载模型含非有限权重")

    def save(self, path):
        if self.main_trained or self.residual_trained:
            self.assert_ready()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(dict(schema="afc.load.3", config=asdict(self.config), main=self.main.state_dict(),
                            residual=self.residual.state_dict(), main_trained=self.main_trained,
                            residual_trained=self.residual_trained, main_version=self.main_version,
                            residual_version=self.residual_version, residual_main_version=self.residual_main_version,
                            normalization=json_value(self.normalization), metadata=json_value(self.metadata),
                            package_version=self.current_version(),
                            input_definition=self.input_definition(),
                            dependencies={"torch": str(torch.__version__), "numpy": np.__version__}), temp)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "afc.load.3":
            raise ValueError("负载模型进给/工况/分位格式不兼容；旧模型需重新训练")
        dependencies = payload.get("dependencies", {})
        if dependencies.get("torch", "").split(".")[0] != str(torch.__version__).split(".")[0]:
            raise ValueError("负载模型PyTorch主版本不兼容")
        if dependencies.get("numpy", "").split(".")[0] != np.__version__.split(".")[0]:
            raise ValueError("负载模型NumPy主版本不兼容")
        config = ModelConfig(**payload["config"])
        config.validate()
        bundle = cls(config)
        bundle.main.load_state_dict(payload["main"], strict=True)
        bundle.residual.load_state_dict(payload["residual"], strict=True)
        for key in ("main_trained", "residual_trained", "main_version", "residual_version", "residual_main_version", "normalization", "metadata"):
            setattr(bundle, key, payload[key])
        expected_inputs = bundle.input_definition()
        if payload.get("input_definition") != expected_inputs or payload.get("package_version") != bundle.current_version():
            raise ValueError("负载模型输入定义/归一化/权重版本不匹配")
        if type(bundle.main_trained) is not bool or type(bundle.residual_trained) is not bool or bundle.main_trained != bundle.residual_trained:
            raise ValueError("负载模型训练状态不完整")
        if bundle.main_trained:
            freeze(bundle.main)
        if bundle.residual_trained:
            freeze(bundle.residual)
            bundle.assert_ready()
        return bundle


def equal_length_batches(items, maximum, *, length):
    """Group independent equal-length items without padding or joined context."""
    if type(maximum) is not int or not 1 <= maximum <= 1024:
        raise ValueError("窗口batch上限必须为1—1024的整数")
    waiting = {}
    for item in items:
        key = length(item)
        group = waiting.setdefault(key, [])
        group.append(item)
        if len(group) == maximum:
            yield group
            waiting[key] = []
    for group in waiting.values():
        if group:
            yield group


def window_batches(items, maximum, *, length, strategy="equal_length"):
    """Padded execution uses a stable length sort; each window stays independent."""
    if strategy == "equal_length":
        yield from equal_length_batches(items, maximum, length=length)
        return
    if strategy != "padded" or type(maximum) is not int or not 1 <= maximum <= 1024:
        raise ValueError("窗口batch策略或上限无效")
    group = []
    for item in sorted(items, key=length):
        group.append(item)
        if len(group) == maximum:
            yield group
            group = []
    if group:
        yield group


def residual_windows(frame, width, *, known_boundaries=False):
    """Never cross a program/execution boundary; invalid points split windows."""
    if type(width) is not int or width < 1:
        raise ValueError("路径窗口长度必须为正整数")
    if frame.empty:
        return
    if known_boundaries:
        from .sample_rules import continuous_ranges
        for first, last in continuous_ranges(frame, known_only=True):
            for start in range(first, last, width):
                yield np.arange(start, min(last, start + width))
        return
    identities = frame["sequence_id" if "sequence_id" in frame else "execution_instance"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(identities[1:] != identities[:-1]) + 1, len(frame)]
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        for start in range(first, last, width):
            yield np.arange(start, min(last, start + width))


def prospective_residual_context(frame, width):
    """Plan complete NC-known context before any supervision selection.

    Returned windows index ``selected``; ``source_indices[window]`` retains
    their original full-grid positions, including windows with no labels.
    The caller selects the feed source before invoking this planner.
    """
    from .sample_rules import continuous_ids
    from .samples import NOMINAL_GEOMETRY_COLUMNS
    data = frame.reset_index(drop=True).copy()
    inputs = PHYSICS_COLUMNS + ["curvature_per_mm", "block_length_mm"] + NOMINAL_GEOMETRY_COLUMNS
    known = np.isfinite(data[inputs].to_numpy(float)).all(axis=1)
    data["sequence_id"] = continuous_ids(data, known, known_only=True)
    indices = np.flatnonzero(known)
    selected = data.iloc[indices].reset_index(drop=True)
    windows = tuple(residual_windows(selected, width, known_boundaries=True))
    return selected, indices, windows


def prediction_policy_version(bundle, calibration=None, memory=None):
    if calibration is None and memory is None:
        return bundle.current_version()
    policy = None if memory is None else {name: getattr(memory, name, None) for name in
        ("schema", "grid_mm", "alpha", "minimum_repeats", "consistency_floor_w")}
    return version_of({"model": bundle.current_version(),
                       "calibration": None if calibration is None else calibration.version,
                       "memory_policy": policy})


def predict_cutting_power(bundle, frame, *, condition=None, memory=None, context=None, round_id=None,
                          feed_mode=None, calibration=None, interval_calibrator=None):
    bundle.assert_ready()
    from .samples import assert_feed_inputs
    nominal = bundle.config.route == "nominal_programmed_residual_v1"
    if feed_mode is None:
        feed_mode = "programmed" if nominal else "predicted"
    if nominal and feed_mode not in ("programmed", "measured"):
        raise ValueError("名义物理残差路线只接受编程输入或显式加工后实测诊断")
    planned_windows = None
    if bundle.execution_window_policy() == "prospective_v1":
        selected, source_indices, planned_windows = prospective_residual_context(frame, bundle.config.window_points)
        if len(source_indices) != len(frame):
            raise ValueError("底层加工前预测需要完整有限输入；请由服务筛选并回填未知位置")
        frame = selected
    assert_feed_inputs(frame, feed_mode)
    physics = frame[PHYSICS_COLUMNS].to_numpy(np.float32)
    physical_quantities(physics)
    bundle.main.eval()
    bundle.residual.eval()
    condition_metadata = condition if isinstance(condition, dict) else context
    if condition is None and context is not None:
        condition = bundle.condition_values(context, len(frame), allow_unknown=calibration is not None)
    if isinstance(condition, dict):
        condition = bundle.condition_values(condition, len(frame), allow_unknown=calibration is not None)
    if calibration is not None:
        calibration.validate_for(bundle, condition_metadata)
    with torch.no_grad():
        cond = torch.as_tensor(condition, dtype=torch.float32, device=bundle.device) if condition is not None else None
        components = bundle.main.components(torch.as_tensor(physics, device=bundle.device), cond)
        shear, edge = components["shear_w"].cpu().numpy(), components["edge_w"].cpu().numpy()
        if calibration is not None:
            shear, edge = shear * calibration.ks_scale, edge * calibration.ke_scale
        main = shear + edge
        norm = bundle.normalization
        features = path_features(frame, main, norm["main_mean_w"], norm["main_scale_w"], bundle.config.feed_reference_mm_min,
                                 route=bundle.config.route)
        residual_quantiles = np.zeros((len(frame), 3))
        condition_bias_w = np.zeros(len(frame))
        data = frame.reset_index(drop=True)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=bundle.device)
        windows = (planned_windows if planned_windows is not None else
                   residual_windows(data, bundle.config.window_points, known_boundaries=nominal))
        strategy = bundle.execution_batch_strategy()
        for group in window_batches(windows, bundle.execution_batch_windows(), length=len, strategy=strategy):
            indices = np.zeros((len(group), max(map(len, group))), dtype=np.int64)
            length_valid = np.zeros_like(indices, dtype=bool)
            for row, values in enumerate(group):
                indices[row, :len(values)] = values
                length_valid[row, :len(values)] = True
            index_tensor = torch.as_tensor(indices, device=bundle.device)
            x = feature_tensor[index_tensor]
            window_condition = cond[index_tensor] if cond is not None else None
            padding = torch.as_tensor(length_valid, device=bundle.device) if strategy == "padded" else None
            values = bundle.residual.forward_quantiles(x, valid=padding, condition=window_condition).cpu().numpy()
            residual_quantiles[indices[length_valid]] = values[length_valid] * norm["residual_scale_w"]
            if calibration is not None:
                effects = bundle.residual.modulation_effects(x, window_condition)
                condition_bias_w[indices[length_valid]] = effects[:, :, 1].cpu().numpy()[length_valid] * norm["residual_scale_w"]
        if calibration is not None:
            residual_quantiles = ((residual_quantiles - condition_bias_w[:, None]) * calibration.residual_scale
                                  + condition_bias_w[:, None] + calibration.residual_bias_w)
    memory_values = np.zeros(len(frame))
    if memory is not None:
        if context is None or round_id is None:
            raise ValueError("记忆预测需要上下文和轮次")
        context = {**context, "main_version": bundle.main_version, "residual_version": bundle.residual_version}
        if calibration is not None:
            context.update(main_version=version_of([bundle.main_version, calibration.version]),
                           residual_version=version_of([bundle.residual_version, calibration.version]))
        memory_values = np.array([memory.predict(context, row.line, row.segment_mm, round_id, row.execution_instance) for _, row in frame.iterrows()])
    contact = physical_quantities(physics)["contact"]
    residual_quantiles = np.where(contact[:, None], residual_quantiles, 0.)
    residual = residual_quantiles[:, 1]
    memory_values = np.where(contact, memory_values, 0.)
    cutting_quantiles = main[:, None] + residual_quantiles + memory_values[:, None]
    prediction_version = prediction_policy_version(bundle, calibration, memory)
    result = dict(main_w=main, shear_w=shear, edge_w=edge, residual_w=residual, memory_w=memory_values,
                  cutting_power_w=cutting_quantiles[:, 1], residual_quantiles_w=residual_quantiles,
                  cutting_quantiles_w=cutting_quantiles, q05_w=cutting_quantiles[:, 0], q50_w=cutting_quantiles[:, 1],
                  q95_w=cutting_quantiles[:, 2], lower_w=cutting_quantiles[:, 0], upper_w=cutting_quantiles[:, 2],
                  quantiles=list(bundle.config.quantiles), interval_kind="direct_neural_quantiles" if bundle.config.residual_output == "quantiles" else "point_regression_comparison",
                  model_version=bundle.current_version(), prediction_version=prediction_version,
                  interval_status="model_prediction_interval" if bundle.config.residual_output == "quantiles" else "point_regression_no_interval",
                  calibration_version=None if calibration is None else calibration.version)
    if interval_calibrator is not None:
        lo, hi = interval_calibrator.apply(cutting_quantiles[:, 0], cutting_quantiles[:, 2],
                                           model_version=prediction_version)
        result.update(calibrated_lower_w=lo, calibrated_upper_w=hi,
                      lower_w=lo, upper_w=hi, interval_status="independently_calibrated_prediction_interval",
                      interval_calibration_version=interval_calibrator.version)
    return result
