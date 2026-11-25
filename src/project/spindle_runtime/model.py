"""Tensor-only cutting and calibrated event-based spindle power inference.

The optional command timeline is explicit. Neural calculations remain FP32,
and event interpolation uses the original float64 calibration precision.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

INPUT_COLUMNS = (
    "ap_mm", "ae_mm", "feed_input_mm_min", "spindle_rpm", "diameter_mm", "teeth",
    "curvature_per_mm", "block_length_mm", "entry_angle_rad", "exit_angle_rad",
    "direction_x", "direction_y", "direction_z", "segment_fraction",
    "previous_block_length_mm", "next_block_length_mm",
    "previous_feed_program_mm_min", "next_feed_program_mm_min",
    "has_previous_block", "has_next_block",
)
OUTPUT_COLUMNS = (
    "total_power_w", "total_lower_w", "total_upper_w", "base_power_w", "cutting_power_w",
)


class SpindlePowerPredictionModel(nn.Module):
    """FP32 features [B,L,20], boolean valid [B,L] -> FP32 [B,L,5]."""
    supports_tensor_io = True
    supports_tensor_export = False
    onnx_export_verified = False

    def __init__(self, config=None, **kwargs):
        super().__init__()
        config = dict(config or {})
        config.update(kwargs)
        architecture = config["architecture"]
        if architecture["route"] != "nominal_programmed_residual_v1":
            raise ValueError("Only the selected nominal residual model is supported")
        if architecture["physical_mode"] != "local" or architecture["coefficient_condition"] != "explicit":
            raise ValueError("Unqualified main model architecture")
        if architecture["residual_output"] != "quantiles":
            raise ValueError("Three quantile residual model required")
        if any(branch["active"] for branch in architecture["condition_spec"]["branches"].values()):
            raise ValueError("Active/new condition branches require a separately verified tensor contract")
        selected = SimpleNamespace(**architecture)
        self.main = PhysicalTrend(selected)
        self.residual = PathResidual(selected)
        norm = config["normalization"]
        self.register_buffer("main_mean_w", torch.tensor(norm["main_mean_w"], dtype=torch.float32))
        self.register_buffer("main_scale_w", torch.tensor(norm["main_scale_w"], dtype=torch.float32))
        self.register_buffer("residual_scale_w", torch.tensor(norm["residual_scale_w"], dtype=torch.float32))
        component = config.get("base_component")
        curve = component if component is not None else config["base_curve"]
        self.register_buffer("base_speeds_rpm", torch.tensor(curve["speeds_rpm"], dtype=torch.float64))
        self.register_buffer("base_powers_w", torch.tensor(curve["powers_w"], dtype=torch.float64))
        self.dynamic_base = DynamicBasePower(component) if component is not None else None
        policy = config.get("prediction_policy", {})
        self.base_extrapolation_default = policy.get("dynamic_base_extrapolation") == "boundary_shape_time_speed_excess_power_scaling_v1"
        self.feed_reference = float(architecture["feed_reference_mm_min"])

    def forward(self, features, valid, time_s=None, commands_rpm=None,
                initial_speed_rpm=None, point_time_s=None, *, base_extrapolation=None):
        # Software validates finite inputs/domain/window identity before invoking
        # the model. Padded positions are numeric zeros and never published.
        raw = features.to(torch.float64)
        physics = raw[..., :6].to(torch.float32)
        main = self.main(physics)
        d = raw[..., 4].clamp_min(1e-9)
        normalized_main = ((main - self.main_mean_w) / self.main_scale_w).to(torch.float64)
        path = torch.stack((
            normalized_main, raw[..., 0] / d, raw[..., 1] / d,
            raw[..., 2] / self.feed_reference, torch.abs(raw[..., 6]) * d,
            raw[..., 7] / d, raw[..., 8] / math.pi, raw[..., 9] / math.pi,
            raw[..., 10], raw[..., 11], raw[..., 12], raw[..., 13],
            raw[..., 14] / d, raw[..., 15] / d,
            raw[..., 16] / self.feed_reference, raw[..., 17] / self.feed_reference,
            raw[..., 18], raw[..., 19],
        ), dim=-1).to(torch.float32)
        residual = self.residual(path, valid) * self.residual_scale_w
        contact = (physics[..., 0] > 0) & (physics[..., 1] > 0) & (physics[..., 2] > 0) & (physics[..., 3] > 0)
        residual = torch.where(contact[..., None], residual, torch.zeros_like(residual))
        cutting = main.to(torch.float64)[..., None] + residual.to(torch.float64)
        # Vectorized piecewise linear interpolation of the frozen steady curve.
        speed = raw[..., 3]
        index = torch.sum(speed[..., None] >= self.base_speeds_rpm, dim=-1) - 1
        index = index.clamp(0, self.base_speeds_rpm.shape[0] - 2)
        x0, x1 = self.base_speeds_rpm[index], self.base_speeds_rpm[index + 1]
        y0, y1 = self.base_powers_w[index], self.base_powers_w[index + 1]
        base = y0 + (speed - x0) * (y1 - y0) / (x1 - x0)
        dynamic_inputs = (time_s, commands_rpm, initial_speed_rpm, point_time_s)
        if any(value is not None for value in dynamic_inputs):
            if any(value is None for value in dynamic_inputs):
                raise ValueError("Dynamic prediction requires time_s, commands_rpm, initial_speed_rpm and point_time_s")
            timeline_power, _, timeline_valid = self.predict_dynamic_base(
                time_s, commands_rpm, initial_speed_rpm, extrapolate=base_extrapolation)
            if point_time_s.shape != valid.shape or time_s.shape[0] not in (1, features.shape[0]):
                raise ValueError("Dynamic timeline batch and point-time shape must match feature batches")
            query = torch.where(valid, point_time_s.to(torch.float64), time_s[:, :1].to(torch.float64))
            base = DynamicBasePower.sample_points(time_s, timeline_power, timeline_valid, query)
        total = cutting + base[..., None]
        result = torch.stack((total[..., 1], total[..., 0], total[..., 2], base, cutting[..., 1]), dim=-1)
        return result.to(torch.float32).masked_fill(~valid[..., None], 0)

    def predict_dynamic_base(self, time_s, commands_rpm, initial_speed_rpm, *, extrapolate=None):
        """Return power, simulated speed and support mask on the full timeline."""
        if self.dynamic_base is None:
            raise ValueError("This historical model needs its complete base component before dynamic prediction")
        if extrapolate is None:
            extrapolate = self.base_extrapolation_default
        if extrapolate:
            try:
                from .base_extrapolation import predict_dynamic_base_extrapolated
            except ImportError:
                from base_extrapolation import predict_dynamic_base_extrapolated
            return predict_dynamic_base_extrapolated(
                self.dynamic_base, time_s, commands_rpm, initial_speed_rpm)
        return self.dynamic_base(time_s, commands_rpm, initial_speed_rpm)


class DynamicBasePower(nn.Module):
    """Tensor implementation of the original afc.base.2 command prediction.

    Each row begins in an explicitly supplied settled state. Unsupported
    transitions and interrupted recovery retain the original invalid mask.
    """

    def __init__(self, component):
        super().__init__()
        if component.get("schema") != "afc.base.2":
            raise ValueError("Dynamic prediction requires the calibrated afc.base.2 event component")
        profiles = component.get("event_profiles", [])
        if not profiles:
            raise ValueError("Dynamic base component has no event profiles")
        self.register_buffer("speeds_rpm", torch.tensor(component["speeds_rpm"], dtype=torch.float64))
        self.register_buffer("powers_w", torch.tensor(component["powers_w"], dtype=torch.float64))
        if (self.speeds_rpm.numel() < 2 or self.speeds_rpm.shape != self.powers_w.shape
                or not torch.isfinite(self.speeds_rpm).all() or not torch.isfinite(self.powers_w).all()
                or not torch.all(torch.diff(self.speeds_rpm) > 0)):
            raise ValueError("Invalid dynamic base steady curve")
        config = component["config"]
        for field in ("settling_absolute_band_w", "settling_relative_band"):
            value = float(config[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid dynamic base recovery tolerance")
            self.register_buffer(field, torch.tensor(value, dtype=torch.float64))
        for kind in ("startup", "braking"):
            selected = sorted((p for p in profiles if p["kind"] == kind), key=lambda p: p["level_rpm"])
            if not selected:
                raise ValueError("Dynamic base component requires startup and braking profiles")
            levels = [float(p["level_rpm"]) for p in selected]
            if any(not math.isfinite(v) or v <= 0 for v in levels) or any(a >= b for a, b in zip(levels, levels[1:])):
                raise ValueError("Invalid or repeated dynamic event speed")
            self.register_buffer(kind + "_domain_rpm", torch.tensor([levels[0], levels[-1]], dtype=torch.float64))
            # A one-level fit supports that exact level. The second interpolation
            # knot duplicates its values and does not extend its valid domain.
            if len(selected) == 1:
                selected = selected * 2
                levels = [levels[0], levels[0] + 1.]
            self.register_buffer(kind + "_levels_rpm", torch.tensor(levels, dtype=torch.float64))
            for field, width in (("phase_times_s", 5), ("power_w", 65), ("speed_rpm", 65),
                                 ("steady_w", None), ("extremum_w", None)):
                values = torch.tensor([p[field] for p in selected], dtype=torch.float64)
                expected = (len(selected),) if width is None else (len(selected), width)
                if values.shape != expected or not torch.isfinite(values).all():
                    raise ValueError("Invalid dynamic event table " + field)
                self.register_buffer(kind + "_" + field, values)
            phase = getattr(self, kind + "_phase_times_s")
            if not torch.all(phase[:, 0] == 0) or not torch.all(torch.diff(phase, dim=1) > 0):
                raise ValueError("Event phase times must increase from zero")
        if set(p["kind"] for p in profiles) != {"startup", "braking"}:
            raise ValueError("Unknown dynamic event kind")

    def _interpolate_profiles(self, kind, level):
        levels = getattr(self, kind + "_levels_rpm")
        clipped = level.clamp(levels[0], levels[-1])
        index = (torch.searchsorted(levels, clipped.contiguous(), right=True) - 1).clamp(0, levels.numel() - 2)
        fraction = (clipped - levels[index]) / (levels[index + 1] - levels[index])
        result = []
        for field in ("phase_times_s", "power_w", "speed_rpm", "steady_w", "extremum_w"):
            table = getattr(self, kind + "_" + field)
            weight = fraction if table.ndim == 1 else fraction[:, None]
            result.append(table[index] + weight * (table[index + 1] - table[index]))
        return result

    def _event_profiles(self, origin, target):
        startup = (origin == 0) & (target > 0)
        braking = (origin > 0) & (target == 0)
        level = torch.maximum(origin, target)
        up = self._interpolate_profiles("startup", level)
        down = self._interpolate_profiles("braking", level)
        phase, power, speed, steady, extremum = [
            torch.where(startup if u.ndim == 1 else startup[:, None], u, d)
            for u, d in zip(up, down)
        ]
        domain = torch.where(startup[:, None], self.startup_domain_rpm, self.braking_domain_rpm)
        supported = (startup | braking) & (level >= domain[:, 0]) & (level <= domain[:, 1])
        sign = torch.where(extremum >= steady, 1., -1.)
        tolerance = torch.maximum(self.settling_absolute_band_w, self.settling_relative_band * torch.abs(steady))
        # Match base_events._condition_recovery after interpolating the event.
        deviation = sign[:, None] * (power[:, 17:48] - steady[:, None])
        deviation = torch.cummin(torch.maximum(deviation, tolerance[:, None] + .001), dim=1).values
        recovery = steady[:, None] + sign[:, None] * deviation
        tail_phase = torch.arange(49, 65, dtype=power.dtype, device=power.device) / 16
        tail = steady[:, None] + sign[:, None] * tolerance[:, None] * (4 - tail_phase)
        conditioned = torch.cat((power[:, :17], recovery,
                                 (steady + sign * tolerance)[:, None], tail), dim=1)
        return phase, conditioned, speed, supported

    def _steady(self, speed):
        clipped = speed.clamp(self.speeds_rpm[0], self.speeds_rpm[-1])
        index = (torch.searchsorted(self.speeds_rpm, clipped.contiguous(), right=True) - 1).clamp(0, self.speeds_rpm.numel() - 2)
        x0, x1 = self.speeds_rpm[index], self.speeds_rpm[index + 1]
        y0, y1 = self.powers_w[index], self.powers_w[index + 1]
        return y0 + (clipped - x0) * (y1 - y0) / (x1 - x0)

    @staticmethod
    def _evaluate(elapsed, event_index, phase_times, power, speed):
        phases = phase_times[event_index]
        phase_index = (torch.sum(elapsed[..., None] >= phases, dim=-1) - 1).clamp(0, 3)
        t0 = torch.gather(phases, -1, phase_index[..., None]).squeeze(-1)
        t1 = torch.gather(phases, -1, (phase_index + 1)[..., None]).squeeze(-1)
        position = (16 * (phase_index + (elapsed - t0) / (t1 - t0))).clamp(0, 64)
        index = torch.floor(position).to(torch.long).clamp(0, 63)
        weight = position - index
        outputs = []
        for table in (power, speed):
            left, right = table[event_index, index], table[event_index, index + 1]
            outputs.append(left + weight * (right - left))
        return outputs

    def forward(self, time_s, commands_rpm, initial_speed_rpm):
        time = time_s.to(torch.float64)
        commands = commands_rpm.to(torch.float64)
        initial = initial_speed_rpm.to(torch.float64)
        if time.ndim != 2 or time.shape[1] < 3 or commands.shape != time.shape or initial.shape != time.shape[:1]:
            raise ValueError("Expected timeline time/commands [B,T>=3] and settled initial speed [B]")
        if not torch.isfinite(time).all() or not torch.isfinite(commands).all() or not torch.isfinite(initial).all():
            raise ValueError("Dynamic timeline and settled initial state must be finite")
        if torch.any(commands < 0) or torch.any(initial < 0):
            raise ValueError("Dynamic command and initial speeds must be nonnegative")
        dt = torch.diff(time, dim=1)
        # Use the same lower median without CUDA's nondeterministic index API.
        median_dt = torch.sort(dt, dim=1, stable=True).values[:, (dt.shape[1] - 1) // 2].unsqueeze(1)
        if torch.any(dt <= 0) or not torch.all(torch.abs(dt - median_dt) <= (1e-9 + .001 * torch.abs(median_dt))):
            raise ValueError("Each dynamic timeline must use strictly increasing, uniform original sample times")
        first = torch.ones_like(commands[:, :1], dtype=torch.bool)
        changes = torch.cat((first, commands[:, 1:] != commands[:, :-1]), dim=1)
        batch_index, sample_index = torch.nonzero(changes, as_tuple=True)
        origins = torch.where(sample_index == 0, initial[batch_index],
                              commands[batch_index, (sample_index - 1).clamp_min(0)])
        targets = commands[batch_index, sample_index]
        onset = time[batch_index, sample_index]
        initial_plateau = (sample_index == 0) & (origins == targets)
        phase, power_profile, speed_profile, supported = self._event_profiles(origins, targets)
        duration = torch.where(initial_plateau, 0., phase[:, -1])
        same_batch = torch.cat((torch.zeros_like(batch_index[:1], dtype=torch.bool),
                                batch_index[1:] == batch_index[:-1]))
        previous_ready = torch.cat((onset[:1], onset[:-1] + duration[:-1]))
        failed_event = ~(supported | initial_plateau) | (same_batch & (onset < previous_ready))
        failures = torch.zeros_like(changes, dtype=torch.long)
        failures[batch_index, sample_index] = failed_event.to(torch.long)
        failed = torch.cumsum(failures, dim=1) > 0
        counts = changes.sum(dim=1)
        offsets = torch.cumsum(counts, dim=0) - counts
        event_index = torch.cumsum(changes.to(torch.long), dim=1) - 1 + offsets[:, None]
        elapsed = time - onset[event_index]
        power, speed = self._evaluate(elapsed, event_index, phase, power_profile, speed_profile)
        prefix = initial_plateau[event_index]
        power = torch.where(prefix, self._steady(initial)[:, None], power)
        speed = torch.where(prefix, initial[:, None], speed)
        positive = self.speeds_rpm[self.speeds_rpm > 0]
        initial_supported = (torch.abs(initial) <= 30) | ((initial >= positive.min() - 30) & (initial <= positive.max() + 30))
        valid = ~failed & (~prefix | initial_supported[:, None])
        power = power.masked_fill(~valid, float("nan"))
        speed = speed.masked_fill(failed, float("nan"))
        return power, speed, valid

    @staticmethod
    def sample_points(time_s, power, valid, query_time_s):
        """Interpolate at positions' times, preserving unsupported intervals."""
        time, query = time_s.to(torch.float64), query_time_s.to(torch.float64)
        if time.shape[0] == 1:
            index = (torch.searchsorted(time[0], query.contiguous(), right=True) - 1).clamp(0, time.shape[1] - 2)
            t0, t1 = time[0, index], time[0, index + 1]
            y0, y1 = power[0, index], power[0, index + 1]
            v0, v1 = valid[0, index], valid[0, index + 1]
        else:
            index = (torch.searchsorted(time.contiguous(), query.contiguous(), right=True) - 1).clamp(0, time.shape[1] - 2)
            t0, t1 = torch.gather(time, 1, index), torch.gather(time, 1, index + 1)
            y0, y1 = torch.gather(power, 1, index), torch.gather(power, 1, index + 1)
            v0, v1 = torch.gather(valid, 1, index), torch.gather(valid, 1, index + 1)
        weight = (query - t0) / (t1 - t0)
        result = torch.where(weight == 0, y0, torch.where(weight == 1, y1, y0 + weight * (y1 - y0)))
        supported = torch.where(weight == 0, v0, torch.where(weight == 1, v1, v0 & v1))
        supported &= torch.isfinite(query) & (query >= time[:, :1]) & (query <= time[:, -1:])
        return result.masked_fill(~supported, float("nan"))


class ConditionEffects(nn.Module):
    """Retain all original parameters; current reference branches are inactive."""
    def __init__(self, config, outputs, enabled=True):
        super().__init__()
        self.spec = config.condition_spec if enabled else None
        self.outputs = outputs
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

    def forward(self, reference_tensor):
        # Export is intentionally restricted to the original inactive branches.
        return reference_tensor.new_zeros((*reference_tensor.shape[:-1], self.outputs))


class PhysicalTrend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        widths = [2, *config.coefficient_widths, 2]
        layers = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.coefficients = nn.Sequential(*layers)
        self.condition_effects = ConditionEffects(config, 2, config.use_condition_effects)

    def forward(self, physics):
        ap, ae, feed, speed, diameter, teeth = physics.unbind(-1)
        vc = math.pi * diameter * speed / 60000
        fz = feed / (teeth * speed).clamp_min(1e-9)
        features = torch.stack((vc / self.config.vc_reference_m_s,
                                fz / self.config.fz_reference_mm), dim=-1)
        positive = F.softplus(self.coefficients(features)) + 1e-6
        effects = self.condition_effects(physics)
        factors = torch.exp(effects.clamp(-8., 8.))
        ks = positive[..., 0] * 1000 * factors[..., 0]
        ke = positive[..., 1] * 10 * factors[..., 1]
        contact = (ap > 0) & (ae > 0) & (feed > 0) & (speed > 0)
        phi = torch.acos((1 - 2 * ae / diameter.clamp_min(1e-9)).clamp(-1 + 1e-7, 1 - 1e-7))
        edge = teeth * ap * phi / (2 * math.pi)
        shear = torch.where(contact, ks * ap * ae * feed / 60000, torch.zeros_like(feed))
        edge_power = torch.where(contact, ke * vc * edge, torch.zeros_like(feed))
        return shear + edge_power


class PathResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(18, config.hidden)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(config.hidden, config.hidden, config.kernel_size,
                      padding=d * (config.kernel_size // 2), dilation=d)
            for d in config.dilations
        ])
        layer = nn.TransformerEncoderLayer(config.hidden, config.attention_heads,
                                           dim_feedforward=4 * config.hidden,
                                           dropout=0., batch_first=True)
        self.attention = nn.TransformerEncoder(layer, config.attention_layers, enable_nested_tensor=False)
        self.output = nn.Linear(config.hidden, 1)
        self.spread = nn.Linear(config.hidden, 2)
        self.condition_effects = ConditionEffects(config, 2, config.use_residual_modulation)

    def forward(self, features, valid):
        x = self.projection(features.masked_fill(~valid[..., None], 0))
        x = x.masked_fill(~valid[..., None], 0)
        for conv in self.convolutions:
            x = (x + F.gelu(conv(x.transpose(1, 2)).transpose(1, 2))).masked_fill(~valid[..., None], 0)
        position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        frequencies = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype)
                                * (-math.log(10000.) / x.shape[-1]))
        encoding = torch.zeros(x.shape[1:], device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        x = self.attention(x + encoding, src_key_padding_mask=~valid)
        median = self.output(x).squeeze(-1)
        spread = F.softplus(self.spread(x)) + 1e-6
        values = torch.stack((median - spread[..., 0], median, median + spread[..., 1]), dim=-1)
        effects = self.condition_effects(features)
        gamma, bias = torch.exp(effects[..., 0].clamp(-8., 8.)), effects[..., 1]
        return (values * gamma[..., None] + bias[..., None]).masked_fill(~valid[..., None], 0)


Model = SpindlePowerPredictionModel


def build_model(config=None):
    return SpindlePowerPredictionModel(config)
