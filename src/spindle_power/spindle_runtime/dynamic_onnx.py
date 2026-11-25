"""ONNX-standard tensor adapter for the frozen afc.base.2 component.

No fitting or calibration changes. This module deliberately preserves the
source's unsupported transitions. It uses standard ONNX NonZero, CumSum and
Loop; compatibility with an NPU compiler is not implied.
"""
from typing import List, Tuple

import torch
from torch import nn


class DynamicBaseONNX(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name, value in original.named_buffers(recurse=False):
            self.register_buffer(name, value.detach().clone())

    def _curve(self, speed: torch.Tensor, clip: bool = False) -> torch.Tensor:
        x = speed.clamp(self.speeds_rpm[0], self.speeds_rpm[-1]) if clip else speed
        index = (torch.sum(x.unsqueeze(-1) >= self.speeds_rpm, dim=-1) - 1).clamp(0, self.speeds_rpm.numel() - 2)
        x0, x1 = self.speeds_rpm[index], self.speeds_rpm[index + 1]
        y0, y1 = self.powers_w[index], self.powers_w[index + 1]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def _interpolate(self, level: torch.Tensor, levels: torch.Tensor,
                     phase: torch.Tensor, power: torch.Tensor, speed: torch.Tensor,
                     steady: torch.Tensor, extremum: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clipped = level.clamp(levels[0], levels[-1])
        index = (torch.sum(clipped.unsqueeze(1) >= levels, dim=1) - 1).clamp(0, levels.numel() - 2)
        fraction = (clipped - levels[index]) / (levels[index + 1] - levels[index])
        weight = fraction.unsqueeze(1)
        return (phase[index] + weight * (phase[index + 1] - phase[index]),
                power[index] + weight * (power[index + 1] - power[index]),
                speed[index] + weight * (speed[index + 1] - speed[index]),
                steady[index] + fraction * (steady[index + 1] - steady[index]),
                extremum[index] + fraction * (extremum[index + 1] - extremum[index]))

    def _profiles(self, origin: torch.Tensor, target: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        startup = (origin == 0) & (target > 0)
        level = torch.maximum(origin, target)
        domain = torch.where(startup.unsqueeze(1), self.startup_domain_rpm, self.braking_domain_rpm)
        anchor = torch.maximum(torch.minimum(level, domain[:, 1]), domain[:, 0])
        ratio = level / anchor
        up = self._interpolate(anchor, self.startup_levels_rpm, self.startup_phase_times_s,
                               self.startup_power_w, self.startup_speed_rpm,
                               self.startup_steady_w, self.startup_extremum_w)
        down = self._interpolate(anchor, self.braking_levels_rpm, self.braking_phase_times_s,
                                 self.braking_power_w, self.braking_speed_rpm,
                                 self.braking_steady_w, self.braking_extremum_w)
        phase = torch.where(startup.unsqueeze(1), up[0], down[0])
        power = torch.where(startup.unsqueeze(1), up[1], down[1])
        speed = torch.where(startup.unsqueeze(1), up[2], down[2])
        steady = torch.where(startup, up[3], down[3])
        extremum = torch.where(startup, up[4], down[4])
        sign = torch.where(extremum >= steady, 1., -1.)
        tolerance = torch.maximum(self.settling_absolute_band_w, self.settling_relative_band * torch.abs(steady))
        deviation = torch.maximum(sign.unsqueeze(1) * (power[:, 17:48] - steady.unsqueeze(1)),
                                  tolerance.unsqueeze(1) + .001)
        # Exactly the same 31-value cumulative minimum as the frozen component.
        columns = torch.jit.annotate(List[torch.Tensor], [])
        running = deviation[:, 0]
        columns.append(running)
        for j in range(1, 31):
            running = torch.minimum(running, deviation[:, j])
            columns.append(running)
        recovery = steady.unsqueeze(1) + sign.unsqueeze(1) * torch.stack(columns, dim=1)
        tail_phase = torch.arange(49, 65, dtype=power.dtype, device=power.device) / 16
        tail = steady.unsqueeze(1) + sign.unsqueeze(1) * tolerance.unsqueeze(1) * (4 - tail_phase)
        conditioned = torch.cat((power[:, :17], recovery, (steady + sign * tolerance).unsqueeze(1), tail), dim=1)
        supported = ((startup | ((origin > 0) & (target == 0))) &
                     (level >= domain[:, 0]) & (level <= domain[:, 1]))
        return phase, conditioned, speed, supported, ratio

    def _evaluate(self, elapsed: torch.Tensor, event_index: torch.Tensor,
                  phase_times: torch.Tensor, power: torch.Tensor, speed: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        phases = phase_times[event_index]
        phase_index = (torch.sum(elapsed.unsqueeze(-1) >= phases, dim=-1) - 1).clamp(0, 3)
        t0 = torch.gather(phases, -1, phase_index.unsqueeze(-1)).squeeze(-1)
        t1 = torch.gather(phases, -1, (phase_index + 1).unsqueeze(-1)).squeeze(-1)
        position = (16 * (phase_index + (elapsed - t0) / (t1 - t0))).clamp(0, 64)
        index = torch.floor(position).to(torch.long).clamp(0, 63)
        weight = position - index
        p0, p1 = power[event_index, index], power[event_index, index + 1]
        s0, s1 = speed[event_index, index], speed[event_index, index + 1]
        return p0 + weight * (p1 - p0), s0 + weight * (s1 - s0)

    def forward(self, time_s: torch.Tensor, commands_rpm: torch.Tensor,
                initial_speed_rpm: torch.Tensor, extrapolate: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time = time_s.to(torch.float64)
        commands = commands_rpm.to(torch.float64)
        initial = initial_speed_rpm.to(torch.float64)
        allow = extrapolate.reshape(-1)[0]
        finite = torch.isfinite(time).all(1) & torch.isfinite(commands).all(1) & torch.isfinite(initial)
        nonnegative = (commands >= 0).all(1) & (initial >= 0)
        dt = time[:, 1:] - time[:, :-1]
        median = torch.sort(dt, dim=1).values[:, (dt.shape[1] - 1) // 2].unsqueeze(1)
        uniform = (dt > 0).all(1) & (torch.abs(dt - median) <= (1e-9 + .001 * torch.abs(median))).all(1)
        # Invalid values cannot index calibration tables. status still rejects.
        commands = torch.where(torch.isfinite(commands), commands, torch.zeros_like(commands))
        initial = torch.where(torch.isfinite(initial), initial, torch.zeros_like(initial))
        first = torch.ones_like(commands[:, :1], dtype=torch.bool)
        changes = torch.cat((first, commands[:, 1:] != commands[:, :-1]), dim=1)
        coordinates = torch.nonzero(changes)
        batches, indices = coordinates[:, 0], coordinates[:, 1]
        origins = torch.where(indices == 0, initial[batches], commands[batches, (indices - 1).clamp_min(0)])
        targets = commands[batches, indices]
        onset = time[batches, indices]
        plateau = (indices == 0) & (origins == targets)
        phase, profile_power, profile_speed, supported, ratios = self._profiles(origins, targets)
        # Metadata selects extrapolation for the ENTIRE request if any row has
        # a positive level outside the steady or either event domain.
        positive_knots = self.speeds_rpm[self.speeds_rpm > 0]
        low = torch.maximum(positive_knots.min(), torch.maximum(self.startup_domain_rpm[0], self.braking_domain_rpm[0]))
        high = torch.minimum(positive_knots.max(), torch.minimum(self.startup_domain_rpm[1], self.braking_domain_rpm[1]))
        outside = (((commands > 0) & ((commands < low) | (commands > high))).any() |
                   ((initial > 0) & ((initial < low) | (initial > high))).any())
        use_ext = allow & outside
        ratio = torch.where(use_ext & ~plateau, ratios, torch.ones_like(ratios))
        duration = torch.where(plateau, 0., phase[:, -1] * ratio)
        same_batch = torch.cat((torch.zeros_like(batches[:1], dtype=torch.bool), batches[1:] == batches[:-1]))
        previous_ready = torch.cat((onset[:1], onset[:-1] + duration[:-1]))
        interrupted = same_batch & (onset < previous_ready)
        valid_transition = plateau | ((origins == 0) & (targets > 0)) | ((origins > 0) & (targets == 0))
        # The extrapolation wrapper raises for these even in-domain requests.
        # Encode those global exceptions as status and no usable outputs.
        unsupported_exception = allow & (~valid_transition).any()
        interrupted_exception = allow & interrupted.any()
        failed_event = ~(supported | plateau) | interrupted
        failures = torch.zeros_like(changes, dtype=torch.long).index_put((batches, indices), failed_event.to(torch.long))
        failed = torch.cumsum(failures, dim=1) > 0
        counts = changes.sum(1)
        offsets = torch.cumsum(counts, dim=0) - counts
        event_index = torch.cumsum(changes.to(torch.long), dim=1) - 1 + offsets.unsqueeze(1)
        elapsed = (time - onset[event_index]) / ratio[event_index]
        anchor_power, anchor_speed = self._evaluate(elapsed, event_index, phase, profile_power, profile_speed)
        scaled_speed = ratio[event_index] * anchor_speed
        scaled_power = self._curve(scaled_speed) + ratio[event_index] * (anchor_power - self._curve(anchor_speed))
        power = torch.where(ratio[event_index] == 1., anchor_power, scaled_power)
        speed = torch.where(ratio[event_index] == 1., anchor_speed, scaled_speed)
        prefix = plateau[event_index]
        initial_base = torch.where(use_ext, self._curve(initial), self._curve(initial, True))
        power = torch.where(prefix, initial_base.unsqueeze(1), power)
        speed = torch.where(prefix, initial.unsqueeze(1), speed)
        initial_supported = (torch.abs(initial) <= 30) | ((initial >= positive_knots.min() - 30) & (initial <= positive_knots.max() + 30))
        strict_valid = ~failed & (~prefix | initial_supported.unsqueeze(1))
        ext_valid = torch.isfinite(power) & torch.isfinite(speed)
        valid = (use_ext & ext_valid) | (~use_ext & strict_valid)
        status = ((~finite).to(torch.long) + 2 * (~nonnegative).to(torch.long) +
                  4 * (~uniform).to(torch.long) +
                  8 * unsupported_exception.to(torch.long) + 16 * interrupted_exception.to(torch.long) +
                  64 * (use_ext & (~ext_valid).any()).to(torch.long))
        # Original Python input validation rejects the whole request.
        accepted = (status == 0).all()
        valid = valid & accepted
        power = power.masked_fill(~valid, float('nan'))
        speed_valid = ((use_ext & ext_valid) | (~use_ext & ~failed)) & accepted
        speed = speed.masked_fill(~speed_valid, float('nan'))
        return power, speed, valid, status


@torch.jit.script
def sample_points_onnx(time_s: torch.Tensor, power: torch.Tensor, valid: torch.Tensor,
                       query_time_s: torch.Tensor) -> torch.Tensor:
    """Exact right-search interpolation using logarithmic-memory ONNX Loop."""
    time, query = time_s.to(torch.float64), query_time_s.to(torch.float64)
    time = time.expand(query.shape[0], time.shape[1])
    power = power.expand(query.shape[0], power.shape[1])
    valid = valid.expand(query.shape[0], valid.shape[1])
    low = torch.zeros_like(query, dtype=torch.long)
    high = torch.full_like(low, time.shape[1])
    while bool(torch.any(low < high)):
        active = low < high
        mid = (low + high) // 2
        value = torch.gather(time, 1, mid.clamp(0, time.shape[1] - 1))
        right = query >= value
        low = torch.where(active & right, mid + 1, low)
        high = torch.where(active & ~right, mid, high)
    index = (low - 1).clamp(0, time.shape[1] - 2)
    t0, t1 = torch.gather(time, 1, index), torch.gather(time, 1, index + 1)
    y0, y1 = torch.gather(power, 1, index), torch.gather(power, 1, index + 1)
    v0, v1 = torch.gather(valid, 1, index), torch.gather(valid, 1, index + 1)
    weight = (query - t0) / (t1 - t0)
    result = torch.where(weight == 0., y0, torch.where(weight == 1., y1, y0 + weight * (y1 - y0)))
    at_left, at_right = weight == 0., weight == 1.
    supported = (at_left & v0) | (~at_left & at_right & v1) | (~at_left & ~at_right & v0 & v1)
    supported = supported & torch.isfinite(query) & (query >= time[:, :1]) & (query <= time[:, -1:])
    return result.masked_fill(~supported, float('nan'))
