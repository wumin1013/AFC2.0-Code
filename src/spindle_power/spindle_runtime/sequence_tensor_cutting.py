"""Tensor implementation of the direct command-domain cutting window policy."""
import math
from typing import Tuple

import torch
from torch import Tensor, nn


class SequenceCutting(nn.Module):
    """Predict ordered lower, median and upper cutting power for one source sequence."""

    def __init__(self, cutting: nn.Module, base_speeds_rpm: Tensor, base_powers_w: Tensor):
        super().__init__()
        self.cutting = cutting
        self.register_buffer("base_speeds_rpm", base_speeds_rpm.detach().clone())
        self.register_buffer("base_powers_w", base_powers_w.detach().clone())
        self.width = 256
        self.batch_size = 128
        self.seam_half_span = 16
        # Store the finite family of tapers so the graph uses exact FP64
        # coefficients without relying on FP64 trigonometric runtime kernels.
        distance = torch.arange(self.seam_half_span, dtype=torch.float64)
        counts = torch.arange(self.seam_half_span + 1, dtype=torch.float64).clamp_min(2)
        phase = distance[None, :] * (math.pi / 2) / (counts[:, None] - 1)
        self.register_buffer("left_tapers", torch.sin(phase) ** 2)
        self.register_buffer("right_tapers", torch.cos(phase) ** 2)

    def _window_ranges(
        self, geometry: Tensor, valid: Tensor, query_path_mm: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        distance = query_path_mm[1:] - query_path_mm[:-1]
        continuation = (
            valid[1:] & valid[:-1]
            & (geometry[1:, 2] == geometry[:-1, 2])
            & (geometry[1:, 3] == geometry[:-1, 3])
            & torch.isfinite(distance) & (distance >= -1e-8)
        )
        fragment_boundary = torch.cat((torch.ones_like(valid[:1]), ~continuation))
        fragments = torch.cumsum(fragment_boundary.to(torch.int64), 0) - 1
        fragment_starts = torch.nonzero(fragment_boundary).flatten()
        fragment_ends = torch.cat((fragment_starts[1:], torch.full_like(fragment_starts[:1], valid.numel())))
        ordinals = torch.arange(valid.numel(), dtype=torch.int64, device=valid.device)
        starts = torch.nonzero(valid & ((ordinals - fragment_starts[fragments]) % self.width == 0)).flatten()
        ends = torch.minimum(starts + self.width, fragment_ends[fragments[starts]])
        return starts, ends, continuation

    def _power_with_window_precision(self, features: Tensor, valid: Tensor) -> Tensor:
        cutting = self.cutting(features, valid)
        speed = features[..., 3].to(torch.float64)
        last = self.base_speeds_rpm.numel() - 2
        level = (torch.sum(speed[..., None] >= self.base_speeds_rpm, dim=-1) - 1).clamp(0, last)
        x0, x1 = self.base_speeds_rpm[level], self.base_speeds_rpm[level + 1]
        y0, y1 = self.base_powers_w[level], self.base_powers_w[level + 1]
        base = y0 + (speed - x0) / (x1 - x0) * (y1 - y0)
        # The published window graph rounds each total and its base separately.
        rounded_base = base.to(torch.float32).to(torch.float64)
        total = (cutting + base[..., None]).to(torch.float32).to(torch.float64)
        return torch.stack((total[..., 0] - rounded_base, cutting[..., 1].to(torch.float32).to(torch.float64),
                            total[..., 2] - rounded_base), dim=-1)

    def _window_power(self, raw: Tensor, starts: Tensor, ends: Tensor, output: Tensor) -> Tensor:
        lengths = ends - starts
        # Order equal-length windows by source ordinal, as in the original stable sort.
        order = torch.argsort(lengths * (raw.size(0) + 1) + starts)
        for begin in range(0, starts.numel(), self.batch_size):
            batch = order[begin:begin + self.batch_size]
            length = lengths[batch]
            column = torch.arange(int(torch.max(length)), dtype=torch.int64, device=raw.device)[None, :]
            valid = column < length[:, None]
            query = torch.where(valid, starts[batch, None] + column, torch.zeros_like(column))
            power = self._power_with_window_precision(raw[query], valid)
            output = output.index_copy(0, query[valid], power[valid])
        return output

    def _smooth_links(
        self, raw: Tensor, geometry: Tensor, segment_mm: Tensor, continuation: Tensor
    ) -> Tuple[Tensor, Tensor]:
        contact = torch.all(raw[:, :4] > 0, dim=1)
        continuous = (
            continuation & contact[1:] & contact[:-1]
            & (geometry[1:, 1] == geometry[:-1, 1])
            & (geometry[1:, 4] == geometry[:-1, 4])
            & (segment_mm[1:] > segment_mm[:-1])
        )
        engagement = raw[:, :2] / raw[:, 4, None]
        change = engagement[1:] - engagement[:-1]
        same_speed = torch.all(torch.abs(raw[1:, 2:4] - raw[:-1, 2:4]) <= 1e-5, dim=1)
        same_curve = torch.abs(raw[1:, 6] - raw[:-1, 6]) <= 1e-5
        tangent_dot = torch.sum(raw[1:, 10:13] * raw[:-1, 10:13], dim=1)
        smooth = (torch.abs(change[:, 0]) <= 0.02) & same_speed & same_curve & (tangent_dot >= math.cos(math.pi / 12))
        radial_change = change[:, 1]
        direction = torch.sign(radial_change)
        smooth_continuity = continuous & smooth
        moderate = smooth_continuity & (torch.abs(radial_change) <= 0.05)
        support = torch.zeros_like(radial_change, dtype=torch.int64)
        for distance in range(1, 3):
            if radial_change.numel() > distance:
                agree = moderate[:-distance] & moderate[distance:]
                if distance == 2:
                    agree = agree & smooth_continuity[1:-1]
                agree = agree & (direction[:-distance] != 0) & (direction[:-distance] == direction[distance:])
                agree = agree & (torch.abs(radial_change[:-distance] - radial_change[distance:]) <= 0.02)
                vote = agree.to(torch.int64)
                padding = torch.zeros_like(support[:distance])
                support = support + torch.cat((vote, padding)) + torch.cat((padding, vote))
        return continuous, smooth & ((torch.abs(radial_change) <= 0.02) | (support >= 2))

    def _repair_seams(
        self, raw: Tensor, geometry: Tensor, segment_mm: Tensor, continuation: Tensor,
        starts: Tensor, ends: Tensor, power: Tensor
    ) -> Tensor:
        continuous, smooth = self._smooth_links(raw, geometry, segment_mm, continuation)
        joinable = continuous & smooth
        seam_ordinals = starts[1:] - 1
        candidates = torch.nonzero((ends[:-1] == starts[1:]) & joinable[seam_ordinals]).flatten()
        cutting = power[:, 1]
        offsets = torch.zeros_like(cutting)
        for candidate in range(candidates.numel()):
            left = int(candidates[candidate])
            boundary = int(starts[left + 1])
            seam = boundary - 1
            from_ordinal, to_ordinal = int(starts[left]), int(ends[left + 1])
            first, last = seam, boundary
            while first > max(from_ordinal, boundary - self.seam_half_span) and bool(joinable[first - 1]):
                first -= 1
            while last < min(to_ordinal - 1, boundary + self.seam_half_span - 1) and bool(joinable[last]):
                last += 1
            if min(boundary - first, last - seam) < 4:
                continue
            neighborhood = cutting[first:last + 1]
            jumps = torch.abs(neighborhood[1:] - neighborhood[:-1])
            bypass = seam - first
            sorted_jumps = torch.sort(torch.cat((jumps[:bypass], jumps[bypass + 1:])))[0]
            count = sorted_jumps.numel()
            median = (sorted_jumps[(count - 1) // 2] + sorted_jumps[count // 2]) * 0.5
            gap = cutting[boundary] - cutting[seam]
            if float(torch.abs(gap)) < max(60.0, 4.0 * float(median)):
                continue
            aux_length = min(self.width, to_ordinal - from_ordinal)
            aux_start = max(from_ordinal, min(boundary - aux_length // 2, to_ordinal - aux_length))
            aux_input = raw[aux_start:aux_start + aux_length][None, :, :]
            aux_valid = torch.ones_like(aux_input[:, :, 0], dtype=torch.bool)
            aux_cut = self.cutting(aux_input, aux_valid)[0, :, 1].to(torch.float32).to(torch.float64)
            aux_gap = aux_cut[boundary - aux_start] - aux_cut[seam - aux_start]
            if bool(torch.abs(aux_gap) > torch.abs(gap) * 0.5):
                continue

            left_count, right_count = boundary - first, last - seam
            left_weights = self.left_tapers[left_count, :left_count]
            right_weights = self.right_tapers[right_count, :right_count]
            correction = torch.cat((left_weights, -right_weights)) * ((gap - aux_gap) / 2)
            correction = torch.cat((torch.zeros_like(correction[:1]), correction[1:-1],
                                    torch.zeros_like(correction[-1:])))
            corrected = neighborhood + correction
            maximum = torch.max(torch.abs(corrected[1:] - corrected[:-1]))
            if bool(maximum > 0.75 * torch.max(jumps)) or bool(torch.any(offsets[first:last + 1] != 0)):
                continue
            ordinals = torch.arange(first, last + 1, dtype=torch.int64, device=raw.device)
            offsets = offsets.index_copy(0, ordinals, correction)
        return power + offsets[:, None]

    def forward(
        self, features: Tensor, feature_valid: Tensor, source_geometry: Tensor,
        segment_mm: Tensor, query_path_mm: Tensor
    ) -> Tensor:
        # Geometry columns are original source ID, NC line, execution instance,
        # sequence group and motion. Only the row ordinal defines adjacency.
        raw = features.to(torch.float32)
        output = torch.full((raw.size(0), 3), float("nan"), dtype=torch.float64, device=raw.device)
        starts, ends, continuation = self._window_ranges(source_geometry, feature_valid, query_path_mm)
        if starts.numel() == 0:
            return output
        output = self._window_power(raw, starts, ends, output)
        return self._repair_seams(raw, source_geometry, segment_mm, continuation, starts, ends, output)
