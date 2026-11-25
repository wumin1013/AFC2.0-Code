"""One tensor-only call for the full command-domain spindle prediction."""
from typing import Tuple

import torch
from torch import Tensor, nn


class SpindleSequenceTensor(nn.Module):
    """Run all original process points and return the merged command-order trace."""

    def __init__(self, cutting, trace, base_speeds_rpm, base_powers_w):
        super().__init__()
        self.cutting = cutting
        self.trace = trace
        self.register_buffer("base_speeds_rpm", base_speeds_rpm.detach().clone())
        self.register_buffer("base_powers_w", base_powers_w.detach().clone())

    def forward(
        self,
        features: Tensor,
        feature_valid: Tensor,
        source_geometry: Tensor,
        segment_mm: Tensor,
        query_path_mm: Tensor,
        command_rpm: Tensor,
        ap_ae_mm: Tensor,
        event_anchor: Tensor,
        event_line: Tensor,
        event_rpm: Tensor,
        extrapolate: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        cutting = self.cutting(features, feature_valid, source_geometry, segment_mm, query_path_mm)
        index = (torch.sum(command_rpm[:, None] >= self.base_speeds_rpm, dim=1) - 1)
        index = index.clamp(0, self.base_speeds_rpm.numel() - 2)
        x0, x1 = self.base_speeds_rpm[index], self.base_speeds_rpm[index + 1]
        y0, y1 = self.base_powers_w[index], self.base_powers_w[index + 1]
        base = y0 + (command_rpm - x0) * ((y1 - y0) / (x1 - x0))
        calibrated = (command_rpm >= self.base_speeds_rpm[0]) & (command_rpm <= self.base_speeds_rpm[-1])
        known = torch.isfinite(command_rpm) & (command_rpm >= 0)
        base = torch.where(known & (calibrated | extrapolate[0]), base, torch.full_like(base, float("nan")))
        base = torch.where(command_rpm == 0, torch.zeros_like(base), base)

        air = (ap_ae_mm >= 0).all(dim=1) & (ap_ae_mm == 0).any(dim=1)
        cutting = torch.where(air[:, None], torch.zeros_like(cutting), cutting)
        total = cutting + base[:, None]
        source_power = torch.stack((total[:, 1], total[:, 0], total[:, 2], base, cutting[:, 1]), dim=1)
        source_valid = torch.isfinite(source_power).all(dim=1)
        trace_power, trace_valid, trace_positions, trace_elapsed, event_status = self.trace(
            source_power, source_valid, source_geometry, event_anchor, event_line, event_rpm
        )
        return source_power, source_valid, trace_power, trace_valid, trace_positions, trace_elapsed, event_status
