"""Build calibrated spindle startup samples and merge them in command order."""

from typing import Tuple

import torch
from torch import Tensor, nn


class SequenceTrace(nn.Module):
    """Run the startup component and place each event ahead of its source anchor."""

    def __init__(self, dynamic, startup_domain_rpm: Tensor):
        super().__init__()
        self.dynamic = dynamic
        self.register_buffer("startup_domain_rpm", startup_domain_rpm.detach().clone())
        nodes = torch.arange(65, dtype=torch.int64)
        phase_index = (nodes // 16).clamp_max(3)
        self.register_buffer("phase_index", phase_index)
        self.register_buffer("phase_fraction", (nodes - 16 * phase_index).to(torch.float64) / 16.0)

    def forward(
        self,
        source_power: Tensor,
        source_valid: Tensor,
        source_geometry: Tensor,
        event_anchor: Tensor,
        event_line: Tensor,
        event_rpm: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        source_count = source_power.size(0)
        in_domain = ((event_rpm >= self.startup_domain_rpm[0]) &
                     (event_rpm <= self.startup_domain_rpm[-1]))
        anchored = (event_anchor >= 0) & (event_anchor < source_count)
        event_status = torch.where(
            in_domain, (~anchored).to(torch.int64), torch.full_like(event_anchor, 2),
        )
        placed = torch.nonzero(event_status == 0).flatten()

        # The calibration path always receives a supported event, including
        # requests with no M03 or only out-of-domain commands. Its first row
        # serves as a sentinel and is absent from the published trace.
        rpm = torch.cat((self.startup_domain_rpm[:1], event_rpm[placed]))
        phase, profile, speed, _, _ = self.dynamic._profiles(torch.zeros_like(rpm), rpm)
        begin = phase[:, self.phase_index]
        end = phase[:, self.phase_index + 1]
        elapsed = begin + self.phase_fraction * (end - begin)
        event_rows = torch.arange(rpm.size(0), dtype=torch.int64, device=rpm.device)[:, None]
        watts, _ = self.dynamic._evaluate(elapsed, event_rows.expand_as(elapsed), phase, profile, speed)
        watts = watts[1:].to(torch.float32).to(torch.float64)
        startup_power = torch.stack((watts, watts, watts, watts, torch.zeros_like(watts)), dim=2)
        startup_power = startup_power.reshape(-1, 5)

        # Each source row occupies a key range. All its M03 samples have keys
        # below that row's key and are ordered by event index and sample index.
        rows = torch.arange(source_count, dtype=torch.int64, device=source_power.device)
        event_slots = event_rpm.size(0) * 65
        stride = event_slots + 1
        startup_keys = (event_anchor[placed, None] * stride + placed[:, None] * 65 +
                        torch.arange(65, dtype=torch.int64, device=source_power.device)[None, :])
        keys = torch.cat((rows * stride + event_slots, startup_keys.flatten()))
        order = torch.sort(keys).indices

        row_id = torch.stack((torch.zeros_like(rows), rows, source_geometry[:, 1],
                              torch.full_like(rows, -1)), dim=1)
        event_ids = placed[:, None].expand(-1, 65).flatten()
        lines = event_line[placed, None].expand(-1, 65).flatten()
        startup_id = torch.stack((torch.ones_like(event_ids), torch.full_like(event_ids, -1),
                                  lines, event_ids), dim=1)
        valid = torch.cat((source_valid, torch.ones_like(event_ids, dtype=torch.bool)))
        time_s = torch.cat((torch.full_like(source_power[:, 0], float("nan")), elapsed[1:].flatten()))
        return (torch.cat((source_power, startup_power))[order], valid[order],
                torch.cat((row_id, startup_id))[order], time_s[order], event_status)
