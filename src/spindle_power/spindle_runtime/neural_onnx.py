"""ONNX-standard cutting inference, using the selected model's original weights.

No fit or parameter change occurs here. Explicit self-attention replaces PyTorch's
backend-specific fused attention so batch and path dimensions remain dynamic.
The current published condition branches are inactive; export refuses active ones.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


class ExportAttentionLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        if layer.norm_first or layer.activation_relu_or_gelu != 1:
            raise ValueError("Only the published post-norm ReLU transformer is supported")
        if layer.self_attn.bias_k is not None or layer.self_attn.bias_v is not None or layer.self_attn.add_zero_attn:
            raise ValueError("Unexpected attention architecture")
        self.heads = int(layer.self_attn.num_heads)
        self.width = int(layer.self_attn.embed_dim)
        self.head_width = self.width // self.heads
        self.in_projection = nn.Linear(self.width, 3 * self.width)
        with torch.no_grad():
            self.in_projection.weight.copy_(layer.self_attn.in_proj_weight)
            self.in_projection.bias.copy_(layer.self_attn.in_proj_bias)
        self.out_projection = deepcopy(layer.self_attn.out_proj)
        self.linear1 = deepcopy(layer.linear1)
        self.linear2 = deepcopy(layer.linear2)
        self.norm1 = deepcopy(layer.norm1)
        self.norm2 = deepcopy(layer.norm2)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length = x.shape[0], x.shape[1]
        qkv = self.in_projection(x).reshape(batch, length, 3, self.heads, self.head_width)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        scores = torch.matmul(q / math.sqrt(float(self.head_width)), k.transpose(-2, -1))
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, v).transpose(1, 2).reshape(batch, length, self.width)
        x = self.norm1(x + self.out_projection(attended))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


class CuttingExport(nn.Module):
    """features float32[B,L,20], valid bool[B,L] -> cutting float64[B,L,3].

The last axis is lower, median, upper in watts. The full export wrapper adds
steady/dynamic base power and applies the final invalid-position mask.
"""
    def __init__(self, original):
        super().__init__()
        config = original.main.config
        if any(branch["active"] for branch in config.condition_spec["branches"].values()):
            raise ValueError("Active condition branches need a separately verified export")
        self.coefficients = deepcopy(original.main.coefficients)
        self.vc_reference = float(config.vc_reference_m_s)
        self.fz_reference = float(config.fz_reference_mm)
        self.feed_reference = float(original.feed_reference)
        for name in ("main_mean_w", "main_scale_w", "residual_scale_w"):
            self.register_buffer(name, getattr(original, name).detach().clone())
        residual = original.residual
        self.projection = deepcopy(residual.projection)
        self.convolutions = deepcopy(residual.convolutions)
        self.attention = nn.ModuleList([ExportAttentionLayer(layer) for layer in residual.attention.layers])
        self.output = deepcopy(residual.output)
        self.spread = deepcopy(residual.spread)
        self.width = int(residual.projection.out_features)
        if self.width % 2:
            raise ValueError("Published positional encoding requires an even hidden width")
        self.eval()

    def main(self, physics: torch.Tensor) -> torch.Tensor:
        ap, ae, feed, speed, diameter, teeth = physics.unbind(-1)
        vc = math.pi * diameter * speed / 60000
        fz = feed / (teeth * speed).clamp_min(1e-9)
        condition = torch.stack((vc / self.vc_reference, fz / self.fz_reference), dim=-1)
        positive = F.softplus(self.coefficients(condition)) + 1e-6
        ks = positive[..., 0] * 1000
        ke = positive[..., 1] * 10
        contact = (ap > 0) & (ae > 0) & (feed > 0) & (speed > 0)
        phi = torch.acos((1 - 2 * ae / diameter.clamp_min(1e-9)).clamp(-1 + 1e-7, 1 - 1e-7))
        edge = teeth * ap * phi / (2 * math.pi)
        shear = torch.where(contact, ks * ap * ae * feed / 60000, torch.zeros_like(feed))
        edge_power = torch.where(contact, ke * vc * edge, torch.zeros_like(feed))
        return shear + edge_power

    def residual(self, path: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid[..., None]
        x = self.projection(torch.where(mask, path, torch.zeros_like(path)))
        x = torch.where(mask, x, torch.zeros_like(x))
        for conv in self.convolutions:
            x = x + F.gelu(conv(x.transpose(1, 2)).transpose(1, 2))
            x = torch.where(mask, x, torch.zeros_like(x))
        position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.width, 2, device=x.device, dtype=x.dtype)
                              * (-math.log(10000.) / self.width))
        phase = position * frequency
        encoding = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).reshape(x.shape[1], self.width)
        x = x + encoding
        for layer in self.attention:
            x = layer(x, valid)
        median = self.output(x).squeeze(-1)
        spread = F.softplus(self.spread(x)) + 1e-6
        values = torch.stack((median - spread[..., 0], median, median + spread[..., 1]), dim=-1)
        return torch.where(mask, values, torch.zeros_like(values))

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        raw = features.to(torch.float64)
        physics = raw[..., :6].to(torch.float32)
        main = self.main(physics)
        diameter = raw[..., 4].clamp_min(1e-9)
        normalized_main = ((main - self.main_mean_w) / self.main_scale_w).to(torch.float64)
        path = torch.stack((
            normalized_main, raw[..., 0] / diameter, raw[..., 1] / diameter,
            raw[..., 2] / self.feed_reference, torch.abs(raw[..., 6]) * diameter,
            raw[..., 7] / diameter, raw[..., 8] / math.pi, raw[..., 9] / math.pi,
            raw[..., 10], raw[..., 11], raw[..., 12], raw[..., 13],
            raw[..., 14] / diameter, raw[..., 15] / diameter,
            raw[..., 16] / self.feed_reference, raw[..., 17] / self.feed_reference,
            raw[..., 18], raw[..., 19],
        ), dim=-1).to(torch.float32)
        residual = self.residual(path, valid) * self.residual_scale_w
        contact = (physics[..., 0] > 0) & (physics[..., 1] > 0) & (physics[..., 2] > 0) & (physics[..., 3] > 0)
        residual = torch.where(contact[..., None], residual, torch.zeros_like(residual))
        return main.to(torch.float64)[..., None] + residual.to(torch.float64)
