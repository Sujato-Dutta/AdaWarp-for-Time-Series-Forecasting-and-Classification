"""Portable single-file TimePro reproduction.

The released implementation relies on custom selective-scan and DCNv4 CUDA
extensions.  This module preserves the paper's operations in ordinary PyTorch:
overlapping patch tokens, bidirectional stable selective state evolution across
variables, learned-offset state propagation over the variable--patch plane,
and a channel-wise prediction head.  It is differentiable and requires no
compiler step on TACC.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        return (x - mean) / std, mean, std

    @staticmethod
    def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


class OffsetStatePropagation(nn.Module):
    """Dynamic interpolation followed by local depthwise state mixing."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.offset = nn.Conv2d(width, 2, kernel_size=3, padding=1)
        self.local = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        # field: B, width, variables, patches
        b, _, variables, patches = field.shape
        offset = 0.5 * torch.tanh(self.offset(field))
        yy = torch.linspace(-1.0, 1.0, variables, device=field.device, dtype=field.dtype)
        xx = torch.linspace(-1.0, 1.0, patches, device=field.device, dtype=field.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)[None].expand(b, -1, -1, -1).clone()
        if patches > 1:
            grid[..., 0] += 2.0 * offset[:, 0] / (patches - 1)
        if variables > 1:
            grid[..., 1] += 2.0 * offset[:, 1] / (variables - 1)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return self.pointwise(F.silu(self.local(sampled)))


class SelectiveVariableScan(nn.Module):
    """Stable input-conditioned state-space scan along the variable axis."""

    def __init__(self, hidden: int, patch_count: int, patch_width: int, dropout: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.patch_count = patch_count
        self.patch_width = patch_width
        self.input_gate = nn.Linear(hidden, 3 * hidden)
        self.log_rate = nn.Parameter(torch.zeros(hidden))
        self.skip = nn.Parameter(torch.ones(hidden))
        self.propagate = OffsetStatePropagation(patch_width)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        drive, delta, gate = self.input_gate(x).chunk(3, dim=-1)
        rate = F.softplus(self.log_rate)[None]
        state = torch.zeros_like(drive[:, 0])
        outputs = []
        for index in range(x.shape[1]):
            step = torch.sigmoid(delta[:, index])
            decay = torch.exp(-step * rate)
            state = decay * state + (1.0 - decay) * torch.tanh(drive[:, index])
            outputs.append(state + self.skip * x[:, index])
        y = torch.stack(outputs, dim=1)
        return y * F.silu(gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward = self._scan(x)
        backward = self._scan(x.flip(dims=[1])).flip(dims=[1])
        state = 0.5 * (forward + backward)
        b, variables, _ = state.shape
        field = state.reshape(b, variables, self.patch_count, self.patch_width).permute(0, 3, 1, 2)
        state = self.propagate(field).permute(0, 2, 3, 1).reshape(b, variables, self.hidden)
        return self.dropout(self.output(self.norm(state)))


class ProBlock(nn.Module):
    def __init__(self, hidden: int, patch_count: int, patch_width: int, dropout: float) -> None:
        super().__init__()
        self.scan = SelectiveVariableScan(hidden, patch_count, patch_width, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scan(x)
        return x + self.ffn(self.norm(x))


class TimePro(nn.Module):
    def __init__(self, seq_len: int, horizon: int, channels: int, *, patch_len: int,
                 stride: int, width: int, layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        del channels
        self.seq_len = seq_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.patch_count = int((seq_len - patch_len) / stride + 2)
        self.revin = RevIN()
        self.embedding = nn.Linear(patch_len, width, bias=False)
        hidden = self.patch_count * width
        self.blocks = nn.ModuleList(
            ProBlock(hidden, self.patch_count, width, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)
        self.projector = nn.Linear(hidden, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, mean, std = self.revin.normalize(inputs)
        x = x.transpose(1, 2)
        x = F.pad(x, (0, self.stride), mode="replicate")
        patches = x.unfold(-1, self.patch_len, self.stride)
        tokens = self.embedding(patches).flatten(start_dim=2)
        for block in self.blocks:
            tokens = block(tokens)
        prediction = self.projector(self.norm(tokens)).transpose(1, 2)
        return self.revin.denormalize(prediction, mean, std)


__all__ = ["TimePro"]
