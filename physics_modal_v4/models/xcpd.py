"""Compact paper-derived xCPD plugin for a DLinear backbone.

The repository named by the xCPD paper did not contain the plugin at the time
this benchmark was frozen.  This file is therefore an auditable reproduction
of the paper equations rather than a claim of released-source execution.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class XCPDPlugin(nn.Module):
    def __init__(self, horizon: int, channels: int, dataset: str, hidden: int, layers: int) -> None:
        super().__init__()
        divisors = {
            "ETTh1": 16, "ETTh2": 16, "ETTm1": 12, "ETTm2": 6,
            "Weather": 2, "Electricity": 3, "Traffic": 1,
        }
        self.horizon = horizon
        self.channels = channels
        self.patch_len = max(1, int(math.ceil(horizon / divisors[dataset])))
        self.patch_count = int(math.ceil(horizon / self.patch_len))
        self.padded_horizon = self.patch_count * self.patch_len
        self.node_count = channels * self.patch_count
        self.embed = nn.Linear(self.patch_len, hidden)
        self.router_mean = nn.Linear(hidden, 3)
        self.router_log_scale = nn.Linear(hidden, 3)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )
        self.node_output = nn.Linear(hidden, self.patch_len)
        self.linear_path = nn.Linear(horizon, horizon)
        self.gnn_gate = nn.Parameter(torch.full((channels,), -4.0))
        self.linear_gate = nn.Parameter(torch.full((channels,), -4.0))
        self.raw_boundaries = nn.Parameter(torch.tensor([-0.7, 0.7]))
        self.routing_threshold = 0.5
        self.graph_density = 0.5
        self.register_buffer("graph_basis", torch.eye(self.node_count), persistent=True)
        self.register_buffer("basis_ready", torch.tensor(False), persistent=True)
        self.last_entropy = torch.tensor(0.0)
        self.last_balance = torch.tensor(0.0)

    def _nodes(self, forecast: torch.Tensor) -> torch.Tensor:
        values = forecast.transpose(1, 2)
        if self.padded_horizon > self.horizon:
            values = F.pad(values, (0, self.padded_horizon - self.horizon), mode="replicate")
        patches = values.reshape(values.shape[0], self.channels, self.patch_count, self.patch_len)
        return self.embed(patches).reshape(values.shape[0], self.node_count, -1)

    @torch.no_grad()
    def fit_shared_basis(self, forecasts: torch.Tensor) -> None:
        nodes = F.normalize(self._nodes(forecasts), dim=-1)
        affinity = torch.einsum("bnd,bmd->nm", nodes, nodes) / nodes.shape[0]
        affinity = affinity.clamp_min(0.0)
        degree = affinity.sum(dim=-1).clamp_min(1e-6)
        inv = degree.rsqrt()
        laplacian = torch.eye(self.node_count, device=affinity.device) - inv[:, None] * affinity * inv[None]
        _, vectors = torch.linalg.eigh(laplacian.float())
        self.graph_basis.copy_(vectors.to(self.graph_basis))
        self.basis_ready.fill_(True)

    def _frequency_bands(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        position = torch.linspace(0.0, 1.0, self.node_count, dtype=dtype, device=device)
        boundaries = torch.sort(torch.sigmoid(self.raw_boundaries))[0]
        low = torch.sigmoid(12.0 * (boundaries[0] - position))
        high = torch.sigmoid(12.0 * (position - boundaries[1]))
        middle = (1.0 - low - high).clamp_min(0.0)
        bands = torch.stack([low, middle, high])
        return bands / bands.sum(dim=0, keepdim=True).clamp_min(1e-6)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        nodes = self._nodes(base)
        basis = self.graph_basis.to(device=nodes.device, dtype=nodes.dtype)
        spectral = torch.einsum("nm,bmd->bnd", basis.T, nodes)
        spectral_energy = spectral.square().sum(dim=-1)
        node_energy = basis.square()[None] * spectral_energy[:, None]
        membership = torch.softmax(
            torch.einsum("bif,kf->bik", node_energy, self._frequency_bands(nodes.dtype, nodes.device)),
            dim=-1,
        )

        route_logits = self.router_mean(nodes)
        if self.training:
            route_logits = route_logits + torch.randn_like(route_logits) * F.softplus(self.router_log_scale(nodes))
        route = route_logits.softmax(dim=-1)
        probability, order = route.sort(dim=-1, descending=True)
        keep_sorted = probability.cumsum(dim=-1) - probability < self.routing_threshold
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        route = route * keep
        route = route / route.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        normalized = F.normalize(nodes, dim=-1)
        similarity = torch.einsum("bnd,bmd->bnm", normalized, normalized).clamp_min(0.0)
        neighbors = max(1, int(self.graph_density * self.node_count))
        cutoff = similarity.topk(neighbors, dim=-1).values[..., -1:]
        similarity = similarity * (similarity >= cutoff)
        hidden = nodes
        for block in self.blocks:
            message = torch.zeros_like(hidden)
            for expert in range(3):
                group = membership[..., expert]
                adjacency = similarity * group[:, :, None] * group[:, None, :]
                adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                expert_message = torch.einsum("bnm,bmd->bnd", adjacency, hidden)
                message = message + route[..., expert, None] * expert_message
            hidden = hidden + block(torch.cat([hidden, message], dim=-1))

        correction = self.node_output(hidden).reshape(
            base.shape[0], self.channels, self.patch_count, self.patch_len
        ).flatten(2)[..., :self.horizon].transpose(1, 2)
        linear = self.linear_path(base.transpose(1, 2)).transpose(1, 2) - base
        self.last_entropy = -(route * route.clamp_min(1e-8).log()).sum(dim=-1).mean()
        usage = route.mean(dim=(0, 1))
        self.last_balance = usage.std() / usage.mean().clamp_min(1e-8)
        return (
            base
            + torch.sigmoid(self.gnn_gate)[None, None] * correction
            + torch.sigmoid(self.linear_gate)[None, None] * linear
        )


class DLinearXCPD(nn.Module):
    def __init__(self, backbone: nn.Module, plugin: XCPDPlugin) -> None:
        super().__init__()
        self.backbone = backbone
        self.plugin = plugin

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.plugin(self.backbone(inputs))

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def auxiliary_loss(self) -> torch.Tensor:
        return -1e-3 * self.plugin.last_entropy + 1e-3 * self.plugin.last_balance


__all__ = ["DLinearXCPD", "XCPDPlugin"]
