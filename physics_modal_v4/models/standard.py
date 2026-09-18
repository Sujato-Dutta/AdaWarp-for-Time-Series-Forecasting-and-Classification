"""Single-file adapters for repository-native comparison models."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


class VarTCNBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        self.norm = nn.BatchNorm2d(width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(2 * width, width, 1), nn.Dropout(dropout),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        mixed = F.gelu(self.norm(self.depthwise(field)))
        return field + self.ffn(mixed)


class VPNetForecaster(nn.Module):
    """Matched variate-patch network used in the existing paper reruns."""

    def __init__(self, seq_len: int, pred_len: int, patch_len: int = 16,
                 embedding_dim: int = 256, depth: int = 2, dropout: float = 0.05,
                 reconstruction_weight: float = 0.1) -> None:
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_len = max(1, min(patch_len, seq_len))
        self.input_patches = math.ceil(seq_len / self.patch_len)
        self.output_patches = math.ceil(pred_len / self.patch_len)
        self.reconstruction_weight = reconstruction_weight
        hidden = max(embedding_dim, 2 * self.patch_len)
        self.encoder = nn.Sequential(nn.Linear(self.patch_len, hidden), nn.GELU(), nn.Linear(hidden, embedding_dim))
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, self.patch_len))
        self.blocks = nn.ModuleList(VarTCNBlock(embedding_dim, dropout) for _ in range(depth))
        self.patch_predictor = nn.Linear(self.input_patches, self.output_patches)

    def _normalize(self, values: torch.Tensor):
        mean = values.mean(1, keepdim=True).detach()
        std = values.std(1, keepdim=True, unbiased=False).detach().clamp_min(1e-5)
        return (values - mean) / std, mean, std

    def _patchify(self, values: torch.Tensor, count: int) -> torch.Tensor:
        target = count * self.patch_len
        values = F.pad(values, (0, 0, 0, max(0, target - values.shape[1])))[:, :target]
        return values.reshape(values.shape[0], count, self.patch_len, values.shape[2]).permute(0, 3, 1, 2)

    def _decode(self, embeddings: torch.Tensor, length: int) -> torch.Tensor:
        decoded = self.decoder(embeddings)
        return decoded.permute(0, 2, 3, 1).reshape(decoded.shape[0], -1, decoded.shape[1])[:, :length]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized, mean, std = self._normalize(inputs)
        encoded = self.encoder_norm(self.encoder(self._patchify(normalized, self.input_patches)))
        field = encoded.permute(0, 3, 1, 2)
        for block in self.blocks:
            field = block(field)
        future = self.patch_predictor(field).permute(0, 2, 3, 1)
        return self._decode(future, self.pred_len) * std + mean

    def auxiliary_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized, _, _ = self._normalize(inputs)
        patches = self._patchify(normalized, self.input_patches)
        reconstruction = self._decode(self.encoder_norm(self.encoder(patches)), self.seq_len)
        return F.l1_loss(reconstruction, normalized)


class FourArgumentAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs, None, None, None)


def _config(seq_len: int, horizon: int, channels: int, **overrides) -> SimpleNamespace:
    values = dict(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=48,
        pred_len=horizon,
        enc_in=channels,
        c_out=channels,
        moving_avg=25,
        d_model=128,
        d_ff=256,
        e_layers=2,
        n_heads=8,
        factor=3,
        dropout=0.1,
        activation="gelu",
        embed="timeF",
        freq="h",
        num_class=2,
        channel_independence=1,
        decomp_method="moving_avg",
        top_k=5,
        down_sampling_layers=1,
        down_sampling_window=2,
        down_sampling_method="avg",
        use_norm=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _module(name: str):
    source = str((ROOT / "TSLibrary").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module(f"models.{name}")


def dlinear(seq_len: int, horizon: int, channels: int) -> nn.Module:
    return FourArgumentAdapter(_module("DLinear").Model(_config(seq_len, horizon, channels)))


def itransformer(seq_len: int, horizon: int, channels: int) -> nn.Module:
    config = _config(seq_len, horizon, channels)
    return FourArgumentAdapter(_module("iTransformer").Model(config))


def timemixer(seq_len: int, horizon: int, channels: int) -> nn.Module:
    config = _config(
        seq_len,
        horizon,
        channels,
        d_model=128,
        d_ff=256,
        e_layers=2,
        down_sampling_layers=1,
        down_sampling_window=2,
        channel_independence=1,
    )
    return FourArgumentAdapter(_module("TimeMixer").Model(config))


def vpnet(seq_len: int, horizon: int, channels: int) -> nn.Module:
    del channels
    return VPNetForecaster(
        seq_len=seq_len,
        pred_len=horizon,
        patch_len=16,
        embedding_dim=256,
        depth=2,
        dropout=0.05,
        reconstruction_weight=0.1,
    )

