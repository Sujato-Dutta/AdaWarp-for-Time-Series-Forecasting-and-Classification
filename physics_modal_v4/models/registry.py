"""Compact registry for the six retained matched comparison models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from physics_modal_v4.models import standard
from physics_modal_v4.models.timepro import TimePro
from physics_modal_v4.models.xcpd import DLinearXCPD, XCPDPlugin


COMPARISON_MODELS = (
    "VPNet",
    "xCPD",
    "TimePro",
    "iTransformer",
    "TimeMixer",
    "DLinear",
)


@dataclass
class ModelSpec:
    model: nn.Module
    epochs: int
    patience: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float = 1e-4
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    scheduler: str = "none"
    provenance: str = "repository-native matched implementation"


def _timepro_settings(dataset: str, horizon: int) -> dict:
    values = dict(patch_len=16, stride=8, width=32, layers=2, batch=16, lr=5e-5)
    table = {
        ("ETTh1", 96): dict(width=16, lr=7e-5),
        ("ETTh1", 192): dict(width=32),
        ("ETTh1", 336): dict(width=32),
        ("ETTh1", 720): dict(width=16),
        ("ETTh2", 96): dict(patch_len=8, stride=4, layers=1, width=16),
        ("ETTh2", 192): dict(patch_len=8, stride=4, layers=1, width=16),
        ("ETTh2", 336): dict(patch_len=8, stride=4, layers=1, width=16),
        ("ETTh2", 720): dict(patch_len=24, stride=12, layers=1, width=32),
        ("ETTm1", 96): dict(layers=1),
        ("ETTm1", 192): dict(layers=2),
        ("ETTm1", 336): dict(layers=2),
        ("ETTm1", 720): dict(layers=2),
        ("ETTm2", 96): dict(width=32),
        ("ETTm2", 192): dict(width=16),
        ("ETTm2", 336): dict(width=16),
        ("ETTm2", 720): dict(width=16),
        ("Weather", 96): dict(patch_len=6, stride=3, width=32, layers=3),
        ("Weather", 192): dict(patch_len=6, stride=3, width=16, layers=3, batch=32, lr=8e-5),
        ("Weather", 336): dict(patch_len=6, stride=3, width=16, layers=3, batch=32, lr=8e-5),
        ("Weather", 720): dict(patch_len=8, stride=4, width=16, layers=3, batch=32, lr=8e-5),
        ("Electricity", 96): dict(patch_len=8, stride=4, width=32, lr=5e-4),
        ("Electricity", 192): dict(patch_len=8, stride=4, width=32, lr=3e-4),
        ("Electricity", 336): dict(patch_len=8, stride=4, width=32, lr=3e-4),
        ("Electricity", 720): dict(patch_len=16, stride=8, width=32, lr=3e-4),
        ("Traffic", 96): dict(patch_len=8, stride=4, width=32, batch=4, lr=3e-4),
        ("Traffic", 192): dict(patch_len=8, stride=4, width=32, batch=4, lr=3e-4),
        ("Traffic", 336): dict(patch_len=8, stride=4, width=32, batch=4, lr=3e-4),
        ("Traffic", 720): dict(patch_len=8, stride=4, width=32, batch=4, lr=3e-4),
    }
    values.update(table[(dataset, horizon)])
    return values


def _timepro(dataset: str, seq_len: int, horizon: int, channels: int) -> ModelSpec:
    cfg = _timepro_settings(dataset, horizon)
    model = TimePro(
        seq_len,
        horizon,
        channels,
        patch_len=cfg["patch_len"],
        stride=cfg["stride"],
        width=cfg["width"],
        layers=cfg["layers"],
    )
    return ModelSpec(
        model,
        10,
        3,
        cfg["batch"],
        cfg["batch"],
        cfg["lr"],
        scheduler="cosine",
        provenance="portable PyTorch reproduction of official TimePro commit 70a20e5",
    )


def _xcpd(dataset: str, seq_len: int, horizon: int, channels: int) -> ModelSpec:
    table = {
        "ETTh1": (256, 1, 10),
        "ETTh2": (256, 1, 10),
        "ETTm1": (256, 2, 10),
        "ETTm2": (128, 2, 10),
        "Weather": (256, 2, 10),
        "Electricity": (512, 2, 15),
        "Traffic": (2048, 2, 30),
    }
    hidden, layers, epochs = table[dataset]
    model = DLinearXCPD(
        standard.dlinear(seq_len, horizon, channels),
        XCPDPlugin(horizon, channels, dataset, hidden, layers),
    )
    batch = 2 if dataset == "Traffic" else (4 if dataset == "Electricity" else 16)
    return ModelSpec(
        model,
        epochs,
        3,
        batch,
        batch,
        1e-4,
        provenance="paper-derived DLinear+xCPD; plugin source unavailable in named repository",
    )


def build_model(name: str, dataset: str, seq_len: int, horizon: int, channels: int) -> ModelSpec:
    if name == "DLinear":
        return ModelSpec(
            standard.dlinear(seq_len, horizon, channels), 10, 3, 32, 32, 1e-4,
            weight_decay=0.0,
        )
    if name == "iTransformer":
        return ModelSpec(
            standard.itransformer(seq_len, horizon, channels), 10, 3, 32, 32, 1e-4,
            weight_decay=0.0,
        )
    if name == "TimeMixer":
        return ModelSpec(
            standard.timemixer(seq_len, horizon, channels), 10, 3, 32, 32, 1e-4,
            weight_decay=0.0,
        )
    if name == "VPNet":
        batch = 4 if channels > 500 else 16
        return ModelSpec(
            standard.vpnet(seq_len, horizon, channels), 10, 3, batch, batch, 1e-3
        )
    if name == "TimePro":
        return _timepro(dataset, seq_len, horizon, channels)
    if name == "xCPD":
        return _xcpd(dataset, seq_len, horizon, channels)
    raise ValueError(f"Unsupported comparison model: {name}")


__all__ = ["COMPARISON_MODELS", "DLinearXCPD", "ModelSpec", "build_model"]
