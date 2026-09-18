"""Leakage-safe data, evaluation, and artifact utilities for final LatentFlow."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from benchmark_adawarp_ltsf import (
    dataset_path,
    load_numeric_csv,
    normalize_train,
    split_lengths,
)
from benchmark_adawarp_mvpf_plus_ltsf import sample_starts


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic")
HORIZONS = (96, 192, 336, 720)
SEEDS = (42, 43, 44, 45, 46)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, starts: Sequence[int], seq_len: int, horizon: int):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        split = start + self.seq_len
        return self.values[start:split], self.values[split : split + self.horizon]


@dataclass(frozen=True)
class TaskData:
    dataset: str
    horizon: int
    values: np.ndarray
    training_starts: np.ndarray
    validation_starts: np.ndarray
    test_starts: np.ndarray


def _sampled_starts(first: int, last: int, maximum: int, seed: int) -> np.ndarray:
    starts = np.arange(first, last, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("No windows are available for this split and horizon.")
    return sample_starts(starts, maximum, seed)


def load_task(
    data_root: Path,
    dataset: str,
    horizon: int,
    *,
    seq_len: int = 96,
    max_train_windows: int = 2048,
    max_validation_windows: int = 1024,
    window_seed: int = 42,
) -> TaskData:
    raw = load_numeric_csv(dataset_path(data_root, dataset))
    train_end, validation_length, declared_test = split_lengths(dataset, len(raw))
    validation_end = train_end + validation_length
    values, _, _ = normalize_train(raw, train_end)

    training = _sampled_starts(
        0,
        train_end - seq_len - horizon + 1,
        max_train_windows,
        window_seed,
    )
    validation_all = np.arange(
        train_end - seq_len,
        validation_end - seq_len - horizon + 1,
        dtype=np.int64,
    )
    validation_all = validation_all[validation_all + seq_len >= train_end]
    if validation_all.size == 0:
        raise ValueError(f"No validation windows for {dataset}, horizon={horizon}.")
    validation = sample_starts(validation_all, max_validation_windows, window_seed + 5_000)

    test_end = min(len(values), validation_end + declared_test) if dataset.startswith("ETT") else len(values)
    test = np.arange(
        validation_end - seq_len,
        test_end - seq_len - horizon + 1,
        dtype=np.int64,
    )
    test = test[test + seq_len >= validation_end]
    if test.size == 0:
        raise ValueError(f"No test windows for {dataset}, horizon={horizon}.")
    return TaskData(
        dataset=dataset,
        horizon=horizon,
        values=values.astype(np.float32),
        training_starts=training,
        validation_starts=validation,
        test_starts=test,
    )


def loader(
    task: TaskData,
    starts: Sequence[int],
    seq_len: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    dataset = WindowDataset(task.values, starts, seq_len, task.horizon)
    return DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: value.to(device) for name, value in state.items()}, strict=True)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    task: TaskData,
    starts: Sequence[int],
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    synchronize: bool = False,
) -> dict[str, float]:
    model.eval()
    squared = absolute = count = 0.0
    elapsed_ms = 0.0
    batches_seen = 0
    for inputs, targets in loader(task, starts, seq_len, batch_size):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if synchronize and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = model(inputs)
        if synchronize and device.type == "cuda":
            torch.cuda.synchronize(device)
        if batches_seen >= 2:
            elapsed_ms += 1000.0 * (time.perf_counter() - started)
        error = prediction - targets
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
        count += error.numel()
        batches_seen += 1
    mse = squared / max(1.0, count)
    return {
        "mse": mse,
        "mae": absolute / max(1.0, count),
        "rmse": math.sqrt(mse),
        "num_windows": float(len(starts)),
        "inference_ms_batch": elapsed_ms / max(1, batches_seen - 2),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


