"""Train VPNet with the final matched MVPF evaluation protocol.

The runner uses train-only dataset normalization, sampled training and
validation windows, validation-best checkpoint selection, and every standard
test window. Metrics are streamed to avoid storing multi-gigabyte predictions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adawarp_experiment_utils import ensure_dir, save_environment
from adawarp_neural_baselines import VPNetForecaster
from benchmark_adawarp_ltsf import dataset_path, load_numeric_csv, normalize_train, split_lengths
from benchmark_adawarp_mvpf_plus_ltsf import (
    WindowDataset,
    clone_state_dict,
    resolve_device,
    split_eval_starts,
    train_starts,
)


DATASETS = ("ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity", "Traffic")
HORIZONS = (96, 192, 336, 720)
FIELDNAMES = (
    "dataset", "horizon", "seed", "seq_len", "model", "mse", "mae", "rmse",
    "num_test_windows", "test_start", "test_end_exclusive", "best_epoch",
    "best_validation_mse", "elapsed_seconds", "checkpoint",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(args: argparse.Namespace, horizon: int) -> dict[str, object]:
    return {
        "seq_len": args.seq_len,
        "pred_len": horizon,
        "patch_len": args.patch_len,
        "embedding_dim": args.width,
        "depth": args.depth,
        "dropout": args.dropout,
        "reconstruction_weight": args.reconstruction_weight,
    }


def build_model(args: argparse.Namespace, horizon: int) -> VPNetForecaster:
    return VPNetForecaster(**model_config(args, horizon))


def full_test_starts(total_len: int, boundary: int, seq_len: int, pred_len: int) -> np.ndarray:
    starts = np.arange(boundary - seq_len, total_len - seq_len - pred_len + 1, dtype=np.int64)
    starts = starts[starts + seq_len >= boundary]
    if starts.size <= 0:
        raise ValueError("No standard test windows are available.")
    return starts


@torch.no_grad()
def evaluate(
    model: nn.Module,
    series: np.ndarray,
    starts: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataset = WindowDataset(series, starts, seq_len=seq_len, pred_len=pred_len)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    total_sq = 0.0
    total_abs = 0.0
    total_count = 0
    was_training = model.training
    model.eval()
    for inputs, targets in loader:
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        prediction = model(inputs, None, None, None)
        error = prediction - targets
        total_sq += float(error.square().sum().detach().cpu())
        total_abs += float(error.abs().sum().detach().cpu())
        total_count += int(error.numel())
    if was_training:
        model.train()
    mse = total_sq / max(1, total_count)
    return {
        "mse": mse,
        "mae": total_abs / max(1, total_count),
        "rmse": math.sqrt(mse),
        "num_windows": float(len(starts)),
    }


def train_model(
    model: VPNetForecaster,
    series: np.ndarray,
    training_starts: np.ndarray,
    validation_starts: np.ndarray,
    *,
    args: argparse.Namespace,
    horizon: int,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    dataset = WindowDataset(series, training_starts, seq_len=args.seq_len, pred_len=horizon)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_state = clone_state_dict(model)
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        forecast_losses = []
        reconstruction_losses = []
        for inputs, targets in loader:
            inputs = inputs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, None, None, None)
            forecast_loss = criterion(prediction, targets)
            reconstruction_loss = model.auxiliary_loss(inputs)
            loss = forecast_loss + args.reconstruction_weight * reconstruction_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            forecast_losses.append(float(forecast_loss.detach().cpu()))
            reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))

        validation = evaluate(
            model, series, validation_starts, seq_len=args.seq_len,
            pred_len=horizon, batch_size=args.eval_batch_size, device=device,
        )
        if validation["mse"] < best_val:
            best_val = float(validation["mse"])
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": float(epoch),
            "training_total": float(np.mean(losses)),
            "training_mse": float(np.mean(forecast_losses)),
            "training_reconstruction": float(np.mean(reconstruction_losses)),
            "validation_mse": float(validation["mse"]),
            "best_validation_mse": float(best_val),
            "best_epoch": float(best_epoch),
        }
        history.append(record)
        print(
            f"epoch={epoch} train={record['training_total']:.6f} "
            f"forecast={record['training_mse']:.6f} val={record['validation_mse']:.6f} "
            f"best={best_val:.6f}@{best_epoch}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    model.load_state_dict({key: value.to(device=device) for key, value in best_state.items()})
    return history, {"best_epoch": float(best_epoch), "best_validation_mse": float(best_val)}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    output = ensure_dir(args.output_root)
    metrics_path = output / "metrics" / "matched_vpnet_standard_test.csv"
    checkpoint_dir = ensure_dir(output / "checkpoints")
    audit_dir = ensure_dir(output / "audit")
    save_environment(audit_dir / "environment_matched_vpnet.json")

    rows: list[dict[str, object]] = [dict(row) for row in read_rows(metrics_path)]
    completed = {(str(row["dataset"]), int(row["horizon"])) for row in rows}

    for dataset_name in args.datasets:
        values = load_numeric_csv(dataset_path(Path(args.data_root), dataset_name))
        train_len, val_len, declared_test_len = split_lengths(dataset_name, len(values))
        val_end = train_len + val_len
        test_end = min(len(values), val_end + declared_test_len)
        normalized, _, _ = normalize_train(values, train_len)

        for horizon in args.horizons:
            key = (dataset_name, horizon)
            if key in completed and not args.force:
                print(f"skip-complete dataset={dataset_name} horizon={horizon}", flush=True)
                continue

            started = time.perf_counter()
            set_seed(args.seed)
            training_starts = train_starts(
                train_len, args.seq_len, horizon, args.max_train_windows, args.seed,
            )
            validation_starts = split_eval_starts(
                val_end, train_len, args.seq_len, horizon,
                args.max_validation_windows, args.seed + 5000,
            )
            test_starts = full_test_starts(test_end, val_end, args.seq_len, horizon)
            print(
                f"matched-vpnet start dataset={dataset_name} h={horizon} "
                f"train={len(training_starts)} val={len(validation_starts)} "
                f"test={len(test_starts)}",
                flush=True,
            )

            model = build_model(args, horizon).to(device=device)
            history, selection = train_model(
                model, normalized, training_starts, validation_starts,
                args=args, horizon=horizon, device=device,
            )
            metrics = evaluate(
                model, normalized, test_starts, seq_len=args.seq_len,
                pred_len=horizon, batch_size=args.eval_batch_size, device=device,
            )
            checkpoint = checkpoint_dir / f"VPNet_{dataset_name}_h{horizon}_seed{args.seed}.pt"
            torch.save(
                {
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "config": model_config(args, horizon),
                    "selection": selection,
                    "history": history,
                    "protocol": {
                        "normalization": "training-split z-score plus VPNet prefix normalization",
                        "training_windows": len(training_starts),
                        "validation_windows": len(validation_starts),
                        "test_windows": len(test_starts),
                        "test_evaluation": "all standard test windows",
                    },
                },
                checkpoint,
            )
            row: dict[str, object] = {
                "dataset": dataset_name,
                "horizon": horizon,
                "seed": args.seed,
                "seq_len": args.seq_len,
                "model": "VPNet",
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "num_test_windows": int(metrics["num_windows"]),
                "test_start": int(test_starts[0]),
                "test_end_exclusive": int(test_starts[-1] + args.seq_len + horizon),
                "best_epoch": int(selection["best_epoch"]),
                "best_validation_mse": selection["best_validation_mse"],
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint),
            }
            rows = [
                existing for existing in rows
                if not (str(existing["dataset"]) == dataset_name and int(existing["horizon"]) == horizon)
            ]
            rows.append(row)
            rows.sort(key=lambda item: (DATASETS.index(str(item["dataset"])), int(item["horizon"])))
            write_rows(metrics_path, rows)
            completed.add(key)
            print(
                f"matched-vpnet result dataset={dataset_name:<11} h={horizon:<3} "
                f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
                f"best_epoch={int(selection['best_epoch'])} "
                f"val={selection['best_validation_mse']:.6f} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patch-len", type=int, default=16)
    p.add_argument("--reconstruction-weight", type=float, default=0.1)
    p.add_argument("--max-train-windows", type=int, default=2048)
    p.add_argument("--max-validation-windows", type=int, default=1024)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--data-root", default="TSLibrary/dataset")
    p.add_argument("--output-root", type=Path, default=Path("mvpf-tests/results/matched_vpnet"))
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())