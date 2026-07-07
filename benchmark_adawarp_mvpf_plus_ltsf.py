"""Run AdaWarp-MVPF+ on standard LTSF datasets.

MVPF+ leaves the original ``adawarp_mvpf.py`` untouched. It evaluates the
ablation-pruned MVPF backbone plus an adaptive linear field skip, using the same
matched LTSF protocol as ``benchmark_adawarp_mvpf_ltsf.py``. This runner uses
validation checkpoint selection from the validation split only, mirroring common
LTSF neural-baseline practice without test leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from adawarp_experiment_utils import ensure_dir, save_environment, write_csv
from adawarp_mvpf_plus import AdaWarpMVPFPlusForecaster
from benchmark_adawarp_ltsf import DATASET_FILES, dataset_path, load_numeric_csv, normalize_train, split_lengths


MODEL_NAME = "AdaWarp-MVPF+"


class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, starts: Sequence[int], *, seq_len: int, pred_len: int):
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        seq_end = start + self.seq_len
        pred_end = seq_end + self.pred_len
        return self.series[start:seq_end], self.series[seq_end:pred_end]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_starts(starts: np.ndarray, max_windows: int, seed: int) -> np.ndarray:
    if starts.size <= max_windows:
        return starts
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(starts, size=max_windows, replace=False))


def train_starts(train_end: int, seq_len: int, pred_len: int, max_windows: int, seed: int) -> np.ndarray:
    starts = np.arange(0, train_end - seq_len - pred_len + 1, dtype=np.int64)
    if starts.size <= 0:
        raise ValueError("Training split is too short for the requested seq_len/pred_len.")
    return sample_starts(starts, max_windows, seed)


def split_eval_starts(
    total_len: int,
    boundary: int,
    seq_len: int,
    pred_len: int,
    max_windows: int,
    seed: int,
) -> np.ndarray:
    starts = np.arange(boundary - seq_len, total_len - seq_len - pred_len + 1, dtype=np.int64)
    starts = starts[starts + seq_len >= boundary]
    if starts.size <= 0:
        raise ValueError("No evaluation windows are available for the requested seq_len/pred_len.")
    return sample_starts(starts, max_windows, seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def evaluate_mse_only(
    model: nn.Module,
    series: np.ndarray,
    starts: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    dataset = WindowDataset(series, starts, seq_len=seq_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=0)
    total_sq = 0.0
    total_count = 0
    was_training = model.training
    model.eval()
    for inputs, targets in loader:
        pred = model(inputs.to(device=device), None, None, None)
        err = pred - targets.to(device=device)
        total_sq += float(err.pow(2).sum().detach().cpu())
        total_count += int(err.numel())
    if was_training:
        model.train()
    return total_sq / max(1, total_count)


def train_model(
    model: AdaWarpMVPFPlusForecaster,
    series: np.ndarray,
    starts: np.ndarray,
    validation_starts: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    patience: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    dataset = WindowDataset(series, starts, seq_len=seq_len, pred_len=pred_len)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history = []
    best_state = clone_state_dict(model)
    best_val_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    model.train()
    for epoch in range(1, epochs + 1):
        parts: dict[str, list[float]] = {"training_mse": []}
        for inputs, targets in loader:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction, aux = model.forward_with_aux(inputs, None, None, None)
            mse = criterion(prediction, targets)
            loss = mse + model.reconstruction_weight * model.auxiliary_loss(inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            parts["training_mse"].append(float(loss.detach().cpu()))
            for key, value in aux.items():
                parts.setdefault(key, []).append(float(value.detach().cpu()))
        val_mse = evaluate_mse_only(
            model,
            series,
            validation_starts,
            seq_len=seq_len,
            pred_len=pred_len,
            batch_size=eval_batch_size,
            device=device,
        )
        record = {key: float(np.mean(values)) for key, values in parts.items()}
        record["epoch"] = epoch
        record["validation_mse"] = float(val_mse)
        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        record["best_validation_mse"] = float(best_val_mse)
        record["best_epoch"] = float(best_epoch)
        history.append(record)
        print(
            "epoch={epoch} train_mse={training_mse:.6f} val_mse={validation_mse:.6f} "
            "best_val={best_validation_mse:.6f}@{best_epoch:.0f} "
            "gate_b={backbone_gate:.3f} gate_l={linear_field_gate:.3f} "
            "lin_conf={linear_component_confidence:.3f}".format(**record),
            flush=True,
        )
        if patience > 0 and stale_epochs >= patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_mse={best_val_mse:.6f}", flush=True)
            break
    model.load_state_dict({key: value.to(device=device) for key, value in best_state.items()})
    return history, {"best_epoch": float(best_epoch), "best_validation_mse": float(best_val_mse)}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    series: np.ndarray,
    starts: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    dataset = WindowDataset(series, starts, seq_len=seq_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=0)
    predictions = []
    targets = []
    model.eval()
    for inputs, batch_targets in loader:
        pred = model(inputs.to(device=device), None, None, None)
        predictions.append(pred.detach().cpu().numpy())
        targets.append(batch_targets.numpy())
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    error = prediction_array - target_array
    metrics = {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "smape": float(np.mean(2.0 * np.abs(error) / np.maximum(np.abs(prediction_array) + np.abs(target_array), 1e-8))),
        "num_eval_windows": float(len(starts)),
    }
    return metrics, prediction_array, target_array


def build_model(args: argparse.Namespace, horizon: int) -> AdaWarpMVPFPlusForecaster:
    return AdaWarpMVPFPlusForecaster(
        args.seq_len,
        horizon,
        patch_lens=args.patch_lens,
        width=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
        num_prototypes=args.num_prototypes,
        max_shift=args.max_shift,
        reconstruction_weight=args.reconstruction_weight,
        use_prototype_memory=args.use_prototype_memory,
        use_frequency_gate=args.use_frequency_gate,
        use_adaptive_shifts=args.use_adaptive_shifts,
        use_adaptive_radius=args.use_adaptive_radius,
        use_component_gate=args.use_component_gate,
        use_trend_decomposition=args.use_trend_decomposition,
        use_linear_field=args.use_linear_field,
    )


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = ensure_dir(args.output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    raw_dir = ensure_dir(output_root / "raw_predictions" / "ltsf_main5")
    audit_dir = ensure_dir(output_root / "audit")
    save_environment(audit_dir / "environment_ltsf_adawarp_mvpf_plus.json")

    rows = []
    for dataset_name in args.datasets:
        values = load_numeric_csv(dataset_path(Path(args.data_root), dataset_name))
        train_len, val_len, _ = split_lengths(dataset_name, len(values))
        train_end = train_len
        val_end = train_len + val_len
        normalized, _, _ = normalize_train(values, train_end)
        for horizon in args.horizons:
            train_window_starts = train_starts(train_end, args.seq_len, horizon, args.max_train_windows, args.seed)
            validation_window_starts = split_eval_starts(
                val_end,
                train_end,
                args.seq_len,
                horizon,
                args.max_validation_windows,
                args.seed + 5_000,
            )
            test_window_starts = split_eval_starts(
                len(normalized),
                val_end,
                args.seq_len,
                horizon,
                args.max_eval_windows,
                args.seed + 10_000,
            )
            for seed in args.seeds:
                set_seed(seed)
                model = build_model(args, horizon).to(device=device)
                history, selection = train_model(
                    model,
                    normalized,
                    train_window_starts,
                    validation_window_starts,
                    seq_len=args.seq_len,
                    pred_len=horizon,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    device=device,
                    seed=seed,
                    patience=args.patience,
                )
                metrics, predictions, targets = evaluate_model(
                    model,
                    normalized,
                    test_window_starts,
                    seq_len=args.seq_len,
                    pred_len=horizon,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                raw_path = raw_dir / f"AdaWarp-MVPFPlus_{dataset_name}_h{horizon}_seed{seed}.npz"
                np.savez_compressed(
                    raw_path,
                    prediction=predictions,
                    target=targets,
                    metadata_json=json.dumps(
                        {
                            "model": MODEL_NAME,
                            "dataset": dataset_name,
                            "horizon": horizon,
                            "seed": seed,
                            "seq_len": args.seq_len,
                            "patch_lens": list(args.patch_lens),
                            "train_history": history,
                            "checkpoint_selection": selection,
                            "normalization": "train_split_zscore",
                            "use_prototype_memory": bool(args.use_prototype_memory),
                            "use_frequency_gate": bool(args.use_frequency_gate),
                            "use_linear_field": bool(args.use_linear_field),
                            "validation_protocol": "prefix windows whose suffix lies in validation split only",
                        },
                        sort_keys=True,
                    ),
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "horizon": horizon,
                        "seq_len": args.seq_len,
                        "model": MODEL_NAME,
                        "mse": metrics["mse"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "smape": metrics["smape"],
                        "num_eval_windows": int(metrics["num_eval_windows"]),
                        "best_epoch": int(selection["best_epoch"]),
                        "best_validation_mse": selection["best_validation_mse"],
                        "raw_prediction_file": str(raw_path),
                    }
                )
                print(
                    f"mvpfplus-ltsf {MODEL_NAME:<13} {dataset_name:<11} h={horizon:<3} seed={seed} "
                    f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
                    f"best_epoch={int(selection['best_epoch'])} val={selection['best_validation_mse']:.6f}",
                    flush=True,
                )

    write_csv(
        metrics_dir / "ltsf_custom_neural_baselines.csv",
        rows,
        fieldnames=[
            "dataset",
            "seed",
            "horizon",
            "seq_len",
            "model",
            "mse",
            "mae",
            "rmse",
            "smape",
            "num_eval_windows",
            "best_epoch",
            "best_validation_mse",
            "raw_prediction_file",
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_FILES), default=["ETTh1", "ETTh2", "Weather", "Electricity", "Traffic"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patch-lens", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--reconstruction-weight", type=float, default=0.03)
    parser.add_argument("--use-prototype-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-frequency-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-adaptive-shifts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-adaptive-radius", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-component-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-trend-decomposition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-linear-field", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-windows", type=int, default=2048)
    parser.add_argument("--max-validation-windows", type=int, default=1024)
    parser.add_argument("--max-eval-windows", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--data-root", default="TSLibrary/dataset")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())