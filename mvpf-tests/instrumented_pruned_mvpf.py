"""Train and audit a validation-pruned MVPF candidate.

This is a candidate experiment, not a replacement for the reported MVPF model.
It removes trend decomposition because the decision was made from aggregate
validation performance, and disables adaptive shifts because prototype memory
is already disabled in the final MVPF configuration.  Every task writes its
checkpoint and metrics immediately so a wall-time interruption preserves work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adawarp_mvpf_plus import AdaWarpMVPFPlusForecaster
from benchmark_adawarp_mvpf_plus_ltsf import (
    WindowDataset,
    dataset_path,
    load_numeric_csv,
    normalize_train,
    resolve_device,
    set_seed,
    split_eval_starts,
    split_lengths,
    train_model,
    train_starts,
)


DATASET_ORDER = ["ETTh1", "ETTm2", "Weather", "ETTh2", "Electricity", "Traffic"]
COMPONENT_NAMES = ["direct", "scale_8", "scale_16", "scale_32"]
LINEAR_NAMES = ["nlinear", "dlinear", "analytic", "persistence"]


def model_config(args: argparse.Namespace, horizon: int) -> dict:
    return {
        "seq_len": args.seq_len,
        "pred_len": horizon,
        "patch_lens": list(args.patch_lens),
        "width": args.d_model,
        "depth": args.depth,
        "dropout": args.dropout,
        "num_prototypes": args.num_prototypes,
        "max_shift": args.max_shift,
        "reconstruction_weight": args.reconstruction_weight,
        "use_prototype_memory": False,
        "use_frequency_gate": False,
        "use_adaptive_shifts": False,
        "use_adaptive_radius": True,
        "use_component_gate": True,
        "use_trend_decomposition": False,
        "use_linear_field": True,
    }


def build_model(args: argparse.Namespace, horizon: int) -> AdaWarpMVPFPlusForecaster:
    config = model_config(args, horizon)
    seq_len = config.pop("seq_len")
    pred_len = config.pop("pred_len")
    return AdaWarpMVPFPlusForecaster(seq_len, pred_len, **config)


def linear_components(head, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last = values[:, -1:, :]
    centered = values - last
    nlinear = head._linear_per_channel(head.center_head, centered) + last
    trend = head._moving_average(values)
    dlinear = head._linear_per_channel(head.trend_head, trend) + head._linear_per_channel(
        head.residual_head, values - trend
    )
    analytic, fit_error = head._analytic_trend(values)
    persistence = last.expand(-1, head.pred_len, -1)
    components = torch.stack([nlinear, dlinear, analytic, persistence], dim=1)
    weights = torch.softmax(head.gate(head._features(values, fit_error)), dim=-1)
    return components, weights, fit_error


@torch.no_grad()
def forward_details(model: AdaWarpMVPFPlusForecaster, x: torch.Tensor) -> dict[str, torch.Tensor]:
    backbone = model.backbone
    normalized, mean, std = backbone._normalize(x)
    trend, residual = backbone._decompose(normalized)
    trend_forecast = normalized.new_zeros(x.shape[0], model.pred_len, x.shape[2])
    direct = backbone._linear_per_channel(backbone.direct_residual_head, residual)
    branch = [module(residual) for module in backbone.branches]
    components_norm = torch.stack([direct, *branch], dim=1)
    component_weights = torch.softmax(backbone.component_gate(backbone._summary_features(normalized, residual)), dim=-1)
    strength = torch.sigmoid(backbone.residual_strength)
    field_norm = trend_forecast + strength * (
        components_norm * component_weights[:, :, None, None]
    ).sum(dim=1)

    linear_norm_components, linear_weights, fit_error = linear_components(model.linear_field, normalized)
    linear_norm = (linear_norm_components * linear_weights[:, :, None, None]).sum(dim=1)
    linear_confidence = linear_weights.max(dim=1, keepdim=True).values
    features = model._summary_features(normalized, residual, linear_confidence, fit_error)
    outer_weights = torch.softmax(model.mean_mixer(features), dim=-1)
    learned_norm = outer_weights[:, 0, None, None] * field_norm + outer_weights[:, 1, None, None] * linear_norm

    return {
        "target_mean": mean,
        "target_std": std,
        "field": field_norm * std + mean,
        "linear": linear_norm * std + mean,
        "learned": learned_norm * std + mean,
        "components": (trend_forecast[:, None] + strength * components_norm) * std[:, None] + mean[:, None],
        "component_weights": component_weights,
        "linear_weights": linear_weights,
        "outer_weights": outer_weights,
        "features": features,
    }


def loader(series: np.ndarray, starts: np.ndarray, seq_len: int, horizon: int, batch_size: int) -> DataLoader:
    dataset = WindowDataset(series, starts, seq_len=seq_len, pred_len=horizon)
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=0)


def rank_values(values: np.ndarray) -> np.ndarray:
    """Average ascending ranks, including exact ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def project_simplex(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1)
    valid = ordered - cssv / indices > 0
    rho = indices[valid][-1]
    theta = cssv[rho - 1] / rho
    return np.maximum(values - theta, 0.0)


@torch.no_grad()
def fit_validation_controls(model, series, starts, args, horizon, device) -> tuple[float, np.ndarray]:
    ff = fl = ll = fy = ly = 0.0
    gram = np.zeros((4, 4), dtype=np.float64)
    rhs = np.zeros(4, dtype=np.float64)
    for inputs, targets in loader(series, starts, args.seq_len, horizon, args.eval_batch_size):
        details = forward_details(model, inputs.to(device))
        y = targets.to(device)
        f, l = details["field"], details["linear"]
        ff += float((f * f).sum().cpu()); fl += float((f * l).sum().cpu()); ll += float((l * l).sum().cpu())
        fy += float((f * y).sum().cpu()); ly += float((l * y).sum().cpu())
        c = details["components"].permute(0, 2, 3, 1).reshape(-1, 4).double().cpu().numpy()
        yy = y.reshape(-1).double().cpu().numpy()
        gram += c.T @ c; rhs += c.T @ yy
    denom = ff - 2.0 * fl + ll
    beta_field = 0.5 if abs(denom) < 1e-12 else float(np.clip((fy - ly - fl + ll) / denom, 0.0, 1.0))
    weights = np.full(4, 0.25, dtype=np.float64)
    lipschitz = max(1e-9, 2.0 * float(np.linalg.eigvalsh(gram).max()))
    for _ in range(1000):
        updated = project_simplex(weights - (2.0 * (gram @ weights - rhs)) / lipschitz)
        if np.max(np.abs(updated - weights)) < 1e-10:
            weights = updated; break
        weights = updated
    return beta_field, weights


def update_metric(state: dict, name: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
    error = prediction - target
    item = state.setdefault(name, [0.0, 0.0, 0])
    item[0] += float(error.pow(2).sum().cpu())
    item[1] += float(error.abs().sum().cpu())
    item[2] += int(error.numel())


@torch.no_grad()
def evaluate(model, series, starts, args, horizon, device, beta_field, component_weights, window_path: Path) -> tuple[dict, list[dict]]:
    totals: dict[str, list[float]] = {}
    rows: list[dict] = []
    model.eval()
    window_index = 0
    for inputs, targets in loader(series, starts, args.seq_len, horizon, args.eval_batch_size):
        details = forward_details(model, inputs.to(device))
        y = targets.to(device)
        f, l, c = details["field"], details["linear"], details["components"]
        learned = details["learned"]
        equal_outer = 0.5 * (f + l)
        fixed_outer = beta_field * f + (1.0 - beta_field) * l
        diff = f - l
        oracle_beta = (((y - l) * diff).sum(dim=(1, 2)) / diff.pow(2).sum(dim=(1, 2)).clamp_min(1e-12)).clamp(0.0, 1.0)
        oracle = oracle_beta[:, None, None] * f + (1.0 - oracle_beta[:, None, None]) * l
        equal_components = c.mean(dim=1)
        fixed_components = (c * torch.as_tensor(component_weights, device=device, dtype=c.dtype)[None, :, None, None]).sum(dim=1)
        predictions = {
            "learned": learned, "field_only": f, "linear_only": l,
            "equal_outer": equal_outer, "validation_outer": fixed_outer,
            "oracle_outer": oracle, "equal_components": equal_components,
            "validation_components": fixed_components,
        }
        for name, prediction in predictions.items():
            update_metric(totals, name, prediction, y)
        for index in range(y.shape[0]):
            field_mse = float((f[index] - y[index]).pow(2).mean().cpu())
            linear_mse = float((l[index] - y[index]).pow(2).mean().cpu())
            row = {
                "window_index": window_index,
                "start": int(starts[window_index]),
                "learned_field_weight": float(details["outer_weights"][index, 0].cpu()),
                "oracle_field_weight": float(oracle_beta[index].cpu()),
                "field_mse": field_mse,
                "linear_mse": linear_mse,
                "linear_minus_field_mse": linear_mse - field_mse,
                "learned_mse": float((learned[index] - y[index]).pow(2).mean().cpu()),
            }
            for j, name in enumerate(["slope", "volatility", "roughness", "level", "linear_confidence", "fit_error"]):
                row[name] = float(details["features"][index, j].cpu())
            for j, name in enumerate(COMPONENT_NAMES):
                row[f"component_weight_{name}"] = float(details["component_weights"][index, j].cpu())
            for j, name in enumerate(LINEAR_NAMES):
                row[f"linear_weight_{name}"] = float(details["linear_weights"][index, j].cpu())
            rows.append(row); window_index += 1
    window_path.parent.mkdir(parents=True, exist_ok=True)
    with window_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = {}
    for name, (sse, sae, count) in totals.items():
        metrics[f"{name}_mse"] = sse / count
        metrics[f"{name}_mae"] = sae / count
    gate = np.asarray([row["learned_field_weight"] for row in rows])
    advantage = np.asarray([row["linear_minus_field_mse"] for row in rows])
    metrics["gate_advantage_spearman"] = float(np.corrcoef(rank_values(gate), rank_values(advantage))[0, 1]) if np.std(gate) > 0 and np.std(advantage) > 0 else float("nan")
    metrics["mean_learned_field_weight"] = float(gate.mean())
    metrics["mean_oracle_field_weight"] = float(np.mean([row["oracle_field_weight"] for row in rows]))
    return metrics, rows


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    output = Path(args.output_root)
    summary_path = output / "metrics" / "pruned_instrumented_summary.csv"
    completed = {(row["dataset"], int(row["horizon"])) for row in read_rows(summary_path)}
    for dataset_name in args.datasets:
        values = load_numeric_csv(dataset_path(Path(args.data_root), dataset_name))
        train_len, val_len, _ = split_lengths(dataset_name, len(values))
        val_end = train_len + val_len
        normalized, _, _ = normalize_train(values, train_len)
        for horizon in args.horizons:
            if (dataset_name, horizon) in completed and not args.force:
                print(f"skip-complete dataset={dataset_name} horizon={horizon}", flush=True); continue
            started = time.perf_counter()
            try:
                set_seed(args.seed)
                tr = train_starts(train_len, args.seq_len, horizon, args.max_train_windows, args.seed)
                va = split_eval_starts(val_end, train_len, args.seq_len, horizon, args.max_validation_windows, args.seed + 5000)
                te = split_eval_starts(len(normalized), val_end, args.seq_len, horizon, args.max_eval_windows, args.seed + 10000)
                model = build_model(args, horizon).to(device)
                history, selection = train_model(
                    model, normalized, tr, va, seq_len=args.seq_len, pred_len=horizon,
                    epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay, device=device,
                    seed=args.seed, patience=args.patience,
                )
                beta_field, component_weights = fit_validation_controls(model, normalized, va, args, horizon, device)
                metrics, _ = evaluate(
                    model, normalized, te, args, horizon, device, beta_field, component_weights,
                    output / "per_window" / f"{dataset_name}_h{horizon}_seed{args.seed}.csv",
                )
                checkpoint = output / "checkpoints" / f"{dataset_name}_h{horizon}_seed{args.seed}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": model_config(args, horizon), "selection": selection, "history": history,
                    "validation_field_weight": beta_field,
                    "validation_component_weights": component_weights.tolist(),
                }, checkpoint)
                row = {
                    "dataset": dataset_name, "horizon": horizon, "seed": args.seed,
                    "best_epoch": int(selection["best_epoch"]),
                    "best_validation_mse": selection["best_validation_mse"],
                    "validation_field_weight": beta_field,
                    "validation_component_weights": json.dumps(component_weights.tolist()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint": str(checkpoint), **metrics,
                }
                rows = [r for r in read_rows(summary_path) if not (r["dataset"] == dataset_name and int(r["horizon"]) == horizon)]
                rows.append(row); rows.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), int(r["horizon"])))
                write_rows(summary_path, rows)
                print(
                    f"pruned-mvpf dataset={dataset_name:<11} h={horizon:<3} "
                    f"mse={metrics['learned_mse']:.6f} mae={metrics['learned_mae']:.6f} "
                    f"best_epoch={int(selection['best_epoch'])} val={selection['best_validation_mse']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"task-failed dataset={dataset_name} horizon={horizon}: {type(exc).__name__}: {exc}", flush=True)
                if args.fail_fast:
                    raise


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DATASET_ORDER)
    p.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patch-lens", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--num-prototypes", type=int, default=8)
    p.add_argument("--max-shift", type=int, default=2)
    p.add_argument("--reconstruction-weight", type=float, default=0.03)
    p.add_argument("--max-train-windows", type=int, default=2048)
    p.add_argument("--max-validation-windows", type=int, default=1024)
    p.add_argument("--max-eval-windows", type=int, default=2048)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--data-root", default="TSLibrary/dataset")
    p.add_argument("--output-root", default="mvpf-tests/results/pruned_instrumented")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
