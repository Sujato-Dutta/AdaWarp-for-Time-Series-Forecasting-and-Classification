"""Seed-42 comparison runs and matched efficiency profiling.

The six retained competitors use the same data windows, train-only
normalization, validation-MSE checkpoint selection, and complete test windows
as finalized LatentFlow.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from physics_modal_v4.experiments.protocol import (
    DATASETS,
    HORIZONS,
    clone_state,
    evaluate,
    load_state,
    load_task,
    loader,
    parameter_count,
    resolve_device,
    seed_all,
    write_csv,
    write_json,
)
from physics_modal_v4.models.registry import (
    COMPARISON_MODELS,
    DLinearXCPD,
    ModelSpec,
    build_model,
)


def task_batch_size(args, dataset: str, requested: int) -> int:
    return min(requested, args.wide_batch_size) if dataset in {"Electricity", "Traffic"} else requested


def checkpoint_path(args) -> Path:
    return (
        Path(args.output_root)
        / "checkpoints"
        / args.model
        / "seed42"
        / args.dataset
        / f"h{args.horizon}.pt"
    )


def metric_path(args) -> Path:
    return (
        Path(args.output_root)
        / "headline"
        / "runs"
        / args.model
        / "seed42"
        / args.dataset
        / f"h{args.horizon}.json"
    )


def efficiency_path(args) -> Path:
    return (
        Path(args.output_root)
        / "efficiency"
        / "runs"
        / args.model
        / args.dataset
        / f"h{args.horizon}.json"
    )


def prediction_loss(spec: ModelSpec, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return spec.loss_fn(prediction, target) if spec.loss_fn is not None else F.mse_loss(prediction, target)


def regularizer(name: str, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if name == "VPNet":
        return 0.1 * model.auxiliary_loss(inputs)
    if isinstance(model, DLinearXCPD):
        return model.auxiliary_loss()
    return inputs.new_zeros(())


def scheduler_for(spec: ModelSpec, optimizer, steps_per_epoch: int):
    if spec.scheduler == "onecycle":
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=spec.learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=spec.epochs,
                pct_start=0.2,
            ),
            True,
        )
    if spec.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec.epochs), False
    return None, False


def fit_model(
    name: str,
    model: nn.Module,
    spec: ModelSpec,
    task,
    args,
    device: torch.device,
    *,
    epochs: int | None = None,
    patience: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    include_regularizer: bool = True,
) -> tuple[list[dict], dict, float]:
    epochs = spec.epochs if epochs is None else epochs
    patience = spec.patience if patience is None else patience
    batch_size = task_batch_size(args, task.dataset, spec.batch_size)
    eval_batch = task_batch_size(args, task.dataset, spec.eval_batch_size)
    training = loader(
        task,
        task.training_starts,
        args.seq_len,
        batch_size,
        shuffle=True,
        seed=42,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=spec.learning_rate if learning_rate is None else learning_rate,
        weight_decay=spec.weight_decay if weight_decay is None else weight_decay,
    )
    schedule, schedule_each_batch = scheduler_for(spec, optimizer, len(training))
    best_state = clone_state(model)
    best_mse = float("inf")
    best_epoch = 0
    history: list[dict] = []
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in training:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = prediction_loss(spec, prediction, targets)
            if include_regularizer:
                loss = loss + regularizer(name, model, inputs)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {name} loss for {task.dataset}, h={task.horizon}, epoch={epoch}."
                )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.gradient_clip,
            )
            optimizer.step()
            if schedule is not None and schedule_each_batch:
                schedule.step()
            total_loss += float(loss.detach())
            optimizer_steps += 1
        if schedule is not None and not schedule_each_batch:
            schedule.step()
        validation = evaluate(
            model,
            task,
            task.validation_starts,
            seq_len=args.seq_len,
            batch_size=eval_batch,
            device=device,
        )
        if validation["mse"] < best_mse:
            best_mse = validation["mse"]
            best_epoch = epoch
            best_state = clone_state(model)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(training)),
                "validation_mse": validation["mse"],
                "validation_mae": validation["mae"],
                "best_validation_mse": best_mse,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"baseline epoch={epoch}/{epochs} model={name} dataset={task.dataset} "
            f"h={task.horizon} train={history[-1]['train_loss']:.6f} "
            f"val={validation['mse']:.6f} best={best_mse:.6f}@{best_epoch}",
            flush=True,
        )
        if patience > 0 and epoch - best_epoch >= patience:
            break
    elapsed_ms = 1000.0 * (time.perf_counter() - started) / max(1, optimizer_steps)
    load_state(model, best_state, device)
    return history, {"best_epoch": best_epoch, "best_validation_mse": best_mse}, elapsed_ms


@torch.no_grad()
def fit_xcpd_basis(model: DLinearXCPD, task, args, device: torch.device) -> None:
    maximum = 8 if task.dataset == "Traffic" else (16 if task.dataset == "Electricity" else 64)
    forecasts = []
    for inputs, _ in loader(
        task,
        task.training_starts[:maximum],
        args.seq_len,
        task_batch_size(args, task.dataset, args.eval_batch_size),
    ):
        forecasts.append(model.backbone(inputs.to(device)).detach())
    model.plugin.fit_shared_basis(torch.cat(forecasts, dim=0))


def train_comparison(args):
    if args.seed != 42:
        raise ValueError("Retained comparison models are evaluated at seed 42 only.")
    output = metric_path(args)
    checkpoint = checkpoint_path(args)
    if output.exists() and checkpoint.exists() and not args.force:
        print(f"skip-complete model={args.model} dataset={args.dataset} h={args.horizon}")
        return
    device = resolve_device(args.device)
    task = load_task(
        Path(args.data_root),
        args.dataset,
        args.horizon,
        seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )
    seed_all(42)
    spec = build_model(args.model, args.dataset, args.seq_len, args.horizon, task.values.shape[1])
    model = spec.model.to(device)
    histories: dict[str, list[dict]] = {}
    if isinstance(model, DLinearXCPD):
        backbone_spec = ModelSpec(
            model.backbone,
            10,
            3,
            spec.batch_size,
            spec.eval_batch_size,
            1e-4,
            weight_decay=1e-4,
        )
        base_history, _, _ = fit_model(
            "DLinear",
            model.backbone,
            backbone_spec,
            task,
            args,
            device,
            include_regularizer=False,
        )
        histories["backbone"] = base_history
        model.freeze_backbone()
        fit_xcpd_basis(model, task, args, device)
    history, selection, train_ms = fit_model(args.model, model, spec, task, args, device)
    histories["model"] = history
    eval_batch = task_batch_size(args, args.dataset, spec.eval_batch_size)
    metrics = evaluate(
        model,
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=eval_batch,
        device=device,
        synchronize=True,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": clone_state(model),
            "model": args.model,
            "dataset": args.dataset,
            "horizon": args.horizon,
            "seed": 42,
            "channels": task.values.shape[1],
            "selection": selection,
            "provenance": spec.provenance,
            "protocol": (
                "input=96; train-only normalization; 2048 train and 1024 validation "
                "windows; validation-MSE checkpoint; all test windows"
            ),
        },
        checkpoint,
    )
    row = {
        "model": args.model,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": 42,
        **metrics,
        **selection,
        "parameters": parameter_count(model),
        "training_ms_optimizer_step": train_ms,
        "checkpoint": str(checkpoint),
        "provenance": spec.provenance,
        "selection_protocol": "validation MSE only; test evaluated once",
    }
    write_json(output, row)
    write_csv(
        Path(args.output_root)
        / "headline"
        / "history"
        / args.model
        / "seed42"
        / args.dataset
        / f"h{args.horizon}.csv",
        [dict(stage=stage, **entry) for stage, entries in histories.items() for entry in entries],
    )
    print(
        f"baseline-result model={args.model:<12} dataset={args.dataset:<11} "
        f"h={args.horizon:<3} seed=42 mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
        flush=True,
    )


def load_comparison(args, device: torch.device):
    checkpoint = checkpoint_path(args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing seed-42 checkpoint {checkpoint}; run baseline first.")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    task = load_task(Path(args.data_root), args.dataset, args.horizon, seq_len=args.seq_len)
    spec = build_model(args.model, args.dataset, args.seq_len, args.horizon, task.values.shape[1])
    model = spec.model.to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if isinstance(model, DLinearXCPD):
        model.freeze_backbone()
    return model, spec, task, checkpoint


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profiler_flops(model: nn.Module, inputs: torch.Tensor) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if inputs.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            model(inputs)
        return float(sum(event.flops or 0 for event in profile.key_averages()))
    except Exception:
        return float("nan")


def profile_comparison(args) -> None:
    if args.seed != 42:
        raise ValueError("Efficiency profiling uses seed 42 only.")
    output = efficiency_path(args)
    if output.exists() and not args.force:
        print(f"skip-complete efficiency model={args.model} dataset={args.dataset} h={args.horizon}")
        return
    device = resolve_device(args.device)
    model, spec, task, checkpoint = load_comparison(args, device)
    batch_size = min(args.profile_batch_size, len(task.test_starts))
    inputs, targets = next(iter(loader(task, task.test_starts[:batch_size], args.seq_len, batch_size)))
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    inference_times = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(args.profile_repeats + 2):
        synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            model(inputs)
        synchronize(device)
        if iteration >= 2:
            inference_times.append(1000.0 * (time.perf_counter() - started))
    peak_inference = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    flops = profiler_flops(model, inputs[:1])

    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    training_times = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(args.profile_repeats + 2):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        prediction = model(inputs)
        loss = prediction_loss(spec, prediction, targets) + regularizer(args.model, model, inputs)
        loss.backward()
        optimizer.step()
        synchronize(device)
        if iteration >= 2:
            training_times.append(1000.0 * (time.perf_counter() - started))
    peak_training = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    row = {
        "model": args.model,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": 42,
        "parameters_total": parameter_count(model),
        "parameters_trainable": parameter_count(model, trainable_only=True),
        "flops_per_sample": flops,
        "gflops_per_sample": flops / 1e9,
        "profile_batch_size": batch_size,
        "inference_ms_batch": sum(inference_times) / len(inference_times),
        "training_ms_iter": sum(training_times) / len(training_times),
        "peak_inference_gpu_mb": peak_inference / (1024**2),
        "peak_training_gpu_mb": peak_training / (1024**2),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "checkpoint": str(checkpoint),
        "flops_note": "torch.profiler count; unsupported custom operations may be omitted",
    }
    write_json(output, row)
    print(
        f"baseline-efficiency model={args.model:<12} dataset={args.dataset:<11} "
        f"h={args.horizon:<3} params={row['parameters_total']} "
        f"gflops={row['gflops_per_sample']:.4f} infer_ms={row['inference_ms_batch']:.3f}",
        flush=True,
    )


def smoke(args) -> None:
    device = resolve_device(args.device)
    inputs = torch.randn(2, 96, 7, device=device)
    for name in COMPARISON_MODELS:
        seed_all(42)
        spec = build_model(name, "ETTh1", 96, 96, 7)
        model = spec.model.to(device)
        prediction = model(inputs)
        if prediction.shape != (2, 96, 7) or not torch.isfinite(prediction).all():
            raise RuntimeError(f"Comparison smoke failed for {name}: {tuple(prediction.shape)}")
        print(f"baseline-smoke passed model={name} shape={tuple(prediction.shape)}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "efficiency"), required=True)
    parser.add_argument("--model", choices=COMPARISON_MODELS, default="DLinear")
    parser.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    parser.add_argument("--horizon", choices=HORIZONS, type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-train-windows", type=int, default=2048)
    parser.add_argument("--max-validation-windows", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--wide-batch-size", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--data-root", default="TSLibrary/dataset")
    parser.add_argument("--output-root", default="physics_modal_v4/results/final")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = parser().parse_args()
    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "train":
        train_comparison(args)
    else:
        profile_comparison(args)


if __name__ == "__main__":
    main()


