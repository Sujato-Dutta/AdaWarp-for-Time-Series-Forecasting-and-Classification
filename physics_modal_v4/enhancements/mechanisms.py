"""Synthetic process recovery and real-channel scaling for LatentFlow."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from benchmark_adawarp_mvpf_plus_ltsf import train_model as train_mvpf
from physics_modal_v4.enhancements.run import (
    MeanAdapter,
    SELECTED_CONFIG,
    _fit_seeded_model,
    assert_no_exchange,
    build_latentflow,
    experiment_args,
    stable_hash,
    task_batch_size,
)
from physics_modal_v4.experiments.protocol import (
    TaskData,
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
from physics_modal_v4.experiments.run import profiler_flops
from physics_modal_v4.experiments.training import (
    backbone_config,
    fit_final_model,
    instantiate_backbone,
    torch_load,
)
from physics_modal_v4.model import PROCESS_SPECS, SENSOR_GROUP_SPECS
from physics_modal_v4.models.registry import build_model


ROOT = Path("physics_modal_v4/enhancements/results")
SYNTHETIC_PROTOCOL = "latentflow-synthetic-mechanism-seven-settings-five-seed-no-exchange-v1"
SCALING_PROTOCOL = "latentflow-real-traffic-channel-scaling-seed42-v1"
SCENARIOS = (
    "reference",
    "interacting",
    "period_drift",
    "trend_curvature",
    "long_ou",
    "abrupt_noisy",
    "long_horizon",
)
SYNTHETIC_VARIANTS = ("early_fusion", "independent_branches", "process_preserving")
CHANNEL_COUNTS = (32, 64, 128, 256, 512, 862)


def register_synthetic_domain() -> None:
    PROCESS_SPECS["SyntheticRevision"] = (
        ("periodic", "periodic", 1.0, 0.65),
        ("smooth", "matern", 0.0, 0.75),
        ("innovation", "ou", 0.0, 0.20),
    )
    SENSOR_GROUP_SPECS["SyntheticRevision"] = {
        "names": ("periodic", "smooth", "innovation"),
        "process_prior": (
            (0.90, 0.05, 0.05),
            (0.05, 0.90, 0.05),
            (0.05, 0.05, 0.90),
        ),
        "group_prior": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    }


def matern_process(rng: np.random.Generator, length: int, channels: int, ell: float) -> np.ndarray:
    frequencies = np.fft.rfftfreq(length)
    spectrum = (ell ** -2 + (2.0 * np.pi * frequencies) ** 2) ** -2
    spectrum /= spectrum.max()
    white = rng.normal(size=(length, channels))
    values = np.fft.irfft(
        np.fft.rfft(white, axis=0) * np.sqrt(spectrum)[:, None],
        n=length,
        axis=0,
    )
    return values / values.std(axis=0, keepdims=True).clip(1e-8)


def ou_process(rng: np.random.Generator, length: int, channels: int, rho: float) -> np.ndarray:
    values = np.zeros((length, channels), dtype=np.float64)
    noise = rng.normal(size=values.shape) * math.sqrt(1.0 - rho * rho)
    for index in range(1, length):
        values[index] = rho * values[index - 1] + noise[index]
    return values


def oracle_design_times(family: str, truth: dict, count: int, length: int = 96) -> np.ndarray:
    """Greedy variance-reduction design under the known generating covariance."""

    points = np.arange(length, dtype=np.float64)
    distance = np.abs(points[:, None] - points[None, :])
    if family == "periodic":
        covariance = np.exp(
            -2.0 * np.sin(np.pi * distance / truth["period_samples"]) ** 2 / 0.65 ** 2
        )
    elif family == "matern":
        scaled = np.sqrt(3.0) * distance / truth["matern_length_samples"]
        covariance = (1.0 + scaled) * np.exp(-scaled)
    elif family == "ou":
        covariance = np.exp(-distance / truth["ou_length_samples"])
    else:
        raise ValueError(family)
    selected: list[int] = []
    residual = np.diag(covariance).copy()
    for _ in range(count):
        index = int(np.argmax(residual))
        selected.append(index)
        kuu = covariance[np.ix_(selected, selected)] + 1e-8 * np.eye(len(selected))
        cross = covariance[:, selected]
        residual = np.diag(covariance) - np.sum(
            cross * np.linalg.solve(kuu, cross.T).T, axis=1
        )
        residual[selected] = -np.inf
    return np.sort(np.asarray(selected, dtype=np.float64))


def chamfer_distance(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.abs(left[:, None] - right[None, :])
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def synthetic_task(seed: int, scenario: str) -> tuple[TaskData, np.ndarray, dict]:
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    length, channels, seq_len = 8192, 8, 96
    horizon = 720 if scenario == "long_horizon" else 96
    time_index = np.arange(length, dtype=np.float64)
    phases = rng.uniform(-np.pi, np.pi, size=(1, channels))
    phase_curve = 2.0 * np.pi * time_index[:, None] / 24.0 + phases
    if scenario == "period_drift":
        phase_curve += 2.0 * np.pi * 0.12 * (time_index[:, None] / length) ** 2 * time_index[:, None] / 24.0
    periodic = 0.65 * np.sin(phase_curve)
    periodic += 0.25 * np.cos(2.0 * np.pi * time_index[:, None] / 168.0 + phases / 2.0)

    ell = 28.0 if scenario == "trend_curvature" else 18.0
    smooth = 0.55 * matern_process(rng, length, channels, ell)
    slope = rng.normal(0.0, 0.12, size=(1, channels))
    centered_time = (time_index[:, None] - length / 2.0) / length
    smooth += slope * centered_time
    if scenario == "trend_curvature":
        smooth += rng.normal(0.0, 0.18, size=(1, channels)) * centered_time.square()

    rho = 0.95 if scenario == "long_ou" else 0.82
    innovation = 0.32 * ou_process(rng, length, channels, rho)
    if scenario == "abrupt_noisy":
        for _ in range(20):
            start = int(rng.integers(64, length - 64))
            duration = int(rng.integers(4, 24))
            innovation[start : start + duration] += rng.normal(0.0, 0.8, size=(1, channels))
        innovation += rng.normal(0.0, 0.16, size=innovation.shape)
    if scenario == "interacting":
        smooth[3:] += 0.22 * periodic[:-3]
        innovation[5:] += 0.20 * np.tanh(smooth[:-5])

    components = np.stack([periodic, smooth, innovation], axis=1)
    measurement_noise = 0.05 if scenario != "abrupt_noisy" else 0.12
    total = components.sum(axis=1) + measurement_noise * rng.normal(size=(length, channels))
    train_end, validation_end = 4915, 6554
    mean = total[:train_end].mean(axis=0, keepdims=True)
    std = total[:train_end].std(axis=0, keepdims=True).clip(1e-6)
    values = ((total - mean) / std).astype(np.float32)
    normalized_components = (components / std[:, None, :]).astype(np.float32)
    normalized_components[:, 1, :] -= mean / std
    training = np.arange(0, train_end - seq_len - horizon + 1, dtype=np.int64)
    validation = np.arange(train_end - seq_len, validation_end - seq_len - horizon + 1, dtype=np.int64)
    testing = np.arange(validation_end - seq_len, length - seq_len - horizon + 1, dtype=np.int64)
    sampler = np.random.default_rng(42)
    training = np.sort(sampler.choice(training, min(2048, len(training)), replace=False))
    validation = np.sort(sampler.choice(validation, min(1024, len(validation)), replace=False))
    truth = {
        "period_samples": 24.0,
        "matern_length_samples": ell,
        "ou_rho": rho,
        "ou_length_samples": -1.0 / math.log(rho),
        "interacting": scenario == "interacting",
    }
    return (
        TaskData("SyntheticRevision", horizon, values, training, validation, testing),
        normalized_components,
        truth,
    )


class UniformRouter(nn.Module):
    def __init__(self, processes: int) -> None:
        super().__init__()
        self.processes = int(processes)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return context.new_zeros((context.shape[0], self.processes))


def synthetic_model(base: nn.Module, task: TaskData, variant: str) -> nn.Module:
    mode = "learned_inducing_elbo" if variant == "early_fusion" else "final"
    model = build_latentflow(
        base,
        task,
        {"latent_dim": 8, "inducing_points": 16},
        mode=mode,
        unfreeze_blocks=2,
    )
    assert_no_exchange(model)
    if variant == "independent_branches":
        model.continuation.process_router = UniformRouter(model.separator.num_processes)
    return model


def synthetic_backbone(args, task: TaskData, device: torch.device, scenario: str) -> dict:
    path = Path(args.output_root) / "synthetic" / "backbones" / scenario / f"seed{args.seed}.pt"
    config = backbone_config(96, task.horizon, (8, 16, 32))
    if path.exists() and not args.force:
        payload = torch_load(path)
        if payload.get("config") == config:
            return payload
    seed_all(args.seed)
    model = instantiate_backbone(config).to(device)
    history, selection = train_mvpf(
        model,
        task.values,
        task.training_starts,
        task.validation_starts,
        seq_len=96,
        pred_len=task.horizon,
        epochs=10,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=7e-4,
        weight_decay=1e-4,
        device=device,
        seed=args.seed,
        patience=0,
    )
    payload = {"state_dict": clone_state(model), "config": config, "selection": selection, "history": history}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


@torch.no_grad()
def collect_components(
    model: nn.Module,
    task: TaskData,
    truth: np.ndarray,
    starts: np.ndarray,
    device: torch.device,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(starts)[:maximum]
    values = torch.as_tensor(task.values, dtype=torch.float32)
    predicted, actual = [], []
    for offset in range(0, len(selected), 16):
        current = selected[offset : offset + 16]
        inputs = torch.stack([values[start : start + 96] for start in current]).to(device)
        backbone = model.continuation.mvpff.backbone
        normalized, local_mean, local_std = backbone._normalize(inputs)
        structural = model.encoder(normalized)
        posterior = model.separator(structural, inputs)
        decoded = model.encoder.decode(posterior["latent_mean"])
        target = torch.as_tensor(
            np.stack([truth[start : start + 96] for start in current]),
            device=device,
        ) / local_std[:, :, None, :]
        target[:, :, 1] -= local_mean / local_std
        predicted.append(decoded.cpu().numpy())
        actual.append(target.cpu().numpy())
    return np.concatenate(predicted), np.concatenate(actual)


def component_recovery(model, task, components, device) -> dict:
    validation_prediction, validation_truth = collect_components(
        model, task, components, task.validation_starts, device, 128
    )
    test_prediction, test_truth = collect_components(
        model, task, components, task.test_starts, device, 256
    )
    processes = validation_truth.shape[2]
    matrix = np.zeros((processes, processes), dtype=np.float64)
    for predicted in range(processes):
        for actual in range(processes):
            left = validation_prediction[:, :, predicted].reshape(-1)
            right = validation_truth[:, :, actual].reshape(-1)
            matrix[predicted, actual] = abs(np.corrcoef(left, right)[0, 1])
    permutation = max(itertools.permutations(range(processes)), key=lambda order: sum(matrix[index, order[index]] for index in range(processes)))
    rows = []
    for predicted, actual in enumerate(permutation):
        validation_left = validation_prediction[:, :, predicted].reshape(-1)
        validation_right = validation_truth[:, :, actual].reshape(-1)
        calibration = float(np.dot(validation_left, validation_right) / max(np.dot(validation_left, validation_left), 1e-8))
        left = calibration * test_prediction[:, :, predicted].reshape(-1)
        right = test_truth[:, :, actual].reshape(-1)
        rows.append(
            {
                "predicted_process": predicted,
                "true_process": actual,
                "correlation": float(np.corrcoef(left, right)[0, 1]),
                "nrmse": float(np.sqrt(np.mean((left - right) ** 2)) / max(np.std(right), 1e-8)),
            }
        )
    return {
        "assignment": rows,
        "component_correlation_mean": float(np.mean([abs(row["correlation"]) for row in rows])),
        "component_nrmse_mean": float(np.mean([row["nrmse"] for row in rows])),
        "assignment_validation_matrix": matrix.tolist(),
    }


def run_synthetic(args) -> None:
    register_synthetic_domain()
    task, components, truth = synthetic_task(args.seed, args.scenario)
    train_args = argparse.Namespace(**vars(args))
    train_args.dataset = task.dataset
    train_args.horizon = task.horizon
    device = resolve_device(args.device)
    payload = synthetic_backbone(args, task, device, args.scenario)
    for variant in SYNTHETIC_VARIANTS:
        output = Path(args.output_root) / "synthetic" / "runs" / args.scenario / variant / f"seed{args.seed}.json"
        if output.exists() and not args.force:
            continue
        base = instantiate_backbone(payload["config"])
        base.load_state_dict(payload["state_dict"], strict=True)
        seed_all(args.seed)
        model = synthetic_model(base, task, variant).to(device)
        history, selection = fit_final_model(
            model,
            task,
            experiment_args(train_args),
            representation_weight=0.01,
            nll_weight=0.02,
            batch_size=task_batch_size(args, "ETTh1"),
            device=device,
        )
        metrics = evaluate(MeanAdapter(model), task, task.test_starts, seq_len=96, batch_size=16, device=device, synchronize=True)
        recovery = component_recovery(model, task, components, device)
        learned = []
        for name, kernel in zip(model.separator.process_names, model.separator.kernels):
            physical = kernel.parameters_physical()
            inducing = kernel.inducing_times().detach().cpu().numpy()
            oracle = oracle_design_times(kernel.family, truth, len(inducing))
            learned_length = float(physical["length"].detach())
            expected_length = (
                truth["matern_length_samples"]
                if kernel.family == "matern"
                else truth["ou_length_samples"] if kernel.family == "ou" else None
            )
            learned.append(
                {
                    "name": name,
                    "family": kernel.family,
                    "length_samples": learned_length,
                    "period_samples": float(physical["period"].detach()) if "period" in physical else None,
                    "period_relative_error": (
                        abs(float(physical["period"].detach()) - truth["period_samples"])
                        / truth["period_samples"]
                        if "period" in physical
                        else None
                    ),
                    "length_relative_error": (
                        abs(learned_length - expected_length) / expected_length
                        if expected_length is not None
                        else None
                    ),
                    "inducing_times": inducing.tolist(),
                    "oracle_design_times": oracle.tolist(),
                    "inducing_design_chamfer_samples": chamfer_distance(inducing, oracle),
                }
            )
        checkpoint = Path(args.output_root) / "synthetic" / "checkpoints" / args.scenario / variant / f"seed{args.seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": clone_state(model), "base": payload, "selection": selection}, checkpoint)
        row = {
            "experiment": "synthetic",
            "model": "LatentFlow",
            "dataset": "SyntheticRevision",
            "enhancement_protocol": SYNTHETIC_PROTOCOL,
            "scenario": args.scenario,
            "variant": variant,
            "seed": args.seed,
            "horizon": task.horizon,
            "cross_process_exchange": False,
            "ground_truth": truth,
            "learned_processes": learned,
            "checkpoint": str(checkpoint),
            "config_hash": stable_hash(
                {
                    "protocol": SYNTHETIC_PROTOCOL,
                    "scenario": args.scenario,
                    "variant": variant,
                    "seed": args.seed,
                    "horizon": task.horizon,
                }
            ),
            **metrics,
            **selection,
            **recovery,
        }
        write_json(output, row)
        write_csv(Path(args.output_root) / "synthetic" / "history" / args.scenario / variant / f"seed{args.seed}.csv", history)
        print(
            f"mechanism-result scenario={args.scenario:<16} variant={variant:<21} seed={args.seed} "
            f"mse={metrics['mse']:.6f} recovery={recovery['component_correlation_mean']:.4f}",
            flush=True,
        )


def subset_task(task: TaskData, indices: np.ndarray) -> TaskData:
    return TaskData(
        task.dataset,
        task.horizon,
        task.values[:, indices].copy(),
        task.training_starts,
        task.validation_starts,
        task.test_starts,
    )


def run_channel_scaling(args) -> None:
    if args.seed != 42 or args.channel_count not in CHANNEL_COUNTS:
        raise ValueError("Channel scaling uses seed 42 and a locked channel count.")
    device = resolve_device(args.device)
    complete = load_task(
        Path(args.data_root), "Traffic", 720, seq_len=96,
        max_train_windows=2048, max_validation_windows=1024, window_seed=42,
    )
    train_end = int(min(complete.training_starts) + 96) if len(complete.training_starts) else 96
    train_end = max(train_end, int(max(complete.training_starts) + 96))
    variance = complete.values[:train_end].var(axis=0)
    order = np.lexsort((np.arange(len(variance)), -variance))
    indices = np.sort(order[: args.channel_count])
    task = subset_task(complete, indices)
    train_args = argparse.Namespace(**vars(args))
    train_args.dataset = "Traffic"
    train_args.horizon = 720
    variant = f"C{args.channel_count}"
    result_root = Path(args.output_root) / "real_channel_scaling"
    lf_path = result_root / "runs" / "LatentFlow" / f"{variant}.json"
    if not lf_path.exists() or args.force:
        started = time.perf_counter()
        seed_all(42)
        config = backbone_config(96, 720, tuple(SELECTED_CONFIG["Traffic"]["patch_lens"]))
        base = instantiate_backbone(config).to(device)
        base_history, base_selection = train_mvpf(
            base, task.values, task.training_starts, task.validation_starts,
            seq_len=96, pred_len=720, epochs=10, batch_size=4, eval_batch_size=4,
            learning_rate=7e-4, weight_decay=1e-4, device=device, seed=42, patience=0,
        )
        model = build_latentflow(base, task, SELECTED_CONFIG["Traffic"], mode="final", unfreeze_blocks=2).to(device)
        assert_no_exchange(model)
        history, selection = fit_final_model(
            model, task, experiment_args(train_args),
            representation_weight=SELECTED_CONFIG["Traffic"]["representation_weight"],
            nll_weight=0.02, batch_size=4, device=device,
        )
        metrics = evaluate(MeanAdapter(model), task, task.test_starts, seq_len=96, batch_size=4, device=device, synchronize=True)
        probe = next(iter(loader(task, task.validation_starts[:1], 96, 1)))[0].to(device)
        flops = profiler_flops(MeanAdapter(model), probe)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = MeanAdapter(model)(probe)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_memory = 0.0
        checkpoint = result_root / "checkpoints" / "LatentFlow" / f"{variant}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": clone_state(model), "base_config": config, "selection": selection, "indices": indices.tolist()}, checkpoint)
        write_json(lf_path, {
            "experiment": "real_channel_scaling", "enhancement_protocol": SCALING_PROTOCOL,
            "model": "LatentFlow", "dataset": "Traffic", "horizon": 720, "seed": 42,
            "channels": args.channel_count, "selection_rule": "top train-split variance with index tie-break",
            "channel_indices": indices.tolist(), "cross_process_exchange": False,
            "parameters": parameter_count(model), "gflops_per_sample": flops / 1e9,
            "peak_inference_memory_mb": peak_memory,
            "throughput_samples_per_second": 4000.0 / max(metrics["inference_ms_batch"], 1e-9),
            "elapsed_seconds": time.perf_counter() - started, "checkpoint": str(checkpoint),
            "config_hash": stable_hash({"protocol": SCALING_PROTOCOL, "model": "LatentFlow", "channels": args.channel_count}),
            **metrics, **selection,
        })
        write_csv(result_root / "history" / "LatentFlow" / f"{variant}.csv", base_history + history)
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tp_path = result_root / "runs" / "TimePro" / f"{variant}.json"
    if not tp_path.exists() or args.force:
        seed_all(42)
        spec = build_model("TimePro", "Traffic", 96, 720, args.channel_count)
        model = spec.model.to(device)
        history, selection, train_ms = _fit_seeded_model(args, "TimePro", model, spec, task, device)
        metrics = evaluate(model, task, task.test_starts, seq_len=96, batch_size=4, device=device, synchronize=True)
        probe = next(iter(loader(task, task.validation_starts[:1], 96, 1)))[0].to(device)
        flops = profiler_flops(model, probe)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(probe)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_memory = 0.0
        checkpoint = result_root / "checkpoints" / "TimePro" / f"{variant}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": clone_state(model), "selection": selection, "indices": indices.tolist()}, checkpoint)
        write_json(tp_path, {
            "experiment": "real_channel_scaling", "enhancement_protocol": SCALING_PROTOCOL,
            "model": "TimePro", "dataset": "Traffic", "horizon": 720, "seed": 42,
            "channels": args.channel_count, "selection_rule": "top train-split variance with index tie-break",
            "channel_indices": indices.tolist(), "cross_process_exchange": False,
            "parameters": parameter_count(model), "gflops_per_sample": flops / 1e9,
            "peak_inference_memory_mb": peak_memory,
            "throughput_samples_per_second": 4000.0 / max(metrics["inference_ms_batch"], 1e-9),
            "training_ms_optimizer_step": train_ms, "checkpoint": str(checkpoint),
            "config_hash": stable_hash({"protocol": SCALING_PROTOCOL, "model": "TimePro", "channels": args.channel_count}),
            **metrics, **selection,
        })
        write_csv(result_root / "history" / "TimePro" / f"{variant}.csv", history)
    print(f"channel-scaling-result channels={args.channel_count} models=2", flush=True)


def run_smoke(args) -> None:
    register_synthetic_domain()
    task, components, truth = synthetic_task(42, "reference")
    base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
    model = synthetic_model(base, task, "process_preserving")
    prediction = model.forward_distribution(torch.randn(2, 96, 8))["mean"]
    if prediction.shape != (2, 96, 8) or components.shape[1] != 3 or truth["period_samples"] != 24.0:
        raise RuntimeError("Mechanism smoke failed.")
    print("LATENTFLOW-MECHANISM-SMOKE PASSED synthetic_processes=3 channel_counts=6", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("smoke", "synthetic", "channel-scaling"), required=True)
    result.add_argument("--scenario", choices=SCENARIOS, default="reference")
    result.add_argument("--channel-count", choices=CHANNEL_COUNTS, type=int, default=32)
    result.add_argument("--seed", choices=(42, 43, 44, 45, 46), type=int, default=42)
    result.add_argument("--seq-len", type=int, default=96)
    result.add_argument("--max-train-windows", type=int, default=2048)
    result.add_argument("--max-validation-windows", type=int, default=1024)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--eval-batch-size", type=int, default=16)
    result.add_argument("--wide-batch-size", type=int, default=4)
    result.add_argument("--effective-batch-size", type=int, default=16)
    result.add_argument("--reference-epochs", type=int, default=2)
    result.add_argument("--extension-epochs", type=int, default=10)
    result.add_argument("--learning-rate", type=float, default=7e-4)
    result.add_argument("--head-learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip", type=float, default=5.0)
    result.add_argument("--nll-weight", type=float, default=0.02)
    result.add_argument("--data-root", default="TSLibrary/dataset")
    result.add_argument("--output-root", default=str(ROOT))
    result.add_argument("--device", default="auto")
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_channel_scaling(args)


if __name__ == "__main__":
    main()
