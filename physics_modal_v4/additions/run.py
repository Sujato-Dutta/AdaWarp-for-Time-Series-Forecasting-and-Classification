"""Isolated additions for the LatentFlow paper.

The module deliberately imports the frozen implementation in ``physics_modal_v4``.
It never writes outside ``physics_modal_v4/additions`` and it preserves the
published data protocol: input length 96, training-only normalization, fixed
sampled train/validation windows, validation-MSE selection, and all test windows.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from benchmark_adawarp_mvpf_plus_ltsf import train_model as train_mvpf
from physics_modal_v4.experiments.baselines import (
    prediction_loss,
    regularizer,
    scheduler_for,
)
from physics_modal_v4.experiments.protocol import (
    DATASETS,
    HORIZONS,
    SEEDS,
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
from physics_modal_v4.experiments.run import (
    FINAL_CANDIDATE,
    MeanAdapter,
    SELECTED_CONFIG,
    build_latentflow,
    load_final,
)
from physics_modal_v4.experiments.training import (
    backbone_config,
    build_backbone,
    chronological_selection,
    ensure_backbone,
    evaluate_distribution,
    fit_final_model,
    instantiate_backbone,
    torch_load,
)
from physics_modal_v4.model import (
    CausalStructuralEncoder,
    LatentFlow,
    MultiStochasticLatentSeparator,
    PROCESS_SPECS,
    SENSOR_GROUP_SPECS,
)
from physics_modal_v4.models.registry import ModelSpec, build_model
from physics_modal_v4.models.standard import VPNetForecaster


ROOT = Path("physics_modal_v4/additions")
DEFAULT_OUTPUT = ROOT / "results"
SOURCE_ROOT = Path("physics_modal_v4/results/final")
TARGET_DATASETS = tuple(DATASETS)
TARGET_VARIANTS = (
    "final_no_exchange",
    "x_to_z",
    "early_fusion",
    "no_multiscale",
    "no_linear_bank",
)
CENTRAL_PROTOCOL = "latentflow_no_exchange_all28_fiveseed_v1"
SYNTHETIC_SETTINGS = ("independent", "interacting")
SYNTHETIC_VARIANTS = ("early_fusion", "independent_branches", "process_preserving", "full")


def batch_size(args, dataset: str, requested: int = 16) -> int:
    return min(requested, args.wide_batch_size) if dataset in {"Electricity", "Traffic"} else requested


def existing_args(args, dataset: str, horizon: int, seed: int = 42, model: str | None = None):
    return SimpleNamespace(
        data_root=args.data_root,
        output_root=args.source_root,
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        model=model,
        seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        wide_batch_size=args.wide_batch_size,
        effective_batch_size=args.effective_batch_size,
        reference_epochs=args.reference_epochs,
        extension_epochs=args.extension_epochs,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        nll_weight=args.nll_weight,
        device=args.device,
        force=args.force,
    )


def additions_args(args, dataset: str, horizon: int, seed: int):
    result = existing_args(args, dataset, horizon, seed)
    result.output_root = args.output_root
    return result


def _metric_path(args, *parts: str) -> Path:
    return Path(args.output_root).joinpath(*parts)


def _save_checkpoint(path: Path, model: nn.Module, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "state_dict": clone_state(model)}, path)


def _train_standard_model(
    name: str,
    spec: ModelSpec,
    task: TaskData,
    args,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict], dict, float]:
    seed_all(seed)
    model = spec.model.to(device)
    requested = batch_size(args, task.dataset, spec.batch_size)
    eval_requested = batch_size(args, task.dataset, spec.eval_batch_size)
    training = loader(
        task, task.training_starts, args.seq_len, requested, shuffle=True, seed=seed
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    schedule, each_batch = scheduler_for(spec, optimizer, len(training))
    best_state, best_mse, best_epoch = clone_state(model), float("inf"), 0
    history: list[dict] = []
    updates = 0
    started = time.perf_counter()
    for epoch in range(1, spec.epochs + 1):
        model.train()
        total = 0.0
        for inputs, targets in training:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = prediction_loss(spec, prediction, targets) + regularizer(name, model, inputs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss at epoch {epoch}.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            if hasattr(model, "project_parameters"):
                model.project_parameters()
            if schedule is not None and each_batch:
                schedule.step()
            total += float(loss.detach())
            updates += 1
        if schedule is not None and not each_batch:
            schedule.step()
        val = evaluate(
            model, task, task.validation_starts, seq_len=args.seq_len,
            batch_size=eval_requested, device=device,
        )
        if val["mse"] < best_mse:
            best_mse, best_epoch, best_state = val["mse"], epoch, clone_state(model)
        history.append({
            "epoch": epoch,
            "training_loss": total / max(1, len(training)),
            "validation_mse": val["mse"],
            "validation_mae": val["mae"],
            "best_validation_mse": best_mse,
        })
        print(
            f"addition-train model={name} dataset={task.dataset} h={task.horizon} "
            f"seed={seed} epoch={epoch} val={val['mse']:.6f} best={best_mse:.6f}@{best_epoch}",
            flush=True,
        )
        if spec.patience > 0 and epoch - best_epoch >= spec.patience:
            break
    load_state(model, best_state, device)
    train_ms = 1000.0 * (time.perf_counter() - started) / max(1, updates)
    return model, history, {"best_epoch": best_epoch, "best_validation_mse": best_mse}, train_ms


def run_timepro(args) -> None:
    """Train one missing TimePro seed on one matched task."""
    if args.seed not in (43, 44, 45, 46):
        raise ValueError("TimePro additions are precisely seeds 43--46; seed 42 already exists.")
    output = _metric_path(
        args, "timepro_multiseed", "runs", f"seed{args.seed}", args.dataset, f"h{args.horizon}.json"
    )
    if output.exists() and not args.force:
        print(f"skip-complete TimePro dataset={args.dataset} h={args.horizon} seed={args.seed}")
        return
    device = resolve_device(args.device)
    task = load_task(
        Path(args.data_root), args.dataset, args.horizon, seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )
    spec = build_model("TimePro", args.dataset, args.seq_len, args.horizon, task.values.shape[1])
    model, history, selection, train_ms = _train_standard_model(
        "TimePro", spec, task, args, args.seed, device
    )
    metrics = evaluate(
        model, task, task.test_starts, seq_len=args.seq_len,
        batch_size=batch_size(args, args.dataset, spec.eval_batch_size),
        device=device, synchronize=True,
    )
    checkpoint = _metric_path(
        args, "timepro_multiseed", "checkpoints", f"seed{args.seed}", args.dataset, f"h{args.horizon}.pt"
    )
    _save_checkpoint(checkpoint, model, {
        "model": "TimePro", "dataset": args.dataset, "horizon": args.horizon,
        "seed": args.seed, "channels": task.values.shape[1], "selection": selection,
        "provenance": spec.provenance,
    })
    row = {
        "model": "TimePro", "dataset": args.dataset, "horizon": args.horizon,
        "seed": args.seed, **metrics, **selection,
        "parameters": parameter_count(model), "training_ms_optimizer_step": train_ms,
        "checkpoint": str(checkpoint),
        "protocol": "matched LatentFlow protocol; validation-MSE checkpoint; all test windows",
    }
    write_json(output, row)
    write_csv(
        _metric_path(args, "timepro_multiseed", "history", f"seed{args.seed}", args.dataset, f"h{args.horizon}.csv"),
        history,
    )
    print(
        f"timepro-multiseed-result dataset={args.dataset:<11} h={args.horizon:<3} "
        f"seed={args.seed} mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}", flush=True,
    )


def _source_final_path(args, dataset: str, horizon: int, seed: int) -> Path:
    return Path(args.source_root) / "headline" / "runs" / "LatentFlow" / f"seed{seed}" / dataset / f"h{horizon}.json"


def _source_ablation_path(args, variant: str, dataset: str, horizon: int) -> Path:
    return Path(args.source_root) / "ablations" / "runs" / variant / dataset / f"h{horizon}.json"


def _central_mode(variant: str) -> str:
    return {
        "final_no_exchange": "final",
        "x_to_z": "x_to_z",
        "early_fusion": "learned_inducing_elbo",
        "no_multiscale": "final",
        "no_linear_bank": "final",
    }[variant]


def _load_backbone_from_final(args, dataset: str, horizon: int, seed: int):
    path = Path(args.source_root) / "checkpoints" / "LatentFlow" / f"seed{seed}" / dataset / f"h{horizon}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing locked LatentFlow checkpoint: {path}")
    payload = torch_load(path)
    if payload.get("candidate") != FINAL_CANDIDATE:
        raise RuntimeError(
            f"Checkpoint {path} is not the revised no-exchange LatentFlow model."
        )
    state = {
        key[len("reference_base."):]: value
        for key, value in payload["state_dict"].items()
        if key.startswith("reference_base.")
    }
    base = instantiate_backbone(payload["base_config"])
    base.load_state_dict(state, strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base, payload


def run_central(args) -> None:
    if args.dataset not in TARGET_DATASETS or args.horizon not in HORIZONS:
        raise ValueError("Central ablations cover every dataset and horizon.")
    if args.variant not in TARGET_VARIANTS:
        raise ValueError(f"Central variant must be one of {TARGET_VARIANTS}.")
    output = _metric_path(
        args,
        "central_ablation",
        "runs",
        args.variant,
        f"seed{args.seed}",
        args.dataset,
        f"h{args.horizon}.json",
    )
    if output.exists() and not args.force:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("central_protocol") == CENTRAL_PROTOCOL
            and existing.get("cross_process_exchange") is False
        ):
            print(
                f"skip-complete central variant={args.variant} dataset={args.dataset} "
                f"h={args.horizon} seed={args.seed}"
            )
            return
        print(
            f"replace-stale central variant={args.variant} dataset={args.dataset} "
            f"h={args.horizon} seed={args.seed}",
            flush=True,
        )

    # The revised headline is exactly the final no-exchange factorial cell.
    # Reuse it only when its candidate metadata confirms the new architecture.
    if args.variant == "final_no_exchange":
        source = _source_final_path(args, args.dataset, args.horizon, args.seed)
    else:
        source = Path("__missing__")
    if source.exists() and not args.force:
        row = json.loads(source.read_text(encoding="utf-8"))
        if (
            row.get("candidate") == FINAL_CANDIDATE
            and row.get("cross_process_exchange") is False
        ):
            row.update(
                {
                    "central_variant": args.variant,
                    "central_protocol": CENTRAL_PROTOCOL,
                    "reused_source": str(source),
                }
            )
            write_json(output, row)
            print(
                f"central-ablation-result variant={args.variant:<19} "
                f"dataset={args.dataset:<11} h={args.horizon:<3} seed={args.seed} "
                f"mse={row['mse']:.6f} mae={row['mae']:.6f} reused=True",
                flush=True,
            )
            return
        raise RuntimeError(
            f"Headline source is not the revised no-exchange model: {source}"
        )

    device = resolve_device(args.device)
    task = load_task(
        Path(args.data_root), args.dataset, args.horizon, seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows, window_seed=42,
    )
    config = SELECTED_CONFIG[args.dataset]

    if args.variant in {"x_to_z", "early_fusion"}:
        base, payload = _load_backbone_from_final(
            args, args.dataset, args.horizon, args.seed
        )
        seed_all(args.seed)
        model = build_latentflow(
            base, task, config, mode=_central_mode(args.variant), unfreeze_blocks=2
        ).to(device)
        history, selection = fit_final_model(
            model,
            task,
            args,
            representation_weight=config["representation_weight"],
            nll_weight=args.nll_weight,
            batch_size=batch_size(args, args.dataset),
            device=device,
        )
        base_config = payload["base_config"]
        selected_single_scale = None
    else:
        candidate_patches = (
            [(int(patch),) for patch in config["patch_lens"]]
            if args.variant == "no_multiscale"
            else [tuple(config["patch_lens"])]
        )
        switches = (
            {"use_linear_field": False}
            if args.variant == "no_linear_bank"
            else None
        )
        best = None
        all_history = []
        for patches in candidate_patches:
            run_args = additions_args(args, args.dataset, args.horizon, args.seed)
            tag = f"central_{args.variant}_patch_{'-'.join(map(str, patches))}"
            backbone_path = ensure_backbone(
                run_args,
                task,
                device,
                patches,
                switch_overrides=switches,
                checkpoint_tag=tag,
            )
            payload = torch_load(backbone_path)
            base = build_backbone(payload, require_final=switches is None)
            seed_all(args.seed)
            candidate = build_latentflow(
                base, task, config, mode="final", unfreeze_blocks=2
            ).to(device)
            candidate_history, candidate_selection = fit_final_model(
                candidate,
                task,
                args,
                representation_weight=config["representation_weight"],
                nll_weight=args.nll_weight,
                batch_size=batch_size(args, args.dataset),
                device=device,
            )
            for item in candidate_history:
                all_history.append({**item, "candidate_patch_lens": "-".join(map(str, patches))})
            score = float(candidate_selection["best_validation_mse"])
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "state": clone_state(candidate),
                    "payload": payload,
                    "selection": candidate_selection,
                    "patches": patches,
                }
            del candidate, base
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if best is None:
            raise RuntimeError("No central ablation candidate was trained.")
        base = build_backbone(best["payload"], require_final=switches is None)
        model = build_latentflow(
            base, task, config, mode="final", unfreeze_blocks=2
        ).to(device)
        load_state(model, best["state"], device)
        history = all_history
        selection = best["selection"]
        base_config = best["payload"]["config"]
        selected_single_scale = (
            int(best["patches"][0]) if args.variant == "no_multiscale" else None
        )

    metrics = evaluate(
        MeanAdapter(model), task, task.test_starts, seq_len=args.seq_len,
        batch_size=batch_size(args, args.dataset), device=device, synchronize=True,
    )
    checkpoint = _metric_path(
        args, "central_ablation", "checkpoints", args.variant,
        f"seed{args.seed}", args.dataset, f"h{args.horizon}.pt",
    )
    _save_checkpoint(checkpoint, model, {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "variant": args.variant,
        "base_config": base_config,
        "selected_config": config,
        "selected_single_scale": selected_single_scale,
        "cross_process_exchange": False,
        "central_protocol": CENTRAL_PROTOCOL,
        "selection": selection,
    })
    row = {
        "model": "LatentFlow", "central_variant": args.variant,
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        **metrics,
        **selection,
        "selected_single_scale": selected_single_scale,
        "cross_process_exchange": False,
        "central_protocol": CENTRAL_PROTOCOL,
        "checkpoint": str(checkpoint),
    }
    write_json(output, row)
    write_csv(
        _metric_path(
            args,
            "central_ablation",
            "history",
            args.variant,
            f"seed{args.seed}",
            args.dataset,
            f"h{args.horizon}.csv",
        ),
        history,
    )
    print(
        f"central-ablation-result variant={args.variant:<19} dataset={args.dataset:<11} "
        f"h={args.horizon:<3} seed={args.seed} "
        f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
        flush=True,
    )


def _deterministic_mask(shape, starts, rate: float, seed: int, device) -> torch.Tensor:
    batch, length, channels = shape
    time_index = torch.arange(length, device=device)[None, :, None]
    channel_index = torch.arange(channels, device=device)[None, None, :]
    absolute = torch.as_tensor(starts, device=device)[:, None, None] + time_index
    # Stateless integer hash: masks are identical across models and batchings.
    hashed = (
        absolute * 1103515245 + channel_index * 12345 + int(seed) * 2654435761
    ) & 0x7FFFFFFF
    return (hashed.to(torch.float64) / float(0x80000000)) < float(rate)


def _prefix_forward_fill(inputs: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
    observed = ~missing
    count = observed.sum(dim=1, keepdim=True).clamp_min(1)
    fallback = (inputs * observed).sum(dim=1, keepdim=True) / count
    filled = inputs.clone()
    last = fallback[:, 0]
    for step in range(inputs.shape[1]):
        current = torch.where(observed[:, step], inputs[:, step], last)
        filled[:, step] = current
        last = current
    return filled


@torch.no_grad()
def _evaluate_masked(model, task, args, rate: float, device, requested_batch: int) -> dict:
    model.eval()
    squared = absolute = count = 0.0
    starts = np.asarray(task.test_starts)
    for offset in range(0, len(starts), requested_batch):
        current = starts[offset: offset + requested_batch]
        values = torch.as_tensor(task.values, dtype=torch.float32)
        inputs = torch.stack([values[s:s + args.seq_len] for s in current]).to(device)
        targets = torch.stack([
            values[s + args.seq_len:s + args.seq_len + task.horizon] for s in current
        ]).to(device)
        mask = _deterministic_mask(inputs.shape, current, rate, args.corruption_seed, device)
        corrupted = _prefix_forward_fill(inputs, mask)
        prediction = model(corrupted)
        error = prediction - targets
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
        count += error.numel()
    return {
        "mse": squared / count, "mae": absolute / count,
        "rmse": math.sqrt(squared / count), "num_windows": len(starts),
    }


def _load_timepro_seed42(args, dataset: str, horizon: int, device):
    checkpoint = Path(args.source_root) / "checkpoints" / "TimePro" / "seed42" / dataset / f"h{horizon}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing TimePro checkpoint: {checkpoint}")
    task = load_task(Path(args.data_root), dataset, horizon, seq_len=args.seq_len)
    spec = build_model("TimePro", dataset, args.seq_len, horizon, task.values.shape[1])
    model = spec.model.to(device)
    model.load_state_dict(torch_load(checkpoint)["state_dict"], strict=True)
    return model, spec, task, checkpoint


def _load_vpnet_seed42(args, dataset: str, horizon: int, device):
    checkpoint = Path(args.source_root) / "checkpoints" / "VPNet" / "seed42" / dataset / f"h{horizon}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing VPNet checkpoint: {checkpoint}")
    task = load_task(Path(args.data_root), dataset, horizon, seq_len=args.seq_len)
    spec = build_model("VPNet", dataset, args.seq_len, horizon, task.values.shape[1])
    model = spec.model.to(device)
    model.load_state_dict(torch_load(checkpoint)["state_dict"], strict=True)
    return model, spec, task, checkpoint


def run_missing(args) -> None:
    if args.dataset not in {"Weather", "Traffic"}:
        raise ValueError("Missing-prefix study is limited to Weather and Traffic.")
    output = _metric_path(args, "missing_prefix", "runs", args.dataset, f"h{args.horizon}.json")
    if output.exists() and not args.force:
        print(f"skip-complete missing dataset={args.dataset} h={args.horizon}")
        return
    device = resolve_device(args.device)
    rows = []
    for name in ("LatentFlow", "TimePro", "VPNet"):
        if name == "LatentFlow":
            model, task, _, checkpoint = load_final(
                existing_args(args, args.dataset, args.horizon, 42), device
            )
            adapter, requested = MeanAdapter(model), batch_size(args, args.dataset)
        elif name == "TimePro":
            model, spec, task, checkpoint = _load_timepro_seed42(args, args.dataset, args.horizon, device)
            adapter, requested = model, batch_size(args, args.dataset, spec.eval_batch_size)
        else:
            model, spec, task, checkpoint = _load_vpnet_seed42(args, args.dataset, args.horizon, device)
            adapter, requested = model, batch_size(args, args.dataset, spec.eval_batch_size)
        for rate in (0.0, 0.2, 0.4):
            metrics = _evaluate_masked(adapter, task, args, rate, device, requested)
            rows.append({
                "model": name, "dataset": args.dataset, "horizon": args.horizon,
                "seed": 42, "missing_rate": rate, **metrics,
                "checkpoint": str(checkpoint),
                "corruption": "elementwise MCAR mask followed by prefix-only forward fill",
                "irregular_time_claim": False,
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(output, {"rows": rows, "scope_note": (
        "Masked regular-grid prefix stress test. Models receive no arbitrary timestamps; "
        "these results do not establish native irregular-time forecasting."
    )})
    write_csv(output.with_suffix(".csv"), rows)
    print(f"missing-prefix-result dataset={args.dataset} h={args.horizon} rows={len(rows)}", flush=True)


def _register_synthetic_domain() -> None:
    PROCESS_SPECS["Synthetic"] = (
        ("periodic", "periodic", 1.0, 0.65),
        ("matern", "matern", 0.0, 2.0),
        ("ou", "ou", 0.0, 0.20),
    )
    SENSOR_GROUP_SPECS["Synthetic"] = {
        "names": ("periodic", "matern", "ou"),
        "process_prior": (
            (0.90, 0.05, 0.05),
            (0.05, 0.90, 0.05),
            (0.05, 0.05, 0.90),
        ),
        "group_prior": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    }


def _fft_matern(rng: np.random.Generator, length: int, channels: int, ell: float) -> np.ndarray:
    frequencies = np.fft.rfftfreq(length)
    spectrum = (ell ** -2 + (2.0 * np.pi * frequencies) ** 2) ** -2
    spectrum /= spectrum.max()
    white = rng.normal(size=(length, channels))
    return np.fft.irfft(np.fft.rfft(white, axis=0) * np.sqrt(spectrum)[:, None], n=length, axis=0)


def _ou_process(rng: np.random.Generator, length: int, channels: int, rho: float = 0.82) -> np.ndarray:
    values = np.zeros((length, channels), dtype=np.float64)
    innovations = rng.normal(size=values.shape) * math.sqrt(1.0 - rho * rho)
    for step in range(1, length):
        values[step] = rho * values[step - 1] + innovations[step]
    return values


def synthetic_task(seed: int, setting: str, seq_len: int = 96, horizon: int = 96) -> tuple[TaskData, np.ndarray, dict]:
    """Known, additive process mixture with optional causal cross-process coupling."""
    if setting not in SYNTHETIC_SETTINGS:
        raise ValueError(setting)
    rng = np.random.default_rng(seed)
    length, channels = 8192, 8
    t = np.arange(length, dtype=np.float64)
    loading = rng.normal(size=(3, channels))
    loading /= np.linalg.norm(loading, axis=1, keepdims=True).clip(1e-8)
    periodic_source = np.stack([
        np.sin(2 * np.pi * t / 24.0 + 0.1),
        0.55 * np.cos(2 * np.pi * t / 168.0 - 0.4),
        0.25 * np.sin(2 * np.pi * t / 12.0 + 0.7),
    ], axis=1)
    periodic = periodic_source @ rng.normal(size=(3, channels)) * 0.65
    matern = _fft_matern(rng, length, channels, ell=18.0)
    matern /= matern.std(axis=0, keepdims=True).clip(1e-8)
    matern *= 0.65
    ou = _ou_process(rng, length, channels) * 0.35
    if setting == "interacting":
        # Coupling is causal and leaves the component labels well-defined.
        matern[3:] += 0.30 * periodic[:-3] @ (loading[0:1].T @ loading[1:2])
        ou[5:] += 0.25 * np.tanh(matern[:-5]) @ (loading[1:2].T @ loading[2:3])
    components = np.stack([periodic, matern, ou], axis=1)
    total = components.sum(axis=1) + 0.05 * rng.normal(size=(length, channels))
    train_end, validation_end = 4915, 6554
    mean = total[:train_end].mean(axis=0, keepdims=True)
    std = total[:train_end].std(axis=0, keepdims=True).clip(1e-6)
    normalized = ((total - mean) / std).astype(np.float32)
    normalized_components = (components / std[:, None, :]).astype(np.float32)
    training = np.arange(0, train_end - seq_len - horizon + 1, dtype=np.int64)
    validation = np.arange(train_end - seq_len, validation_end - seq_len - horizon + 1, dtype=np.int64)
    test = np.arange(validation_end - seq_len, length - seq_len - horizon + 1, dtype=np.int64)
    rng_windows = np.random.default_rng(42)
    training = np.sort(rng_windows.choice(training, min(2048, len(training)), replace=False))
    validation = np.sort(rng_windows.choice(validation, min(1024, len(validation)), replace=False))
    task = TaskData("Synthetic", horizon, normalized, training, validation, test)
    truth = {
        "period_samples": 24.0,
        "matern_length_samples": 18.0,
        "ou_rho": 0.82,
        "ou_length_samples": float(-1.0 / math.log(0.82)),
    }
    return task, normalized_components, truth


def _oracle_inducing_times(family: str, truth: dict, count: int, length: int = 96) -> np.ndarray:
    """Greedy variance-reduction design under the known generating kernel."""
    x = np.arange(length, dtype=np.float64)
    distance = np.abs(x[:, None] - x[None, :])
    if family == "periodic":
        covariance = np.exp(-2.0 * np.sin(np.pi * distance / truth["period_samples"]) ** 2 / 0.65 ** 2)
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
        residual = np.diag(covariance) - np.sum(cross * np.linalg.solve(kuu, cross.T).T, axis=1)
        residual[selected] = -np.inf
    return np.sort(np.asarray(selected, dtype=np.float64))


def _chamfer_timestamp_error(left: np.ndarray, right: np.ndarray) -> float:
    pairwise = np.abs(left[:, None] - right[None, :])
    return float(0.5 * (pairwise.min(axis=1).mean() + pairwise.min(axis=0).mean()))


class SyntheticLatentFlow(LatentFlow):
    def __init__(self, *values, independent_uniform: bool = False, **kwargs):
        self.independent_uniform = bool(independent_uniform)
        super().__init__(*values, **kwargs)

    def _enforce_ablation_freezes(self) -> None:
        super()._enforce_ablation_freezes()
        if self.independent_uniform:
            with torch.no_grad():
                for parameter in self.continuation.process_router.parameters():
                    parameter.zero_()
            for parameter in self.continuation.process_router.parameters():
                parameter.requires_grad_(False)


def _synthetic_model(base, variant: str) -> SyntheticLatentFlow:
    mode = {
        "early_fusion": "learned_inducing_elbo",
        "independent_branches": "process_preserving",
        "process_preserving": "process_preserving",
        "full": "final",
    }[variant]
    return SyntheticLatentFlow(
        base, dataset="Synthetic", channels=8, inducing_points=16,
        latent_dim=8, ablation_mode=mode, continuation_unfreeze_blocks=2,
        independent_uniform=variant == "independent_branches",
    )


def _fit_synthetic_extension(model, task, args, device) -> tuple[list[dict], dict]:
    """Select among trained extension epochs; the reference is reported separately."""
    from physics_modal_v4.experiments.training import fit_reference_head

    selection_starts = chronological_selection(task.validation_starts)
    history = fit_reference_head(model, task, selection_starts, args, args.batch_size, device)
    model.configure_stage("C")
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    training = loader(task, task.training_starts, args.seq_len, args.batch_size, shuffle=True, seed=args.seed)
    best_state, best_mse, best_epoch = clone_state(model), float("inf"), 0
    for epoch in range(1, args.extension_epochs + 1):
        model.train()
        total = 0.0
        for inputs, targets in training:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss, _, _, _, _ = model.training_objective(
                inputs, targets, sample_dual=None, fenchel_weight=0.0,
                nll_weight=args.nll_weight, representation_weight=0.01,
                augmented_weight=0.0, reconstruction_weight=0.0,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), args.gradient_clip)
            optimizer.step()
            model.project_primal()
            total += float(loss.detach())
        validation = evaluate_distribution(
            model, task, selection_starts, seq_len=args.seq_len,
            batch_size=args.batch_size, device=device,
        )
        if validation["mse"] < best_mse:
            best_mse, best_epoch, best_state = validation["mse"], epoch, clone_state(model)
        history.append({
            "stage": "extension", "epoch": epoch,
            "training_loss": total / len(training),
            "validation_mse": validation["mse"], "best_validation_mse": best_mse,
        })
        print(
            f"synthetic-train variant={args.variant} setting={args.setting} seed={args.seed} "
            f"epoch={epoch} val={validation['mse']:.6f} best={best_mse:.6f}@{best_epoch}", flush=True,
        )
    load_state(model, best_state, device)
    model.select_reference_only(False)
    return history, {"best_epoch": best_epoch, "best_validation_mse": best_mse, "best_stage": "extension"}


def _decoded_processes(model: LatentFlow, inputs: torch.Tensor) -> tuple[torch.Tensor, dict]:
    backbone = model.continuation.mvpff.backbone
    normalized, _, _ = backbone._normalize(inputs)
    structural = model.encoder(normalized)
    posterior = model.separator(structural, inputs)
    return model.encoder.decode(posterior["latent_mean"]), posterior


@torch.no_grad()
def _collect_component_estimates(model, task, components, starts, args, device, maximum: int):
    chosen = np.asarray(starts)[:maximum]
    values = torch.as_tensor(task.values, dtype=torch.float32)
    estimates, targets = [], []
    for offset in range(0, len(chosen), args.batch_size):
        current = chosen[offset:offset + args.batch_size]
        inputs = torch.stack([values[s:s + args.seq_len] for s in current]).to(device)
        predicted, _ = _decoded_processes(model, inputs)
        local_std = inputs.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-5)
        truth = torch.as_tensor(
            np.stack([components[s:s + args.seq_len] for s in current]), device=device
        ) / local_std[:, :, None, :]
        estimates.append(predicted.cpu().numpy())
        targets.append(truth.cpu().numpy())
    return np.concatenate(estimates), np.concatenate(targets)


def _component_recovery(model, task, components, args, device) -> dict:
    from scipy.optimize import linear_sum_assignment

    val_est, val_true = _collect_component_estimates(
        model, task, components, task.validation_starts, args, device, 128
    )
    test_est, test_true = _collect_component_estimates(
        model, task, components, task.test_starts, args, device, 256
    )
    process_count = val_true.shape[2]
    correlations = np.zeros((process_count, process_count))
    for predicted in range(process_count):
        p = val_est[:, :, predicted].reshape(-1)
        for true in range(process_count):
            y = val_true[:, :, true].reshape(-1)
            correlations[predicted, true] = abs(np.corrcoef(p, y)[0, 1])
    predicted_indices, true_indices = linear_sum_assignment(-np.nan_to_num(correlations))
    rows, test_correlations, normalized_errors = [], [], []
    for predicted, true in zip(predicted_indices, true_indices):
        pv = val_est[:, :, predicted].reshape(-1)
        tv = val_true[:, :, true].reshape(-1)
        calibration = float(np.dot(pv, tv) / max(1e-8, np.dot(pv, pv)))
        pt = calibration * test_est[:, :, predicted].reshape(-1)
        tt = test_true[:, :, true].reshape(-1)
        correlation = float(np.corrcoef(pt, tt)[0, 1])
        nrmse = float(np.sqrt(np.mean((pt - tt) ** 2)) / max(1e-8, np.std(tt)))
        rows.append({
            "predicted_process": int(predicted), "true_process": int(true),
            "correlation": correlation, "nrmse": nrmse,
        })
        test_correlations.append(abs(correlation))
        normalized_errors.append(nrmse)
    return {
        "assignment": rows,
        "mean_absolute_component_correlation": float(np.mean(test_correlations)),
        "mean_component_nrmse": float(np.mean(normalized_errors)),
        "process_identity_accuracy": float(np.mean(predicted_indices == true_indices)),
        "validation_absolute_correlation_matrix": correlations.tolist(),
    }


def run_synthetic(args) -> None:
    if args.setting not in SYNTHETIC_SETTINGS or args.variant not in SYNTHETIC_VARIANTS:
        raise ValueError("Invalid synthetic setting or variant.")
    _register_synthetic_domain()
    output = _metric_path(
        args, "synthetic", "runs", args.setting, args.variant, f"seed{args.seed}.json"
    )
    if output.exists() and not args.force:
        print(f"skip-complete synthetic setting={args.setting} variant={args.variant} seed={args.seed}")
        return
    device = resolve_device(args.device)
    task, components, truth = synthetic_task(args.seed, args.setting, args.seq_len, 96)
    base_path = _metric_path(args, "synthetic", "backbones", args.setting, f"seed{args.seed}.pt")
    if base_path.exists() and not args.force:
        base_payload = torch_load(base_path)
        base = build_backbone(base_payload)
    else:
        seed_all(args.seed)
        config = backbone_config(args.seq_len, 96, (8, 16, 32))
        base_train = instantiate_backbone(config).to(device)
        history, selection = train_mvpf(
            base_train, task.values, task.training_starts, task.validation_starts,
            seq_len=args.seq_len, pred_len=96, epochs=10, batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, device=device, seed=args.seed, patience=0,
        )
        _save_checkpoint(base_path, base_train, {
            "config": config, "selection": selection, "history": history,
        })
        base = build_backbone(torch_load(base_path))
        del base_train
    seed_all(args.seed)
    model = _synthetic_model(base, args.variant).to(device)
    history, selection = _fit_synthetic_extension(model, task, args, device)
    metrics = evaluate(
        MeanAdapter(model), task, task.test_starts, seq_len=args.seq_len,
        batch_size=args.eval_batch_size, device=device, synchronize=True,
    )
    recovery = _component_recovery(model, task, components, args, device)
    learned = []
    for name, kernel in zip(model.separator.process_names, model.separator.kernels):
        physical = kernel.parameters_physical()
        inducing_times = kernel.inducing_times().detach().cpu().numpy()
        oracle_times = _oracle_inducing_times(
            kernel.family, truth, len(inducing_times), args.seq_len
        )
        item = {
            "name": name, "family": kernel.family,
            "inducing_times": inducing_times.tolist(),
            "oracle_inducing_times": oracle_times.tolist(),
            "inducing_chamfer_error_samples": _chamfer_timestamp_error(
                inducing_times, oracle_times
            ),
            "learned_length_samples": float(physical["length"].detach()),
        }
        if kernel.raw_period_adjustment is not None:
            item["learned_period_samples"] = float(physical["period"].detach())
            item["period_relative_error"] = abs(
                item["learned_period_samples"] - truth["period_samples"]
            ) / truth["period_samples"]
        elif kernel.family == "matern":
            item["length_relative_error"] = abs(
                item["learned_length_samples"] - truth["matern_length_samples"]
            ) / truth["matern_length_samples"]
        elif kernel.family == "ou":
            item["length_relative_error"] = abs(
                item["learned_length_samples"] - truth["ou_length_samples"]
            ) / truth["ou_length_samples"]
        learned.append(item)
    checkpoint = _metric_path(
        args, "synthetic", "checkpoints", args.setting, args.variant, f"seed{args.seed}.pt"
    )
    _save_checkpoint(checkpoint, model, {
        "setting": args.setting, "variant": args.variant,
        "seed": args.seed, "selection": selection,
    })
    row = {
        "model": "LatentFlow", "setting": args.setting, "variant": args.variant,
        "seed": args.seed, **metrics, **selection, **recovery,
        "ground_truth": truth, "learned_processes": learned,
        "checkpoint": str(checkpoint),
    }
    write_json(output, row)
    write_csv(
        _metric_path(args, "synthetic", "history", args.setting, args.variant, f"seed{args.seed}.csv"),
        history,
    )
    print(
        f"synthetic-result setting={args.setting:<11} variant={args.variant:<20} "
        f"seed={args.seed} mse={metrics['mse']:.6f} "
        f"recovery_corr={recovery['mean_absolute_component_correlation']:.4f}", flush=True,
    )


class VPNetProcessSeparatorControl(nn.Module):
    """A deliberately simple stochastic separator placed before matched VPNet."""

    def __init__(self, dataset: str, channels: int, seq_len: int, horizon: int, config: dict):
        super().__init__()
        self.encoder = CausalStructuralEncoder(config["latent_dim"])
        self.separator = MultiStochasticLatentSeparator(
            dataset, channels=channels, seq_len=seq_len, pred_len=horizon,
            latent_dim=config["latent_dim"], inducing_points=config["inducing_points"],
        )
        self.vpnet = VPNetForecaster(seq_len, horizon)
        self.representation_weight = float(config["representation_weight"])
        self._filtered: torch.Tensor | None = None
        self._representation: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=1, keepdim=True).detach()
        std = inputs.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(1e-5)
        normalized = (inputs - mean) / std
        structural = self.encoder(normalized)
        posterior = self.separator(structural, inputs)
        self._filtered = self.encoder.decode(posterior["reconstruction"]) * std + mean
        self._representation = posterior["diagnostics"]["representation_objective"]
        return self.vpnet(self._filtered)

    def auxiliary_loss(self, _: torch.Tensor) -> torch.Tensor:
        if self._filtered is None or self._representation is None:
            raise RuntimeError("Call forward before auxiliary_loss.")
        # baselines.regularizer multiplies this return value by 0.1.
        return self.vpnet.auxiliary_loss(self._filtered) + 10.0 * self.representation_weight * self._representation

    @torch.no_grad()
    def project_parameters(self) -> None:
        self.encoder.project()
        self.separator.project()


def run_vpnet_control(args) -> None:
    output = _metric_path(args, "vpnet_separator_control", "runs", args.dataset, f"h{args.horizon}.json")
    if output.exists() and not args.force:
        print(f"skip-complete VPNet-control dataset={args.dataset} h={args.horizon}")
        return
    device = resolve_device(args.device)
    task = load_task(
        Path(args.data_root), args.dataset, args.horizon, seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows, window_seed=42,
    )
    config = SELECTED_CONFIG[args.dataset]
    seed_all(42)
    model = VPNetProcessSeparatorControl(
        args.dataset, task.values.shape[1], args.seq_len, args.horizon, config
    )
    vp_spec = build_model("VPNet", args.dataset, args.seq_len, args.horizon, task.values.shape[1])
    spec = ModelSpec(
        model=model, epochs=vp_spec.epochs, patience=vp_spec.patience,
        batch_size=vp_spec.batch_size, eval_batch_size=vp_spec.eval_batch_size,
        learning_rate=vp_spec.learning_rate, weight_decay=vp_spec.weight_decay,
        provenance="matched VPNet with the simplified LatentFlow stochastic separator frontend",
    )
    model, history, selection, train_ms = _train_standard_model(
        "VPNet", spec, task, args, 42, device
    )
    metrics = evaluate(
        model, task, task.test_starts, seq_len=args.seq_len,
        batch_size=batch_size(args, args.dataset, spec.eval_batch_size),
        device=device, synchronize=True,
    )
    checkpoint = _metric_path(args, "vpnet_separator_control", "checkpoints", args.dataset, f"h{args.horizon}.pt")
    _save_checkpoint(checkpoint, model, {
        "model": "VPNet+StochasticSeparator", "dataset": args.dataset,
        "horizon": args.horizon, "seed": 42, "selection": selection,
    })
    row = {
        "model": "VPNet+StochasticSeparator", "dataset": args.dataset,
        "horizon": args.horizon, "seed": 42, **metrics, **selection,
        "parameters": parameter_count(model), "training_ms_optimizer_step": train_ms,
        "checkpoint": str(checkpoint),
        "control_scope": "early stochastic separation followed by matched VPNet; no process-preserving continuation or cross-process exchange",
    }
    write_json(output, row)
    write_csv(
        _metric_path(args, "vpnet_separator_control", "history", args.dataset, f"h{args.horizon}.csv"), history
    )
    print(
        f"vpnet-separator-result dataset={args.dataset:<11} h={args.horizon:<3} "
        f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}", flush=True,
    )


@torch.no_grad()
def run_process_plot(args) -> None:
    if args.dataset not in {"Weather", "Traffic"}:
        raise ValueError("Process visualization is defined for Weather and Traffic.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = resolve_device(args.device)
    source_args = existing_args(args, args.dataset, args.horizon, 42)
    model, task, _, checkpoint = load_final(source_args, device)
    start = int(task.test_starts[0])  # fixed chronologically; never selected using target error
    values = torch.as_tensor(task.values, dtype=torch.float32)
    inputs = values[start:start + args.seq_len][None].to(device)
    target = values[start + args.seq_len:start + args.seq_len + args.horizon]
    distribution = model.forward_distribution(inputs)
    mean = distribution["mean"][0].cpu().numpy()
    variance = distribution["variance"][0].cpu().numpy()
    channel = 0 if args.dataset == "Weather" else int(inputs[0].var(dim=0).argsort()[inputs.shape[2] // 2])
    effects: dict[str, np.ndarray] = {}
    effect_variances: dict[str, float] = {}
    per_process = []
    decoder_energy = float(model.encoder.decoder.weight[0].square().sum())
    raw_scale = float(inputs[0, :, channel].std(unbiased=False))
    for index, (name, kernel) in enumerate(zip(model.separator.process_names, model.separator.kernels)):
        model.disabled_process = index
        without = model.forward_distribution(inputs)["mean"][0].cpu().numpy()
        effect = mean - without
        effects.setdefault(kernel.family, np.zeros_like(effect))
        effects[kernel.family] += effect
        latent_variance = float(
            distribution["diagnostics"][f"process_future_variance_{name}"]
        )
        effect_variances[kernel.family] = effect_variances.get(kernel.family, 0.0) + (
            latent_variance * decoder_energy * raw_scale * raw_scale
        )
        per_process.append({
            "index": index, "name": name, "family": kernel.family,
            "effect_rms": float(np.sqrt(np.mean(effect ** 2))),
            "future_variance_summary": float(
                distribution["diagnostics"][f"process_future_variance_{name}"]
            ),
        })
    model.disabled_process = None

    output_dir = _metric_path(args, "process_visualization", args.dataset, f"h{args.horizon}")
    output_dir.mkdir(parents=True, exist_ok=True)
    x_past = np.arange(-args.seq_len, 0)
    x_future = np.arange(args.horizon)
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 4.8), sharex=False)
    axes[0].plot(x_past, inputs[0, :, channel].cpu(), color="#222222", lw=1.2, label="prefix")
    axes[0].plot(x_future, target[:, channel], color="#777777", lw=1.0, label="target")
    axes[0].plot(x_future, mean[:, channel], color="#1764ab", lw=1.6, label="LatentFlow")
    band = 1.645 * np.sqrt(variance[:, channel])
    axes[0].fill_between(x_future, mean[:, channel] - band, mean[:, channel] + band, color="#1764ab", alpha=0.14)
    axes[0].axvline(0, color="#555555", lw=0.8, ls="--")
    axes[0].set_ylabel("normalized value")
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    palette = {"periodic": "#d95f02", "matern": "#1b9e77", "ou": "#7570b3"}
    for family in ("periodic", "matern", "ou"):
        if family in effects:
            curve = effects[family][:, channel]
            approximate_std = math.sqrt(max(0.0, effect_variances[family]))
            axes[1].plot(x_future, curve, lw=1.4, color=palette[family], label=family)
            axes[1].fill_between(
                x_future, curve - approximate_std, curve + approximate_std,
                color=palette[family], alpha=0.09,
            )
    axes[1].axhline(0, color="#999999", lw=0.7)
    axes[1].set_xlabel("forecast step")
    axes[1].set_ylabel("leave-out effect")
    axes[1].legend(ncol=3, frameon=False, fontsize=8)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"process_decomposition.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": 42,
        "test_start": start, "channel": channel, "checkpoint": str(checkpoint),
        "processes": per_process,
        "interpretation_limit": (
            "Curves are leave-one-process intervention effects, grouped by kernel family. "
            "Bands are projected latent-posterior variance summaries, not calibrated branchwise "
            "forecast intervals. The panel is qualitative evidence of distinct learned dynamics, "
            "not proof of physical semantics."
        ),
    }
    write_json(output_dir / "metadata.json", metadata)
    curve_rows = []
    for step in range(args.horizon):
        row = {
            "forecast_step": step + 1,
            "target": float(target[step, channel]),
            "prediction": float(mean[step, channel]),
            "predictive_variance": float(variance[step, channel]),
        }
        for family in ("periodic", "matern", "ou"):
            if family in effects:
                row[f"{family}_leave_out_effect"] = float(effects[family][step, channel])
                row[f"{family}_approximate_variance"] = effect_variances[family]
        curve_rows.append(row)
    write_csv(output_dir / "process_decomposition.csv", curve_rows)
    print(f"process-plot-result dataset={args.dataset} h={args.horizon} channel={channel}", flush=True)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_scaling(args) -> None:
    device = resolve_device(args.device)
    counts = (7, 21, 64, 128, 321, 512, 862, 1024, 2048)
    rows = []
    for channels in counts:
        seed_all(42)
        base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
        config = SELECTED_CONFIG["Traffic"]
        model = LatentFlow(
            base, dataset="Traffic", channels=channels,
            inducing_points=config["inducing_points"], latent_dim=config["latent_dim"],
            ablation_mode="final", continuation_unfreeze_blocks=2,
        ).to(device).eval()
        inputs = torch.randn(1, 96, channels, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for iteration in range(args.profile_repeats + 2):
            _synchronize(device)
            started = time.perf_counter()
            model.forward_distribution(inputs)
            _synchronize(device)
            if iteration >= 2:
                timings.append(1000.0 * (time.perf_counter() - started))
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        processes = model.separator.num_processes
        rows.append({
            "channels": channels, "latency_ms": float(np.mean(timings)),
            "latency_sd_ms": float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0,
            "peak_gpu_memory_mb": peak / (1024 ** 2),
            "parameters": parameter_count(model),
            "dense_covariance_reference_mb": processes * channels * channels * 4 / (1024 ** 2),
            "batch_size": 1, "input_length": 96, "horizon": 96,
            "implementation_dense_channel_covariance": False,
        })
        del model, base, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = _metric_path(args, "channel_scaling", "channel_scaling.json")
    write_json(output, {"rows": rows, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"})
    write_csv(output.with_suffix(".csv"), rows)
    print(f"channel-scaling-result counts={len(rows)} max_channels={counts[-1]}", flush=True)


def run_smoke(args) -> None:
    device = resolve_device(args.device)
    seed_all(42)
    inputs = torch.randn(2, 96, 7, device=device)
    base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
    latent = LatentFlow(
        base, dataset="ETTh1", channels=7, inducing_points=8, latent_dim=4,
        ablation_mode="final", continuation_unfreeze_blocks=2,
    ).to(device)
    prediction = MeanAdapter(latent)(inputs)
    control = VPNetProcessSeparatorControl(
        "ETTh1", 7, 96, 96,
        {"latent_dim": 4, "inducing_points": 8, "representation_weight": 0.01},
    ).to(device)
    control_prediction = control(inputs)
    control_loss = F.mse_loss(control_prediction, prediction.detach()) + 0.1 * control.auxiliary_loss(inputs)
    control_loss.backward()
    mask = _deterministic_mask(inputs.shape, [0, 1], 0.2, 99173, device)
    filled = _prefix_forward_fill(inputs, mask)
    _register_synthetic_domain()
    synthetic = _synthetic_model(
        instantiate_backbone(backbone_config(96, 96, (8, 16, 32))), "full"
    ).to(device)
    synthetic_prediction = MeanAdapter(synthetic)(torch.randn(2, 96, 8, device=device))
    if not (
        prediction.shape == control_prediction.shape == (2, 96, 7)
        and synthetic_prediction.shape == (2, 96, 8)
        and torch.isfinite(filled).all()
        and torch.isfinite(control_loss)
    ):
        raise RuntimeError("Additions smoke failed shape or finite checks.")
    print(
        f"additions-smoke passed latent={tuple(prediction.shape)} "
        f"control={tuple(control_prediction.shape)} synthetic={tuple(synthetic_prediction.shape)}",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--mode",
        choices=("smoke", "timepro", "synthetic", "central", "missing", "process", "vpnet_control", "scaling"),
        required=True,
    )
    result.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    result.add_argument("--horizon", choices=HORIZONS, type=int, default=96)
    result.add_argument("--seed", choices=SEEDS, type=int, default=42)
    result.add_argument("--variant", default="full")
    result.add_argument("--setting", choices=SYNTHETIC_SETTINGS, default="independent")
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
    result.add_argument("--profile-repeats", type=int, default=10)
    result.add_argument("--corruption-seed", type=int, default=99173)
    result.add_argument("--data-root", default="TSLibrary/dataset")
    result.add_argument("--source-root", default=str(SOURCE_ROOT))
    result.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    result.add_argument("--device", default="auto")
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    actions = {
        "smoke": run_smoke,
        "timepro": run_timepro,
        "synthetic": run_synthetic,
        "central": run_central,
        "missing": run_missing,
        "process": run_process_plot,
        "vpnet_control": run_vpnet_control,
        "scaling": run_scaling,
    }
    actions[args.mode](args)


if __name__ == "__main__":
    main()
