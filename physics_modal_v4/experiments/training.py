"""Training primitives for the locked LatentFlow model.

This module preserves the protocol used to select the finalized LatentFlow:
two epochs for the reference distribution head, ten epochs for the LatentFlow extension,
validation-MSE checkpointing, two unfrozen MVPF continuation blocks, and no
primal-dual updates or cross-process exchange.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import torch
from torch import nn

from adawarp_mvpf_plus import AdaWarpMVPFPlusForecaster
from benchmark_adawarp_mvpf_plus_ltsf import train_model as train_mvpf
from physics_modal_v4.experiments.protocol import (
    TaskData,
    clone_state,
    load_state,
    loader,
    seed_all,
)
from physics_modal_v4.model import LatentFlow


FINAL_BACKBONE_SWITCHES = {
    "use_prototype_memory": False,
    "use_frequency_gate": False,
    "use_adaptive_shifts": False,
    "use_adaptive_radius": True,
    "use_component_gate": True,
    "use_trend_decomposition": False,
    "use_linear_field": True,
}


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def backbone_config(
    seq_len: int,
    horizon: int,
    patch_lens: tuple[int, ...],
    *,
    switch_overrides: dict[str, bool] | None = None,
) -> dict:
    config = {
        "seq_len": int(seq_len),
        "pred_len": int(horizon),
        "patch_lens": list(patch_lens),
        "width": 128,
        "depth": 2,
        "dropout": 0.05,
        "num_prototypes": 8,
        "max_shift": 2,
        "reconstruction_weight": 0.03,
        **FINAL_BACKBONE_SWITCHES,
    }
    config.update(switch_overrides or {})
    return config


def instantiate_backbone(
    config: dict,
    *,
    require_final: bool = True,
) -> AdaWarpMVPFPlusForecaster:
    values = dict(config)
    if require_final:
        for name, expected in FINAL_BACKBONE_SWITCHES.items():
            if values.get(name) is not expected:
                raise ValueError(
                    f"Backbone is not finalized: {name}={values.get(name)!r}, "
                    f"expected {expected!r}."
                )
    seq_len = int(values.pop("seq_len"))
    pred_len = int(values.pop("pred_len"))
    return AdaWarpMVPFPlusForecaster(seq_len, pred_len, **values)


def build_backbone(
    payload: dict,
    *,
    require_final: bool = True,
) -> AdaWarpMVPFPlusForecaster:
    model = instantiate_backbone(payload["config"], require_final=require_final)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def ensure_backbone(
    args,
    task: TaskData,
    device: torch.device,
    patch_lens: tuple[int, ...],
    *,
    switch_overrides: dict[str, bool] | None = None,
    checkpoint_tag: str = "",
) -> Path:
    patch_tag = "-".join(str(value) for value in patch_lens)
    root = Path(args.output_root) / "checkpoints" / "backbone"
    if checkpoint_tag:
        root = root / checkpoint_tag
    checkpoint = (
        root
        / f"patch_{patch_tag}"
        / f"seed{args.seed}"
        / f"{task.dataset}_h{task.horizon}.pt"
    )
    if checkpoint.exists() and not args.force:
        payload = torch_load(checkpoint)
        expected = backbone_config(
            args.seq_len,
            task.horizon,
            patch_lens,
            switch_overrides=switch_overrides,
        )
        if payload.get("config") == expected:
            return checkpoint

    seed_all(args.seed)
    config = backbone_config(
        args.seq_len,
        task.horizon,
        patch_lens,
        switch_overrides=switch_overrides,
    )
    model = instantiate_backbone(config, require_final=not bool(switch_overrides)).to(device)
    history, selection = train_mvpf(
        model,
        task.values,
        task.training_starts,
        task.validation_starts,
        seq_len=args.seq_len,
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
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": clone_state(model),
            "config": config,
            "selection": selection,
            "history": history,
            "checkpoint_tag": checkpoint_tag or "final",
            "protocol": (
                "train-only normalization; 10 epochs; validation-best checkpoint; "
                "test split untouched"
            ),
        },
        checkpoint,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint

def chronological_selection(starts, calibration_fraction: float = 0.0):
    ordered = sorted(int(value) for value in starts)
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1).")
    if calibration_fraction == 0.0:
        return torch.as_tensor(ordered, dtype=torch.long).numpy()
    split = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * (1.0 - calibration_fraction)))),
    )
    return torch.as_tensor(ordered[:split], dtype=torch.long).numpy()


@torch.no_grad()
def evaluate_distribution(
    model: LatentFlow,
    task: TaskData,
    starts,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared = absolute = nll = 0.0
    count = 0
    for inputs, targets in loader(task, starts, seq_len, batch_size):
        inputs = inputs.to(device)
        targets = targets.to(device)
        distribution = model.forward_distribution(inputs)
        error = distribution["mean"] - targets
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
        nll += float(model.mixture_nll(targets, distribution)) * targets.numel()
        count += targets.numel()
    return {
        "mse": squared / max(1, count),
        "mae": absolute / max(1, count),
        "nll": nll / max(1, count),
    }


@torch.no_grad()
def evaluate_reference_nll(
    model: LatentFlow,
    task: TaskData,
    starts,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for inputs, targets in loader(task, starts, seq_len, batch_size):
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss = model.reference_nll(inputs, targets)
        total += float(loss) * targets.numel()
        count += targets.numel()
    return total / max(1, count)


def fit_reference_head(
    model: LatentFlow,
    task: TaskData,
    selection_starts,
    args,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    model.configure_stage("B")
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    training = loader(
        task,
        task.training_starts,
        args.seq_len,
        batch_size,
        shuffle=True,
        seed=args.seed,
    )
    best_state = clone_state(model.reference_head)
    best_nll = float("inf")
    history = []
    for epoch in range(1, args.reference_epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for inputs, targets in training:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.reference_nll(inputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), args.gradient_clip)
            optimizer.step()
            model.project_primal()
            total += float(loss.detach())
            steps += 1
        validation_nll = evaluate_reference_nll(
            model,
            task,
            selection_starts,
            seq_len=args.seq_len,
            batch_size=batch_size,
            device=device,
        )
        if validation_nll < best_nll:
            best_nll = validation_nll
            best_state = clone_state(model.reference_head)
        history.append(
            {
                "stage": "reference",
                "epoch": epoch,
                "train_nll": total / max(1, steps),
                "validation_nll": validation_nll,
                "best_validation_nll": best_nll,
            }
        )
        print(
            f"reference epoch={epoch} train_nll={total / max(1, steps):.6f} "
            f"val_nll={validation_nll:.6f} best={best_nll:.6f}",
            flush=True,
        )
    load_state(model.reference_head, best_state, device)
    model.initialize_mixture_from_reference()
    return history


def fit_extension(
    model: LatentFlow,
    task: TaskData,
    selection_starts,
    args,
    batch_size: int,
    device: torch.device,
    *,
    representation_weight: float,
    nll_weight: float,
    history: list[dict],
) -> tuple[list[dict], dict]:
    model.select_reference_only(True)
    initial = evaluate_distribution(
        model,
        task,
        selection_starts,
        seq_len=args.seq_len,
        batch_size=batch_size,
        device=device,
    )
    best_state = clone_state(model)
    best_mse = initial["mse"]
    best_epoch = 0
    best_stage = "reference"

    model.configure_stage("C")
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    accumulation = max(1, math.ceil(args.effective_batch_size / batch_size))
    started_training = time.perf_counter()
    optimizer_steps = 0
    training = loader(
        task,
        task.training_starts,
        args.seq_len,
        batch_size,
        shuffle=True,
        seed=args.seed,
    )
    for epoch in range(1, args.extension_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        steps = 0
        for step, (inputs, targets) in enumerate(training, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss, parts, _, _, _ = model.training_objective(
                inputs,
                targets,
                sample_dual=None,
                fenchel_weight=0.0,
                nll_weight=nll_weight,
                representation_weight=representation_weight,
                augmented_weight=0.0,
                reconstruction_weight=0.0,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite final LatentFlow loss at epoch={epoch}, step={step}."
                )
            (loss / accumulation).backward()
            for name, value in parts.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            steps += 1
            if step % accumulation == 0 or step == len(training):
                nn.utils.clip_grad_norm_(model.trainable_parameters(), args.gradient_clip)
                optimizer.step()
                model.project_primal()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        validation = evaluate_distribution(
            model,
            task,
            selection_starts,
            seq_len=args.seq_len,
            batch_size=batch_size,
            device=device,
        )
        if validation["mse"] < best_mse:
            best_mse = validation["mse"]
            best_epoch = epoch
            best_stage = "extension"
            best_state = clone_state(model)
        row = {
            "stage": "extension",
            "epoch": epoch,
            "validation_mse": validation["mse"],
            "validation_mae": validation["mae"],
            "validation_nll": validation["nll"],
            "best_validation_mse": best_mse,
            **{name: value / max(1, steps) for name, value in sums.items()},
        }
        history.append(row)
        print(
            f"extension epoch={epoch}/{args.extension_epochs} "
            f"loss={row['total']:.6f} mse={row['point_mse']:.6f} "
            f"val={validation['mse']:.6f} best={best_mse:.6f}@{best_stage}:{best_epoch}",
            flush=True,
        )

    load_state(model, best_state, device)
    model.select_reference_only(best_stage == "reference")
    return history, {
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_validation_mse": best_mse,
        "training_ms_optimizer_step": (
            1000.0 * (time.perf_counter() - started_training) / max(1, optimizer_steps)
        ),
    }


def fit_final_model(
    model: LatentFlow,
    task: TaskData,
    args,
    *,
    representation_weight: float,
    nll_weight: float,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    selection_starts = chronological_selection(task.validation_starts)
    history = fit_reference_head(
        model, task, selection_starts, args, batch_size, device
    )
    return fit_extension(
        model,
        task,
        selection_starts,
        args,
        batch_size,
        device,
        representation_weight=representation_weight,
        nll_weight=nll_weight,
        history=history,
    )




