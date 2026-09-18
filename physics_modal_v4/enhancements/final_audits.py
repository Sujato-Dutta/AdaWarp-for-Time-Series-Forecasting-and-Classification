"""Final reviewer-facing controls for the locked no-exchange LatentFlow model.

This module adds only audits requested after architecture freeze:
three prior-dependence controls, reference-versus-extension accounting,
the early-fusion/X-to-Z identity audit, and dataset-wise mechanism summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from physics_modal_v4.enhancements.run import (
    CENTRAL_PROTOCOL,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    assert_no_exchange,
    experiment_args,
    fit_design,
    stable_hash,
    validate_locked_protocol,
)
from physics_modal_v4.experiments.protocol import (
    DATASETS,
    HORIZONS,
    SEEDS,
    clone_state,
    load_state,
    load_task,
    loader,
    resolve_device,
    seed_all,
    write_csv,
    write_json,
)
from physics_modal_v4.experiments.run import (
    FINAL_CANDIDATE,
    SELECTED_CONFIG,
    build_latentflow,
    task_batch_size,
)
from physics_modal_v4.experiments.training import (
    chronological_selection,
    evaluate_distribution,
    fit_reference_head,
    instantiate_backbone,
    torch_load,
)


PRIOR_PROTOCOL = "latentflow-prior-controls-all28-three-seed-no-exchange-v1"
REFERENCE_PROTOCOL = "latentflow-reference-extension-audit-all28-five-seed-v1"
FUSION_PROTOCOL = "latentflow-fusion-identity-audit-all28-five-seed-v1"

PRIOR_VARIANTS = {
    "no_sensor_group_priors": "no_sensor_group_priors",
    "permuted_process_families": "permuted_process_families",
    "neutral_process_scales": "neutral_process_scales",
}
CONTROL_MODES = {
    "early_fusion": "learned_inducing_elbo",
    "x_to_z": "x_to_z",
}


def _clustered_interval(values, clusters, *, seed: int, draws: int):
    """Vectorized cluster bootstrap with the same estimand as aggregate.py."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=str)
    names = np.unique(clusters)
    if not len(values):
        return float("nan"), float("nan")
    if len(names) == 1:
        point = float(values.mean())
        return point, point
    sums = np.asarray([values[clusters == name].sum() for name in names])
    counts = np.asarray([(clusters == name).sum() for name in names])
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    block = 4096
    for start in range(0, draws, block):
        size = min(block, draws - start)
        selected = rng.integers(0, len(names), size=(size, len(names)))
        estimates[start:start + size] = (
            sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_path(args, experiment: str, variant: str) -> Path:
    return (
        Path(args.output_root) / experiment / "runs" / variant
        / f"seed{args.seed}" / args.dataset / f"h{args.horizon}.json"
    )


def _audit_checkpoint_path(args, experiment: str, variant: str) -> Path:
    return (
        Path(args.output_root) / experiment / "checkpoints" / variant
        / f"seed{args.seed}" / args.dataset / f"h{args.horizon}.pt"
    )


def _source_checkpoint(args, variant: str) -> Path:
    if variant == "final_no_exchange":
        return (
            Path(args.source_root) / "checkpoints" / "LatentFlow"
            / f"seed{args.seed}" / args.dataset / f"h{args.horizon}.pt"
        )
    return (
        Path(args.output_root) / "central" / "checkpoints" / variant
        / f"seed{args.seed}" / args.dataset / f"h{args.horizon}.pt"
    )


def _source_metric(args, variant: str) -> Path:
    if variant == "final_no_exchange":
        return (
            Path(args.source_root) / "headline" / "runs" / "LatentFlow"
            / f"seed{args.seed}" / args.dataset / f"h{args.horizon}.json"
        )
    return (
        Path(args.output_root) / "central" / "runs" / variant
        / f"seed{args.seed}" / args.dataset / f"h{args.horizon}.json"
    )


def _protocol(args, name: str) -> dict:
    payload = {
        "audit_protocol": name,
        "candidate": FINAL_CANDIDATE,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "train_windows": args.max_train_windows,
        "validation_windows": args.max_validation_windows,
        "extension_epochs": args.extension_epochs,
        "cross_process_exchange": False,
        "primal_dual": False,
    }
    return {**payload, "audit_config_hash": stable_hash(payload)}


def run_prior_control(args) -> None:
    if args.seed not in (42, 43, 44):
        raise ValueError("Prior controls are locked to seeds 42, 43, and 44.")
    if args.variant not in PRIOR_VARIANTS:
        raise ValueError(f"Unknown prior control: {args.variant}")
    fit_design(
        args,
        experiment="prior_controls",
        variant=args.variant,
        scale="multiscale",
        bank="on",
        continuation="preserved",
        protocol=PRIOR_PROTOCOL,
        ablation_mode=PRIOR_VARIANTS[args.variant],
        namespace="prior_controls",
    )
    _audit_prior_control_selection(args)


def _audit_prior_control_selection(args) -> None:
    """Re-evaluate an existing control with its saved validation selector."""
    output = _result_path(args, "prior_control_selected_audit", args.variant)
    if output.exists() and not args.force:
        print(
            f"skip-complete prior-selected-audit variant={args.variant} "
            f"dataset={args.dataset} h={args.horizon} seed={args.seed}",
            flush=True,
        )
        return

    source = _result_path(args, "prior_controls", args.variant)
    checkpoint = (
        Path(args.output_root) / "prior_controls" / "checkpoints"
        / args.variant / f"seed{args.seed}" / args.dataset
        / f"h{args.horizon}.pt"
    )
    if not source.exists() or not checkpoint.exists():
        raise FileNotFoundError(
            f"Prior-control audit requires {source} and {checkpoint}."
        )
    source_row = _json(source)
    payload = torch_load(checkpoint)
    task = _load_task(args)
    device = resolve_device(args.device)
    base = instantiate_backbone(payload["base_config"])
    model = build_latentflow(
        base,
        task,
        payload.get("selected_config", SELECTED_CONFIG[args.dataset]),
        mode=str(payload.get("ablation_mode", args.variant)),
        unfreeze_blocks=int(payload.get("continuation_unfreeze_blocks", 2)),
    ).to(device)
    assert_no_exchange(model)
    model.load_state_dict(payload["state_dict"], strict=True)
    selected_stage = str(payload["selection"]["best_stage"])
    model.select_reference_only(selected_stage == "reference")
    selected = evaluate_distribution(
        model,
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=task_batch_size(args, args.dataset),
        device=device,
    )
    row = {
        **source_row,
        "mse": selected["mse"],
        "mae": selected["mae"],
        "nll": selected["nll"],
        "validation_selected_stage": selected_stage,
        "source_reported_mse": float(source_row["mse"]),
        "source_reported_mae": float(source_row["mae"]),
        "source_metric_mse_difference": (
            selected["mse"] - float(source_row["mse"])
        ),
        "selector_state_repaired": True,
        **_protocol(args, PRIOR_PROTOCOL),
    }
    write_json(output, row)
    print(
        f"prior-selected-audit-result variant={args.variant} "
        f"dataset={args.dataset} h={args.horizon} seed={args.seed} "
        f"selected={selected_stage} mse={selected['mse']:.6f} "
        f"source_delta={row['source_metric_mse_difference']:+.9f}",
        flush=True,
    )


def _load_model(args, task, variant: str):
    path = _source_checkpoint(args, variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. The audit requires final headline and five-seed "
            "central checkpoints, not metrics alone."
        )
    payload = torch_load(path)
    if variant == "final_no_exchange" and payload.get("candidate") != FINAL_CANDIDATE:
        raise RuntimeError(f"Stale final checkpoint: {path}")
    base = instantiate_backbone(payload["base_config"])
    mode = "final" if variant == "final_no_exchange" else CONTROL_MODES[variant]
    model = build_latentflow(
        base,
        task,
        SELECTED_CONFIG[args.dataset],
        mode=mode,
        unfreeze_blocks=2,
    ).to(resolve_device(args.device))
    assert_no_exchange(model)
    model.load_state_dict(payload["state_dict"], strict=True)
    selection = payload.get("selection", {})
    return model, payload, path, selection


def _evaluate_mode(model, task, args, reference_only: bool) -> dict[str, float]:
    model.select_reference_only(reference_only)
    return evaluate_distribution(
        model,
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=task_batch_size(args, args.dataset),
        device=resolve_device(args.device),
    )


def _fit_best_raw_extension(model, task, args, cache: Path) -> tuple[dict, bool]:
    """Return the best extension-only validation candidate, excluding fallback."""
    device = resolve_device(args.device)
    if cache.exists() and not args.force:
        payload = torch_load(cache)
        if payload.get("audit_protocol") == REFERENCE_PROTOCOL:
            load_state(model, payload["state_dict"], device)
            model.select_reference_only(False)
            return payload["selection"], False

    model.configure_stage("C")
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batch_size = task_batch_size(args, args.dataset)
    accumulation = max(1, math.ceil(args.effective_batch_size / batch_size))
    training = loader(
        task,
        task.training_starts,
        args.seq_len,
        batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_starts = chronological_selection(task.validation_starts)
    representation_weight = (
        0.0 if model.structural_only
        else SELECTED_CONFIG[args.dataset]["representation_weight"]
    )
    best_state = None
    best_mse = float("inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, args.extension_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (inputs, targets) in enumerate(training, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss, _, _, _, _ = model.training_objective(
                inputs,
                targets,
                sample_dual=None,
                fenchel_weight=0.0,
                nll_weight=args.nll_weight,
                representation_weight=representation_weight,
                augmented_weight=0.0,
                reconstruction_weight=0.0,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fallback-audit extension loss.")
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(training):
                nn.utils.clip_grad_norm_(model.trainable_parameters(), args.gradient_clip)
                optimizer.step()
                model.project_primal()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate_distribution(
            model,
            task,
            validation_starts,
            seq_len=args.seq_len,
            batch_size=batch_size,
            device=device,
        )
        if validation["mse"] < best_mse:
            best_mse = validation["mse"]
            best_epoch = epoch
            best_state = clone_state(model)
        print(
            f"raw-extension-audit variant={args.variant} dataset={args.dataset} "
            f"h={args.horizon} seed={args.seed} epoch={epoch} "
            f"val={validation['mse']:.6f} best={best_mse:.6f}@{best_epoch}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("No raw extension checkpoint was produced.")
    load_state(model, best_state, device)
    model.select_reference_only(False)
    selection = {
        "best_raw_extension_epoch": best_epoch,
        "best_raw_extension_validation_mse": best_mse,
        "elapsed_seconds": time.perf_counter() - started,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "selection": selection,
            **_protocol(args, REFERENCE_PROTOCOL),
        },
        cache,
    )
    return selection, True


def _fresh_pre_extension_model(args, task, variant: str, payload: dict):
    """Replay the original seeded initialization and reference-head fit."""
    device = resolve_device(args.device)
    base = instantiate_backbone(payload["base_config"])
    seed_all(args.seed)
    mode = "final" if variant == "final_no_exchange" else CONTROL_MODES[variant]
    model = build_latentflow(
        base, task, SELECTED_CONFIG[args.dataset],
        mode=mode, unfreeze_blocks=2,
    ).to(device)
    assert_no_exchange(model)
    fit_reference_head(
        model,
        task,
        chronological_selection(task.validation_starts),
        args,
        task_batch_size(args, args.dataset),
        device,
    )
    return model


def _candidate_audit(args, task, variant: str) -> tuple[nn.Module, dict]:
    model, payload, source, selection = _load_model(args, task, variant)
    selected_stage = str(selection.get("best_stage", "unknown"))
    reference = _evaluate_mode(model, task, args, True)
    extension_retrained = False
    if selected_stage == "extension":
        extension = _evaluate_mode(model, task, args, False)
        raw_selection = {
            "best_raw_extension_epoch": int(selection.get("best_epoch", -1)),
            "best_raw_extension_validation_mse": float(
                selection.get("best_validation_mse", float("nan"))
            ),
        }
    elif selected_stage == "reference":
        cache = _audit_checkpoint_path(args, "reference_extension_audit", variant)
        cached = cache.exists() and not args.force
        if not cached:
            # The selected checkpoint retained the pre-extension candidate but
            # not the discarded extension or RNG state. Replay from the frozen
            # backbone and original seed; mark the result explicitly because
            # GPU kernels need not be bitwise deterministic across executions.
            del model
            if resolve_device(args.device).type == "cuda":
                torch.cuda.empty_cache()
            model = _fresh_pre_extension_model(args, task, variant, payload)
        raw_selection, extension_retrained = _fit_best_raw_extension(
            model, task, args, cache
        )
        extension = _evaluate_mode(model, task, args, False)
    else:
        raise RuntimeError(f"Unknown checkpoint selection stage {selected_stage!r}: {source}")
    selected = reference if selected_stage == "reference" else extension
    source_metrics = _json(_source_metric(args, variant))
    source_delta = selected["mse"] - float(source_metrics["mse"])
    source_match = abs(source_delta) <= 5e-6
    if not source_match:
        print(
            f"selector-audit-warning variant={variant} dataset={args.dataset} "
            f"h={args.horizon} seed={args.seed} "
            f"source_mse={float(source_metrics['mse']):.9f} "
            f"selected_checkpoint_mse={selected['mse']:.9f} "
            f"delta={source_delta:+.9f}",
            flush=True,
        )
    model.select_reference_only(selected_stage == "reference")
    row = {
        "variant": variant,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "validation_selected_stage": selected_stage,
        "extension_retrained": extension_retrained,
        "extension_replay_from_seed": bool(
            selected_stage == "reference" and extension_retrained
        ),
        "source_reported_mse": float(source_metrics["mse"]),
        "source_reported_mae": float(source_metrics["mae"]),
        "source_metric_mse_difference": source_delta,
        "source_metric_matches_selected_checkpoint": source_match,
        **{f"reference_{k}": v for k, v in reference.items()},
        **{f"extension_{k}": v for k, v in extension.items()},
        **{f"selected_{k}": v for k, v in selected.items()},
        **raw_selection,
    }
    return model, row


@torch.no_grad()
def _prediction_difference(left, right, task, args, *, left_reference: bool, right_reference: bool) -> dict:
    left.select_reference_only(left_reference)
    right.select_reference_only(right_reference)
    left.eval()
    right.eval()
    device = resolve_device(args.device)
    count = 0
    absolute = squared = maximum = 0.0
    exact = True
    close = True
    left_hash = hashlib.sha256()
    right_hash = hashlib.sha256()
    for inputs, _ in loader(
        task, task.test_starts, args.seq_len,
        task_batch_size(args, args.dataset),
    ):
        inputs = inputs.to(device)
        a = left.forward_distribution(inputs)["mean"].detach().cpu().contiguous()
        b = right.forward_distribution(inputs)["mean"].detach().cpu().contiguous()
        delta = (a - b).double()
        count += delta.numel()
        absolute += float(delta.abs().sum())
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        exact = exact and torch.equal(a, b)
        close = close and torch.allclose(a, b, rtol=1e-6, atol=1e-7)
        left_hash.update(a.numpy().tobytes())
        right_hash.update(b.numpy().tobytes())
    return {
        "mean_absolute_difference": absolute / max(1, count),
        "root_mean_square_difference": math.sqrt(squared / max(1, count)),
        "maximum_absolute_difference": maximum,
        "bitwise_identical": exact,
        "allclose_rtol1e_6_atol1e_7": close,
        "left_prediction_sha256": left_hash.hexdigest(),
        "right_prediction_sha256": right_hash.hexdigest(),
    }


def _load_task(args):
    return load_task(
        Path(args.data_root), args.dataset, args.horizon,
        seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )


def run_reference_audit(args) -> None:
    args.variant = "final_no_exchange"
    output = _result_path(args, "reference_extension_audit", args.variant)
    if output.exists() and not args.force:
        print(f"skip-complete reference-audit dataset={args.dataset} h={args.horizon} seed={args.seed}")
        return
    task = _load_task(args)
    _, row = _candidate_audit(args, task, args.variant)
    row.update(_protocol(args, REFERENCE_PROTOCOL))
    write_json(output, row)
    print(
        f"reference-audit-result dataset={args.dataset} h={args.horizon} seed={args.seed} "
        f"selected={row['validation_selected_stage']} reference_mse={row['reference_mse']:.6f} "
        f"extension_mse={row['extension_mse']:.6f} selected_mse={row['selected_mse']:.6f} "
        f"retrained={row['extension_retrained']}", flush=True,
    )


def run_fusion_audit(args) -> None:
    output = _result_path(args, "fusion_anomaly_audit", "early_fusion_vs_x_to_z")
    if output.exists() and not args.force:
        print(f"skip-complete fusion-audit dataset={args.dataset} h={args.horizon} seed={args.seed}")
        return
    task = _load_task(args)
    args.variant = "early_fusion"
    early, early_row = _candidate_audit(args, task, "early_fusion")
    args.variant = "x_to_z"
    structural, structural_row = _candidate_audit(args, task, "x_to_z")
    extension_difference = _prediction_difference(
        early, structural, task, args,
        left_reference=False, right_reference=False,
    )
    selected_difference = _prediction_difference(
        early, structural, task, args,
        left_reference=early_row["validation_selected_stage"] == "reference",
        right_reference=structural_row["validation_selected_stage"] == "reference",
    )
    row = {
        "variant": "early_fusion_vs_x_to_z",
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        **{f"early_{k}": v for k, v in early_row.items() if k not in {"variant", "dataset", "horizon", "seed"}},
        **{f"x_to_z_{k}": v for k, v in structural_row.items() if k not in {"variant", "dataset", "horizon", "seed"}},
        **{f"extension_{k}": v for k, v in extension_difference.items()},
        **{f"selected_{k}": v for k, v in selected_difference.items()},
        **_protocol(args, FUSION_PROTOCOL),
    }
    write_json(output, row)
    print(
        f"fusion-audit-result dataset={args.dataset} h={args.horizon} seed={args.seed} "
        f"extension_identical={row['extension_bitwise_identical']} "
        f"extension_max_abs={row['extension_maximum_absolute_difference']:.9g} "
        f"selected_identical={row['selected_bitwise_identical']} "
        f"selected_max_abs={row['selected_maximum_absolute_difference']:.9g}", flush=True,
    )


def _collect(root: Path, experiment: str, protocol: str) -> pd.DataFrame:
    rows = []
    for path in sorted((root / experiment / "runs").glob("**/*.json")):
        row = _json(path)
        if row.get("audit_protocol", row.get("enhancement_protocol")) == protocol:
            rows.append(row)
    return pd.DataFrame(rows)


def _require(frame: pd.DataFrame, expected: int, name: str, allow: bool) -> None:
    if len(frame) != expected:
        message = f"{name}: expected {expected} rows, found {len(frame)}"
        if not allow:
            raise RuntimeError(message)
        print("WARNING", message, flush=True)


def _statistics_root(output_root: Path) -> Path:
    for candidate in (
        output_root / "statistics",
        output_root / "final_enhancement_statistics",
    ):
        if (candidate / "central_runs.csv").exists():
            return candidate
    raise FileNotFoundError(
        f"No central_runs.csv under {output_root / 'statistics'} or "
        f"{output_root / 'final_enhancement_statistics'}."
    )


def _summarize_prior(args, output: Path) -> int:
    root = Path(args.output_root)
    controls = _collect(root, "prior_control_selected_audit", PRIOR_PROTOCOL)
    _require(controls, 252, "selected prior controls", args.allow_incomplete)
    if controls.empty:
        return 0
    final = pd.read_csv(_statistics_root(root) / "central_runs.csv")
    final = final[(final.variant == "final_no_exchange") & final.seed.isin([42, 43, 44])]
    _require(final, 84, "prior reference", args.allow_incomplete)
    merged = controls.merge(
        final[["dataset", "horizon", "seed", "mse", "mae"]],
        on=["dataset", "horizon", "seed"], suffixes=("", "_final"), validate="many_to_one",
    )
    rows = []
    for (variant, dataset), group in merged.groupby(["variant", "dataset"], sort=False):
        for metric in ("mse", "mae"):
            delta = 100.0 * (group[metric].to_numpy() / group[f"{metric}_final"].to_numpy() - 1.0)
            lo, hi = _clustered_interval(
                delta.tolist(), group.seed.astype(str).tolist(),
                seed=2027, draws=args.bootstrap_draws,
            )
            rows.append({
                "variant": variant, "dataset": dataset, "metric": metric,
                "cells": len(group), "mean": float(group[metric].mean()),
                "final_mean": float(group[f"{metric}_final"].mean()),
                "mean_relative_degradation_percent": float(delta.mean()),
                "seed_clustered_ci95_low_percent": lo,
                "seed_clustered_ci95_high_percent": hi,
                "control_better": int((delta < -1e-8).sum()),
                "ties": int((np.abs(delta) <= 1e-8).sum()),
                "control_worse": int((delta > 1e-8).sum()),
            })
    write_csv(output / "prior_control_runs.csv", controls.to_dict("records"))
    write_csv(output / "prior_control_dataset_effects.csv", rows)
    return len(controls)


def _summarize_reference(args, output: Path) -> int:
    frame = _collect(Path(args.output_root), "reference_extension_audit", REFERENCE_PROTOCOL)
    _require(frame, 140, "reference/extension audit", args.allow_incomplete)
    if frame.empty:
        return 0
    rows = []
    for dataset, group in list(frame.groupby("dataset", sort=False)) + [("Overall", frame)]:
        ref_better = (group.reference_mse < group.extension_mse - 1e-10)
        ext_better = (group.extension_mse < group.reference_mse - 1e-10)
        rows.append({
            "dataset": dataset, "fits": len(group),
            "validation_selected_reference": int((group.validation_selected_stage == "reference").sum()),
            "validation_selected_extension": int((group.validation_selected_stage == "extension").sum()),
            "test_reference_better": int(ref_better.sum()),
            "test_extension_better": int(ext_better.sum()),
            "test_ties": int((~ref_better & ~ext_better).sum()),
            "reference_mse": float(group.reference_mse.mean()),
            "raw_extension_mse": float(group.extension_mse.mean()),
            "selected_mse": float(group.selected_mse.mean()),
            "fallback_regret_vs_test_oracle_percent": float(
                100.0 * (group.selected_mse / group[["reference_mse", "extension_mse"]].min(axis=1) - 1.0).mean()
            ),
        })
    write_csv(output / "reference_extension_runs.csv", frame.to_dict("records"))
    write_csv(output / "reference_extension_by_dataset.csv", rows)
    return len(frame)


def _summarize_fusion(args, output: Path) -> int:
    frame = _collect(Path(args.output_root), "fusion_anomaly_audit", FUSION_PROTOCOL)
    _require(frame, 140, "fusion anomaly audit", args.allow_incomplete)
    if frame.empty:
        return 0
    rows = []
    for dataset, group in list(frame.groupby("dataset", sort=False)) + [("Overall", frame)]:
        rows.append({
            "dataset": dataset, "fits": len(group),
            "early_reference_selections": int((group.early_validation_selected_stage == "reference").sum()),
            "x_to_z_reference_selections": int((group.x_to_z_validation_selected_stage == "reference").sum()),
            "selected_bitwise_identical": int(group.selected_bitwise_identical.sum()),
            "selected_allclose": int(group.selected_allclose_rtol1e_6_atol1e_7.sum()),
            "extension_bitwise_identical": int(group.extension_bitwise_identical.sum()),
            "extension_allclose": int(group.extension_allclose_rtol1e_6_atol1e_7.sum()),
            "maximum_selected_absolute_difference": float(group.selected_maximum_absolute_difference.max()),
            "maximum_extension_absolute_difference": float(group.extension_maximum_absolute_difference.max()),
            "mean_selected_absolute_difference": float(group.selected_mean_absolute_difference.mean()),
            "mean_extension_absolute_difference": float(group.extension_mean_absolute_difference.mean()),
        })
    write_csv(output / "fusion_anomaly_runs.csv", frame.to_dict("records"))
    write_csv(output / "fusion_anomaly_by_dataset.csv", rows)
    return len(frame)


def _mechanism_breakdown(args, output: Path) -> int:
    root = _statistics_root(Path(args.output_root))
    central = pd.read_csv(root / "central_runs.csv")
    comparisons = {
        "preserved_vs_early_fusion": "early_fusion",
        "full_vs_single_scale": "no_multiscale",
        "full_vs_no_bank": "no_linear_bank",
    }
    final = central[central.variant == "final_no_exchange"]
    rows = []
    for label, variant in comparisons.items():
        control = central[central.variant == variant]
        merged = final.merge(
            control, on=["dataset", "horizon", "seed"],
            suffixes=("_full", "_control"), validate="one_to_one",
        )
        for dataset, group in merged.groupby("dataset", sort=False):
            for metric in ("mse", "mae"):
                degradation = 100.0 * (
                    group[f"{metric}_control"].to_numpy()
                    / group[f"{metric}_full"].to_numpy() - 1.0
                )
                lo, hi = _clustered_interval(
                    degradation.tolist(), group.seed.astype(str).tolist(),
                    seed=1729, draws=args.bootstrap_draws,
                )
                rows.append({
                    "comparison": label, "dataset": dataset, "metric": metric,
                    "cells": len(group),
                    "full_mean": float(group[f"{metric}_full"].mean()),
                    "control_mean": float(group[f"{metric}_control"].mean()),
                    "control_relative_degradation_percent": float(degradation.mean()),
                    "seed_clustered_ci95_low_percent": lo,
                    "seed_clustered_ci95_high_percent": hi,
                    "full_better": int((degradation > 1e-8).sum()),
                    "ties": int((np.abs(degradation) <= 1e-8).sum()),
                    "control_better": int((degradation < -1e-8).sum()),
                })
    write_csv(output / "mechanism_effects_by_dataset.csv", rows)
    factorial = pd.read_csv(root / "factorial_effects.csv")
    selected = factorial[
        (factorial.dataset != "ALL")
        & (factorial.metric.isin(["mse", "mae"]))
        & (factorial.effect.isin(["scale", "bank", "continuation"]))
    ]
    write_csv(output / "factorial_main_effects_by_dataset.csv", selected.to_dict("records"))
    return len(rows)


def run_analysis(args) -> None:
    output = Path(args.analysis_output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "prior_rows": _summarize_prior(args, output),
        "reference_audit_rows": _summarize_reference(args, output),
        "fusion_audit_rows": _summarize_fusion(args, output),
        "mechanism_summary_rows": _mechanism_breakdown(args, output),
    }
    (output / "final_audit_status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"latentflow-final-audits complete {summary}", flush=True)


def run_smoke(args) -> None:
    device = resolve_device(args.device)
    from physics_modal_v4.experiments.training import backbone_config
    task = type("Task", (), {"dataset": "ETTh1", "values": np.zeros((200, 7), np.float32)})()
    inputs = torch.randn(2, 96, 7, device=device)
    for mode in PRIOR_VARIANTS.values():
        base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
        model = build_latentflow(
            base, task, {"latent_dim": 4, "inducing_points": 8},
            mode=mode, unfreeze_blocks=2,
        ).to(device)
        assert_no_exchange(model)
        output = model.forward_distribution(inputs)["mean"]
        if output.shape != (2, 96, 7) or not torch.isfinite(output).all():
            raise RuntimeError(f"Prior-control smoke failed for {mode}.")
    if len(PRIOR_VARIANTS) * 3 * len(DATASETS) * len(HORIZONS) != 252:
        raise RuntimeError("Prior control cardinality is not 252.")
    checkpoint_count = "not_checked"
    if args.check_checkpoints:
        missing = []
        original = (args.dataset, args.horizon, args.seed)
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    args.dataset, args.horizon, args.seed = dataset, horizon, seed
                    for variant in ("final_no_exchange", "early_fusion", "x_to_z"):
                        for path in (_source_checkpoint(args, variant), _source_metric(args, variant)):
                            if not path.exists():
                                missing.append(str(path))
        args.dataset, args.horizon, args.seed = original
        if missing:
            preview = "\n".join(missing[:12])
            raise FileNotFoundError(
                f"Checkpoint audit prerequisites missing ({len(missing)} files).\n{preview}"
            )
        checkpoint_count = 420
    print(
        "LATENTFLOW-FINAL-AUDITS-SMOKE PASSED prior_cells=252 "
        f"audit_groups=140 checkpoint_sets={checkpoint_count}", flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=(
        "smoke", "prior-control-task", "reference-audit-task",
        "fusion-audit-task", "analysis",
    ), required=True)
    result.add_argument("--variant", choices=tuple(PRIOR_VARIANTS), default="no_sensor_group_priors")
    result.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    result.add_argument("--horizon", choices=HORIZONS, type=int, default=96)
    result.add_argument("--seed", choices=SEEDS, type=int, default=42)
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
    result.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    result.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    result.add_argument("--analysis-output", default=str(DEFAULT_OUTPUT / "final_audits"))
    result.add_argument("--device", default="auto")
    result.add_argument("--bootstrap-draws", type=int, default=20_000)
    result.add_argument("--allow-incomplete", action="store_true")
    result.add_argument("--check-checkpoints", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode != "analysis":
        validate_locked_protocol(args)
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "prior-control-task":
        run_prior_control(args)
    elif args.mode == "reference-audit-task":
        run_reference_audit(args)
    elif args.mode == "fusion-audit-task":
        run_fusion_audit(args)
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
