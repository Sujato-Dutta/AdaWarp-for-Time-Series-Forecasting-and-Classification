"""Controlled revision experiments for finalized no-exchange LatentFlow.

This runner owns experiments that are not part of the release benchmark:
five-seed central removals, a balanced 2x2x2 factorial, full-data checks,
multi-seed comparator checks, and channel-order sensitivity. It reuses the
locked parent model and protocol but writes all artifacts under enhancements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

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
    profiler_flops,
    task_batch_size,
)
from physics_modal_v4.experiments.training import (
    backbone_config,
    build_backbone,
    ensure_backbone,
    fit_final_model,
    instantiate_backbone,
    torch_load,
)
from physics_modal_v4.models.registry import DLinearXCPD, ModelSpec, build_model


ROOT = Path("physics_modal_v4/enhancements")
DEFAULT_OUTPUT = ROOT / "results"
DEFAULT_SOURCE = Path("physics_modal_v4/results/final")

CENTRAL_PROTOCOL = "latentflow-central-all28-five-seed-no-exchange-v2-selector-safe"
FACTORIAL_PROTOCOL = "latentflow-factorial-2x2x2-all28-three-seed-no-exchange-v2-selector-safe"
FULL_DATA_PROTOCOL = "latentflow-full-data-all28-seed42-no-exchange-v2-selector-safe"
COMPARATOR_PROTOCOL = "strongest-nontimepro-by-dataset-seeds43-46-v1"
CHANNEL_PROTOCOL = "latentflow-channel-order-pilot-seed42-no-exchange-v2-selector-safe"
CHANNEL_EXPANSION_PROTOCOL = (
    "latentflow-channel-order-expansion-seed42-no-exchange-v2-selector-safe"
)
CAPACITY_PROTOCOL = "latentflow-capacity-controls-all28-seed42-no-exchange-v2-selector-safe"
DIAGNOSTIC_PROTOCOL = "latentflow-gate-bank-diagnostics-seed42-no-exchange-v1"
MATCHED_CONTROL_PROTOCOL = "latentflow-matched-nongp-multibranch-all28-seed42-v1"
MATCHED_CONTROL_MULTI_PROTOCOL = (
    "latentflow-matched-nongp-multibranch-all28-seeds43-46-v1"
)

DATA_BUDGET_PROTOCOL = 'latentflow-nested-data-budget-all28-seed42-no-exchange-v2-selector-safe'
DATA_BUDGETS = (128, 512, 2048, 'all')
TIMEPRO_DATA_BUDGET_PROTOCOL = (
    "timepro-nested-data-budget-all28-seed42-fixed-validation-v1"
)
TIMEPRO_DATA_BUDGETS = (128, 512, "all")

CENTRAL_VARIANTS = (
    "final_no_exchange",
    "x_to_z",
    "early_fusion",
    "no_multiscale",
    "no_linear_bank",
)
FACTOR_LEVELS = {
    "scale": ("single", "multiscale"),
    "bank": ("off", "on"),
    "continuation": ("early", "preserved"),
}
STRONGEST_NON_TIMEPRO = {
    "ETTh1": "iTransformer",
    "ETTh2": "TimeMixer",
    "ETTm1": "xCPD",
    "ETTm2": "TimeMixer",
    "Weather": "VPNet",
    "Electricity": "VPNet",
    "Traffic": "VPNet",
}
DIAGNOSTIC_TASKS = (
    ("ETTh1", 96),
    ("ETTh1", 720),
    ("ETTm1", 336),
    ("ETTm1", 720),
    ("Weather", 720),
    ("Electricity", 336),
    ("Traffic", 192),
)


class FieldOnlyMixer(nn.Module):
    """Parameter-free replacement used when the complete linear bank is removed."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_zeros((features.shape[0], 2))


def remove_linear_bank(base: nn.Module) -> nn.Module:
    """Physically remove the bank and its learned outer router from a backbone."""

    base.use_linear_field = False
    base.linear_field = nn.Identity()
    base.mean_mixer = FieldOnlyMixer()
    return base


def instantiate_design_backbone(config: dict, *, bank: str) -> nn.Module:
    base = instantiate_backbone(config, require_final=bank == "on")
    return base if bank == "on" else remove_linear_bank(base)


def stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def experiment_args(
    args,
    *,
    max_train_windows: int | None = None,
    max_validation_windows: int | None = None,
):
    return SimpleNamespace(
        data_root=args.data_root,
        output_root=args.output_root,
        dataset=args.dataset,
        horizon=args.horizon,
        seed=args.seed,
        seq_len=args.seq_len,
        max_train_windows=(
            args.max_train_windows if max_train_windows is None else max_train_windows
        ),
        max_validation_windows=(
            args.max_validation_windows
            if max_validation_windows is None
            else max_validation_windows
        ),
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
        force=args.force,
    )


def protocol_payload(args, protocol: str) -> dict:
    payload = {
        "enhancement_protocol": protocol,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "max_train_windows": args.max_train_windows,
        "max_validation_windows": args.max_validation_windows,
        "reference_epochs": args.reference_epochs,
        "extension_epochs": args.extension_epochs,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "batch_size": args.batch_size,
        "wide_batch_size": args.wide_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "nll_weight": args.nll_weight,
        "selected_config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in SELECTED_CONFIG[args.dataset].items()
        },
        "final_candidate": FINAL_CANDIDATE,
        "cross_process_exchange": False,
        "primal_dual": False,
    }
    return {**payload, "config_hash": stable_hash(payload)}


def validate_locked_protocol(args, *, allow_training_window_override: bool = False) -> None:
    path = ROOT / "protocol.json"
    locked = json.loads(path.read_text(encoding="utf-8"))
    if locked["architecture"]["candidate"] != FINAL_CANDIDATE:
        raise RuntimeError("protocol.json candidate does not match release code.")
    expected = {
        "sequence_length": args.seq_len,
        "training_windows": args.max_train_windows,
        "validation_windows": args.max_validation_windows,
        "reference_epochs": args.reference_epochs,
        "extension_epochs": args.extension_epochs,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "nll_weight": args.nll_weight,
        "effective_batch_size": args.effective_batch_size,
    }
    for name, actual in expected.items():
        if name == 'training_windows' and allow_training_window_override:
            continue
        locked_value = (
            locked["selection"][name]
            if name == "validation_windows"
            else locked["training"][name]
        )
        if actual != locked_value:
            raise RuntimeError(
                f"Locked protocol mismatch for {name}: {actual!r} != {locked_value!r}."
            )
    if list(DATASETS) != locked["datasets"] or list(HORIZONS) != locked["horizons"]:
        raise RuntimeError("Locked dataset or horizon matrix does not match code.")
    if STRONGEST_NON_TIMEPRO != locked["strongest_non_timepro_comparator_selected_from_seed42"]:
        raise RuntimeError("Locked strongest-comparator map does not match code.")
    for dataset, config in SELECTED_CONFIG.items():
        normalized = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in config.items()
        }
        if normalized != locked["selected_hyperparameters"][dataset]:
            raise RuntimeError(f"Locked hyperparameters disagree for {dataset}.")


def artifact_path(
    args,
    experiment: str,
    kind: str,
    variant: str,
    suffix: str,
) -> Path:
    return (
        Path(args.output_root)
        / experiment
        / kind
        / variant
        / f"seed{args.seed}"
        / args.dataset
        / f"h{args.horizon}.{suffix}"
    )


def result_path(args, experiment: str, variant: str) -> Path:
    return artifact_path(args, experiment, "runs", variant, "json")


def checkpoint_path(args, experiment: str, variant: str) -> Path:
    return artifact_path(args, experiment, "checkpoints", variant, "pt")


def history_path(args, experiment: str, variant: str) -> Path:
    return artifact_path(args, experiment, "history", variant, "csv")


def source_metric_path(args) -> Path:
    return (
        Path(args.source_root)
        / "headline"
        / "runs"
        / "LatentFlow"
        / f"seed{args.seed}"
        / args.dataset
        / f"h{args.horizon}.json"
    )


def source_checkpoint_path(args) -> Path:
    return (
        Path(args.source_root)
        / "checkpoints"
        / "LatentFlow"
        / f"seed{args.seed}"
        / args.dataset
        / f"h{args.horizon}.pt"
    )


def assert_no_exchange(model: nn.Module) -> None:
    continuation = getattr(model, "continuation", None)
    if continuation is None:
        raise TypeError("Expected a LatentFlow model with a continuation module.")
    allocated = sum(len(blocks) for blocks in continuation.exchange_blocks)
    named = [name for name, _ in model.named_parameters() if "exchange_blocks" in name]
    if allocated or named or not continuation.disable_cross_exchange:
        raise RuntimeError(
            f"No-exchange invariant failed: modules={allocated}, parameters={named}."
        )


def restore_validation_selector(model: nn.Module, selection: dict) -> bool:
    """Restore the discrete inference path that is not part of state_dict."""

    stage = str(selection.get("best_stage", ""))
    if stage not in {"reference", "extension"}:
        raise RuntimeError(f"Invalid validation-selected stage: {stage!r}")
    reference_only = stage == "reference"
    model.select_reference_only(reference_only)
    if bool(model.reference_only) != reference_only:
        raise RuntimeError("Failed to restore LatentFlow validation selector state.")
    return reference_only


def _load_source_payload(args) -> dict:
    path = source_checkpoint_path(args)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing revised LatentFlow checkpoint {path}. Run headline first."
        )
    payload = torch_load(path)
    if payload.get("candidate") != FINAL_CANDIDATE:
        raise RuntimeError(
            f"Checkpoint {path} is {payload.get('candidate', 'unknown')}, "
            f"not {FINAL_CANDIDATE}."
        )
    return payload


def load_source_backbone(args):
    payload = _load_source_payload(args)
    state = {
        name[len("reference_base."):]: value
        for name, value in payload["state_dict"].items()
        if name.startswith("reference_base.")
    }
    base = instantiate_backbone(payload["base_config"])
    base.load_state_dict(state, strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base, payload


def load_source_model(args, task: TaskData, device: torch.device):
    payload = _load_source_payload(args)
    base = instantiate_backbone(payload["base_config"])
    model = build_latentflow(
        base,
        task,
        SELECTED_CONFIG[args.dataset],
        mode="final",
        unfreeze_blocks=2,
    ).to(device)
    assert_no_exchange(model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _existing_valid(path: Path, protocol: str, *, force: bool) -> dict | None:
    if force or not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("enhancement_protocol") != protocol:
        print(f"replace-stale path={path}", flush=True)
        return None
    if row.get("cross_process_exchange") is not False:
        raise RuntimeError(f"Exchange-enabled enhancement artifact is invalid: {path}")
    return row


def reuse_final_result(args, experiment: str, variant: str, protocol: str) -> dict:
    source = source_metric_path(args)
    if not source.exists():
        raise FileNotFoundError(f"Missing revised headline metric {source}.")
    row = json.loads(source.read_text(encoding="utf-8"))
    if (
        row.get("candidate") != FINAL_CANDIDATE
        or row.get("cross_process_exchange") is not False
    ):
        raise RuntimeError(f"Refusing stale or exchange-enabled result: {source}")
    updated = {
        **row,
        "experiment": experiment,
        "variant": variant,
        "scale": "multiscale",
        "bank": "on",
        "continuation": "preserved",
        "structural_only": False,
        "continuation_unfreeze_blocks": 2,
        "reused_source": str(source),
        **protocol_payload(args, protocol),
    }
    write_json(result_path(args, experiment, variant), updated)
    return updated


def _load_or_train_backbone(
    args,
    task: TaskData,
    device: torch.device,
    patches: tuple[int, ...],
    *,
    bank: str,
    namespace: str,
    use_source_when_exact: bool,
):
    selected = tuple(SELECTED_CONFIG[args.dataset]["patch_lens"])
    if use_source_when_exact and bank == "on" and patches == selected:
        return load_source_backbone(args)
    switches = None if bank == "on" else {"use_linear_field": False}
    scale = "single" if len(patches) == 1 else "multiscale"
    path = ensure_backbone(
        experiment_args(args),
        task,
        device,
        patches,
        switch_overrides=switches,
        checkpoint_tag=f"{namespace}_scale-{scale}_bank-{bank}",
    )
    payload = torch_load(path)
    base = build_backbone(payload, require_final=bank == "on")
    if bank == "off":
        remove_linear_bank(base)
    return base, payload


def _mode_and_weight(continuation: str, structural_only: bool, config: dict):
    if structural_only:
        return "x_to_z", 0.0
    if continuation == "early":
        return "learned_inducing_elbo", config["representation_weight"]
    if continuation == "preserved":
        return "final", config["representation_weight"]
    raise ValueError(f"Unknown continuation level: {continuation}")


def fit_design(
    args,
    *,
    experiment: str,
    variant: str,
    scale: str,
    bank: str,
    continuation: str,
    protocol: str,
    structural_only: bool = False,
    task: TaskData | None = None,
    namespace: str | None = None,
    use_source_backbone: bool = True,
    ablation_mode: str | None = None,
) -> dict:
    output = result_path(args, experiment, variant)
    existing = _existing_valid(output, protocol, force=args.force)
    if existing is not None:
        print(
            f"skip-complete experiment={experiment} variant={variant} "
            f"dataset={args.dataset} h={args.horizon} seed={args.seed}",
            flush=True,
        )
        return existing

    device = resolve_device(args.device)
    task = task or load_task(
        Path(args.data_root),
        args.dataset,
        args.horizon,
        seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )
    config = SELECTED_CONFIG[args.dataset]
    patch_candidates = (
        [tuple(config["patch_lens"])]
        if scale == "multiscale"
        else [(int(patch),) for patch in config["patch_lens"]]
    )
    mode, representation_weight = _mode_and_weight(
        continuation, structural_only, config
    )
    if ablation_mode is not None:
        mode = str(ablation_mode)
    started = time.perf_counter()
    best = None
    all_history: list[dict] = []
    namespace = namespace or experiment

    for patches in patch_candidates:
        base, base_payload = _load_or_train_backbone(
            args,
            task,
            device,
            patches,
            bank=bank,
            namespace=namespace,
            use_source_when_exact=use_source_backbone,
        )
        seed_all(args.seed)
        model = build_latentflow(
            base,
            task,
            config,
            mode=mode,
            unfreeze_blocks=2,
        ).to(device)
        assert_no_exchange(model)
        history, selection = fit_final_model(
            model,
            task,
            experiment_args(args),
            representation_weight=representation_weight,
            nll_weight=args.nll_weight,
            batch_size=task_batch_size(args, args.dataset),
            device=device,
        )
        for entry in history:
            all_history.append(
                {
                    **entry,
                    "candidate_patch_lens": "-".join(map(str, patches)),
                }
            )
        candidate = {
            "score": float(selection["best_validation_mse"]),
            "state_dict": clone_state(model),
            "base_config": (
                base_payload["config"]
                if "config" in base_payload
                else base_payload["base_config"]
            ),
            "selection": selection,
            "patches": patches,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("No candidate model was fitted.")
    base = instantiate_design_backbone(best["base_config"], bank=bank)
    seed_all(args.seed)
    model = build_latentflow(
        base,
        task,
        config,
        mode=mode,
        unfreeze_blocks=2,
    ).to(device)
    assert_no_exchange(model)
    load_state(model, best["state_dict"], device)
    # reference_only is runtime state and is intentionally absent from
    # state_dict. Restore it before any metric is computed from the rebuild.
    reference_only = restore_validation_selector(model, best["selection"])
    metrics = evaluate(
        MeanAdapter(model),
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=task_batch_size(args, args.dataset),
        device=device,
        synchronize=True,
    )
    meta = protocol_payload(args, protocol)
    checkpoint = checkpoint_path(args, experiment, variant)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": clone_state(model),
            "base_config": best["base_config"],
            "selected_config": config,
            "selection": best["selection"],
            "inference_reference_only": reference_only,
            "selector_state_serialized": True,
            "variant": variant,
            "scale": scale,
            "bank": bank,
            "continuation": continuation,
            "structural_only": structural_only,
            "ablation_mode": mode,
            "selected_patch_lens": list(best["patches"]),
            **meta,
        },
        checkpoint,
    )
    row = {
        "model": "LatentFlow",
        "experiment": experiment,
        "variant": variant,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "scale": scale,
        "bank": bank,
        "continuation": continuation,
        "structural_only": structural_only,
        "ablation_mode": mode,
        "selected_patch_lens": "-".join(map(str, best["patches"])),
        "cross_process_exchange": False,
        "continuation_unfreeze_blocks": 2,
        "validation_selected_stage": best["selection"]["best_stage"],
        "inference_reference_only": reference_only,
        "selector_state_restored": True,
        "parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        **metrics,
        **best["selection"],
        **meta,
    }
    write_json(output, row)
    write_csv(history_path(args, experiment, variant), all_history)
    print(
        f"enhancement-result experiment={experiment:<10} variant={variant:<38} "
        f"dataset={args.dataset:<11} h={args.horizon:<3} seed={args.seed} "
        f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
        flush=True,
    )
    return row


def run_central_task(args) -> None:
    output = result_path(args, "central", "final_no_exchange")
    if _existing_valid(output, CENTRAL_PROTOCOL, force=args.force) is None:
        row = reuse_final_result(
            args, "central", "final_no_exchange", CENTRAL_PROTOCOL
        )
        print(
            f"enhancement-result experiment=central variant=final_no_exchange "
            f"dataset={args.dataset} h={args.horizon} seed={args.seed} "
            f"mse={float(row['mse']):.6f} mae={float(row['mae']):.6f} reused=True",
            flush=True,
        )

    designs = {
        "x_to_z": {
            "scale": "multiscale",
            "bank": "on",
            "continuation": "preserved",
            "structural_only": True,
        },
        "early_fusion": {
            "scale": "multiscale",
            "bank": "on",
            "continuation": "early",
        },
        "no_multiscale": {
            "scale": "single",
            "bank": "on",
            "continuation": "preserved",
        },
        "no_linear_bank": {
            "scale": "multiscale",
            "bank": "off",
            "continuation": "preserved",
        },
    }
    for variant, design in designs.items():
        fit_design(
            args,
            experiment="central",
            variant=variant,
            namespace="central",
            protocol=CENTRAL_PROTOCOL,
            **design,
        )


def run_matched_control_task(args) -> None:
    """Fit the equal-stage, process-preserving non-GP multibranch control."""
    if args.seed != 42:
        raise ValueError("The first matched-control pass is locked to seed 42.")
    fit_design(
        args,
        experiment="matched_controls",
        variant="deterministic_multibranch",
        scale="multiscale",
        bank="on",
        continuation="preserved",
        protocol=MATCHED_CONTROL_PROTOCOL,
        namespace="matched_controls",
        ablation_mode="deterministic_multibranch",
    )


def run_matched_control_multiseed_task(args) -> None:
    """Extend the matched non-GP control to the remaining headline seeds."""
    if args.seed not in (43, 44, 45, 46):
        raise ValueError("The multiseed matched control is locked to seeds 43--46.")
    fit_design(
        args,
        experiment="matched_controls_multiseed",
        variant="deterministic_multibranch",
        scale="multiscale",
        bank="on",
        continuation="preserved",
        protocol=MATCHED_CONTROL_MULTI_PROTOCOL,
        namespace="matched_controls_multiseed",
        ablation_mode="deterministic_multibranch",
    )

def factorial_variant(scale: str, bank: str, continuation: str) -> str:
    return f"scale-{scale}__bank-{bank}__continuation-{continuation}"


def _central_reuse_variant(
    scale: str,
    bank: str,
    continuation: str,
) -> str | None:
    return {
        ("multiscale", "on", "preserved"): "final_no_exchange",
        ("multiscale", "on", "early"): "early_fusion",
        ("single", "on", "preserved"): "no_multiscale",
        ("multiscale", "off", "preserved"): "no_linear_bank",
    }.get((scale, bank, continuation))


def run_factorial_task(args) -> None:
    if args.seed not in (42, 43, 44):
        raise ValueError("The factorial is locked to seeds 42, 43, and 44.")
    for scale in FACTOR_LEVELS["scale"]:
        for bank in FACTOR_LEVELS["bank"]:
            for continuation in FACTOR_LEVELS["continuation"]:
                variant = factorial_variant(scale, bank, continuation)
                output = result_path(args, "factorial", variant)
                if _existing_valid(
                    output, FACTORIAL_PROTOCOL, force=args.force
                ) is not None:
                    continue
                central_variant = _central_reuse_variant(
                    scale, bank, continuation
                )
                if central_variant is not None:
                    source = result_path(args, "central", central_variant)
                    if not source.exists():
                        raise FileNotFoundError(
                            f"Missing prerequisite central result {source}."
                        )
                    row = json.loads(source.read_text(encoding="utf-8"))
                    if row.get("enhancement_protocol") != CENTRAL_PROTOCOL:
                        raise RuntimeError(f"Stale central prerequisite: {source}")
                    reused = {
                        **row,
                        "experiment": "factorial",
                        "variant": variant,
                        "scale": scale,
                        "bank": bank,
                        "continuation": continuation,
                        "enhancement_protocol": FACTORIAL_PROTOCOL,
                        "config_hash": protocol_payload(
                            args, FACTORIAL_PROTOCOL
                        )["config_hash"],
                        "reused_source": str(source),
                    }
                    write_json(output, reused)
                    print(
                        f"enhancement-result experiment=factorial "
                        f"variant={variant} dataset={args.dataset} "
                        f"h={args.horizon} seed={args.seed} "
                        f"mse={float(row['mse']):.6f} reused=True",
                        flush=True,
                    )
                    continue
                fit_design(
                    args,
                    experiment="factorial",
                    variant=variant,
                    scale=scale,
                    bank=bank,
                    continuation=continuation,
                    namespace="central",
                    protocol=FACTORIAL_PROTOCOL,
                )


def replace_decoder_hidden(base: nn.Module, hidden: int) -> nn.Module:
    """Change only the nonlinear patch decoders' hidden capacity."""

    hidden = int(hidden)
    if hidden < 1:
        raise ValueError("Decoder hidden width must be positive.")
    for branch in base.backbone.branches:
        width = int(branch.encoder_norm.normalized_shape[0])
        dropout = next(
            (module.p for module in branch.decoder if isinstance(module, nn.Dropout)),
            0.0,
        )
        branch.decoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, branch.patch_len),
        )
    return base


def capacity_backbone(
    config: dict,
    *,
    bank: str,
    decoder_hidden: int,
) -> nn.Module:
    base = instantiate_design_backbone(config, bank=bank)
    return replace_decoder_hidden(base, decoder_hidden)


def matched_decoder_hidden(
    task: TaskData,
    config: dict,
    patches: tuple[int, ...],
    *,
    bank: str,
    target_parameters: int,
) -> tuple[int, int]:
    base_config = backbone_config(
        96,
        task.horizon,
        patches,
        switch_overrides=None if bank == "on" else {"use_linear_field": False},
    )

    def count(hidden: int) -> int:
        base = capacity_backbone(
            base_config,
            bank=bank,
            decoder_hidden=hidden,
        )
        model = build_latentflow(
            base,
            task,
            config,
            mode="final",
            unfreeze_blocks=2,
        )
        assert_no_exchange(model)
        return parameter_count(model)

    at_128 = count(128)
    increment = max(1, count(129) - at_128)
    estimate = max(1, 128 + round((target_parameters - at_128) / increment))
    candidates = range(max(1, estimate - 3), estimate + 4)
    selected = min(candidates, key=lambda value: abs(count(value) - target_parameters))
    return int(selected), int(count(selected))


def ensure_capacity_backbone(
    args,
    task: TaskData,
    device: torch.device,
    patches: tuple[int, ...],
    *,
    bank: str,
    decoder_hidden: int,
    variant: str,
) -> dict:
    patch_tag = "-".join(map(str, patches))
    checkpoint = (
        Path(args.output_root)
        / "capacity"
        / "checkpoints"
        / "backbone"
        / variant
        / f"patch_{patch_tag}_decoder_{decoder_hidden}"
        / f"seed{args.seed}"
        / args.dataset
        / f"h{args.horizon}.pt"
    )
    switches = None if bank == "on" else {"use_linear_field": False}
    config = backbone_config(
        args.seq_len,
        args.horizon,
        patches,
        switch_overrides=switches,
    )
    if checkpoint.exists() and not args.force:
        payload = torch_load(checkpoint)
        if (
            payload.get("config") == config
            and int(payload.get("decoder_hidden", -1)) == decoder_hidden
            and payload.get("bank") == bank
        ):
            return payload

    seed_all(args.seed)
    base = capacity_backbone(
        config,
        bank=bank,
        decoder_hidden=decoder_hidden,
    ).to(device)
    history, selection = train_mvpf(
        base,
        task.values,
        task.training_starts,
        task.validation_starts,
        seq_len=args.seq_len,
        pred_len=args.horizon,
        epochs=10,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=7e-4,
        weight_decay=1e-4,
        device=device,
        seed=args.seed,
        patience=0,
    )
    payload = {
        "state_dict": clone_state(base),
        "config": config,
        "decoder_hidden": decoder_hidden,
        "bank": bank,
        "selection": selection,
        "history": history,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    return payload


def load_capacity_backbone(payload: dict) -> nn.Module:
    base = capacity_backbone(
        payload["config"],
        bank=payload["bank"],
        decoder_hidden=int(payload["decoder_hidden"]),
    )
    base.load_state_dict(payload["state_dict"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base


def fit_capacity_control(
    args,
    *,
    variant: str,
    scale: str,
    bank: str,
) -> dict:
    output = result_path(args, "capacity", variant)
    existing = _existing_valid(output, CAPACITY_PROTOCOL, force=args.force)
    if existing is not None:
        return existing
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
    config = SELECTED_CONFIG[args.dataset]
    source = json.loads(source_metric_path(args).read_text(encoding="utf-8"))
    if (
        source.get("candidate") != FINAL_CANDIDATE
        or source.get("cross_process_exchange") is not False
    ):
        raise RuntimeError("Capacity matching requires the revised no-exchange result.")
    target_parameters = int(source["parameters"])
    candidates = (
        [tuple(config["patch_lens"])]
        if scale == "multiscale"
        else [(int(patch),) for patch in config["patch_lens"]]
    )
    best = None
    all_history = []
    started = time.perf_counter()
    for patches in candidates:
        decoder_hidden, matched_parameters = matched_decoder_hidden(
            task,
            config,
            patches,
            bank=bank,
            target_parameters=target_parameters,
        )
        payload = ensure_capacity_backbone(
            args,
            task,
            device,
            patches,
            bank=bank,
            decoder_hidden=decoder_hidden,
            variant=variant,
        )
        base = load_capacity_backbone(payload)
        seed_all(args.seed)
        model = build_latentflow(
            base,
            task,
            config,
            mode="final",
            unfreeze_blocks=2,
        ).to(device)
        assert_no_exchange(model)
        history, selection = fit_final_model(
            model,
            task,
            experiment_args(args),
            representation_weight=config["representation_weight"],
            nll_weight=args.nll_weight,
            batch_size=task_batch_size(args, args.dataset),
            device=device,
        )
        all_history.extend(
            {
                **entry,
                "candidate_patch_lens": "-".join(map(str, patches)),
                "decoder_hidden": decoder_hidden,
                "matched_parameters_before_fit": matched_parameters,
            }
            for entry in history
        )
        candidate = {
            "score": float(selection["best_validation_mse"]),
            "state_dict": clone_state(model),
            "payload": payload,
            "patches": patches,
            "decoder_hidden": decoder_hidden,
            "selection": selection,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("No capacity-matched candidate was fitted.")
    base = load_capacity_backbone(best["payload"])
    model = build_latentflow(
        base,
        task,
        config,
        mode="final",
        unfreeze_blocks=2,
    ).to(device)
    assert_no_exchange(model)
    load_state(model, best["state_dict"], device)
    reference_only = restore_validation_selector(model, best["selection"])
    adapter = MeanAdapter(model)
    metrics = evaluate(
        adapter,
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=task_batch_size(args, args.dataset),
        device=device,
        synchronize=True,
    )
    probe = next(iter(loader(task, task.validation_starts[:1], args.seq_len, 1)))[0].to(device)
    flops = profiler_flops(adapter, probe)
    meta = protocol_payload(args, CAPACITY_PROTOCOL)
    checkpoint = checkpoint_path(args, "capacity", variant)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": clone_state(model),
            "base_payload": best["payload"],
            "selection": best["selection"],
            "inference_reference_only": reference_only,
            "selector_state_serialized": True,
            "variant": variant,
            **meta,
        },
        checkpoint,
    )
    row = {
        "model": "LatentFlow",
        "experiment": "capacity",
        "variant": variant,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "scale": scale,
        "bank": bank,
        "selected_patch_lens": "-".join(map(str, best["patches"])),
        "decoder_hidden": best["decoder_hidden"],
        "target_parameters": target_parameters,
        "parameters": parameter_count(model),
        "parameter_difference_percent": 100.0 * (parameter_count(model) - target_parameters) / target_parameters,
        "gflops_per_sample": flops / 1e9,
        "cross_process_exchange": False,
        "validation_selected_stage": best["selection"]["best_stage"],
        "inference_reference_only": reference_only,
        "selector_state_restored": True,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        **metrics,
        **best["selection"],
        **meta,
    }
    write_json(output, row)
    write_csv(history_path(args, "capacity", variant), all_history)
    print(
        f"capacity-result variant={variant:<30} dataset={args.dataset:<11} "
        f"h={args.horizon:<3} mse={metrics['mse']:.6f} "
        f"params={row['parameters']} target={target_parameters}",
        flush=True,
    )
    return row


def reuse_central_for_capacity(args, source_variant: str, target_variant: str) -> dict:
    output = result_path(args, "capacity", target_variant)
    existing = _existing_valid(output, CAPACITY_PROTOCOL, force=args.force)
    if existing is not None:
        return existing
    source = result_path(args, "central", source_variant)
    if not source.exists():
        raise FileNotFoundError(f"Missing central prerequisite: {source}")
    row = json.loads(source.read_text(encoding="utf-8"))
    if row.get("enhancement_protocol") != CENTRAL_PROTOCOL:
        raise RuntimeError(f"Stale central prerequisite: {source}")
    updated = {
        **row,
        "experiment": "capacity",
        "variant": target_variant,
        "reused_source": str(source),
        **protocol_payload(args, CAPACITY_PROTOCOL),
    }
    write_json(output, updated)
    return updated


def run_capacity_task(args) -> None:
    if args.seed != 42:
        raise ValueError("Capacity controls use seed 42 only.")
    reuse_central_for_capacity(args, "no_multiscale", "single_scale_selected")
    reuse_central_for_capacity(args, "no_linear_bank", "no_linear_bank")
    fit_capacity_control(
        args,
        variant="single_scale_width_matched",
        scale="single",
        bank="on",
    )
    fit_capacity_control(
        args,
        variant="no_bank_decoder_matched",
        scale="multiscale",
        bank="off",
    )


def _fit_seeded_model(
    args,
    name: str,
    model: nn.Module,
    spec: ModelSpec,
    task: TaskData,
    device: torch.device,
    *,
    include_regularizer: bool = True,
):
    batch = (
        min(spec.batch_size, args.wide_batch_size)
        if task.dataset in {"Electricity", "Traffic"}
        else spec.batch_size
    )
    eval_batch = (
        min(spec.eval_batch_size, args.wide_batch_size)
        if task.dataset in {"Electricity", "Traffic"}
        else spec.eval_batch_size
    )
    training = loader(
        task,
        task.training_starts,
        args.seq_len,
        batch,
        shuffle=True,
        seed=args.seed,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    schedule, each_batch = scheduler_for(spec, optimizer, len(training))
    best_state = clone_state(model)
    best_mse = float("inf")
    best_epoch = 0
    history = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, spec.epochs + 1):
        model.train()
        running = 0.0
        for inputs, targets in training:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = prediction_loss(spec, prediction, targets)
            if include_regularizer:
                loss = loss + regularizer(name, model, inputs)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {name} loss for {task.dataset}, "
                    f"h={task.horizon}, epoch={epoch}."
                )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            if schedule is not None and each_batch:
                schedule.step()
            running += float(loss.detach())
            steps += 1
        if schedule is not None and not each_batch:
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
                "train_loss": running / max(1, len(training)),
                "validation_mse": validation["mse"],
                "validation_mae": validation["mae"],
                "best_validation_mse": best_mse,
            }
        )
        print(
            f"enhancement-baseline epoch={epoch}/{spec.epochs} model={name} "
            f"dataset={task.dataset} h={task.horizon} "
            f"train={history[-1]['train_loss']:.6f} "
            f"val={validation['mse']:.6f} best={best_mse:.6f}@{best_epoch}",
            flush=True,
        )
        if spec.patience > 0 and epoch - best_epoch >= spec.patience:
            break
    load_state(model, best_state, device)
    elapsed = 1000.0 * (time.perf_counter() - started) / max(1, steps)
    return history, {
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
    }, elapsed


def _fit_seeded_baseline(
    args,
    model_name: str,
    protocol: str,
    experiment: str,
    *,
    task: TaskData | None = None,
    variant: str | None = None,
) -> dict:
    variant = variant or model_name
    output = result_path(args, experiment, variant)
    existing = _existing_valid(output, protocol, force=args.force)
    if existing is not None:
        return existing
    device = resolve_device(args.device)
    task = task or load_task(
        Path(args.data_root),
        args.dataset,
        args.horizon,
        seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )
    seed_all(args.seed)
    spec = build_model(
        model_name,
        args.dataset,
        args.seq_len,
        args.horizon,
        task.values.shape[1],
    )
    model = spec.model.to(device)
    histories = []
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
        base_history, _, _ = _fit_seeded_model(
            args,
            "DLinear",
            model.backbone,
            backbone_spec,
            task,
            device,
            include_regularizer=False,
        )
        histories.extend(
            {"stage": "backbone", **entry} for entry in base_history
        )
        model.freeze_backbone()
        maximum = (
            8
            if args.dataset == "Traffic"
            else (16 if args.dataset == "Electricity" else 64)
        )
        forecasts = []
        basis_batch = (
            min(spec.eval_batch_size, args.wide_batch_size)
            if args.dataset in {"Electricity", "Traffic"}
            else spec.eval_batch_size
        )
        for inputs, _ in loader(
            task,
            task.training_starts[:maximum],
            args.seq_len,
            basis_batch,
        ):
            forecasts.append(model.backbone(inputs.to(device)).detach())
        model.plugin.fit_shared_basis(torch.cat(forecasts, dim=0))

    history, selection, train_ms = _fit_seeded_model(
        args, model_name, model, spec, task, device
    )
    histories.extend({"stage": "model", **entry} for entry in history)
    eval_batch = (
        min(spec.eval_batch_size, args.wide_batch_size)
        if args.dataset in {"Electricity", "Traffic"}
        else spec.eval_batch_size
    )
    metrics = evaluate(
        model,
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=eval_batch,
        device=device,
        synchronize=True,
    )
    meta = protocol_payload(args, protocol)
    checkpoint = checkpoint_path(args, experiment, variant)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": clone_state(model),
            "model": model_name,
            "selection": selection,
            "channels": task.values.shape[1],
            "provenance": spec.provenance,
            **meta,
        },
        checkpoint,
    )
    row = {
        "model": model_name,
        "experiment": experiment,
        "variant": variant,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "cross_process_exchange": False,
        "parameters": parameter_count(model),
        "training_ms_optimizer_step": train_ms,
        "checkpoint": str(checkpoint),
        "provenance": spec.provenance,
        **metrics,
        **selection,
        **meta,
    }
    write_json(output, row)
    write_csv(history_path(args, experiment, variant), histories)
    print(
        f"enhancement-baseline-result model={model_name:<12} "
        f"dataset={args.dataset:<11} h={args.horizon:<3} seed={args.seed} "
        f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
        flush=True,
    )
    return row


def run_strongest_task(args) -> None:
    if args.seed == 42:
        raise ValueError("Seed 42 exists already; run only seeds 43--46.")
    _fit_seeded_baseline(
        args,
        STRONGEST_NON_TIMEPRO[args.dataset],
        COMPARATOR_PROTOCOL,
        "strongest_comparator",
    )


def run_full_data_task(args) -> None:
    if args.seed != 42:
        raise ValueError("The full-data check is restricted to seed 42.")
    full_limit = 2_147_483_647
    full_args = argparse.Namespace(**vars(args))
    full_args.max_train_windows = full_limit
    full_args.max_validation_windows = full_limit
    task = load_task(
        Path(args.data_root),
        args.dataset,
        args.horizon,
        seq_len=args.seq_len,
        max_train_windows=full_limit,
        max_validation_windows=full_limit,
        window_seed=42,
    )
    fit_design(
        full_args,
        experiment="full_data",
        variant="LatentFlow",
        scale="multiscale",
        bank="on",
        continuation="preserved",
        task=task,
        namespace="full_data",
        use_source_backbone=False,
        protocol=FULL_DATA_PROTOCOL,
    )
    _fit_seeded_baseline(
        full_args,
        "TimePro",
        FULL_DATA_PROTOCOL,
        "full_data",
    )


def nested_budget_task(args, budget: int | str) -> TaskData:
    '''Load one task with nested train subsets and a fixed validation set.'''

    full_limit = 2_147_483_647
    task = load_task(
        Path(args.data_root),
        args.dataset,
        args.horizon,
        seq_len=args.seq_len,
        max_train_windows=full_limit,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )
    all_starts = task.training_starts
    if all_starts.size <= 2048:
        anchor = all_starts.copy()
    else:
        anchor_rng = np.random.default_rng(42)
        anchor = np.sort(anchor_rng.choice(all_starts, size=2048, replace=False))

    if budget == 'all':
        selected = all_starts
    elif int(budget) == 2048:
        selected = anchor
    else:
        requested = int(budget)
        nested_order = np.random.default_rng(9_042).permutation(anchor)
        selected = np.sort(nested_order[: min(requested, anchor.size)])
    return TaskData(
        dataset=task.dataset,
        horizon=task.horizon,
        values=task.values,
        training_starts=selected,
        validation_starts=task.validation_starts,
        test_starts=task.test_starts,
    )


def run_data_budget_task(args) -> None:
    if args.seed != 42:
        raise ValueError('The data-budget curve is restricted to seed 42.')
    budget: int | str = args.train_budget
    if budget != 'all':
        budget = int(budget)
    if budget not in DATA_BUDGETS:
        raise ValueError(f'Unsupported train budget: {budget}')

    task = nested_budget_task(args, budget)
    limit = 2_147_483_647 if budget == 'all' else int(budget)
    curve_args = argparse.Namespace(**vars(args))
    curve_args.max_train_windows = limit
    curve_args.max_validation_windows = 1024
    label = 'all' if budget == 'all' else f'{int(budget):04d}'
    variant = f'train_{label}'

    if budget == 2048:
        row = reuse_final_result(
            curve_args,
            'data_budget',
            variant,
            DATA_BUDGET_PROTOCOL,
        )
    else:
        row = fit_design(
            curve_args,
            experiment='data_budget',
            variant=variant,
            scale='multiscale',
            bank='on',
            continuation='preserved',
            task=task,
            namespace=f'data_budget_{label}',
            use_source_backbone=False,
            protocol=DATA_BUDGET_PROTOCOL,
        )
    row.update(
        {
            'train_budget': budget,
            'train_budget_label': label,
            'num_training_windows': int(task.training_starts.size),
            'num_validation_windows': int(task.validation_starts.size),
            'nested_anchor_windows': 2048,
            'nested_subset_seed': 9042,
            'validation_window_seed': 5042,
        }
    )
    write_json(result_path(curve_args, 'data_budget', variant), row)
    mse = float(row.get('mse'))
    mae = float(row.get('mae'))
    print(
        f'enhancement-result experiment=data_budget variant={variant:<10} '
        f'dataset={args.dataset:<11} h={args.horizon:<3} seed=42 '
        f'train_windows={len(task.training_starts)} '
        f'mse={mse:.6f} mae={mae:.6f}',
        flush=True,
    )


def run_timepro_data_budget_task(args) -> None:
    """Fit TimePro on LatentFlow's nested origin-budget protocol."""
    if args.seed != 42:
        raise ValueError("The TimePro data-budget curve is restricted to seed 42.")
    budget: int | str = args.train_budget
    if budget != "all":
        budget = int(budget)
    if budget not in TIMEPRO_DATA_BUDGETS:
        raise ValueError(f"Unsupported TimePro train budget: {budget}")

    task = nested_budget_task(args, budget)
    limit = 2_147_483_647 if budget == "all" else int(budget)
    curve_args = argparse.Namespace(**vars(args))
    curve_args.max_train_windows = limit
    # Keep validation fixed across the complete budget curve.
    curve_args.max_validation_windows = 1024
    label = "all" if budget == "all" else f"{int(budget):04d}"
    variant = f"train_{label}"
    row = _fit_seeded_baseline(
        curve_args,
        "TimePro",
        TIMEPRO_DATA_BUDGET_PROTOCOL,
        "timepro_data_budget",
        task=task,
        variant=variant,
    )
    row.update(
        {
            "train_budget": budget,
            "train_budget_label": label,
            "num_training_windows": int(task.training_starts.size),
            "num_validation_windows": int(task.validation_starts.size),
            "nested_anchor_windows": 2048,
            "nested_subset_seed": 9042,
            "validation_window_seed": 5042,
        }
    )
    write_json(result_path(curve_args, "timepro_data_budget", variant), row)
    print(
        f"enhancement-baseline-result experiment=timepro_data_budget "
        f"variant={variant:<10} dataset={args.dataset:<11} "
        f"h={args.horizon:<3} seed=42 train_windows={len(task.training_starts)} "
        f"mse={float(row['mse']):.6f} mae={float(row['mae']):.6f}",
        flush=True,
    )


def permuted_task(
    task: TaskData,
    permutation_seed: int,
) -> tuple[TaskData, list[int]]:
    rng = np.random.default_rng(permutation_seed)
    permutation = rng.permutation(task.values.shape[1])
    return TaskData(
        dataset=task.dataset,
        horizon=task.horizon,
        values=task.values[:, permutation].copy(),
        training_starts=task.training_starts,
        validation_starts=task.validation_starts,
        test_starts=task.test_starts,
    ), permutation.tolist()


def run_channel_order(args, *, expanded: bool = False) -> None:
    if args.dataset not in {"Electricity", "Traffic"} or args.horizon != 720:
        raise ValueError("Channel-order audit covers Electricity/Traffic at H=720.")
    if args.seed != 42:
        raise ValueError("Channel-order audit uses model seed 42.")
    allowed = {4817, 7919, 104729} if expanded else {1729, 3253}
    if args.permutation_seed not in allowed:
        raise ValueError(
            f"Permutation seed {args.permutation_seed} is not in the locked "
            f"{'expansion' if expanded else 'pilot'} set {sorted(allowed)}."
        )
    experiment = "channel_order_expanded" if expanded else "channel_order"
    protocol = CHANNEL_EXPANSION_PROTOCOL if expanded else CHANNEL_PROTOCOL
    variant = f"permutation_{args.permutation_seed}"
    output = result_path(args, experiment, variant)
    if _existing_valid(output, protocol, force=args.force) is not None:
        return
    task = load_task(
        Path(args.data_root),
        args.dataset,
        args.horizon,
        seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )
    task, permutation = permuted_task(task, args.permutation_seed)
    row = fit_design(
        args,
        experiment=experiment,
        variant=variant,
        scale="multiscale",
        bank="on",
        continuation="preserved",
        task=task,
        namespace=variant,
        use_source_backbone=False,
        protocol=protocol,
    )
    row["permutation_seed"] = args.permutation_seed
    row["permutation"] = permutation
    write_json(output, row)


def _linear_bank_components(head: nn.Module, values: torch.Tensor) -> torch.Tensor:
    last = values[:, -1:, :]
    centered = values - last
    nlinear = head._linear_per_channel(head.center_head, centered) + last
    trend = head._moving_average(values)
    residual = values - trend
    dlinear = head._linear_per_channel(head.trend_head, trend)
    dlinear = dlinear + head._linear_per_channel(head.residual_head, residual)
    analytic, _ = head._analytic_trend(values)
    persistence = last.expand(-1, head.pred_len, -1)
    return torch.stack([nlinear, dlinear, analytic, persistence], dim=1)


@torch.no_grad()
def _diagnostic_components(model: nn.Module, inputs: torch.Tensor) -> dict:
    distribution = model.forward_distribution(inputs)
    continuation = model.continuation
    backbone = continuation.mvpff.backbone
    normalized, window_mean, window_std = backbone._normalize(inputs)
    structural = model.encoder(normalized)
    posterior = model.separator(structural, inputs)
    reconstructed = model.encoder.decode(posterior["reconstruction"])
    trend, residual = backbone._decompose(reconstructed)
    trend_forecast = (
        backbone._linear_per_channel(backbone.trend_head, trend)
        if backbone.use_trend_decomposition
        else torch.zeros(
            inputs.shape[0], model.pred_len, inputs.shape[2], device=inputs.device
        )
    )
    direct = backbone._linear_per_channel(backbone.direct_residual_head, residual)
    process_weights, context = continuation._weights(
        posterior["latent_mean"], posterior["posterior_past_variance"]
    )
    branches = []
    for index, branch in enumerate(backbone.branches):
        process_forecast, _ = continuation._process_branch(
            index, branch, posterior["latent_mean"], context
        )
        branches.append(
            (process_forecast * process_weights[:, :, None, None]).sum(dim=1)
        )
    field_components = torch.stack([direct, *branches], dim=1)
    if backbone.use_component_gate:
        field_weights = torch.softmax(
            backbone.component_gate(backbone._summary_features(reconstructed, residual)),
            dim=-1,
        )
    else:
        field_weights = field_components.new_full(
            (inputs.shape[0], field_components.shape[1]),
            1.0 / field_components.shape[1],
        )
    strength = torch.sigmoid(backbone.residual_strength)
    field_deployed = trend_forecast + strength * (
        field_components * field_weights[:, :, None, None]
    ).sum(dim=1)

    bank = continuation.mvpff.linear_field
    bank_components = _linear_bank_components(bank, reconstructed)
    bank_deployed, bank_weights, fit_error = bank(reconstructed)
    confidence = bank_weights.max(dim=1, keepdim=True).values
    outer_weights = torch.softmax(
        continuation.mvpff.mean_mixer(
            continuation.mvpff._summary_features(
                reconstructed, residual, confidence, fit_error
            )
        ),
        dim=-1,
    )

    def raw(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 4:
            return values * window_std[:, None] + window_mean[:, None]
        return values * window_std + window_mean

    return {
        "final": distribution["mean"],
        "field_components": raw(
            trend_forecast[:, None] + strength * field_components
        ),
        "field_deployed": raw(field_deployed),
        "field_weights": field_weights,
        "bank_components": raw(bank_components),
        "bank_deployed": raw(bank_deployed),
        "bank_weights": bank_weights,
        "process_weights": process_weights,
        "outer_weights": outer_weights,
        "trend": raw(trend_forecast),
    }


def _metric_state(names: list[str]) -> dict[str, dict[str, float]]:
    return {name: {"squared": 0.0, "absolute": 0.0, "count": 0.0} for name in names}


def _accumulate_metric(state: dict, name: str, prediction, target) -> None:
    difference = prediction - target
    state[name]["squared"] += float(difference.square().sum())
    state[name]["absolute"] += float(difference.abs().sum())
    state[name]["count"] += float(difference.numel())


def _pearson(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_diagnostics(args) -> None:
    if (args.dataset, args.horizon) not in DIAGNOSTIC_TASKS or args.seed != 42:
        raise ValueError("Diagnostics are locked to the seven predeclared seed-42 cells.")
    output = result_path(args, "diagnostics", "analysis")
    existing = _existing_valid(output, DIAGNOSTIC_PROTOCOL, force=args.force)
    if existing is not None:
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
    model, payload = load_source_model(args, task, device)
    process_names = list(model.separator.process_names)
    scale_names = ["direct"] + [
        f"patch_{branch.patch_len}" for branch in model.continuation.mvpff.backbone.branches
    ]
    bank_names = ["centered_linear", "decomposition_linear", "analytic_trend", "persistence"]
    metric_names = (
        ["final", "field_deployed", "field_uniform", "field_permuted", "bank_deployed", "bank_uniform", "bank_permuted", "bank_sample_oracle", "trend"]
        + [f"field_{name}" for name in scale_names]
        + [f"bank_{name}" for name in bank_names]
    )
    metrics = _metric_state(metric_names)
    bank_weight_values: list[float] = []
    bank_quality_values: list[float] = []
    field_weight_values: list[float] = []
    field_quality_values: list[float] = []
    bank_weight_sum = np.zeros(4, dtype=np.float64)
    field_weight_sum = np.zeros(len(scale_names), dtype=np.float64)
    process_weight_sum = np.zeros(len(process_names), dtype=np.float64)
    outer_weight_sum = np.zeros(2, dtype=np.float64)
    samples = 0
    evaluation_batch = task_batch_size(args, args.dataset)

    for inputs, target in loader(task, task.test_starts, args.seq_len, evaluation_batch):
        inputs = inputs.to(device)
        target = target.to(device)
        parts = _diagnostic_components(model, inputs)
        batch = inputs.shape[0]
        samples += batch
        bank_weight_sum += parts["bank_weights"].sum(dim=0).cpu().numpy()
        field_weight_sum += parts["field_weights"].sum(dim=0).cpu().numpy()
        process_weight_sum += parts["process_weights"].sum(dim=0).cpu().numpy()
        outer_weight_sum += parts["outer_weights"].sum(dim=0).cpu().numpy()

        _accumulate_metric(metrics, "final", parts["final"], target)
        _accumulate_metric(metrics, "field_deployed", parts["field_deployed"], target)
        _accumulate_metric(metrics, "bank_deployed", parts["bank_deployed"], target)
        _accumulate_metric(metrics, "trend", parts["trend"], target)
        _accumulate_metric(
            metrics,
            "field_uniform",
            parts["field_components"].mean(dim=1),
            target,
        )
        _accumulate_metric(
            metrics,
            "bank_uniform",
            parts["bank_components"].mean(dim=1),
            target,
        )
        field_permuted = (
            parts["field_components"]
            * parts["field_weights"].roll(1, dims=0)[:, :, None, None]
        ).sum(dim=1)
        bank_permuted = (
            parts["bank_components"]
            * parts["bank_weights"].roll(1, dims=0)[:, :, None, None]
        ).sum(dim=1)
        _accumulate_metric(metrics, "field_permuted", field_permuted, target)
        _accumulate_metric(metrics, "bank_permuted", bank_permuted, target)

        bank_sample_mse = (
            parts["bank_components"] - target[:, None]
        ).square().mean(dim=(2, 3))
        selected = bank_sample_mse.argmin(dim=1)
        oracle = parts["bank_components"][torch.arange(batch, device=device), selected]
        _accumulate_metric(metrics, "bank_sample_oracle", oracle, target)
        bank_weight_values.extend(parts["bank_weights"].cpu().numpy().ravel().tolist())
        bank_quality_values.extend((-bank_sample_mse).cpu().numpy().ravel().tolist())

        field_sample_mse = (
            parts["field_components"] - target[:, None]
        ).square().mean(dim=(2, 3))
        field_weight_values.extend(parts["field_weights"].cpu().numpy().ravel().tolist())
        field_quality_values.extend((-field_sample_mse).cpu().numpy().ravel().tolist())
        for index, name in enumerate(scale_names):
            _accumulate_metric(
                metrics,
                f"field_{name}",
                parts["field_components"][:, index],
                target,
            )
        for index, name in enumerate(bank_names):
            _accumulate_metric(
                metrics,
                f"bank_{name}",
                parts["bank_components"][:, index],
                target,
            )

    if samples == 0:
        raise RuntimeError("No diagnostic test windows were evaluated.")
    bank_global = torch.as_tensor(bank_weight_sum / samples, device=device, dtype=torch.float32)
    field_global = torch.as_tensor(field_weight_sum / samples, device=device, dtype=torch.float32)
    metrics.update(_metric_state(["bank_global", "field_global"]))
    for inputs, target in loader(task, task.test_starts, args.seq_len, evaluation_batch):
        inputs = inputs.to(device)
        target = target.to(device)
        parts = _diagnostic_components(model, inputs)
        _accumulate_metric(
            metrics,
            "bank_global",
            (parts["bank_components"] * bank_global[None, :, None, None]).sum(dim=1),
            target,
        )
        _accumulate_metric(
            metrics,
            "field_global",
            (parts["field_components"] * field_global[None, :, None, None]).sum(dim=1),
            target,
        )

    member_rows = [
        {
            "component": name,
            "mse": values["squared"] / max(values["count"], 1.0),
            "mae": values["absolute"] / max(values["count"], 1.0),
        }
        for name, values in metrics.items()
    ]
    meta = protocol_payload(args, DIAGNOSTIC_PROTOCOL)
    row = {
        "model": "LatentFlow",
        "experiment": "diagnostics",
        "variant": "analysis",
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "cross_process_exchange": False,
        "bank_weight_quality_correlation": _pearson(bank_weight_values, bank_quality_values),
        "field_weight_quality_correlation": _pearson(field_weight_values, field_quality_values),
        "mean_bank_weights": (bank_weight_sum / samples).tolist(),
        "mean_field_weights": (field_weight_sum / samples).tolist(),
        "mean_process_weights": (process_weight_sum / samples).tolist(),
        "mean_outer_weights": (outer_weight_sum / samples).tolist(),
        "bank_names": bank_names,
        "field_names": scale_names,
        "process_names": process_names,
        "component_metrics": member_rows,
        "checkpoint": str(source_checkpoint_path(args)),
        "source_candidate": payload["candidate"],
        **meta,
    }
    write_json(output, row)
    write_csv(
        history_path(args, "diagnostics", "component_metrics"),
        [{"dataset": args.dataset, "horizon": args.horizon, **entry} for entry in member_rows],
    )
    print(
        f"diagnostic-result dataset={args.dataset:<11} h={args.horizon:<3} "
        f"bank_corr={row['bank_weight_quality_correlation']:+.4f} "
        f"field_corr={row['field_weight_quality_correlation']:+.4f}",
        flush=True,
    )


def replay_design_metric(
    args,
    *,
    experiment: str,
    variant: str,
    protocol: str,
) -> dict:
    """Repair a v1 metric by evaluating its untouched selected checkpoint."""

    output = result_path(args, experiment, variant)
    checkpoint = checkpoint_path(args, experiment, variant)
    if not output.exists() or not checkpoint.exists():
        raise FileNotFoundError(
            f"Metric repair requires both {output} and {checkpoint}."
        )
    source_row = json.loads(output.read_text(encoding="utf-8"))
    payload = torch_load(checkpoint)
    task = load_task(
        Path(args.data_root),
        args.dataset,
        args.horizon,
        seq_len=args.seq_len,
        max_train_windows=args.max_train_windows,
        max_validation_windows=args.max_validation_windows,
        window_seed=42,
    )
    bank = str(payload.get("bank", source_row.get("bank", "on")))
    base = instantiate_design_backbone(payload["base_config"], bank=bank)
    config = payload.get("selected_config", SELECTED_CONFIG[args.dataset])
    mode = str(payload.get("ablation_mode", source_row.get("ablation_mode", "final")))
    unfreeze_blocks = int(
        payload.get(
            "continuation_unfreeze_blocks",
            source_row.get("continuation_unfreeze_blocks", 2),
        )
    )
    device = resolve_device(args.device)
    model = build_latentflow(
        base,
        task,
        config,
        mode=mode,
        unfreeze_blocks=unfreeze_blocks,
    ).to(device)
    assert_no_exchange(model)
    load_state(model, payload["state_dict"], device)
    selection = payload.get("selection", {})
    reference_only = restore_validation_selector(model, selection)
    metrics = evaluate(
        MeanAdapter(model),
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=task_batch_size(args, args.dataset),
        device=device,
        synchronize=True,
    )
    previous_mse = float(source_row["mse"])
    previous_mae = float(source_row["mae"])
    repaired = {
        **source_row,
        **metrics,
        **selection,
        **protocol_payload(args, protocol),
        "validation_selected_stage": selection["best_stage"],
        "inference_reference_only": reference_only,
        "selector_state_restored": True,
        "selector_repair_evaluation_only": True,
        "pre_repair_mse": previous_mse,
        "pre_repair_mae": previous_mae,
        "repair_mse_delta": float(metrics["mse"]) - previous_mse,
        "repair_mae_delta": float(metrics["mae"]) - previous_mae,
        "checkpoint": str(checkpoint),
    }
    write_json(output, repaired)
    print(
        f"selector-repair-result experiment={experiment:<9} variant={variant:<48} "
        f"dataset={args.dataset:<11} h={args.horizon:<3} seed={args.seed} "
        f"selected={selection['best_stage']} old_mse={previous_mse:.6f} "
        f"mse={metrics['mse']:.6f} delta={metrics['mse'] - previous_mse:+.6f}",
        flush=True,
    )
    return repaired


def run_repair_central_task(args) -> None:
    """Replay all five central cells for one dataset/horizon/seed group."""

    reuse_final_result(args, "central", "final_no_exchange", CENTRAL_PROTOCOL)
    print(
        f"selector-repair-result experiment=central variant=final_no_exchange "
        f"dataset={args.dataset:<11} h={args.horizon:<3} seed={args.seed} reused=headline",
        flush=True,
    )
    for variant in ("x_to_z", "early_fusion", "no_multiscale", "no_linear_bank"):
        replay_design_metric(
            args,
            experiment="central",
            variant=variant,
            protocol=CENTRAL_PROTOCOL,
        )


def run_repair_factorial_task(args) -> None:
    """Replay all eight factorial cells for one dataset/horizon/seed group."""

    if args.seed not in (42, 43, 44):
        raise ValueError("Factorial repair is locked to seeds 42, 43, and 44.")
    for scale in FACTOR_LEVELS["scale"]:
        for bank in FACTOR_LEVELS["bank"]:
            for continuation in FACTOR_LEVELS["continuation"]:
                variant = factorial_variant(scale, bank, continuation)
                central_variant = _central_reuse_variant(scale, bank, continuation)
                output = result_path(args, "factorial", variant)
                if central_variant is not None:
                    source = result_path(args, "central", central_variant)
                    if not source.exists():
                        raise FileNotFoundError(
                            f"Run selector-repair-central first; missing {source}."
                        )
                    row = json.loads(source.read_text(encoding="utf-8"))
                    if row.get("enhancement_protocol") != CENTRAL_PROTOCOL:
                        raise RuntimeError(
                            f"Central prerequisite is not selector-safe v2: {source}"
                        )
                    repaired = {
                        **row,
                        "experiment": "factorial",
                        "variant": variant,
                        "scale": scale,
                        "bank": bank,
                        "continuation": continuation,
                        "reused_source": str(source),
                        "selector_repair_evaluation_only": True,
                        **protocol_payload(args, FACTORIAL_PROTOCOL),
                    }
                    write_json(output, repaired)
                    print(
                        f"selector-repair-result experiment=factorial variant={variant:<48} "
                        f"dataset={args.dataset:<11} h={args.horizon:<3} seed={args.seed} "
                        f"mse={float(repaired['mse']):.6f} reused=central",
                        flush=True,
                    )
                else:
                    replay_design_metric(
                        args,
                        experiment="factorial",
                        variant=variant,
                        protocol=FACTORIAL_PROTOCOL,
                    )


def run_smoke(args) -> None:
    device = resolve_device(args.device)
    seed_all(42)
    inputs = torch.randn(2, 96, 7, device=device)
    task = SimpleNamespace(
        dataset="ETTh1",
        values=np.zeros((200, 7), dtype=np.float32),
    )
    config = {"latent_dim": 4, "inducing_points": 8}
    for mode in ("final", "learned_inducing_elbo", "x_to_z", "deterministic_multibranch"):
        base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
        model = build_latentflow(
            base,
            task,
            config,
            mode=mode,
            unfreeze_blocks=2,
        ).to(device)
        assert_no_exchange(model)
        output = model.forward_distribution(inputs)["mean"]
        if output.shape != (2, 96, 7) or not torch.isfinite(output).all():
            raise RuntimeError(f"Smoke failure for mode={mode}.")
    # Regression test for selector state, which is intentionally absent from
    # state_dict. A rebuild starts in stage A and must be switched explicitly.
    base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
    selected = build_latentflow(
        base, task, config, mode="learned_inducing_elbo", unfreeze_blocks=2
    ).to(device)
    selected.configure_stage("C")
    selected.select_reference_only(False)
    selected.eval()
    expected = selected.forward_distribution(inputs)["mean"].detach()
    selected_state = clone_state(selected)
    rebuilt_base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
    rebuilt = build_latentflow(
        rebuilt_base, task, config, mode="learned_inducing_elbo", unfreeze_blocks=2
    ).to(device)
    load_state(rebuilt, selected_state, device)
    if not rebuilt.reference_only:
        raise RuntimeError("Smoke precondition failed: rebuild must start reference-only.")
    restore_validation_selector(rebuilt, {"best_stage": "extension"})
    rebuilt.eval()
    replay = rebuilt.forward_distribution(inputs)["mean"].detach()
    if not torch.allclose(expected, replay, rtol=1e-7, atol=1e-7):
        maximum = float((expected - replay).abs().max())
        raise RuntimeError(f"Selector round-trip changed predictions: max={maximum}")

    factorial = {
        factorial_variant(scale, bank, continuation)
        for scale in FACTOR_LEVELS["scale"]
        for bank in FACTOR_LEVELS["bank"]
        for continuation in FACTOR_LEVELS["continuation"]
    }
    if len(factorial) != 8:
        raise RuntimeError("Factorial does not contain eight unique cells.")
    if len(CENTRAL_VARIANTS) * len(DATASETS) * len(HORIZONS) * len(SEEDS) != 700:
        raise RuntimeError("Central cardinality is not 700.")
    if len(DATA_BUDGETS) * len(DATASETS) * len(HORIZONS) != 112:
        raise RuntimeError('Data-budget curve cardinality is not 112.')
    if len(factorial) * len(DATASETS) * len(HORIZONS) * 3 != 672:
        raise RuntimeError("Factorial cardinality is not 672.")
    print(
        "LATENTFLOW-ENHANCEMENTS-SMOKE PASSED "
        "selector_roundtrip=passed exchange_modules=0 "
        "central_cells=700 factorial_cells=672 "
        "central_task_groups=140 factorial_task_groups=84",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--mode",
        choices=(
            'data-budget-task',
            'timepro-data-budget-task',
            "smoke",
            "central-task",
            "factorial-task",
            "selector-repair-central-task",
            "selector-repair-factorial-task",
            "strongest-task",
            "full-data-task",
            "channel-order",
            "channel-order-expanded",
            "capacity-task",
            "diagnostics-task",
            "matched-control-task",
            "matched-control-multiseed-task",
        ),
        required=True,
    )
    result.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    result.add_argument("--horizon", choices=HORIZONS, type=int, default=96)
    result.add_argument("--seed", choices=SEEDS, type=int, default=42)
    result.add_argument("--permutation-seed", type=int, default=1729)
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
    result.add_argument("--device", default="auto")
    result.add_argument("--force", action="store_true")
    result.add_argument(
        '--train-budget',
        choices=('128', '512', '2048', 'all'),
        default='2048',
    )
    return result


def main() -> None:
    args = parser().parse_args()
    validate_locked_protocol(
        args,
        allow_training_window_override=args.mode in {
            'data-budget-task',
            'timepro-data-budget-task',
        },
    )
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "central-task":
        run_central_task(args)
    elif args.mode == "factorial-task":
        run_factorial_task(args)
    elif args.mode == "selector-repair-central-task":
        run_repair_central_task(args)
    elif args.mode == "selector-repair-factorial-task":
        run_repair_factorial_task(args)
    elif args.mode == "strongest-task":
        run_strongest_task(args)
    elif args.mode == "full-data-task":
        run_full_data_task(args)
    elif args.mode == 'data-budget-task':
        run_data_budget_task(args)
    elif args.mode == 'timepro-data-budget-task':
        run_timepro_data_budget_task(args)
    elif args.mode == "capacity-task":
        run_capacity_task(args)
    elif args.mode == "diagnostics-task":
        run_diagnostics(args)
    elif args.mode == "matched-control-task":
        run_matched_control_task(args)
    elif args.mode == "matched-control-multiseed-task":
        run_matched_control_multiseed_task(args)
    elif args.mode == "channel-order-expanded":
        run_channel_order(args, expanded=True)
    else:
        run_channel_order(args)


if __name__ == "__main__":
    main()
