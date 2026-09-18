"""Final LatentFlow benchmark, ablations, and efficiency profiler."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from physics_modal_v4.experiments.protocol import (
    DATASETS,
    HORIZONS,
    SEEDS,
    clone_state,
    evaluate,
    load_task,
    loader,
    parameter_count,
    resolve_device,
    seed_all,
    write_csv,
    write_json,
)
from physics_modal_v4.experiments.training import (
    build_backbone,
    ensure_backbone,
    fit_final_model,
    instantiate_backbone,
    torch_load,
)
from physics_modal_v4.model import LatentFlow


FINAL_CANDIDATE = "latentflow_process_preserving_unfreeze2"

# Selected exclusively by mean validation MSE across horizons; test data were
# not queried during this search.
SELECTED_CONFIG = {
    "ETTh1": {"latent_dim": 16, "inducing_points": 32, "patch_lens": (16, 32, 48), "representation_weight": 0.05},
    "ETTh2": {"latent_dim": 8, "inducing_points": 32, "patch_lens": (16, 32, 48), "representation_weight": 0.001},
    "ETTm1": {"latent_dim": 8, "inducing_points": 32, "patch_lens": (8, 24, 48), "representation_weight": 0.01},
    "ETTm2": {"latent_dim": 8, "inducing_points": 16, "patch_lens": (8, 16, 32), "representation_weight": 0.001},
    "Weather": {"latent_dim": 16, "inducing_points": 32, "patch_lens": (8, 16, 32), "representation_weight": 0.001},
    "Electricity": {"latent_dim": 16, "inducing_points": 32, "patch_lens": (8, 16, 32), "representation_weight": 0.001},
    "Traffic": {"latent_dim": 16, "inducing_points": 32, "patch_lens": (8, 16, 32), "representation_weight": 0.001},
}
BACKBONE_LADDER = {
    "single_scale_field": {
        "patch_lens": (16,),
        "switches": {
            "use_adaptive_radius": False,
            "use_component_gate": False,
            "use_linear_field": False,
        },
    },
    "multiscale_field": {
        "patch_lens": None,
        "switches": {
            "use_adaptive_radius": False,
            "use_component_gate": False,
            "use_linear_field": False,
        },
    },
    "adaptive_local_field": {
        "patch_lens": None,
        "switches": {
            "use_adaptive_radius": True,
            "use_component_gate": False,
            "use_linear_field": False,
        },
    },
    "component_gated_field": {
        "patch_lens": None,
        "switches": {
            "use_adaptive_radius": True,
            "use_component_gate": True,
            "use_linear_field": False,
        },
    },
    "linear_field_bank": {
        "patch_lens": None,
        "switches": {},
    },
}
FINAL_ABLATION = "continuation_last2"
UNFREEZE_DEPTHS = {
    "continuation_frozen": 0,
    "continuation_last1": 1,
    "continuation_last2": 2,
    "continuation_full": -1,
}
ABLATIONS = (
    *BACKBONE_LADDER,
    "x_to_z",
    "single_stochastic",
    "multi_stochastic",
    "learned_inducing_elbo",
    "process_preserving",
    "fixed_inducing",
    "random_process_families",
    "no_sensor_priors",
    *UNFREEZE_DEPTHS,
    "no_nll",
)

class MeanAdapter(nn.Module):
    def __init__(self, model: LatentFlow):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.forward_distribution(inputs)["mean"]


class BackboneAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, None, None, None)


def task_batch_size(args, dataset: str) -> int:
    return args.wide_batch_size if dataset in {"Electricity", "Traffic"} else args.batch_size


def final_checkpoint(args) -> Path:
    return (
        Path(args.output_root)
        / "checkpoints"
        / "LatentFlow"
        / f"seed{args.seed}"
        / args.dataset
        / f"h{args.horizon}.pt"
    )


def headline_metric_path(args) -> Path:
    return (
        Path(args.output_root)
        / "headline"
        / "runs"
        / "LatentFlow"
        / f"seed{args.seed}"
        / args.dataset
        / f"h{args.horizon}.json"
    )


def build_latentflow(base: nn.Module, task, config: dict, *, mode: str, unfreeze_blocks: int) -> LatentFlow:
    return LatentFlow(
        base,
        dataset=task.dataset,
        channels=task.values.shape[1],
        inducing_points=config["inducing_points"],
        latent_dim=config["latent_dim"],
        horizon_groups=4,
        ablation_mode=mode,
        continuation_unfreeze_blocks=unfreeze_blocks,
    )


def train_final(args, *, save_checkpoint: bool = True) -> tuple[LatentFlow, object, dict, list[dict]]:
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
    backbone_path = ensure_backbone(args, task, device, config["patch_lens"])
    backbone_payload = torch_load(backbone_path)
    base = build_backbone(backbone_payload)
    seed_all(args.seed)
    model = build_latentflow(base, task, config, mode="final", unfreeze_blocks=2).to(device)
    batch_size = task_batch_size(args, args.dataset)
    history, selection = fit_final_model(
        model,
        task,
        args,
        representation_weight=config["representation_weight"],
        nll_weight=args.nll_weight,
        batch_size=batch_size,
        device=device,
    )
    if save_checkpoint:
        path = final_checkpoint(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": clone_state(model),
                "base_config": backbone_payload["config"],
                "selected_config": config,
                "dataset": args.dataset,
                "horizon": args.horizon,
                "channels": task.values.shape[1],
                "seed": args.seed,
                "selection": selection,
                "candidate": FINAL_CANDIDATE,
                "protocol": (
                    "input=96; train-only normalization; 2048 train and 1024 "
                    "validation windows; validation-MSE checkpoint; all test windows"
                ),
            },
            path,
        )
    return model, task, selection, history


def run_headline(args) -> None:
    output = headline_metric_path(args)
    if output.exists() and not args.force:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("candidate") == FINAL_CANDIDATE:
            print(f"skip-complete headline dataset={args.dataset} h={args.horizon} seed={args.seed}")
            return
        print(
            f"replace-stale headline dataset={args.dataset} h={args.horizon} "
            f"seed={args.seed} old_candidate={existing.get('candidate', 'unknown')}",
            flush=True,
        )
    started = time.perf_counter()
    model, task, selection, history = train_final(args)
    device = resolve_device(args.device)
    batch_size = task_batch_size(args, args.dataset)
    metrics = evaluate(
        MeanAdapter(model),
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=batch_size,
        device=device,
        synchronize=True,
    )
    config = SELECTED_CONFIG[args.dataset]
    row = {
        "model": "LatentFlow",
        "candidate": FINAL_CANDIDATE,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        **metrics,
        **selection,
        "latent_dim": config["latent_dim"],
        "inducing_points": config["inducing_points"],
        "patch_lens": "-".join(map(str, config["patch_lens"])),
        "representation_weight": config["representation_weight"],
        "nll_weight": args.nll_weight,
        "continuation_unfreeze_blocks": 2,
        "cross_process_exchange": False,
        "primal_dual": False,
        "parameters": parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(final_checkpoint(args)),
        "selection_protocol": "validation MSE only; test evaluated once",
    }
    write_json(output, row)
    write_csv(
        Path(args.output_root)
        / "headline"
        / "history"
        / f"seed{args.seed}"
        / args.dataset
        / f"h{args.horizon}.csv",
        history,
    )
    print(
        f"latentflow-result dataset={args.dataset:<11} h={args.horizon:<3} "
        f"seed={args.seed} mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
        flush=True,
    )


def _copy_ablation_source(args, variant: str, source: Path, output: Path) -> bool:
    if not source.exists():
        return False
    row = json.loads(source.read_text(encoding="utf-8"))
    source_model = row.get("model", "")
    row.update(
        source_model=source_model,
        model="LatentFlow",
        variant=variant,
        ablation_seed=42,
        ablation_protocol="single seed 42; validation checkpointing; identical test windows",
        primal_dual=False,
    )
    write_json(output, row)
    print(
        f"latentflow-ablation-result variant={variant:<26} "
        f"dataset={args.dataset:<11} h={args.horizon:<3} "
        f"mse={float(row['mse']):.6f} mae={float(row['mae']):.6f}",
        flush=True,
    )
    return True


def run_ablation(args) -> None:
    if args.seed != 42:
        raise ValueError("Every LatentFlow ablation is restricted to the single seed 42.")
    output = (
        Path(args.output_root)
        / "ablations"
        / "runs"
        / args.variant
        / args.dataset
        / f"h{args.horizon}.json"
    )
    if output.exists() and not args.force:
        print(f"skip-complete ablation={args.variant} dataset={args.dataset} h={args.horizon}")
        return

    if args.variant == FINAL_ABLATION:
        source = headline_metric_path(args)
        if not _copy_ablation_source(args, args.variant, source, output):
            raise FileNotFoundError(
                f"Missing seed-42 LatentFlow result {source}. Run headline before ablations."
            )
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
    config = SELECTED_CONFIG[args.dataset]
    batch_size = task_batch_size(args, args.dataset)
    started = time.perf_counter()
    history: list[dict] = []
    unfreeze_blocks: int | None = None
    backbone_switches: dict[str, bool] = {}

    if args.variant in BACKBONE_LADDER:
        ladder = BACKBONE_LADDER[args.variant]
        patches = ladder["patch_lens"] or config["patch_lens"]
        backbone_switches = dict(ladder["switches"])
        if backbone_switches:
            backbone_path = ensure_backbone(
                args,
                task,
                device,
                patches,
                switch_overrides=backbone_switches,
                checkpoint_tag=args.variant,
            )
            backbone_payload = torch_load(backbone_path)
            base = build_backbone(backbone_payload, require_final=False)
        else:
            backbone_path = ensure_backbone(args, task, device, patches)
            backbone_payload = torch_load(backbone_path)
            base = build_backbone(backbone_payload)
        model: nn.Module = BackboneAdapter(base.to(device))
        selection = {
            "best_epoch": int(backbone_payload["selection"]["best_epoch"]),
            "best_stage": "backbone",
            "best_validation_mse": float(
                backbone_payload["selection"]["best_validation_mse"]
            ),
        }
    else:
        backbone_path = ensure_backbone(args, task, device, config["patch_lens"])
        backbone_payload = torch_load(backbone_path)
        base = build_backbone(backbone_payload)
        mode = args.variant
        if args.variant in UNFREEZE_DEPTHS or args.variant == "no_nll":
            mode = "final"
        unfreeze_blocks = UNFREEZE_DEPTHS.get(
            args.variant,
            0 if args.variant == "process_preserving" else 2,
        )
        representation_weight = (
            0.0
            if args.variant in {"x_to_z", "single_stochastic", "multi_stochastic"}
            else config["representation_weight"]
        )
        nll_weight = 0.0 if args.variant == "no_nll" else args.nll_weight
        seed_all(42)
        latentflow = build_latentflow(
            base,
            task,
            config,
            mode=mode,
            unfreeze_blocks=unfreeze_blocks,
        ).to(device)
        history, selection = fit_final_model(
            latentflow,
            task,
            args,
            representation_weight=representation_weight,
            nll_weight=nll_weight,
            batch_size=batch_size,
            device=device,
        )
        model = MeanAdapter(latentflow)

    metrics = evaluate(
        model,
        task,
        task.test_starts,
        seq_len=args.seq_len,
        batch_size=batch_size,
        device=device,
    )
    row = {
        "model": "LatentFlow",
        "variant": args.variant,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": 42,
        **metrics,
        **selection,
        "elapsed_seconds": time.perf_counter() - started,
        "selected_hyperparameters": True,
        "backbone_patch_lens": "-".join(
            map(
                str,
                (BACKBONE_LADDER[args.variant]["patch_lens"] or config["patch_lens"])
                if args.variant in BACKBONE_LADDER
                else config["patch_lens"],
            )
        ),
        "backbone_switches": backbone_switches,
        "continuation_unfreeze_blocks": unfreeze_blocks,
        "primal_dual": False,
        "ablation_protocol": (
            "single seed 42; same train/validation/test windows; "
            "validation-MSE checkpoint selection"
        ),
    }
    write_json(output, row)
    if history:
        write_csv(
            Path(args.output_root)
            / "ablations"
            / "history"
            / args.variant
            / args.dataset
            / f"h{args.horizon}.csv",
            history,
        )
    print(
        f"latentflow-ablation-result variant={args.variant:<26} "
        f"dataset={args.dataset:<11} h={args.horizon:<3} "
        f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
        flush=True,
    )


def load_final(args, device: torch.device):
    path = final_checkpoint(args)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing final checkpoint {path}. Run the seed-42 headline task first."
        )
    payload = torch_load(path)
    if payload.get("candidate") != FINAL_CANDIDATE:
        raise RuntimeError(
            f"Checkpoint {path} belongs to {payload.get('candidate', 'unknown')}; "
            "rerun the revised no-exchange headline task first."
        )
    base = instantiate_backbone(payload["base_config"])
    task = load_task(Path(args.data_root), args.dataset, args.horizon, seq_len=args.seq_len)
    model = build_latentflow(
        base,
        task,
        payload["selected_config"],
        mode="final",
        unfreeze_blocks=2,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.configure_stage("C")
    model.select_reference_only(payload["selection"]["best_stage"] == "reference")
    return model, task, payload, path


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_efficiency(args) -> None:
    if args.seed != 42:
        raise ValueError("Efficiency profiling uses the locked seed-42 checkpoint.")
    output = (
        Path(args.output_root)
        / "efficiency"
        / "runs"
        / "LatentFlow"
        / args.dataset
        / f"h{args.horizon}.json"
    )
    if output.exists() and not args.force:
        print(f"skip-complete efficiency dataset={args.dataset} h={args.horizon}")
        return
    device = resolve_device(args.device)
    model, task, payload, checkpoint = load_final(args, device)
    batch_size = min(args.profile_batch_size, len(task.test_starts))
    inputs, targets = next(iter(loader(task, task.test_starts[:batch_size], args.seq_len, batch_size)))
    inputs = inputs.to(device)
    targets = targets.to(device)
    adapter = MeanAdapter(model)

    adapter.eval()
    inference_times = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(args.profile_repeats + 2):
        synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            adapter(inputs)
        synchronize(device)
        if iteration >= 2:
            inference_times.append(1000.0 * (time.perf_counter() - started))
    peak_inference = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    flops = profiler_flops(adapter, inputs[:1])

    model.configure_stage("C")
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    training_times = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    config = SELECTED_CONFIG[args.dataset]
    for iteration in range(args.profile_repeats + 2):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        _, loss, _, _, _, _ = model.training_objective(
            inputs,
            targets,
            sample_dual=None,
            fenchel_weight=0.0,
            nll_weight=args.nll_weight,
            representation_weight=config["representation_weight"],
            augmented_weight=0.0,
            reconstruction_weight=0.0,
        )
        loss.backward()
        optimizer.step()
        model.project_primal()
        synchronize(device)
        if iteration >= 2:
            training_times.append(1000.0 * (time.perf_counter() - started))
    peak_training = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    row = {
        "model": "LatentFlow",
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
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
        f"latentflow-efficiency dataset={args.dataset:<11} h={args.horizon:<3} "
        f"params={row['parameters_total']} gflops={row['gflops_per_sample']:.4f} "
        f"infer_ms={row['inference_ms_batch']:.3f}",
        flush=True,
    )


def run_smoke(args) -> None:
    device = resolve_device(args.device)
    from physics_modal_v4.experiments.training import backbone_config

    inputs = torch.randn(2, 96, 7, device=device)
    targets = torch.randn(2, 96, 7, device=device)

    # Validate every cumulative backbone configuration without fitting a checkpoint.
    for variant, ladder in BACKBONE_LADDER.items():
        patches = ladder["patch_lens"] or (8, 16, 32)
        config = backbone_config(
            96,
            96,
            patches,
            switch_overrides=ladder["switches"],
        )
        backbone = instantiate_backbone(
            config,
            require_final=not bool(ladder["switches"]),
        ).to(device)
        prediction = backbone(inputs, None, None, None)
        if prediction.shape != targets.shape or not torch.isfinite(prediction).all():
            raise RuntimeError(f"Backbone smoke failed for {variant}.")

    base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
    seed_all(42)
    model = LatentFlow(
        base,
        dataset="ETTh1",
        channels=7,
        inducing_points=8,
        latent_dim=4,
        ablation_mode="final",
        continuation_unfreeze_blocks=2,
    ).to(device)
    if any(len(blocks) for blocks in model.continuation.exchange_blocks):
        raise RuntimeError("Final LatentFlow must not allocate cross-process exchange modules.")
    if any(key.startswith("deterministic_process") for key in model.state_dict()):
        raise RuntimeError("Final LatentFlow unexpectedly allocated matched-control parameters.")
    model.configure_stage("C")
    distribution, loss, _, _, _, _ = model.training_objective(
        inputs,
        targets,
        sample_dual=None,
        fenchel_weight=0.0,
        nll_weight=0.02,
        representation_weight=0.01,
        augmented_weight=0.0,
        reconstruction_weight=0.0,
    )
    loss.backward()
    if distribution["mean"].shape != targets.shape or not torch.isfinite(loss):
        raise RuntimeError("Final LatentFlow smoke failed shape/finite checks.")

    trainable_by_depth = {}
    for name, depth in UNFREEZE_DEPTHS.items():
        probe_base = instantiate_backbone(backbone_config(96, 96, (8, 16, 32)))
        probe = LatentFlow(
            probe_base,
            dataset="ETTh1",
            channels=7,
            inducing_points=8,
            latent_dim=4,
            ablation_mode="final",
            continuation_unfreeze_blocks=depth,
        ).to(device)
        probe.configure_stage("C")
        trainable_by_depth[name] = parameter_count(probe, trainable_only=True)
    ordered = [
        trainable_by_depth["continuation_frozen"],
        trainable_by_depth["continuation_last1"],
        trainable_by_depth["continuation_last2"],
        trainable_by_depth["continuation_full"],
    ]
    if not (ordered[0] < ordered[1] <= ordered[2] < ordered[3]):
        raise RuntimeError(f"Unexpected continuation parameter ladder: {trainable_by_depth}")
    print(
        f"latentflow-smoke passed shape={tuple(distribution['mean'].shape)} "
        f"loss={float(loss.detach()):.6f} exchange_modules=0 "
        f"trainable={trainable_by_depth}",
        flush=True,
    )

def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "headline", "ablation", "efficiency"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    parser.add_argument("--horizon", choices=HORIZONS, type=int, default=96)
    parser.add_argument("--seed", choices=SEEDS, type=int, default=42)
    parser.add_argument("--variant", choices=ABLATIONS, default=FINAL_ABLATION)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-train-windows", type=int, default=2048)
    parser.add_argument("--max-validation-windows", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--wide-batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--reference-epochs", type=int, default=2)
    parser.add_argument("--extension-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--nll-weight", type=float, default=0.02)
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
        run_smoke(args)
    elif args.mode == "headline":
        run_headline(args)
    elif args.mode == "ablation":
        run_ablation(args)
    else:
        run_efficiency(args)


if __name__ == "__main__":
    main()












