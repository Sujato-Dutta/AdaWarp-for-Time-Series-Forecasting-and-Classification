"""Run one limited-budget H=720 replication or MVPF ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import instrumented_pruned_mvpf as final_runner


EXPERIMENTS = (
    "mvpf",
    "patchtst",
    "global_outer",
    "no_variable_mixing",
    "single_scale_8",
    "permute_1",
    "permute_2",
)
PERMUTATION_SEEDS = {"permute_1": 1729, "permute_2": 3253}


class GlobalOuterGate(nn.Module):
    """A single learned field/linear mixture shared across samples."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.65, 0.15], dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits[None].expand(features.shape[0], -1)


def remove_variable_mixing(model: nn.Module) -> None:
    """Set the variable-axis kernel to one while retaining patch-axis mixing."""

    for scale in model.backbone.branches:
        for block in scale.blocks:
            old = block.branches[0]
            patch_kernel = int(old.kernel_size[1])
            block.branches = nn.ModuleList(
                [
                    nn.Conv2d(
                        old.in_channels,
                        old.out_channels,
                        kernel_size=(1, patch_kernel),
                        padding=(0, patch_kernel // 2),
                        groups=old.groups,
                        bias=False,
                    )
                ]
            )
            block.use_adaptive_radius = False
            block.radius_gate = None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def fit_validation_controls_dynamic(
    model: nn.Module,
    series: np.ndarray,
    starts: np.ndarray,
    args: argparse.Namespace,
    horizon: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Validation-only controls for an arbitrary number of field components."""

    ff = fl = ll = fy = ly = 0.0
    num_components = len(model.backbone.branches) + 1
    gram = np.zeros((num_components, num_components), dtype=np.float64)
    rhs = np.zeros(num_components, dtype=np.float64)
    for inputs, targets in final_runner.loader(
        series, starts, args.seq_len, horizon, args.eval_batch_size
    ):
        details = final_runner.forward_details(model, inputs.to(device))
        target = targets.to(device)
        field, linear = details["field"], details["linear"]
        ff += float((field * field).sum().cpu())
        fl += float((field * linear).sum().cpu())
        ll += float((linear * linear).sum().cpu())
        fy += float((field * target).sum().cpu())
        ly += float((linear * target).sum().cpu())
        components = details["components"].permute(0, 2, 3, 1).reshape(-1, num_components)
        component_target = target.reshape(-1)
        values = components.double().cpu().numpy()
        target_values = component_target.double().cpu().numpy()
        gram += values.T @ values
        rhs += values.T @ target_values

    denominator = ff - 2.0 * fl + ll
    beta = 0.5 if abs(denominator) < 1e-12 else float(
        np.clip((fy - ly - fl + ll) / denominator, 0.0, 1.0)
    )
    weights = np.full(num_components, 1.0 / num_components, dtype=np.float64)
    lipschitz = max(1e-9, 2.0 * float(np.linalg.eigvalsh(gram).max()))
    for _ in range(1000):
        updated = final_runner.project_simplex(
            weights - (2.0 * (gram @ weights - rhs)) / lipschitz
        )
        if np.max(np.abs(updated - weights)) < 1e-10:
            weights = updated
            break
        weights = updated
    return beta, weights


def run_mvpf(args: argparse.Namespace, output: Path) -> dict:
    task_root = output / "tasks" / f"{args.experiment}_{args.dataset}_h720_seed{args.seed}"
    runner_args = final_runner.parser().parse_args([])
    runner_args.datasets = [args.dataset]
    runner_args.horizons = [720]
    runner_args.seed = args.seed
    runner_args.data_root = args.data_root
    runner_args.output_root = str(task_root)
    runner_args.device = args.device
    runner_args.max_eval_windows = 1_000_000_000
    runner_args.fail_fast = True
    runner_args.force = args.force

    if args.experiment == "single_scale_8":
        runner_args.patch_lens = [8]
        final_runner.COMPONENT_NAMES = ["direct", "scale_8"]
        final_runner.fit_validation_controls = fit_validation_controls_dynamic

    original_builder = final_runner.build_model

    def build_variant(namespace: argparse.Namespace, horizon: int) -> nn.Module:
        model = original_builder(namespace, horizon)
        if args.experiment == "global_outer":
            model.mean_mixer = GlobalOuterGate()
        elif args.experiment == "no_variable_mixing":
            remove_variable_mixing(model)
        return model

    final_runner.build_model = build_variant

    if args.experiment in PERMUTATION_SEEDS:
        original_loader = final_runner.load_numeric_csv
        permutation_rng = np.random.default_rng(PERMUTATION_SEEDS[args.experiment])

        def load_permuted(path: Path) -> np.ndarray:
            values = original_loader(path)
            permutation = permutation_rng.permutation(values.shape[1])
            permutation_path = task_root / "audit" / "channel_permutation.json"
            permutation_path.parent.mkdir(parents=True, exist_ok=True)
            permutation_path.write_text(
                json.dumps(
                    {
                        "experiment": args.experiment,
                        "permutation_seed": PERMUTATION_SEEDS[args.experiment],
                        "permutation": permutation.tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return values[:, permutation]

        final_runner.load_numeric_csv = load_permuted

    final_runner.run(runner_args)
    metric_path = task_root / "metrics" / "pruned_instrumented_summary.csv"
    with metric_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {metric_path}, found {len(rows)}.")
    row = rows[0]
    window_path = task_root / "per_window" / f"{args.dataset}_h720_seed{args.seed}.csv"
    with window_path.open(encoding="utf-8") as handle:
        num_test_windows = max(0, sum(1 for _ in handle) - 1)
    return {
        "experiment": args.experiment,
        "model": "MVPF",
        "dataset": args.dataset,
        "horizon": 720,
        "seed": args.seed,
        "mse": float(row["learned_mse"]),
        "mae": float(row["learned_mae"]),
        "best_epoch": int(float(row["best_epoch"])),
        "best_validation_mse": float(row["best_validation_mse"]),
        "checkpoint": row["checkpoint"],
        "num_test_windows": num_test_windows,
        "permutation_seed": PERMUTATION_SEEDS.get(args.experiment),
        "task_root": str(task_root),
    }


def run_patchtst(args: argparse.Namespace, output: Path) -> dict:
    task_root = output / "tasks" / f"patchtst_{args.dataset}_h720_seed{args.seed}"
    command = [
        sys.executable,
        str(ROOT / "scripts/tacc/run_tslibrary_ltsf_baselines.py"),
        "--datasets", args.dataset,
        "--models", "PatchTST",
        "--horizons", "720",
        "--seeds", str(args.seed),
        "--seq-len", "96",
        "--label-len", "48",
        "--train-epochs", "10",
        "--batch-size", "32",
        "--learning-rate", "0.0001",
        "--d-model", "128",
        "--d-ff", "256",
        "--n-heads", "8",
        "--e-layers", "2",
        "--d-layers", "1",
        "--patience", "3",
        "--output-root", str(task_root.resolve()),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    metric_path = task_root / "metrics" / "ltsf_tslibrary_baselines.csv"
    with metric_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {metric_path}, found {len(rows)}.")
    row = rows[0]
    return {
        "experiment": "patchtst",
        "model": "PatchTST",
        "dataset": args.dataset,
        "horizon": 720,
        "seed": args.seed,
        "mse": float(row["mse"]),
        "mae": float(row["mae"]),
        "result_dir": row["result_dir"],
        "pred_file": row["pred_file"],
        "true_file": row["true_file"],
        "task_root": str(task_root),
    }


def run(args: argparse.Namespace) -> None:
    if args.experiment in PERMUTATION_SEEDS and args.dataset != "Traffic":
        raise ValueError("Permutation experiments are defined only for Traffic.")
    if args.experiment == "patchtst" and args.seed == 42:
        raise ValueError("The existing seed-42 PatchTST result should be reused.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    output = Path(args.output_root)
    destination = output / "results" / f"{args.experiment}_{args.dataset}_h720_seed{args.seed}.json"
    if destination.exists() and not args.force:
        print(f"skip-complete {destination}", flush=True)
        return
    started = time.perf_counter()
    print(
        f"rescue-start experiment={args.experiment} dataset={args.dataset} seed={args.seed}",
        flush=True,
    )
    payload = run_patchtst(args, output) if args.experiment == "patchtst" else run_mvpf(args, output)
    payload["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(destination, payload)
    print(
        f"rescue-result experiment={args.experiment:<22} dataset={args.dataset:<11} "
        f"seed={args.seed} mse={payload['mse']:.6f} mae={payload['mae']:.6f} "
        f"elapsed={payload['elapsed_seconds']:.1f}s",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    p.add_argument("--dataset", choices=("Electricity", "Traffic"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="TSLibrary/dataset")
    p.add_argument("--output-root", default="mvpf-tests/results/rescue_720")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())