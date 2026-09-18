"""Checkpoint-level submission audits for the frozen LatentFlow architecture.

These routines do not alter the released model. They replay archived fits,
measure numerical behavior, and produce provenance needed by the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_adawarp_ltsf import dataset_path
from physics_modal_v4.enhancements.final_audits import (
    _candidate_audit,
    _load_model,
    _load_task,
    _source_checkpoint,
    _source_metric,
)
from physics_modal_v4.enhancements.run import (
    DATA_BUDGET_PROTOCOL,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    MATCHED_CONTROL_MULTI_PROTOCOL,
    MATCHED_CONTROL_PROTOCOL,
    TIMEPRO_DATA_BUDGET_PROTOCOL,
)
from physics_modal_v4.experiments.protocol import (
    DATASETS,
    HORIZONS,
    SEEDS,
    loader,
    parameter_count,
    resolve_device,
    write_json,
)
from physics_modal_v4.experiments.run import MeanAdapter, profiler_flops
from physics_modal_v4.experiments.training import (
    chronological_selection,
    evaluate_distribution,
    torch_load,
)


AUDIT_PROTOCOL = "latentflow-submission-checks-v1"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        item = np.ascontiguousarray(array)
        digest.update(str(item.dtype).encode())
        digest.update(np.asarray(item.shape, dtype=np.int64).tobytes())
        digest.update(item.tobytes())
    return digest.hexdigest()


def result_path(args, name: str) -> Path:
    path = (
        Path(args.output_root) / "submission_checks" / name
        / f"seed{args.seed}" / args.dataset / f"h{args.horizon}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def exact_gaussian_unit_test() -> dict:
    """Check component recovery and dense/inducing posterior moments."""
    torch.manual_seed(2027)
    dtype = torch.float64
    factor_1 = torch.randn(5, 2, dtype=dtype)
    factor_2 = torch.randn(5, 1, dtype=dtype)
    a1 = factor_1 @ factor_1.T
    a2 = factor_2 @ factor_2.T
    total = a1 + a2
    noise = 0.17
    observed = torch.randn(5, 3, dtype=dtype)
    alpha = torch.linalg.solve(total + noise * torch.eye(5, dtype=dtype), observed)
    components = [a1 @ alpha, a2 @ alpha]
    aggregate = sum(components)
    recovered = [matrix @ torch.linalg.pinv(total) @ aggregate for matrix in (a1, a2)]
    recovery_error = max(
        float((left - right).abs().max())
        for left, right in zip(components, recovered)
    )

    times = torch.arange(8, dtype=dtype)
    covariance = torch.exp(-0.5 * (times[:, None] - times[None, :]).square() / 2.3**2)
    past = covariance[:5, :5]
    cross = covariance[5:, :5]
    future = covariance[5:, 5:]
    dense_mean = cross @ torch.linalg.solve(
        past + noise * torch.eye(5, dtype=dtype), observed
    )
    dense_variance = future - cross @ torch.linalg.solve(
        past + noise * torch.eye(5, dtype=dtype), cross.T
    )
    # With every past timestamp used as an inducing timestamp, the Nyström
    # quantities equal the dense past and cross covariances.
    chol = torch.linalg.cholesky(past + 1e-10 * torch.eye(5, dtype=dtype))
    q_past = past @ torch.cholesky_solve(past, chol)
    q_cross = cross @ torch.cholesky_solve(past, chol)
    sparse_mean = q_cross @ torch.linalg.solve(
        q_past + noise * torch.eye(5, dtype=dtype), observed
    )
    sparse_variance = future - q_cross @ torch.linalg.solve(
        q_past + noise * torch.eye(5, dtype=dtype), q_cross.T
    )
    mean_error = float((dense_mean - sparse_mean).abs().max())
    variance_error = float((dense_variance - sparse_variance).abs().max())
    passed = recovery_error < 1e-8 and mean_error < 1e-7 and variance_error < 1e-7
    return {
        "audit_protocol": AUDIT_PROTOCOL,
        "component_recovery_max_abs": recovery_error,
        "dense_sparse_mean_max_abs": mean_error,
        "dense_sparse_variance_max_abs": variance_error,
        "passed": passed,
    }


def run_provenance(args) -> None:
    task = _load_task(args)
    checkpoint = _source_checkpoint(args, "final_no_exchange")
    metric = _source_metric(args, "final_no_exchange")
    payload = torch_load(checkpoint)
    metric_row = json.loads(metric.read_text(encoding="utf-8"))
    source_csv = dataset_path(Path(args.data_root), args.dataset)
    coverage = {}
    for name, starts in (
        ("train", task.training_starts),
        ("validation", task.validation_starts),
        ("test", task.test_starts),
    ):
        occupied = np.zeros(len(task.values), dtype=bool)
        for start in starts:
            occupied[int(start): int(start) + args.seq_len + args.horizon] = True
        coverage[f"{name}_origins"] = int(len(starts))
        coverage[f"{name}_unique_timepoints"] = int(occupied.sum())
    row = {
        "audit_protocol": AUDIT_PROTOCOL,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "metric_sha256": sha256_file(metric),
        "dataset_sha256": sha256_file(source_csv),
        "window_index_sha256": sha256_arrays(
            task.training_starts, task.validation_starts, task.test_starts
        ),
        "selected_stage": payload["selection"]["best_stage"],
        "selected_epoch": int(payload["selection"]["best_epoch"]),
        "selected_validation_mse": float(payload["selection"]["best_validation_mse"]),
        "test_mse": float(metric_row["mse"]),
        "test_mae": float(metric_row["mae"]),
        "candidate": payload.get("candidate"),
        "selected_config": payload.get("selected_config"),
        **coverage,
    }
    write_json(result_path(args, "provenance"), row)
    print(
        f"submission-provenance dataset={args.dataset} h={args.horizon} "
        f"seed={args.seed} stage={row['selected_stage']} "
        f"checkpoint={row['checkpoint_sha256'][:12]}", flush=True,
    )


@torch.no_grad()
def run_traffic_replay(args) -> None:
    if (args.dataset, args.horizon, args.seed) != ("Traffic", 720, 42):
        raise ValueError("Canonical replay is locked to Traffic H=720 seed=42.")
    task = _load_task(args)
    model, payload, checkpoint, selection = _load_model(
        args, task, "final_no_exchange"
    )
    reference_only = selection["best_stage"] == "reference"
    model.select_reference_only(reference_only)
    model.eval()
    device = resolve_device(args.device)
    digest_one, digest_four = hashlib.sha256(), hashlib.sha256()
    count = 0
    abs_delta = squared_delta = maximum = 0.0
    squared_one = squared_four = absolute_one = absolute_four = 0.0
    for inputs, targets in loader(task, task.test_starts, args.seq_len, 4):
        targets = targets.to(device)
        inputs = inputs.to(device)
        prediction_four = model.forward_distribution(inputs)["mean"]
        prediction_one = torch.cat([
            model.forward_distribution(inputs[index:index + 1])["mean"]
            for index in range(inputs.shape[0])
        ], dim=0)
        delta = (prediction_one - prediction_four).double()
        count += delta.numel()
        abs_delta += float(delta.abs().sum())
        squared_delta += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        for left, right in zip(prediction_one.cpu(), prediction_four.cpu()):
            digest_one.update(left.contiguous().numpy().tobytes())
            digest_four.update(right.contiguous().numpy().tobytes())
        error_one = prediction_one - targets
        error_four = prediction_four - targets
        squared_one += float(error_one.square().sum())
        squared_four += float(error_four.square().sum())
        absolute_one += float(error_one.abs().sum())
        absolute_four += float(error_four.abs().sum())
    tolerance = maximum <= args.replay_atol
    row = {
        "audit_protocol": AUDIT_PROTOCOL,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "selected_stage": selection["best_stage"],
        "selected_epoch": int(selection["best_epoch"]),
        "batch1_mse": squared_one / count,
        "batch4_mse": squared_four / count,
        "batch1_mae": absolute_one / count,
        "batch4_mae": absolute_four / count,
        "mean_absolute_prediction_difference": abs_delta / count,
        "rms_prediction_difference": math.sqrt(squared_delta / count),
        "maximum_absolute_prediction_difference": maximum,
        "batch1_prediction_sha256": digest_one.hexdigest(),
        "batch4_prediction_sha256": digest_four.hexdigest(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "window_index_sha256": sha256_arrays(task.test_starts),
        "absolute_tolerance": args.replay_atol,
        "passed": tolerance,
        "config": payload.get("selected_config"),
    }
    write_json(result_path(args, "traffic_replay"), row)
    print(
        "traffic-replay-result "
        f"max_abs={maximum:.9g} mean_abs={row['mean_absolute_prediction_difference']:.9g} "
        f"mse_b1={row['batch1_mse']:.9f} mse_b4={row['batch4_mse']:.9f} "
        f"passed={tolerance}", flush=True,
    )


@torch.no_grad()
def run_numerical(args) -> None:
    task = _load_task(args)
    model, _, _, _ = _load_model(args, task, "final_no_exchange")
    model.select_reference_only(False)
    model.eval()
    device = resolve_device(args.device)
    inputs, _ = next(iter(loader(task, task.test_starts[:1], args.seq_len, 1)))
    inputs = inputs.to(device)
    distribution = model.forward_distribution(inputs)
    posterior = distribution["latent_posterior"]
    normalized, _, _ = model.continuation.mvpff.backbone._normalize(inputs)
    structural = model.encoder(normalized)
    sparse, past_cov, _, _, _ = model.separator._sparse_terms(structural)
    _, group_weights, _ = model.separator.sensor_geometry(inputs)
    noise = torch.nn.functional.softplus(model.separator.raw_noise_std).clamp_min(1e-4)
    identity = torch.eye(model.seq_len, device=device, dtype=structural.dtype)
    observations = [
        (weights[:, None, None] * past_cov).sum(dim=0)
        + (noise.square() + model.separator.jitter) * identity
        for weights in group_weights
    ]
    observation_eigenvalues = [torch.linalg.eigvalsh(item) for item in observations]
    kernel_eigenvalues = []
    physical = {}
    boundary_hits = total_raw = 0

    def count_boundary(tensor, low: float, high: float) -> None:
        nonlocal boundary_hits, total_raw
        values = tensor.detach()
        tolerance = 1e-5
        total_raw += values.numel()
        boundary_hits += int(
            ((values <= low + tolerance) | (values >= high - tolerance)).sum()
        )

    for index, (kernel, terms) in enumerate(zip(model.separator.kernels, sparse)):
        inducing = terms["inducing_times"]
        kuu = kernel.kernel(inducing, inducing)
        kernel_eigenvalues.append(torch.linalg.eigvalsh(kuu))
        physical[f"process_{index}_{kernel.name}"] = {
            key: float(value.detach())
            for key, value in kernel.parameters_physical().items()
        }
        count_boundary(kernel.raw_amplitude, -8.0, 5.0)
        count_boundary(
            kernel.raw_length,
            -8.0,
            max(32.0, 14.0 * kernel.samples_day),
        )
        count_boundary(kernel.raw_inducing_gaps, -8.0, 8.0)
        if kernel.raw_decay is not None:
            count_boundary(
                kernel.raw_decay,
                -8.0,
                max(64.0, 28.0 * kernel.samples_day),
            )
            count_boundary(kernel.raw_period_adjustment, -5.0, 5.0)
    adapter_norms = [float(module.weight.detach().norm()) for module in model.continuation.patch_adapters]
    row = {
        "audit_protocol": AUDIT_PROTOCOL,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "latent_rms": float(posterior["latent_mean"].square().mean().sqrt()),
        "posterior_variance_mean": float(posterior["posterior_past_variance"].mean()),
        "decoder_weight_norm": float(model.encoder.decoder.weight.detach().norm()),
        "adapter_weight_norm_min": min(adapter_norms),
        "adapter_weight_norm_max": max(adapter_norms),
        "observation_noise_std": float(noise),
        "observation_min_eigenvalue": min(float(v.min()) for v in observation_eigenvalues),
        "observation_max_condition_number": max(float(v.max() / v.min()) for v in observation_eigenvalues),
        "kernel_min_eigenvalue": min(float(v.min()) for v in kernel_eigenvalues),
        "raw_parameter_boundary_rate": boundary_hits / max(1, total_raw),
        "predictive_variance_min": float(distribution["variance"].min()),
        "all_finite": bool(all(torch.isfinite(value).all() for value in (
            posterior["latent_mean"], posterior["posterior_past_variance"],
            distribution["mean"], distribution["variance"],
        ))),
        "process_parameters": physical,
    }
    write_json(result_path(args, "numerical"), row)
    print(
        f"numerical-audit-result dataset={args.dataset} h={args.horizon} "
        f"min_eig={row['observation_min_eigenvalue']:.3e} "
        f"max_cond={row['observation_max_condition_number']:.3e} "
        f"finite={row['all_finite']}", flush=True,
    )


def run_selector(args) -> None:
    args.variant = "final_no_exchange"
    task = _load_task(args)
    model, test = _candidate_audit(args, task, "final_no_exchange")
    starts = chronological_selection(task.validation_starts)
    device = resolve_device(args.device)
    model.select_reference_only(True)
    reference = evaluate_distribution(
        model, task, starts, seq_len=args.seq_len,
        batch_size=args.wide_batch_size if args.dataset in {"Electricity", "Traffic"} else args.batch_size,
        device=device,
    )
    model.select_reference_only(False)
    extension = evaluate_distribution(
        model, task, starts, seq_len=args.seq_len,
        batch_size=args.wide_batch_size if args.dataset in {"Electricity", "Traffic"} else args.batch_size,
        device=device,
    )
    row = {
        "audit_protocol": AUDIT_PROTOCOL,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "selected_stage": test["validation_selected_stage"],
        "validation_reference_mse": reference["mse"],
        "validation_extension_mse": extension["mse"],
        "validation_extension_minus_reference_mse": extension["mse"] - reference["mse"],
        "test_reference_mse": test["reference_mse"],
        "test_extension_mse": test["extension_mse"],
        "test_extension_minus_reference_mse": test["extension_mse"] - test["reference_mse"],
        "extension_retrained": test["extension_retrained"],
    }
    write_json(result_path(args, "selector_validation_test"), row)
    print(
        f"selector-audit-result dataset={args.dataset} h={args.horizon} seed={args.seed} "
        f"val_delta={row['validation_extension_minus_reference_mse']:+.6f} "
        f"test_delta={row['test_extension_minus_reference_mse']:+.6f}", flush=True,
    )


@torch.no_grad()
def run_fusion_profile(args) -> None:
    task = _load_task(args)
    inputs, _ = next(iter(loader(task, task.test_starts[:1], args.seq_len, 1)))
    inputs = inputs.to(resolve_device(args.device))
    profiles = {}
    for variant in ("final_no_exchange", "early_fusion", "x_to_z"):
        model, _, _, selection = _load_model(args, task, variant)
        model.configure_stage("C")
        active = parameter_count(model, trainable_only=True)
        model.select_reference_only(False)
        model.eval()
        profiles[variant] = {
            "parameters_total": parameter_count(model),
            "parameters_active_stage_c": active,
            "gflops_extension_per_sample": profiler_flops(MeanAdapter(model), inputs) / 1e9,
            "validation_selected_stage": selection["best_stage"],
        }
    row = {
        "audit_protocol": AUDIT_PROTOCOL,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "profiles": profiles,
    }
    write_json(result_path(args, "fusion_profile"), row)
    print(
        f"fusion-profile-result dataset={args.dataset} h={args.horizon} seed={args.seed} "
        f"preserved_gflops={profiles['final_no_exchange']['gflops_extension_per_sample']:.5f} "
        f"early_gflops={profiles['early_fusion']['gflops_extension_per_sample']:.5f}", flush=True,
    )


def run_aggregate(args) -> None:
    root = Path(args.output_root)
    audit_root = root / "submission_checks"
    output = audit_root / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    expected = {
        "provenance": 140,
        "selector_validation_test": 140,
        "numerical": 28,
        "fusion_profile": 28,
        "traffic_replay": 1,
    }
    status = {"audit_protocol": AUDIT_PROTOCOL}
    for name, count in expected.items():
        rows = []
        for path in sorted((audit_root / name).glob("seed*/*/h*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("audit_protocol") == AUDIT_PROTOCOL:
                row["artifact"] = str(path)
                rows.append(row)
        frame = pd.json_normalize(rows, sep=".")
        frame.to_csv(output / f"{name}.csv", index=False)
        status[f"{name}_rows"] = len(frame)
        if len(frame) != count and not args.allow_incomplete:
            raise RuntimeError(f"{name}: expected {count} rows, found {len(frame)}")
    matched = []
    for path in sorted((root / "matched_controls" / "runs").glob("**/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("enhancement_protocol") == "latentflow-matched-nongp-multibranch-all28-seed42-v1":
            row["artifact"] = str(path)
            matched.append(row)
    pd.json_normalize(matched, sep=".").to_csv(
        output / "matched_non_gp_multibranch.csv", index=False
    )
    status["matched_control_rows"] = len(matched)
    if len(matched) != 28 and not args.allow_incomplete:
        raise RuntimeError(
            f"matched controls: expected 28 rows, found {len(matched)}"
        )
    write_json(output / "status.json", status)
    print("LATENTFLOW-SUBMISSION-AUDITS COMPLETE", status, flush=True)


def _collect_protocol_rows(root: Path, pattern: str, protocol: str) -> list[dict]:
    rows = []
    for path in sorted(root.glob(pattern)):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("enhancement_protocol") == protocol:
            row["artifact"] = str(path)
            rows.append(row)
    return rows


def _comparison_summary(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    summaries = []
    groups = [("Overall", frame)] + list(frame.groupby("dataset", sort=False))
    for dataset, group in groups:
        for metric in ("mse", "mae"):
            left_values = group[f"{metric}_{left}"]
            right_values = group[f"{metric}_{right}"]
            delta = left_values - right_values
            summaries.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    f"mean_{left}": float(left_values.mean()),
                    f"mean_{right}": float(right_values.mean()),
                    f"{left}_gain_percent": float(
                        100.0 * (right_values.mean() - left_values.mean())
                        / right_values.mean()
                    ),
                    f"{left}_wins": int((delta < -1e-12).sum()),
                    "ties": int((delta.abs() <= 1e-12).sum()),
                    f"{right}_wins": int((delta > 1e-12).sum()),
                    "runs": int(len(group)),
                }
            )
    return pd.DataFrame(summaries)


def run_remaining_aggregate(args) -> None:
    """Aggregate the final multiseed control and matched budget curve."""
    root = Path(args.output_root)
    source = Path(args.source_root)
    output = root / "remaining_checks" / "statistics"
    output.mkdir(parents=True, exist_ok=True)

    controls = _collect_protocol_rows(
        root, "matched_controls/runs/**/*.json", MATCHED_CONTROL_PROTOCOL
    )
    controls.extend(
        _collect_protocol_rows(
            root,
            "matched_controls_multiseed/runs/**/*.json",
            MATCHED_CONTROL_MULTI_PROTOCOL,
        )
    )
    if len(controls) != 140:
        raise RuntimeError(f"matched non-GP controls: expected 140 rows, found {len(controls)}")
    control = pd.json_normalize(controls, sep=".")
    latent_rows = []
    for path in sorted((source / "headline" / "runs" / "LatentFlow").glob("seed*/*/h*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("candidate") == "latentflow_process_preserving_unfreeze2":
            latent_rows.append(row)
    if len(latent_rows) != 140:
        raise RuntimeError(f"LatentFlow five-seed rows: expected 140, found {len(latent_rows)}")
    latent = pd.DataFrame(latent_rows)
    keys = ["dataset", "horizon", "seed"]
    matched = latent[keys + ["mse", "mae"]].merge(
        control[keys + ["mse", "mae", "validation_selected_stage"]],
        on=keys,
        suffixes=("_latentflow", "_control"),
        validate="one_to_one",
    )
    matched.to_csv(output / "matched_non_gp_five_seed_pairs.csv", index=False)
    control.to_csv(output / "matched_non_gp_five_seed_runs.csv", index=False)
    matched_tasks = (
        matched.groupby(["dataset", "horizon"], sort=False)
        .agg(
            mse_latentflow=("mse_latentflow", "mean"),
            mae_latentflow=("mae_latentflow", "mean"),
            mse_control=("mse_control", "mean"),
            mae_control=("mae_control", "mean"),
            seeds=("seed", "size"),
        )
        .reset_index()
    )
    matched_tasks.to_csv(
        output / "matched_non_gp_five_seed_task_means.csv", index=False
    )
    _comparison_summary(matched_tasks, "latentflow", "control").to_csv(
        output / "matched_non_gp_five_seed_summary.csv", index=False
    )
    (
        matched.groupby("seed", sort=True)
        .agg(
            mse_latentflow=("mse_latentflow", "mean"),
            mae_latentflow=("mae_latentflow", "mean"),
            mse_control=("mse_control", "mean"),
            mae_control=("mae_control", "mean"),
        )
        .reset_index()
        .to_csv(output / "matched_non_gp_five_seed_by_seed.csv", index=False)
    )

    timepro_rows = _collect_protocol_rows(
        root, "timepro_data_budget/runs/**/*.json", TIMEPRO_DATA_BUDGET_PROTOCOL
    )
    if len(timepro_rows) != 84:
        raise RuntimeError(f"new TimePro budget rows: expected 84, found {len(timepro_rows)}")
    for path in sorted((source / "headline" / "runs" / "TimePro" / "seed42").glob("*/h*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row.update(
            {
                "train_budget": 2048,
                "train_budget_label": "2048",
                "num_training_windows": 2048,
                "num_validation_windows": 1024,
                "nested_anchor_windows": 2048,
                "nested_subset_seed": 9042,
                "validation_window_seed": 5042,
                "artifact": str(path),
            }
        )
        timepro_rows.append(row)
    if len(timepro_rows) != 112:
        raise RuntimeError(f"complete TimePro budget curve: expected 112, found {len(timepro_rows)}")
    timepro = pd.json_normalize(timepro_rows, sep=".")

    latent_budget_rows = _collect_protocol_rows(
        root, "data_budget/runs/**/*.json", DATA_BUDGET_PROTOCOL
    )
    if len(latent_budget_rows) != 112:
        raise RuntimeError(
            f"LatentFlow budget curve: expected 112 rows, found {len(latent_budget_rows)}"
        )
    latent_budget = pd.json_normalize(latent_budget_rows, sep=".")
    curve_columns = [
        "model", "dataset", "horizon", "seed", "train_budget",
        "train_budget_label", "num_training_windows", "num_validation_windows",
        "mse", "mae", "best_epoch", "best_validation_mse", "artifact",
    ]
    curve = pd.concat(
        [latent_budget.reindex(columns=curve_columns), timepro.reindex(columns=curve_columns)],
        ignore_index=True,
    )
    curve.to_csv(output / "latentflow_timepro_budget_runs.csv", index=False)
    summary = (
        curve.groupby(["model", "train_budget_label", "dataset"], sort=False)
        .agg(mean_mse=("mse", "mean"), mean_mae=("mae", "mean"), tasks=("mse", "size"))
        .reset_index()
    )
    overall = (
        curve.groupby(["model", "train_budget_label"], sort=False)
        .agg(mean_mse=("mse", "mean"), mean_mae=("mae", "mean"), tasks=("mse", "size"))
        .reset_index()
    )
    overall.insert(2, "dataset", "Overall")
    pd.concat([overall, summary], ignore_index=True).to_csv(
        output / "latentflow_timepro_budget_summary.csv", index=False
    )
    budget_pairs = latent_budget[
        ["dataset", "horizon", "train_budget_label", "mse", "mae"]
    ].merge(
        timepro[["dataset", "horizon", "train_budget_label", "mse", "mae"]],
        on=["dataset", "horizon", "train_budget_label"],
        suffixes=("_latentflow", "_timepro"),
        validate="one_to_one",
    )
    budget_pairs.to_csv(output / "latentflow_timepro_budget_pairs.csv", index=False)
    budget_summaries = []
    for budget, group in budget_pairs.groupby("train_budget_label", sort=False):
        item = _comparison_summary(group, "latentflow", "timepro")
        item.insert(0, "train_budget_label", budget)
        budget_summaries.append(item)
    pd.concat(budget_summaries, ignore_index=True).to_csv(
        output / "latentflow_timepro_budget_comparison.csv", index=False
    )

    status = {
        "matched_control_rows": int(len(control)),
        "matched_pairs": int(len(matched)),
        "matched_tasks": int(len(matched_tasks)),
        "timepro_budget_rows": int(len(timepro)),
        "latentflow_budget_rows": int(len(latent_budget)),
        "budget_pairs": int(len(budget_pairs)),
    }
    write_json(output / "status.json", status)
    print("LATENTFLOW-REMAINING-CHECKS COMPLETE", status, flush=True)

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=(
        "smoke", "provenance", "traffic-replay", "numerical",
        "selector", "fusion-profile", "aggregate", "remaining-aggregate",
    ), required=True)
    result.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    result.add_argument("--horizon", choices=HORIZONS, type=int, default=96)
    result.add_argument("--seed", choices=SEEDS, type=int, default=42)
    result.add_argument("--seq-len", type=int, default=96)
    result.add_argument("--max-train-windows", type=int, default=2048)
    result.add_argument("--max-validation-windows", type=int, default=1024)
    result.add_argument("--batch-size", type=int, default=16)
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
    result.add_argument("--replay-atol", type=float, default=2e-5)
    result.add_argument("--force", action="store_true")
    result.add_argument("--allow-incomplete", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode == "smoke":
        result = exact_gaussian_unit_test()
        if not result["passed"]:
            raise RuntimeError(result)
        output = Path(args.output_root) / "submission_checks" / "gaussian_unit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
        print("LATENTFLOW-SUBMISSION-CHECKS-SMOKE PASSED", result, flush=True)
    elif args.mode == "provenance":
        run_provenance(args)
    elif args.mode == "traffic-replay":
        run_traffic_replay(args)
    elif args.mode == "numerical":
        run_numerical(args)
    elif args.mode == "selector":
        run_selector(args)
    elif args.mode == "fusion-profile":
        run_fusion_profile(args)
    elif args.mode == "remaining-aggregate":
        run_remaining_aggregate(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
