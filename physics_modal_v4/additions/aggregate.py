"""Aggregate the isolated LatentFlow additions into auditable CSVs and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physics_modal_v4.experiments.protocol import DATASETS, HORIZONS, SEEDS, write_csv, write_json


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, allow_incomplete: bool) -> dict | None:
    if path.exists():
        return read_json(path)
    if allow_incomplete:
        return None
    raise FileNotFoundError(path)


def ci(values, level: float = 0.95):
    values = np.asarray(values, dtype=float)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _timepro_path(additions: Path, source: Path, seed: int, dataset: str, horizon: int) -> Path:
    if seed == 42:
        return source / "headline" / "runs" / "TimePro" / "seed42" / dataset / f"h{horizon}.json"
    return additions / "timepro_multiseed" / "runs" / f"seed{seed}" / dataset / f"h{horizon}.json"


def _latentflow_path(source: Path, seed: int, dataset: str, horizon: int) -> Path:
    return source / "headline" / "runs" / "LatentFlow" / f"seed{seed}" / dataset / f"h{horizon}.json"


def aggregate_timepro(args, additions: Path, output: Path) -> None:
    source = Path(args.source_root)
    rows = []
    tensor = {metric: np.full((len(DATASETS), len(HORIZONS), len(SEEDS), 2), np.nan) for metric in ("mse", "mae")}
    for di, dataset in enumerate(DATASETS):
        for hi, horizon in enumerate(HORIZONS):
            for si, seed in enumerate(SEEDS):
                lf = require(_latentflow_path(source, seed, dataset, horizon), args.allow_incomplete)
                tp = require(_timepro_path(additions, source, seed, dataset, horizon), args.allow_incomplete)
                if lf is None or tp is None:
                    continue
                for mi, item in enumerate((lf, tp)):
                    for metric in tensor:
                        tensor[metric][di, hi, si, mi] = item[metric]
                rows.extend([
                    {"model": "LatentFlow", "dataset": dataset, "horizon": horizon, "seed": seed, "mse": lf["mse"], "mae": lf["mae"]},
                    {"model": "TimePro", "dataset": dataset, "horizon": horizon, "seed": seed, "mse": tp["mse"], "mae": tp["mae"]},
                ])
    write_csv(output / "timepro_five_seed_runs.csv", rows)
    summary, inference = [], []
    rng = np.random.default_rng(20260903)
    for metric, values in tensor.items():
        if np.isnan(values).any():
            if not args.allow_incomplete:
                raise RuntimeError(f"Incomplete five-seed {metric} tensor.")
            continue
        for di, dataset in enumerate(DATASETS):
            for model_index, model in enumerate(("LatentFlow", "TimePro")):
                seed_means = values[di, :, :, model_index].mean(axis=0)
                summary.append({
                    "dataset": dataset, "metric": metric, "model": model,
                    "mean": float(seed_means.mean()), "sd_across_seeds": float(seed_means.std(ddof=1)),
                    "tasks": 4, "seeds": 5,
                })
        task_means = values.mean(axis=2)
        difference = task_means[:, :, 1] - task_means[:, :, 0]
        relative = 100.0 * difference / task_means[:, :, 1]
        boot = np.empty(args.bootstrap_repetitions)
        for repeat in range(args.bootstrap_repetitions):
            dataset_indices = rng.integers(0, len(DATASETS), len(DATASETS))
            seed_indices = rng.integers(0, len(SEEDS), len(SEEDS))
            sampled = values[dataset_indices][:, :, seed_indices]
            per_cell = 100.0 * (sampled[..., 1] - sampled[..., 0]) / sampled[..., 1]
            boot[repeat] = per_cell.mean()
        lower, upper = ci(boot)
        from scipy.stats import wilcoxon
        stat, pvalue = wilcoxon(difference.reshape(-1), alternative="greater", zero_method="wilcox")
        wins = int((difference > 1e-12).sum())
        losses = int((difference < -1e-12).sum())
        inference.append({
            "metric": metric,
            "latentflow_mean": float(values[..., 0].mean()),
            "timepro_mean": float(values[..., 1].mean()),
            "relative_improvement_percent": float(relative.mean()),
            "dataset_seed_cluster_bootstrap_lower_95": lower,
            "dataset_seed_cluster_bootstrap_upper_95": upper,
            "paired_task_wilcoxon_statistic": float(stat),
            "paired_task_wilcoxon_one_sided_p": float(pvalue),
            "wins": wins, "ties": 28 - wins - losses, "losses": losses,
            "tasks": 28, "seeds": 5, "bootstrap_repetitions": args.bootstrap_repetitions,
        })
    write_csv(output / "timepro_five_seed_dataset_summary.csv", summary)
    write_csv(output / "timepro_five_seed_inference.csv", inference)


def aggregate_central(args, additions: Path, output: Path) -> None:
    variants = (
        "final_no_exchange",
        "x_to_z",
        "early_fusion",
        "no_multiscale",
        "no_linear_bank",
    )
    rows, summary = [], []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for variant in variants:
                values = []
                for seed in SEEDS:
                    path = (
                        additions
                        / "central_ablation"
                        / "runs"
                        / variant
                        / f"seed{seed}"
                        / dataset
                        / f"h{horizon}.json"
                    )
                    row = require(path, args.allow_incomplete)
                    if row is not None:
                        values.append(row)
                        rows.append({
                            "dataset": dataset,
                            "horizon": horizon,
                            "variant": variant,
                            "seed": seed,
                            "mse": row["mse"],
                            "mae": row["mae"],
                        })
                if values:
                    summary.append({
                        "dataset": dataset,
                        "horizon": horizon,
                        "variant": variant,
                        "seeds": len(values),
                        "mse_mean": float(np.mean([v["mse"] for v in values])),
                        "mse_sd": float(np.std([v["mse"] for v in values], ddof=1)) if len(values) > 1 else 0.0,
                        "mae_mean": float(np.mean([v["mae"] for v in values])),
                        "mae_sd": float(np.std([v["mae"] for v in values], ddof=1)) if len(values) > 1 else 0.0,
                    })
    write_csv(output / "central_ablation_runs.csv", rows)
    write_csv(output / "central_ablation_summary.csv", summary)


def aggregate_synthetic(args, additions: Path, output: Path) -> None:
    rows, assignments, kernels = [], [], []
    for setting in ("independent", "interacting"):
        for variant in ("early_fusion", "independent_branches", "process_preserving", "full"):
            for seed in SEEDS:
                path = additions / "synthetic" / "runs" / setting / variant / f"seed{seed}.json"
                item = require(path, args.allow_incomplete)
                if item is None:
                    continue
                rows.append({
                    "setting": setting, "variant": variant, "seed": seed,
                    "mse": item["mse"], "mae": item["mae"],
                    "component_correlation": item["mean_absolute_component_correlation"],
                    "component_nrmse": item["mean_component_nrmse"],
                    "identity_accuracy": item["process_identity_accuracy"],
                    "inducing_chamfer_error": float(np.mean([
                        process["inducing_chamfer_error_samples"]
                        for process in item["learned_processes"]
                    ])),
                })
                assignments.extend({"setting": setting, "variant": variant, "seed": seed, **row} for row in item["assignment"])
                kernels.extend({
                    "setting": setting, "variant": variant, "seed": seed, **process
                } for process in item["learned_processes"])
    write_csv(output / "synthetic_runs.csv", rows)
    write_csv(output / "synthetic_assignments.csv", assignments)
    write_csv(output / "synthetic_kernel_recovery.csv", kernels)
    summary = []
    for setting in ("independent", "interacting"):
        for variant in ("early_fusion", "independent_branches", "process_preserving", "full"):
            selected = [row for row in rows if row["setting"] == setting and row["variant"] == variant]
            if not selected:
                continue
            summary.append({
                "setting": setting, "variant": variant, "seeds": len(selected),
                **{f"{name}_mean": float(np.mean([r[name] for r in selected])) for name in ("mse", "mae", "component_correlation", "component_nrmse", "identity_accuracy", "inducing_chamfer_error")},
                **{f"{name}_sd": float(np.std([r[name] for r in selected], ddof=1)) if len(selected) > 1 else 0.0 for name in ("mse", "component_correlation")},
            })
    write_csv(output / "synthetic_summary.csv", summary)


def aggregate_other(args, additions: Path, output: Path) -> None:
    missing = []
    for path in sorted((additions / "missing_prefix" / "runs").glob("**/*.json")):
        missing.extend(read_json(path)["rows"])
    write_csv(output / "missing_prefix.csv", missing)
    control = []
    for path in sorted((additions / "vpnet_separator_control" / "runs").glob("**/*.json")):
        item = read_json(path)
        dataset, horizon = item["dataset"], int(item["horizon"])
        vpnet = require(
            Path(args.source_root) / "headline" / "runs" / "VPNet" / "seed42" / dataset / f"h{horizon}.json",
            args.allow_incomplete,
        )
        latentflow = require(
            Path(args.source_root) / "headline" / "runs" / "LatentFlow" / "seed42" / dataset / f"h{horizon}.json",
            args.allow_incomplete,
        )
        if vpnet is not None and latentflow is not None:
            item.update({
                "vpnet_mse": vpnet["mse"], "latentflow_mse": latentflow["mse"],
                "separator_gain_over_vpnet_percent": 100.0 * (vpnet["mse"] - item["mse"]) / vpnet["mse"],
                "latentflow_gain_over_separator_control_percent": 100.0 * (item["mse"] - latentflow["mse"]) / item["mse"],
            })
        control.append(item)
    write_csv(output / "vpnet_separator_control.csv", control)
    scaling = additions / "channel_scaling" / "channel_scaling.json"
    if scaling.exists():
        write_csv(output / "channel_scaling.csv", read_json(scaling)["rows"])


def plots(output: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def csv_rows(name):
        path = output / name
        return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = csv_rows("timepro_five_seed_dataset_summary.csv")
    if rows:
        datasets = list(DATASETS)
        figure, axis = plt.subplots(figsize=(7.6, 3.1))
        x = np.arange(len(datasets)); width = 0.36
        for offset, model, color in ((-width / 2, "LatentFlow", "#1764ab"), (width / 2, "TimePro", "#d95f02")):
            chosen = [next(r for r in rows if r["dataset"] == d and r["metric"] == "mse" and r["model"] == model) for d in datasets]
            axis.bar(x + offset, [float(r["mean"]) for r in chosen], width, yerr=[float(r["sd_across_seeds"]) for r in chosen], label=model, color=color, capsize=2)
        axis.set_xticks(x, datasets, rotation=25, ha="right"); axis.set_ylabel("MSE (mean over horizons)"); axis.legend(frameon=False)
        figure.tight_layout(); figure.savefig(figure_dir / "timepro_five_seed.pdf", bbox_inches="tight"); plt.close(figure)
    rows = csv_rows("synthetic_summary.csv")
    if rows:
        figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
        variants = ("early_fusion", "independent_branches", "process_preserving", "full")
        for setting, marker in (("independent", "o"), ("interacting", "s")):
            chosen = [next(r for r in rows if r["setting"] == setting and r["variant"] == v) for v in variants]
            axes[0].plot(range(4), [float(r["mse_mean"]) for r in chosen], marker=marker, label=setting)
            axes[1].plot(range(4), [float(r["component_correlation_mean"]) for r in chosen], marker=marker, label=setting)
        for axis in axes:
            axis.set_xticks(range(4), ["early\nfusion", "independent", "preserve", "full"]); axis.legend(frameon=False)
        axes[0].set_ylabel("Forecast MSE"); axes[1].set_ylabel("Component |correlation|")
        figure.tight_layout(); figure.savefig(figure_dir / "synthetic_mechanism.pdf", bbox_inches="tight"); plt.close(figure)
    rows = csv_rows("central_ablation_summary.csv")
    if rows:
        variants = (
            "x_to_z",
            "early_fusion",
            "no_multiscale",
            "no_linear_bank",
            "final_no_exchange",
        )
        means = [
            np.mean([float(r["mse_mean"]) for r in rows if r["variant"] == variant])
            for variant in variants
        ]
        figure, axis = plt.subplots(figsize=(5.8, 2.9))
        axis.bar(
            range(5),
            means,
            color=("#bdbdbd", "#9ecae1", "#fdae6b", "#74c476", "#08519c"),
        )
        axis.set_xticks(
            range(5),
            ("X to Z", "early fusion", "single scale", "no bank", "LatentFlow"),
        )
        axis.set_ylabel("MSE (28 tasks, 5 seeds)")
        figure.tight_layout(); figure.savefig(figure_dir / "central_multiseed_ablation.pdf", bbox_inches="tight"); plt.close(figure)
    rows = csv_rows("missing_prefix.csv")
    if rows:
        figure, axis = plt.subplots(figsize=(5.8, 3.0))
        for model, color, marker in (("LatentFlow", "#1764ab", "o"), ("TimePro", "#d95f02", "s"), ("VPNet", "#1b9e77", "^")):
            chosen = [r for r in rows if r["model"] == model]
            baseline = {
                (r["dataset"], r["horizon"]): float(r["mse"])
                for r in chosen if float(r["missing_rate"]) == 0.0
            }
            rates, degradation = [], []
            for rate in (0.0, 0.2, 0.4):
                current = [r for r in chosen if float(r["missing_rate"]) == rate]
                rates.append(100 * rate)
                degradation.append(np.mean([
                    100.0 * (float(r["mse"]) / baseline[(r["dataset"], r["horizon"])] - 1.0)
                    for r in current
                ]))
            axis.plot(rates, degradation, marker=marker, color=color, label=model)
        axis.set_xlabel("masked prefix values (%)"); axis.set_ylabel("MSE degradation (%)")
        axis.legend(frameon=False); figure.tight_layout()
        figure.savefig(figure_dir / "missing_prefix_stress.pdf", bbox_inches="tight"); plt.close(figure)
    rows = csv_rows("vpnet_separator_control.csv")
    if rows and all("vpnet_mse" in row for row in rows):
        values = (
            np.mean([float(r["vpnet_mse"]) for r in rows]),
            np.mean([float(r["mse"]) for r in rows]),
            np.mean([float(r["latentflow_mse"]) for r in rows]),
        )
        figure, axis = plt.subplots(figsize=(4.8, 2.9))
        axis.bar(range(3), values, color=("#bdbdbd", "#74a9cf", "#1764ab"))
        axis.set_xticks(range(3), ("VPNet", "+ separator", "LatentFlow"))
        axis.set_ylabel("Mean MSE (28 tasks)")
        figure.tight_layout(); figure.savefig(figure_dir / "vpnet_separator_control.pdf", bbox_inches="tight"); plt.close(figure)
    rows = csv_rows("channel_scaling.csv")
    if rows:
        channels = np.asarray([float(r["channels"]) for r in rows])
        figure, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
        axes[0].plot(channels, [float(r["latency_ms"]) for r in rows], "o-", color="#1764ab")
        axes[1].plot(channels, [float(r["peak_gpu_memory_mb"]) for r in rows], "o-", color="#1b9e77", label="LatentFlow")
        axes[1].plot(channels, [float(r["dense_covariance_reference_mb"]) for r in rows], "--", color="#d95f02", label="dense $C^2$ reference")
        for axis in axes: axis.set_xscale("log"); axis.set_xlabel("channels")
        axes[0].set_ylabel("latency (ms)"); axes[1].set_ylabel("memory (MiB)"); axes[1].set_yscale("log"); axes[1].legend(frameon=False, fontsize=8)
        figure.tight_layout(); figure.savefig(figure_dir / "channel_scaling.pdf", bbox_inches="tight"); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="physics_modal_v4/additions/results")
    parser.add_argument("--source-root", default="physics_modal_v4/results/final")
    parser.add_argument("--output-root", default="physics_modal_v4/additions/results/summary")
    parser.add_argument("--bootstrap-repetitions", type=int, default=50000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    additions, output = Path(args.input_root), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    aggregate_timepro(args, additions, output)
    aggregate_central(args, additions, output)
    aggregate_synthetic(args, additions, output)
    aggregate_other(args, additions, output)
    plots(output)
    write_json(output / "manifest.json", {
        "timepro_expected_runs": 112,
        "central_expected_runs": 700,
        "synthetic_expected_runs": 40,
        "missing_expected_tasks": 8,
        "vpnet_control_expected_tasks": 28,
        "process_visualizations": 2,
        "channel_scaling_profiles": 9,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    })
    print(f"additions-aggregate complete output={output}")


if __name__ == "__main__":
    main()
