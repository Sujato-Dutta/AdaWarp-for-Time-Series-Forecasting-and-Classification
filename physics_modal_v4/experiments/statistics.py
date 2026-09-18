"""Aggregate the final seven-model benchmark, ablations, and efficiency evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physics_modal_v4.experiments.protocol import (
    DATASETS,
    HORIZONS,
    SEEDS,
    write_csv,
    write_json,
)
from physics_modal_v4.experiments.run import ABLATIONS, FINAL_ABLATION
from physics_modal_v4.models.registry import COMPARISON_MODELS


MODELS = ("LatentFlow",) + COMPARISON_MODELS
METRICS = ("mse", "mae")


def json_rows(root: Path) -> list[dict]:
    if not root.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.json"))
    ]


def canonical_model(row: dict) -> str:
    name = str(row.get("model", ""))
    return name


def bootstrap_ci(values: np.ndarray, seed: int = 17_239) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    draws = values[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and abs(values[order[end]] - values[order[cursor]]) <= 1e-12:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return ranks


def wilcoxon_less(reference: np.ndarray, competitor: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(reference, competitor, alternative="less").pvalue)
    except Exception:
        differences = np.asarray(reference) - np.asarray(competitor)
        differences = differences[np.abs(differences) > 1e-12]
        if not len(differences):
            return 1.0
        rng = np.random.default_rng(7_291)
        signs = rng.choice((-1.0, 1.0), size=(100_000, len(differences)))
        null = (signs * np.abs(differences)[None]).mean(axis=1)
        return float((1 + np.sum(null <= differences.mean())) / (1 + len(null)))


def holm_adjust(rows: list[dict], key: str, output_key: str) -> None:
    if not rows:
        return
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][key]))
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * float(rows[index][key]))
        running = max(running, adjusted)
        rows[index][output_key] = running


def aggregate_headline(root: Path, output: Path, allow_incomplete: bool):
    rows = json_rows(root / "headline" / "runs")
    indexed: dict[tuple[str, str, int, int], dict] = {}
    for row in rows:
        model = canonical_model(row)
        if model in MODELS:
            indexed[(model, row["dataset"], int(row["horizon"]), int(row["seed"]))] = row

    expected = {
        ("LatentFlow", dataset, horizon, seed)
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
    }
    expected |= {
        (model, dataset, horizon, 42)
        for model in COMPARISON_MODELS
        for dataset in DATASETS
        for horizon in HORIZONS
    }
    missing = sorted(expected - set(indexed))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Headline matrix incomplete: {len(missing)}/{len(expected)} missing; "
            f"first={missing[:8]}"
        )

    raw_rows = []
    for key, row in sorted(indexed.items()):
        model, dataset, horizon, seed = key
        raw_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                "seed": seed,
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
            }
        )

    task_rows = []
    task_lookup: dict[tuple[str, str, int, str], float] = {}
    for model in MODELS:
        seeds = SEEDS if model == "LatentFlow" else (42,)
        for dataset in DATASETS:
            for horizon in HORIZONS:
                available = [
                    indexed[(model, dataset, horizon, seed)]
                    for seed in seeds
                    if (model, dataset, horizon, seed) in indexed
                ]
                for metric in METRICS:
                    if not available:
                        continue
                    values = np.asarray([float(row[metric]) for row in available])
                    low, high = bootstrap_ci(values)
                    record = {
                        "model": model,
                        "dataset": dataset,
                        "horizon": horizon,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "ci95_low": low,
                        "ci95_high": high,
                        "seeds": len(values),
                        "paper_value": (
                            f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"
                            if len(values) > 1
                            else f"{values.mean():.4f}"
                        ),
                    }
                    task_rows.append(record)
                    task_lookup[(model, dataset, horizon, metric)] = record["mean"]

    dataset_rows = []
    overall_rows = []
    rank_rows = []
    win_rows = []
    for metric in METRICS:
        for model in MODELS:
            for dataset in DATASETS:
                values = [
                    task_lookup[(model, dataset, horizon, metric)]
                    for horizon in HORIZONS
                    if (model, dataset, horizon, metric) in task_lookup
                ]
                if values:
                    dataset_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "metric": metric,
                            "mean_across_horizons": float(np.mean(values)),
                            "horizons": len(values),
                        }
                    )
            values = [
                task_lookup[(model, dataset, horizon, metric)]
                for dataset in DATASETS
                for horizon in HORIZONS
                if (model, dataset, horizon, metric) in task_lookup
            ]
            if values:
                low, high = bootstrap_ci(np.asarray(values))
                overall_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "mean_across_tasks": float(np.mean(values)),
                        "task_ci95_low": low,
                        "task_ci95_high": high,
                        "tasks": len(values),
                    }
                )

        model_ranks = {model: [] for model in MODELS}
        wins = {model: 0 for model in MODELS}
        complete_tasks = 0
        for dataset in DATASETS:
            for horizon in HORIZONS:
                if not all((model, dataset, horizon, metric) in task_lookup for model in MODELS):
                    continue
                values = np.asarray(
                    [task_lookup[(model, dataset, horizon, metric)] for model in MODELS]
                )
                ranks = average_ranks(values)
                complete_tasks += 1
                best = values.min()
                for index, model in enumerate(MODELS):
                    model_ranks[model].append(float(ranks[index]))
                    if abs(values[index] - best) <= 1e-12:
                        wins[model] += 1
        for model in MODELS:
            if model_ranks[model]:
                rank_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "average_rank": float(np.mean(model_ranks[model])),
                        "rank_std": float(np.std(model_ranks[model], ddof=1)),
                        "tasks": complete_tasks,
                    }
                )
                win_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "wins": wins[model],
                        "tasks": complete_tasks,
                    }
                )

    comparisons = []
    for metric in METRICS:
        metric_rows = []
        for competitor in COMPARISON_MODELS:
            keys = [
                (dataset, horizon)
                for dataset in DATASETS
                for horizon in HORIZONS
                if ("LatentFlow", dataset, horizon, metric) in task_lookup
                and (competitor, dataset, horizon, metric) in task_lookup
            ]
            if not keys:
                continue
            latentflow = np.asarray([task_lookup[("LatentFlow", d, h, metric)] for d, h in keys])
            baseline = np.asarray(
                [task_lookup[(competitor, d, h, metric)] for d, h in keys]
            )
            delta = latentflow - baseline
            relative = 100.0 * (baseline - latentflow) / np.maximum(baseline, 1e-12)
            low, high = bootstrap_ci(relative)
            metric_rows.append(
                {
                    "reference": "LatentFlow",
                    "competitor": competitor,
                    "metric": metric,
                    "tasks": len(keys),
                    "wins": int(np.sum(delta < -1e-6)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-6)),
                    "losses": int(np.sum(delta > 1e-6)),
                    "mean_relative_gain_percent": float(relative.mean()),
                    "bootstrap_task_ci95_low": low,
                    "bootstrap_task_ci95_high": high,
                    "wilcoxon_one_sided_p": wilcoxon_less(latentflow, baseline),
                }
            )
        holm_adjust(metric_rows, "wilcoxon_one_sided_p", "wilcoxon_holm_p")
        comparisons.extend(metric_rows)

    omnibus = []
    try:
        from scipy.stats import friedmanchisquare

        for metric in METRICS:
            arrays = []
            for model in MODELS:
                values = [
                    task_lookup[(model, dataset, horizon, metric)]
                    for dataset in DATASETS
                    for horizon in HORIZONS
                    if all(
                        (candidate, dataset, horizon, metric) in task_lookup
                        for candidate in MODELS
                    )
                ]
                arrays.append(np.asarray(values))
            if arrays and len(arrays[0]):
                statistic, p_value = friedmanchisquare(*arrays)
                omnibus.append(
                    {
                        "metric": metric,
                        "test": "Friedman",
                        "tasks": len(arrays[0]),
                        "models": len(arrays),
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                )
    except Exception:
        pass

    write_csv(output / "headline_seed_runs.csv", raw_rows)
    write_csv(output / "headline_task_mean_std.csv", task_rows)
    write_csv(output / "headline_dataset_summary.csv", dataset_rows)
    write_csv(output / "headline_overall.csv", overall_rows)
    write_csv(output / "headline_average_rank.csv", rank_rows)
    write_csv(output / "headline_wins.csv", win_rows)
    write_csv(output / "headline_latentflow_pairwise.csv", comparisons)
    write_csv(output / "headline_omnibus.csv", omnibus)
    write_json(
        output / "headline_completeness.json",
        {
            "expected_runs": len(expected),
            "observed_runs": len(expected & set(indexed)),
            "missing": [list(item) for item in missing],
            "design": {
                "LatentFlow": list(SEEDS),
                "competitors": [42],
                "note": "LatentFlow reports five-seed mean/std; competitors are matched seed-42 runs.",
            },
        },
    )
    return task_rows, dataset_rows, overall_rows, rank_rows, win_rows


def aggregate_ablations(root: Path, output: Path, allow_incomplete: bool) -> list[dict]:
    rows = json_rows(root / "ablations" / "runs")
    indexed = {
        (row["variant"], row["dataset"], int(row["horizon"])): row for row in rows
    }
    expected = {
        (variant, dataset, horizon)
        for variant in ABLATIONS
        for dataset in DATASETS
        for horizon in HORIZONS
    }
    missing = sorted(expected - set(indexed))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Ablation matrix incomplete: {len(missing)}/{len(expected)} missing."
        )
    summary = []
    for variant in ABLATIONS:
        for metric in METRICS:
            pairs = []
            for dataset in DATASETS:
                for horizon in HORIZONS:
                    current = indexed.get((variant, dataset, horizon))
                    reference_row = indexed.get((FINAL_ABLATION, dataset, horizon))
                    if current is not None and reference_row is not None:
                        pairs.append((float(current[metric]), float(reference_row[metric])))
            if not pairs:
                continue
            current = np.asarray([pair[0] for pair in pairs])
            reference = np.asarray([pair[1] for pair in pairs])
            delta = current - reference
            summary.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "tasks": len(pairs),
                    "mean": float(current.mean()),
                    "latentflow_mean": float(reference.mean()),
                    "degradation_vs_latentflow_percent": float(
                        100.0 * (current.mean() / reference.mean() - 1.0)
                    ),
                    "better_than_latentflow": int(np.sum(delta < -1e-6)),
                    "ties_with_latentflow": int(np.sum(np.abs(delta) <= 1e-6)),
                    "worse_than_latentflow": int(np.sum(delta > 1e-6)),
                }
            )
    write_csv(output / "ablation_tasks.csv", rows)
    write_csv(output / "ablation_summary.csv", summary)
    write_json(
        output / "ablation_completeness.json",
        {
            "expected_runs": len(expected),
            "observed_runs": len(expected & set(indexed)),
            "missing": [list(item) for item in missing],
            "seed": 42,
            "reference": FINAL_ABLATION,
        },
    )
    return summary


def aggregate_efficiency(root: Path, output: Path, allow_incomplete: bool):
    rows = json_rows(root / "efficiency" / "runs")
    indexed = {
        (canonical_model(row), row["dataset"], int(row["horizon"])): row
        for row in rows
        if canonical_model(row) in MODELS
    }
    expected = {
        (model, dataset, horizon)
        for model in MODELS
        for dataset in DATASETS
        for horizon in HORIZONS
    }
    missing = sorted(expected - set(indexed))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Efficiency matrix incomplete: {len(missing)}/{len(expected)} missing."
        )
    fields = (
        "parameters_total",
        "parameters_trainable",
        "gflops_per_sample",
        "inference_ms_batch",
        "training_ms_iter",
        "peak_inference_gpu_mb",
        "peak_training_gpu_mb",
    )
    task_rows = []
    for key, row in sorted(indexed.items()):
        model, dataset, horizon = key
        task_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                **{field: float(row[field]) for field in fields},
                "gpu": row.get("gpu", ""),
                "profile_batch_size": int(row.get("profile_batch_size", 1)),
            }
        )
    dataset_rows = []
    overall_rows = []
    for model in MODELS:
        for dataset in DATASETS:
            subset = [
                row
                for row in task_rows
                if row["model"] == model and row["dataset"] == dataset
            ]
            if not subset:
                continue
            record = {"model": model, "dataset": dataset, "tasks": len(subset)}
            for field in fields:
                record[f"mean_{field}"] = float(
                    np.nanmean([float(row[field]) for row in subset])
                )
            dataset_rows.append(record)
        subset = [row for row in task_rows if row["model"] == model]
        if subset:
            record = {"model": model, "tasks": len(subset)}
            for field in fields:
                record[f"mean_{field}"] = float(
                    np.nanmean([float(row[field]) for row in subset])
                )
            overall_rows.append(record)
    write_csv(output / "efficiency_tasks.csv", task_rows)
    write_csv(output / "efficiency_by_dataset.csv", dataset_rows)
    write_csv(output / "efficiency_overall.csv", overall_rows)
    write_json(
        output / "efficiency_completeness.json",
        {
            "expected_runs": len(expected),
            "observed_runs": len(expected & set(indexed)),
            "missing": [list(item) for item in missing],
            "seed": 42,
            "profile_batch_size": 1,
        },
    )
    return dataset_rows, overall_rows


def save_figure(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")


def generate_plots(
    output: Path,
    dataset_rows: list[dict],
    overall_rows: list[dict],
    rank_rows: list[dict],
    win_rows: list[dict],
    ablation_summary: list[dict],
    efficiency_overall: list[dict],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        write_json(output / "figures" / "plot_error.json", {"error": str(error)})
        return

    colors = {
        "LatentFlow": "#9E1B32",
        "VPNet": "#1F77B4",
        "xCPD": "#2CA02C",
        "TimePro": "#FF7F0E",
        "iTransformer": "#9467BD",
        "TimeMixer": "#8C564B",
        "DLinear": "#7F7F7F",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    figures = output / "figures"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for axis, metric in zip(axes, METRICS):
        subset = {row["model"]: row for row in rank_rows if row["metric"] == metric}
        ordered = sorted(subset, key=lambda model: subset[model]["average_rank"])
        axis.barh(
            ordered[::-1],
            [subset[model]["average_rank"] for model in ordered[::-1]],
            color=[colors[model] for model in ordered[::-1]],
        )
        axis.set_xlabel(f"Average {metric.upper()} rank (lower is better)")
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "average_rank")
    plt.close(fig)

    mse_rows = [row for row in dataset_rows if row["metric"] == "mse"]
    complete_datasets = [
        dataset
        for dataset in DATASETS
        if any(row["dataset"] == dataset for row in mse_rows)
    ]
    if complete_datasets:
        fig, axis = plt.subplots(figsize=(11.5, 4.2))
        x = np.arange(len(complete_datasets))
        width = 0.11
        best = np.asarray(
            [
                min(
                    row["mean_across_horizons"]
                    for row in mse_rows
                    if row["dataset"] == dataset
                )
                for dataset in complete_datasets
            ]
        )
        for index, model in enumerate(MODELS):
            lookup = {
                row["dataset"]: row["mean_across_horizons"]
                for row in mse_rows
                if row["model"] == model
            }
            values = np.asarray(
                [lookup.get(dataset, np.nan) for dataset in complete_datasets]
            )
            axis.bar(
                x + (index - 3) * width,
                values / best,
                width,
                label=model,
                color=colors[model],
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_xticks(x, complete_datasets, rotation=20)
        axis.set_ylabel("MSE / best dataset MSE")
        axis.legend(ncol=4, frameon=False, loc="upper left")
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        save_figure(fig, figures, "dataset_relative_mse")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for axis, metric in zip(axes, METRICS):
        subset = {row["model"]: row["wins"] for row in win_rows if row["metric"] == metric}
        axis.bar(MODELS, [subset.get(model, 0) for model in MODELS], color=[colors[m] for m in MODELS])
        axis.set_title(f"{metric.upper()} task wins")
        axis.set_ylabel("Wins across 28 tasks")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "task_wins")
    plt.close(fig)

    latentflow_tasks = [
        row
        for row in json_rows(output.parent / "headline" / "runs" / "LatentFlow")
        if int(row.get("seed", -1)) in SEEDS
    ]
    if latentflow_tasks:
        means, stds = [], []
        for dataset in DATASETS:
            seed_means = []
            for seed in SEEDS:
                values = [
                    float(row["mse"])
                    for row in latentflow_tasks
                    if row["dataset"] == dataset and int(row["seed"]) == seed
                ]
                if values:
                    seed_means.append(float(np.mean(values)))
            means.append(float(np.mean(seed_means)) if seed_means else np.nan)
            stds.append(float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0)
        fig, axis = plt.subplots(figsize=(8.6, 3.8))
        axis.errorbar(
            np.arange(len(DATASETS)),
            means,
            yerr=stds,
            fmt="o",
            capsize=4,
            color=colors["LatentFlow"],
        )
        axis.set_xticks(np.arange(len(DATASETS)), DATASETS, rotation=20)
        axis.set_ylabel("LatentFlow mean MSE across horizons")
        axis.set_title("Five-seed stability")
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        save_figure(fig, figures, "latentflow_seed_stability")
        plt.close(fig)

    mse_ablation = [
        row
        for row in ablation_summary
        if row["metric"] == "mse" and row["variant"] != FINAL_ABLATION
    ]
    if mse_ablation:
        ordered = sorted(
            mse_ablation,
            key=lambda row: row["degradation_vs_latentflow_percent"],
        )
        labels = [row["variant"].replace("_", " ") for row in ordered]
        values = [row["degradation_vs_latentflow_percent"] for row in ordered]
        colors_ablation = [
            "#2E8B57" if value < 0.0 else "#B23A48" for value in values
        ]
        fig, axis = plt.subplots(figsize=(8.8, 6.2))
        axis.barh(labels, values, color=colors_ablation)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Mean MSE change relative to LatentFlow (%)")
        axis.set_title("Seed-42 component and adaptation ablations")
        axis.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        save_figure(fig, figures, "ablation_mse_degradation")
        plt.close(fig)
    if efficiency_overall:
        perf = {
            row["model"]: row["mean_across_tasks"]
            for row in overall_rows
            if row["metric"] == "mse"
        }
        efficiency = {row["model"]: row for row in efficiency_overall}
        common = [model for model in MODELS if model in perf and model in efficiency]
        if common:
            fig, axis = plt.subplots(figsize=(6.4, 4.4))
            for model in common:
                x_value = efficiency[model]["mean_gflops_per_sample"]
                y_value = perf[model]
                axis.scatter(x_value, y_value, s=70, color=colors[model], label=model)
                axis.annotate(model, (x_value, y_value), xytext=(5, 3), textcoords="offset points")
            if all(float(efficiency[m]["mean_gflops_per_sample"]) > 0 for m in common):
                axis.set_xscale("log")
            axis.set_xlabel("Mean GFLOPs per sample")
            axis.set_ylabel("Mean MSE across 28 tasks")
            axis.grid(alpha=0.2)
            fig.tight_layout()
            save_figure(fig, figures, "accuracy_efficiency_pareto")
            plt.close(fig)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="physics_modal_v4/results/final")
    parser.add_argument("--output-root", default="physics_modal_v4/results/final/statistics")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = parser().parse_args()
    root = Path(args.input_root)
    output = Path(args.output_root)
    task_rows, dataset_rows, overall_rows, rank_rows, win_rows = aggregate_headline(
        root, output, args.allow_incomplete
    )
    ablation_summary = aggregate_ablations(root, output, args.allow_incomplete)
    _, efficiency_overall = aggregate_efficiency(root, output, args.allow_incomplete)
    generate_plots(
        output,
        dataset_rows,
        overall_rows,
        rank_rows,
        win_rows,
        ablation_summary,
        efficiency_overall,
    )
    print(
        f"latentflow-statistics complete output={output} headline_rows={len(task_rows)} "
        f"efficiency_models={len(efficiency_overall)}"
    )


if __name__ == "__main__":
    main()









