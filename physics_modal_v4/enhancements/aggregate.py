"""Strict aggregation and paired inference for LatentFlow revision experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from physics_modal_v4.enhancements.run import (
    CAPACITY_PROTOCOL,
    CENTRAL_PROTOCOL,
    CENTRAL_VARIANTS,
    CHANNEL_EXPANSION_PROTOCOL,
    CHANNEL_PROTOCOL,
    COMPARATOR_PROTOCOL,
    DIAGNOSTIC_PROTOCOL,
    DATA_BUDGET_PROTOCOL,
    FACTORIAL_PROTOCOL,
    FULL_DATA_PROTOCOL,
    STRONGEST_NON_TIMEPRO,
    factorial_variant,
)
from physics_modal_v4.experiments.protocol import DATASETS, HORIZONS, SEEDS, write_csv
from physics_modal_v4.experiments.run import FINAL_CANDIDATE
from physics_modal_v4.enhancements.mechanisms import SCALING_PROTOCOL, SYNTHETIC_PROTOCOL


DEFAULT_INPUT = Path("physics_modal_v4/enhancements/results")
DEFAULT_SOURCE = Path("physics_modal_v4/results/final")
TOLERANCE = 1e-12


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_rows(root: Path, experiment: str, protocol: str) -> list[dict]:
    rows = []
    for path in sorted((root / experiment / "runs").glob("**/*.json")):
        row = read_json(path)
        if row.get("enhancement_protocol") != protocol:
            raise RuntimeError(f"Protocol mismatch in {path}.")
        if row.get("cross_process_exchange") is not False:
            raise RuntimeError(f"Exchange-enabled artifact is forbidden: {path}.")
        rows.append({**row, "result_path": str(path), "result_sha256": sha256(path)})
    return rows


def require_count(label: str, rows: list[dict], expected: int, allow: bool) -> None:
    if len(rows) != expected:
        message = f"{label}: expected {expected} rows, found {len(rows)}"
        if not allow:
            raise RuntimeError(message)
        print(f"WARNING {message}", flush=True)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def grouped_summary(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        mse_mean, mse_std = mean_std(float(row["mse"]) for row in group)
        mae_mean, mae_std = mean_std(float(row["mae"]) for row in group)
        output.append(
            {
                **dict(zip(keys, key)),
                "runs": len(group),
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
            }
        )
    return output


def clustered_interval(
    values: list[float],
    clusters: list[str],
    *,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(float(value))
    names = sorted(grouped)
    if len(names) == 1:
        point = float(np.mean(values))
        return point, point
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        selected = rng.choice(names, size=len(names), replace=True)
        sample = [value for name in selected for value in grouped[str(name)]]
        estimates[index] = np.mean(sample)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_effects(
    rows: list[dict],
    *,
    reference: str,
    variants: Iterable[str],
    variant_key: str,
    cluster: Callable[[dict], str],
    draws: int,
) -> list[dict]:
    indexed = {
        (row[variant_key], row["dataset"], int(row["horizon"]), int(row["seed"])): row
        for row in rows
    }
    output = []
    for variant_index, variant in enumerate(variants):
        if variant == reference:
            continue
        for metric in ("mse", "mae"):
            differences = []
            relative = []
            clusters = []
            better = ties = worse = 0
            for dataset in DATASETS:
                for horizon in HORIZONS:
                    for seed in SEEDS:
                        ref = indexed.get((reference, dataset, horizon, seed))
                        alt = indexed.get((variant, dataset, horizon, seed))
                        if ref is None or alt is None:
                            continue
                        difference = float(alt[metric]) - float(ref[metric])
                        differences.append(difference)
                        relative.append(100.0 * difference / max(float(ref[metric]), 1e-12))
                        clusters.append(cluster(ref))
                        if difference < -TOLERANCE:
                            better += 1
                        elif difference > TOLERANCE:
                            worse += 1
                        else:
                            ties += 1
            low, high = clustered_interval(
                relative,
                clusters,
                seed=1729 + variant_index * 17 + (0 if metric == "mse" else 1),
                draws=draws,
            )
            output.append(
                {
                    "reference": reference,
                    "variant": variant,
                    "metric": metric,
                    "paired_cells": len(differences),
                    "mean_absolute_degradation": float(np.mean(differences)) if differences else float("nan"),
                    "mean_relative_degradation_percent": float(np.mean(relative)) if relative else float("nan"),
                    "clustered_ci95_low_percent": low,
                    "clustered_ci95_high_percent": high,
                    "variant_better": better,
                    "ties": ties,
                    "variant_worse": worse,
                }
            )
    return output


def aggregate_central(args, rows: list[dict], output: Path) -> None:
    require_count("central", rows, 700, args.allow_incomplete)
    write_csv(output / "central_runs.csv", rows)
    write_csv(
        output / "central_task_summary.csv",
        grouped_summary(rows, ("variant", "dataset", "horizon")),
    )
    write_csv(
        output / "central_dataset_summary.csv",
        grouped_summary(rows, ("variant", "dataset")),
    )
    effects = paired_effects(
        rows,
        reference="final_no_exchange",
        variants=CENTRAL_VARIANTS,
        variant_key="variant",
        cluster=lambda row: f"{row['dataset']}::seed{row['seed']}",
        draws=args.bootstrap_draws,
    )
    write_csv(output / "central_paired_effects.csv", effects)


def factorial_cells() -> list[str]:
    return [
        factorial_variant(scale, bank, continuation)
        for scale in ("single", "multiscale")
        for bank in ("off", "on")
        for continuation in ("early", "preserved")
    ]


def factorial_contrasts(rows: list[dict], draws: int) -> list[dict]:
    encodings = {
        "scale": {"single": -1, "multiscale": 1},
        "bank": {"off": -1, "on": 1},
        "continuation": {"early": -1, "preserved": 1},
    }
    effects = {
        "scale": ("scale",),
        "bank": ("bank",),
        "continuation": ("continuation",),
        "scale_x_bank": ("scale", "bank"),
        "scale_x_continuation": ("scale", "continuation"),
        "bank_x_continuation": ("bank", "continuation"),
        "scale_x_bank_x_continuation": ("scale", "bank", "continuation"),
    }
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["horizon"]), int(row["seed"]))].append(row)
    output = []
    for metric in ("mse", "mae"):
        for effect_index, (effect, factors) in enumerate(effects.items()):
            records = []
            for (dataset, horizon, seed), group in grouped.items():
                if len(group) != 8:
                    continue
                coefficient_sum = 0.0
                for row in group:
                    sign = int(np.prod([encodings[factor][row[factor]] for factor in factors]))
                    coefficient_sum += sign * float(row[metric])
                contrast = coefficient_sum / 4.0
                records.append((dataset, horizon, seed, contrast))
            for dataset_scope in ("ALL", *DATASETS):
                selected = [record for record in records if dataset_scope == "ALL" or record[0] == dataset_scope]
                values = [record[3] for record in selected]
                clusters = [record[0] for record in selected]
                low, high = clustered_interval(
                    values,
                    clusters,
                    seed=3253 + effect_index * 31 + (0 if metric == "mse" else 1),
                    draws=draws,
                )
                output.append(
                    {
                        "dataset": dataset_scope,
                        "effect": effect,
                        "metric": metric,
                        "paired_dataset_horizon_seed_cells": len(values),
                        "contrast_plus_minus": float(np.mean(values)) if values else float("nan"),
                        "clustered_ci95_low": low,
                        "clustered_ci95_high": high,
                        "interpretation": "negative contrast favors the +1 level(s)",
                    }
                )
    return output


def aggregate_factorial(args, rows: list[dict], output: Path) -> None:
    require_count("factorial", rows, 672, args.allow_incomplete)
    write_csv(output / "factorial_runs.csv", rows)
    write_csv(
        output / "factorial_cell_summary.csv",
        grouped_summary(rows, ("variant", "scale", "bank", "continuation", "dataset")),
    )
    write_csv(
        output / "factorial_effects.csv",
        factorial_contrasts(rows, args.bootstrap_draws),
    )


def source_headline(source: Path, model: str, seed: int, dataset: str, horizon: int) -> dict:
    path = source / "headline" / "runs" / model / f"seed{seed}" / dataset / f"h{horizon}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    row = read_json(path)
    if model == "LatentFlow" and (
        row.get("candidate") != FINAL_CANDIDATE
        or row.get("cross_process_exchange") is not False
    ):
        raise RuntimeError(f"Stale LatentFlow source result: {path}")
    return {**row, "result_path": str(path), "result_sha256": sha256(path)}


def aggregate_comparator(args, rows: list[dict], source: Path, output: Path) -> None:
    require_count("strongest comparator additions", rows, 112, args.allow_incomplete)
    combined = []
    try:
        for dataset in DATASETS:
            model = STRONGEST_NON_TIMEPRO[dataset]
            for horizon in HORIZONS:
                for seed in SEEDS:
                    if seed == 42:
                        combined.append(source_headline(source, model, seed, dataset, horizon))
                    else:
                        match = [
                            row for row in rows
                            if row["model"] == model
                            and row["dataset"] == dataset
                            and int(row["horizon"]) == horizon
                            and int(row["seed"]) == seed
                        ]
                        if len(match) != 1:
                            raise RuntimeError(
                                f"Expected one {model} row for {dataset}, h={horizon}, seed={seed}."
                            )
                        combined.append(match[0])
                for seed in SEEDS:
                    combined.append(source_headline(source, "LatentFlow", seed, dataset, horizon))
    except (FileNotFoundError, RuntimeError) as error:
        if not args.allow_incomplete:
            raise
        print(f"WARNING comparator aggregation incomplete: {error}", flush=True)
    write_csv(output / "strongest_comparator_runs.csv", combined)
    write_csv(
        output / "strongest_comparator_summary.csv",
        grouped_summary(combined, ("model", "dataset", "horizon")),
    )
    indexed = {
        (row["model"], row["dataset"], int(row["horizon"]), int(row["seed"])): row
        for row in combined
    }
    effects = []
    for dataset_scope in ("ALL", *DATASETS):
        for metric in ("mse", "mae"):
            gains = []
            clusters = []
            wins = ties = losses = 0
            for dataset in DATASETS:
                if dataset_scope != "ALL" and dataset != dataset_scope:
                    continue
                baseline = STRONGEST_NON_TIMEPRO[dataset]
                for horizon in HORIZONS:
                    for seed in SEEDS:
                        ours = indexed.get(("LatentFlow", dataset, horizon, seed))
                        other = indexed.get((baseline, dataset, horizon, seed))
                        if ours is None or other is None:
                            continue
                        delta = float(other[metric]) - float(ours[metric])
                        gains.append(100.0 * delta / max(float(other[metric]), 1e-12))
                        clusters.append(f"{dataset}::seed{seed}")
                        if delta > TOLERANCE:
                            wins += 1
                        elif delta < -TOLERANCE:
                            losses += 1
                        else:
                            ties += 1
            low, high = clustered_interval(
                gains,
                clusters,
                seed=4409 + (0 if metric == "mse" else 1),
                draws=args.bootstrap_draws,
            )
            effects.append(
                {
                    "dataset": dataset_scope,
                    "metric": metric,
                    "paired_cells": len(gains),
                    "latentflow_mean_gain_percent": float(np.mean(gains)) if gains else float("nan"),
                    "clustered_ci95_low_percent": low,
                    "clustered_ci95_high_percent": high,
                    "latentflow_wins": wins,
                    "ties": ties,
                    "latentflow_losses": losses,
                }
            )
    write_csv(output / "strongest_comparator_paired_effects.csv", effects)


def aggregate_other(args, root: Path, output: Path) -> list[dict]:
    all_rows = []
    capacity = json_rows(root, "capacity", CAPACITY_PROTOCOL)
    require_count("capacity controls", capacity, 112, args.allow_incomplete)
    write_csv(output / "capacity_runs.csv", capacity)
    write_csv(
        output / "capacity_summary.csv",
        grouped_summary(capacity, ("variant", "dataset")),
    )
    all_rows.extend(capacity)

    full = json_rows(root, "full_data", FULL_DATA_PROTOCOL)
    require_count("full data", full, 56, args.allow_incomplete)
    write_csv(output / "full_data_runs.csv", full)
    write_csv(output / "full_data_summary.csv", grouped_summary(full, ("model", "dataset")))
    all_rows.extend(full)

    data_budget = json_rows(root, 'data_budget', DATA_BUDGET_PROTOCOL)
    require_count('data-budget curve', data_budget, 112, args.allow_incomplete)
    data_budget = sorted(
        data_budget,
        key=lambda row: (
            int(row['num_training_windows']),
            DATASETS.index(row['dataset']),
            int(row['horizon']),
        ),
    )
    write_csv(output / 'data_budget_runs.csv', data_budget)
    write_csv(
        output / 'data_budget_dataset_summary.csv',
        grouped_summary(data_budget, ('train_budget_label', 'dataset')),
    )
    write_csv(
        output / 'data_budget_overall_summary.csv',
        grouped_summary(data_budget, ('train_budget_label',)),
    )
    anchor = {
        (row['dataset'], int(row['horizon'])): row
        for row in data_budget
        if row['train_budget_label'] == '2048'
    }
    deltas = []
    for row in data_budget:
        reference = anchor.get((row['dataset'], int(row['horizon'])))
        if reference is None:
            continue
        deltas.append(
            {
                'train_budget_label': row['train_budget_label'],
                'num_training_windows': row['num_training_windows'],
                'dataset': row['dataset'],
                'horizon': row['horizon'],
                'mse': row['mse'],
                'mae': row['mae'],
                'mse_change_vs_2048_percent': 100.0
                * (float(row['mse']) - float(reference['mse']))
                / max(float(reference['mse']), 1e-12),
                'mae_change_vs_2048_percent': 100.0
                * (float(row['mae']) - float(reference['mae']))
                / max(float(reference['mae']), 1e-12),
            }
        )
    write_csv(output / 'data_budget_task_deltas.csv', deltas)
    all_rows.extend(data_budget)

    diagnostics = json_rows(root, "diagnostics", DIAGNOSTIC_PROTOCOL)
    require_count("diagnostics", diagnostics, 7, args.allow_incomplete)
    diagnostic_summary = []
    diagnostic_components = []
    for row in diagnostics:
        diagnostic_summary.append(
            {
                key: row.get(key)
                for key in (
                    "dataset",
                    "horizon",
                    "seed",
                    "bank_weight_quality_correlation",
                    "field_weight_quality_correlation",
                    "mean_bank_weights",
                    "mean_field_weights",
                    "mean_process_weights",
                    "mean_outer_weights",
                )
            }
        )
        diagnostic_components.extend(
            {
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                **entry,
            }
            for entry in row["component_metrics"]
        )
    write_csv(output / "diagnostic_summary.csv", diagnostic_summary)
    write_csv(output / "diagnostic_component_metrics.csv", diagnostic_components)
    all_rows.extend(diagnostics)

    channel = json_rows(root, "channel_order", CHANNEL_PROTOCOL)
    require_count("channel order", channel, 4, args.allow_incomplete)
    channel_expanded = json_rows(
        root, "channel_order_expanded", CHANNEL_EXPANSION_PROTOCOL
    )
    require_count(
        "channel order expansion", channel_expanded, 6, args.allow_incomplete
    )
    channel.extend(channel_expanded)
    write_csv(output / "channel_order_runs.csv", channel)
    all_rows.extend(channel)

    synthetic = json_rows(root, "synthetic", SYNTHETIC_PROTOCOL)
    require_count("synthetic mechanism", synthetic, 105, args.allow_incomplete)
    write_csv(output / "synthetic_runs.csv", synthetic)
    synthetic_groups: dict[tuple, list[dict]] = defaultdict(list)
    process_rows = []
    assignment_rows = []
    for row in synthetic:
        synthetic_groups[(row["scenario"], row["variant"])].append(row)
        process_rows.extend(
            {
                "scenario": row["scenario"],
                "variant": row["variant"],
                "seed": row["seed"],
                **process,
            }
            for process in row["learned_processes"]
        )
        assignment_rows.extend(
            {
                "scenario": row["scenario"],
                "variant": row["variant"],
                "seed": row["seed"],
                **assignment,
            }
            for assignment in row["assignment"]
        )
    synthetic_summary = []
    for (scenario, variant), group in sorted(synthetic_groups.items()):
        entry = {"scenario": scenario, "variant": variant, "runs": len(group)}
        for metric in ("mse", "mae", "component_correlation_mean", "component_nrmse_mean"):
            mean, std = mean_std(float(row[metric]) for row in group)
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        synthetic_summary.append(entry)
    write_csv(output / "synthetic_summary.csv", synthetic_summary)
    write_csv(output / "synthetic_process_parameters.csv", process_rows)
    write_csv(output / "synthetic_component_assignments.csv", assignment_rows)
    all_rows.extend(synthetic)

    scaling = json_rows(root, "real_channel_scaling", SCALING_PROTOCOL)
    require_count("real channel scaling", scaling, 12, args.allow_incomplete)
    write_csv(output / "real_channel_scaling.csv", sorted(scaling, key=lambda row: (int(row["channels"]), row["model"])))
    all_rows.extend(scaling)
    return all_rows


def write_manifest(output: Path, rows: list[dict]) -> None:
    manifest = []
    for row in rows:
        manifest.append(
            {
                "experiment": row.get("experiment"),
                "variant": row.get("variant"),
                "model": row.get("model"),
                "dataset": row.get("dataset"),
                "horizon": row.get("horizon"),
                "seed": row.get("seed"),
                "config_hash": row.get("config_hash"),
                "checkpoint": row.get("checkpoint"),
                "result_path": row.get("result_path"),
                "result_sha256": row.get("result_sha256"),
                "cross_process_exchange": row.get("cross_process_exchange"),
            }
        )
    write_csv(output / "run_manifest.csv", manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", default=str(DEFAULT_INPUT))
    result.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    result.add_argument("--output-root", default=str(DEFAULT_INPUT / "statistics"))
    result.add_argument("--bootstrap-draws", type=int, default=20_000)
    result.add_argument("--allow-incomplete", action="store_true")
    result.add_argument(
        "--core-only",
        action="store_true",
        help="Aggregate only selector-safe central and factorial results.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    root = Path(args.input_root)
    source = Path(args.source_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    central = json_rows(root, "central", CENTRAL_PROTOCOL)
    factorial = json_rows(root, "factorial", FACTORIAL_PROTOCOL)
    comparator = json_rows(root, "strongest_comparator", COMPARATOR_PROTOCOL)
    aggregate_central(args, central, output)
    aggregate_factorial(args, factorial, output)
    if args.core_only:
        write_manifest(output, central + factorial)
        summary = {
            "central_rows": len(central),
            "factorial_rows": len(factorial),
            "selector_safe": True,
            "output_root": str(output),
        }
        (output / "aggregation_status.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"latentflow-selector-safe-core-aggregate complete {summary}", flush=True)
        return
    aggregate_comparator(args, comparator, source, output)
    other = aggregate_other(args, root, output)
    write_manifest(output, central + factorial + comparator + other)
    summary = {
        "central_rows": len(central),
        "factorial_rows": len(factorial),
        "strongest_comparator_new_rows": len(comparator),
        "other_rows": len(other),
        "output_root": str(output),
    }
    (output / "aggregation_status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"latentflow-enhancements-aggregate complete {summary}", flush=True)


if __name__ == "__main__":
    main()
