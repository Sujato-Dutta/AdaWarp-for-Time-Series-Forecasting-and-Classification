"""Combine seed 42 with rescue results and report ablation deltas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Electricity", "Traffic")
SEED42 = {
    ("MVPF", "Electricity"): {"mse": 0.23087487303069237, "mae": 0.31484298284118595},
    ("PatchTST", "Electricity"): {"mse": 0.25582972168922424, "mae": 0.3363006114959717},
    ("MVPF", "Traffic"): {"mse": 0.5056212378325112, "mae": 0.33024772081989495},
    ("PatchTST", "Traffic"): {"mse": 0.5769842863082886, "mae": 0.3694722056388855},
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for item in items for key in item})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def result(root: Path, experiment: str, dataset: str, seed: int) -> dict | None:
    path = root / "results" / f"{experiment}_{dataset}_h720_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def baselines() -> dict[tuple[str, str], dict]:
    output = {key: dict(value) for key, value in SEED42.items()}
    mvpf = ROOT / "mvpf-tests/results/pruned_instrumented_820583/metrics/standard_test_metrics.csv"
    tasks = ROOT / "mvpf-tests/results/metrics/task_metrics_24.csv"
    if mvpf.exists():
        for row in rows(mvpf):
            if row["dataset"] in DATASETS and int(row["horizon"]) == 720:
                output[("MVPF", row["dataset"])] = row
    if tasks.exists():
        for row in rows(tasks):
            if row["method"] == "PatchTST" and row["dataset"] in DATASETS and int(row["horizon"]) == 720:
                output[("PatchTST", row["dataset"])] = row
    return output


def run(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    base = baselines()
    replication = []
    for dataset in DATASETS:
        for model, experiment in (("MVPF", "mvpf"), ("PatchTST", "patchtst")):
            values = [{
                "seed": 42,
                "mse": float(base[(model, dataset)]["mse"]),
                "mae": float(base[(model, dataset)]["mae"]),
            }]
            for seed in (43, 44):
                item = result(root, experiment, dataset, seed)
                if item:
                    values.append(item)
            row = {"dataset": dataset, "model": model, "n": len(values)}
            for metric in ("mse", "mae"):
                sample = [float(item[metric]) for item in values]
                row[f"mean_{metric}"] = statistics.mean(sample)
                row[f"std_{metric}"] = statistics.stdev(sample) if len(sample) > 1 else ""
            replication.append(row)

    ablations = []
    for dataset in DATASETS:
        for experiment in ("global_outer", "no_variable_mixing", "single_scale_8"):
            item = result(root, experiment, dataset, 42)
            if not item:
                continue
            row = {"dataset": dataset, "experiment": experiment}
            for metric in ("mse", "mae"):
                reference = float(base[("MVPF", dataset)][metric])
                value = float(item[metric])
                row[f"baseline_{metric}"] = reference
                row[metric] = value
                row[f"delta_{metric}_percent"] = 100.0 * (value - reference) / reference
            ablations.append(row)

    permutations = []
    for experiment in ("permute_1", "permute_2"):
        item = result(root, experiment, "Traffic", 42)
        if not item:
            continue
        row = {"experiment": experiment, "permutation_seed": item["permutation_seed"]}
        for metric in ("mse", "mae"):
            reference = float(base[("MVPF", "Traffic")][metric])
            value = float(item[metric])
            row[f"baseline_{metric}"] = reference
            row[metric] = value
            row[f"delta_{metric}_percent"] = 100.0 * (value - reference) / reference
        permutations.append(row)

    summary = root / "summary"
    write(summary / "three_seed_reproducibility.csv", replication)
    write(summary / "retrained_ablations.csv", ablations)
    write(summary / "channel_permutations.csv", permutations)
    for item in replication:
        print(
            f"{item['model']:<8} {item['dataset']:<11} n={item['n']} "
            f"MSE={item['mean_mse']} +/- {item['std_mse']} "
            f"MAE={item['mean_mae']} +/- {item['std_mae']}"
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="mvpf-tests/results/rescue_720")
    return p


if __name__ == "__main__":
    run(parser().parse_args())