"""Compare the pruned candidate against the reported MVPF task metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(args: argparse.Namespace) -> None:
    baseline_rows = read_csv(Path(args.baseline))
    candidate_rows = read_csv(Path(args.candidate))
    baseline = {
        (row["dataset"], int(row["horizon"])): row
        for row in baseline_rows
        if row.get("method", row.get("model")) in {"AdaWarp-MVPF+", "AdaWarp-MVPF", "MVPF"}
    }
    candidate = {(row["dataset"], int(row["horizon"])): row for row in candidate_rows}
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise RuntimeError("No shared dataset/horizon tasks were found.")
    print(f"shared_tasks={len(shared)}")
    for metric in ("mse", "mae"):
        old = np.asarray([float(baseline[key][metric]) for key in shared])
        new = np.asarray([float(candidate[key][f"learned_{metric}"]) for key in shared])
        relative = 100.0 * (new - old) / old
        delta = new - old
        wins = int((delta < -1e-12).sum())
        ties = int((np.abs(delta) <= 1e-12).sum())
        losses = int((delta > 1e-12).sum())
        print(
            f"{metric.upper()}: candidate_wins/ties/losses={wins}/{ties}/{losses} "
            f"mean_relative_delta={relative.mean():+.4f}% median_relative_delta={np.median(relative):+.4f}%"
        )
    print("\nPer task (negative delta is better):")
    for key in shared:
        old = baseline[key]
        new = candidate[key]
        mse_delta = 100.0 * (float(new["learned_mse"]) - float(old["mse"])) / float(old["mse"])
        mae_delta = 100.0 * (float(new["learned_mae"]) - float(old["mae"])) / float(old["mae"])
        print(f"{key[0]:<11} h={key[1]:<3} mse={mse_delta:+7.3f}% mae={mae_delta:+7.3f}%")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", default="mvpf-tests/results/metrics/task_metrics_24.csv")
    return p


if __name__ == "__main__":
    run(parser().parse_args())

