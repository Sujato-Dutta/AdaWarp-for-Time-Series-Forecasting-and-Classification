"""Aggregate final AdaWarp-MVPF ablation runs.

This aggregates validation-checkpointed final-MVPF ablations produced by
``scripts/tacc/submit_adawarp_mvpf_plus_ablations.sh``. The code filenames keep
``plus`` for reproducibility, but the paper-facing model is AdaWarp-MVPF.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Iterable


METRIC_FILE = "ltsf_custom_neural_baselines.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def discover_ablation_files(root: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(root.rglob(METRIC_FILE)):
        rel = path.relative_to(root)
        if len(rel.parts) < 4:
            continue
        ablation = rel.parts[0]
        yield ablation, path


def numeric(row: dict[str, object], key: str) -> float:
    return float(row[key])


def key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row["dataset"]), str(row["horizon"]), str(row.get("seed", 42)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root produced by submit_adawarp_mvpf_plus_ablations.sh")
    parser.add_argument("--full-root", default=None, help="Optional final full-model root for delta calculations")
    parser.add_argument("--output-dir", default=None, help="Defaults to ROOT/metrics")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "metrics"

    rows: list[dict[str, object]] = []
    for ablation, path in discover_ablation_files(root):
        for row in read_rows(path):
            item: dict[str, object] = dict(row)
            item["paper_model"] = "AdaWarp-MVPF"
            item["implementation"] = "adawarp_mvpf_plus"
            item["ablation"] = ablation
            item["source_file"] = str(path)
            rows.append(item)

    if not rows:
        raise SystemExit(f"No {METRIC_FILE} files found under {root}")

    common_fields = [
        "paper_model",
        "implementation",
        "ablation",
        "dataset",
        "horizon",
        "seed",
        "seq_len",
        "model",
        "mse",
        "mae",
        "rmse",
        "smape",
        "num_eval_windows",
        "best_epoch",
        "best_validation_mse",
        "raw_prediction_file",
        "source_file",
    ]
    write_rows(output_dir / "mvpf_final_ablation_all.csv", rows, common_fields)

    by_ablation: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_ablation.setdefault(str(row["ablation"]), []).append(row)

    summary: list[dict[str, object]] = []
    for ablation, group in sorted(by_ablation.items()):
        summary.append({
            "ablation": ablation,
            "n": len(group),
            "mean_mse": mean(numeric(row, "mse") for row in group),
            "mean_mae": mean(numeric(row, "mae") for row in group),
            "mean_rmse": mean(numeric(row, "rmse") for row in group),
            "mean_best_epoch": mean(numeric(row, "best_epoch") for row in group if str(row.get("best_epoch", "")) != ""),
        })

    full_rows: list[dict[str, object]] = []
    if args.full_root:
        full_root = Path(args.full_root)
        for path in sorted(full_root.rglob(METRIC_FILE)):
            for row in read_rows(path):
                item: dict[str, object] = dict(row)
                item["paper_model"] = "AdaWarp-MVPF"
                item["implementation"] = "adawarp_mvpf_plus"
                item["ablation"] = "full"
                item["source_file"] = str(path)
                full_rows.append(item)
        full_by_key = {key(row): row for row in full_rows}
        deltas: list[dict[str, object]] = []
        for row in rows:
            full = full_by_key.get(key(row))
            if full is None:
                continue
            mse = numeric(row, "mse")
            mae = numeric(row, "mae")
            full_mse = numeric(full, "mse")
            full_mae = numeric(full, "mae")
            deltas.append({
                "ablation": row["ablation"],
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "seed": row.get("seed", 42),
                "mse": mse,
                "full_mse": full_mse,
                "delta_mse": mse - full_mse,
                "rel_delta_mse_pct": 100.0 * (mse - full_mse) / max(abs(full_mse), 1e-12),
                "mae": mae,
                "full_mae": full_mae,
                "delta_mae": mae - full_mae,
                "rel_delta_mae_pct": 100.0 * (mae - full_mae) / max(abs(full_mae), 1e-12),
                "ablation_worse_mse": int(mse > full_mse),
                "ablation_worse_mae": int(mae > full_mae),
            })
        if deltas:
            write_rows(
                output_dir / "mvpf_final_ablation_deltas.csv",
                deltas,
                [
                    "ablation",
                    "dataset",
                    "horizon",
                    "seed",
                    "mse",
                    "full_mse",
                    "delta_mse",
                    "rel_delta_mse_pct",
                    "mae",
                    "full_mae",
                    "delta_mae",
                    "rel_delta_mae_pct",
                    "ablation_worse_mse",
                    "ablation_worse_mae",
                ],
            )
            delta_by_ablation: dict[str, list[dict[str, object]]] = {}
            for row in deltas:
                delta_by_ablation.setdefault(str(row["ablation"]), []).append(row)
            for item in summary:
                group = delta_by_ablation.get(str(item["ablation"]), [])
                if not group:
                    continue
                item["mean_rel_delta_mse_pct"] = mean(float(row["rel_delta_mse_pct"]) for row in group)
                item["mean_rel_delta_mae_pct"] = mean(float(row["rel_delta_mae_pct"]) for row in group)
                item["worse_mse_count"] = sum(int(row["ablation_worse_mse"]) for row in group)
                item["worse_mae_count"] = sum(int(row["ablation_worse_mae"]) for row in group)
                item["matched_full_count"] = len(group)

    summary_fields = [
        "ablation",
        "n",
        "mean_mse",
        "mean_mae",
        "mean_rmse",
        "mean_best_epoch",
        "matched_full_count",
        "mean_rel_delta_mse_pct",
        "mean_rel_delta_mae_pct",
        "worse_mse_count",
        "worse_mae_count",
    ]
    write_rows(output_dir / "mvpf_final_ablation_summary.csv", summary, summary_fields)
    print(f"wrote {output_dir / 'mvpf_final_ablation_all.csv'}")
    print(f"wrote {output_dir / 'mvpf_final_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
