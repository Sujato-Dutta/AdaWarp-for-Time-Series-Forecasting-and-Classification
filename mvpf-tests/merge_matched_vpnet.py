"""Merge matched VPNet metrics into the 192-row final task table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity", "Traffic")
HORIZONS = (96, 192, 336, 720)
METHODS = ("MVPF", "DLinear", "PatchTST", "TimesNet", "iTransformer", "TimeMixer", "FEDformer", "VPNet")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def run(args: argparse.Namespace) -> None:
    base = read_csv(args.task_table)
    matched = read_csv(args.vpnet)
    lookup = {(row["dataset"], int(row["horizon"])): row for row in matched}
    expected_tasks = {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}
    if set(lookup) != expected_tasks:
        missing = sorted(expected_tasks - set(lookup))
        extra = sorted(set(lookup) - expected_tasks)
        raise RuntimeError(f"Matched VPNet task mismatch: missing={missing}, extra={extra}")

    output_rows = []
    for row in base:
        updated = dict(row)
        if row["method"] == "VPNet":
            source = lookup[(row["dataset"], int(row["horizon"]))]
            updated["mse"] = source["mse"]
            updated["mae"] = source["mae"]
            updated["seed"] = source["seed"]
            updated["seq_len"] = source["seq_len"]
            updated["source_file"] = str(args.vpnet)
            updated["listed_prediction_file"] = ""
        output_rows.append(updated)

    keys = [(row["dataset"], int(row["horizon"]), row["method"]) for row in output_rows]
    expected = {
        (dataset, horizon, method)
        for dataset in DATASETS
        for horizon in HORIZONS
        for method in METHODS
    }
    if len(output_rows) != 192 or set(keys) != expected or len(keys) != len(set(keys)):
        raise RuntimeError("Merged task table failed the 192-row completeness audit.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.output} with 24 matched VPNet rows and {len(output_rows)} total rows")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vpnet", type=Path, required=True)
    p.add_argument(
        "--task-table",
        type=Path,
        default=Path("mvpf-tests/results/metrics/task_metrics_24.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("mvpf-tests/results/metrics/task_metrics_24_matched_vpnet.csv"),
    )
    return p


if __name__ == "__main__":
    run(parser().parse_args())