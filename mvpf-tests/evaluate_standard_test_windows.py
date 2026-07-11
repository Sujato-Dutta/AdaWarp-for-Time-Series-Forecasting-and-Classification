"""Re-evaluate final pruned MVPF checkpoints on full standard LTSF test windows.

The training runner caps evaluation windows for throughput.  This script uses
the stored validation-best checkpoints and streams over exactly the test
windows used by TSLibrary: fixed 12/4/4 month ETT splits and chronological
70/10/20 splits for the remaining datasets.  No parameters are updated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adawarp_mvpf_plus import AdaWarpMVPFPlusForecaster
from benchmark_adawarp_ltsf import dataset_path, load_numeric_csv, normalize_train, split_lengths
from benchmark_adawarp_mvpf_plus_ltsf import WindowDataset


DATASETS = ("ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity", "Traffic")
HORIZONS = (96, 192, 336, 720)


def test_starts(dataset: str, total: int, seq_len: int, horizon: int) -> tuple[np.ndarray, int, int]:
    train, validation, declared_test = split_lengths(dataset, total)
    boundary = train + validation
    test_end = boundary + declared_test if dataset.startswith("ETT") else total
    starts = np.arange(boundary - seq_len, test_end - seq_len - horizon + 1, dtype=np.int64)
    starts = starts[starts + seq_len >= boundary]
    if starts.size == 0:
        raise ValueError(f"No standard test windows for {dataset}, horizon={horizon}.")
    return starts, train, test_end


def build_model(config: dict) -> AdaWarpMVPFPlusForecaster:
    config = dict(config)
    seq_len = int(config.pop("seq_len"))
    pred_len = int(config.pop("pred_len"))
    return AdaWarpMVPFPlusForecaster(seq_len, pred_len, **config)


@torch.inference_mode()
def evaluate(model, series: np.ndarray, starts: np.ndarray, seq_len: int, horizon: int, batch_size: int, device):
    data = WindowDataset(series, starts, seq_len=seq_len, pred_len=horizon)
    batches = DataLoader(data, batch_size=min(batch_size, len(data)), shuffle=False, num_workers=0)
    squared = absolute = 0.0
    count = 0
    model.eval()
    for inputs, targets in batches:
        prediction = model(inputs.to(device), None, None, None)
        error = prediction - targets.to(device)
        squared += float(error.square().sum().cpu())
        absolute += float(error.abs().sum().cpu())
        count += int(error.numel())
    return squared / count, absolute / count


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint_root = Path(args.checkpoint_root)
    output = Path(args.output)
    existing = []
    if output.exists() and not args.force:
        with output.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    completed = {(row["dataset"], int(row["horizon"])) for row in existing}
    rows = existing
    for dataset in args.datasets:
        values = load_numeric_csv(dataset_path(Path(args.data_root), dataset))
        for horizon in args.horizons:
            if (dataset, horizon) in completed and not args.force:
                print(f"skip {dataset} h={horizon}", flush=True)
                continue
            checkpoint = checkpoint_root / f"{dataset}_h{horizon}_seed42.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = payload["config"]
            model = build_model(config).to(device)
            model.load_state_dict(payload["state_dict"])
            starts, train_end, test_end = test_starts(dataset, len(values), int(config["seq_len"]), horizon)
            normalized, _, _ = normalize_train(values, train_end)
            started = time.perf_counter()
            mse, mae = evaluate(
                model, normalized, starts, int(config["seq_len"]), horizon,
                args.batch_size if dataset not in {"Electricity", "Traffic"} else args.wide_batch_size,
                device,
            )
            row = {
                "dataset": dataset,
                "horizon": horizon,
                "seed": 42,
                "seq_len": int(config["seq_len"]),
                "model": "MVPF",
                "mse": mse,
                "mae": mae,
                "num_test_windows": len(starts),
                "test_start": int(starts.min()),
                "test_end_exclusive": test_end,
                "best_epoch": int(payload["selection"]["best_epoch"]),
                "best_validation_mse": payload["selection"]["best_validation_mse"],
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint),
            }
            rows = [old for old in rows if not (old["dataset"] == dataset and int(old["horizon"]) == horizon)]
            rows.append(row)
            rows.sort(key=lambda item: (DATASETS.index(item["dataset"]), int(item["horizon"])))
            write_csv(output, rows)
            print(
                f"standard-test MVPF {dataset:<11} h={horizon:<3} windows={len(starts):<5} "
                f"mse={mse:.6f} mae={mae:.6f}",
                flush=True,
            )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    p.add_argument("--checkpoint-root", default="mvpf-tests/results/pruned_instrumented_820583/checkpoints")
    p.add_argument("--output", default="mvpf-tests/results/pruned_instrumented_820583/metrics/standard_test_metrics.csv")
    p.add_argument("--data-root", default="TSLibrary/dataset")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--wide-batch-size", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
