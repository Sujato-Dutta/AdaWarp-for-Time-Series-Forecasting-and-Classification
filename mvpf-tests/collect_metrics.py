"""Collect the six-dataset, eight-method LTSF table into one audited CSV."""

from __future__ import annotations

from pathlib import Path

from common import DATASETS, HERE, HORIZONS, METHODS, RESULTS, ROOT, read_csv, write_csv, write_json


FIVE_DATASET_FOLDERS = {
    "DLinear": "D-Linear",
    "PatchTST": "PatchTST",
    "TimesNet": "TimesNet",
    "iTransformer": "iTransformer",
    "TimeMixer": "TimeMixer",
    "FEDformer": "FEDformer",
    "VPNet": "VPNet",
}
ETTM2_FOLDERS = {
    "MVPF": "MVPF",
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "TimesNet": "TimesNet",
    "iTransformer": "iTransformer",
    "TimeMixer": "Timemixer",
    "FEDformer": "FEDFormer",
    "VPNet": "VPNet",
}


def _normalized_row(row: dict[str, str], method: str, source: Path) -> dict[str, object]:
    return {
        "dataset": row["dataset"],
        "horizon": int(row["horizon"]),
        "seed": int(row["seed"]),
        "seq_len": int(row["seq_len"]),
        "method": method,
        "mse": float(row["mse"]),
        "mae": float(row["mae"]),
        "source_file": str(source.relative_to(ROOT)),
        "listed_prediction_file": row.get("raw_prediction_file") or row.get("pred_file") or "",
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for dataset in (name for name in DATASETS if name != "ETTm2"):
        source = ROOT / "results" / "ltsf_5" / "MVPF_Cpt" / dataset / "metrics" / "ltsf_main5_combined.csv"
        for row in read_csv(source):
            if row["dataset"] == dataset and int(row["seed"]) == 42:
                rows.append(_normalized_row(row, "MVPF", source))
    for method, folder in FIVE_DATASET_FOLDERS.items():
        source = ROOT / "results" / "ltsf_5" / folder / "metrics" / "ltsf_main5_combined.csv"
        for row in read_csv(source):
            if row["dataset"] in DATASETS and row["dataset"] != "ETTm2" and int(row["seed"]) == 42:
                rows.append(_normalized_row(row, method, source))
    for method, folder in ETTM2_FOLDERS.items():
        source = ROOT / "results" / "ettm2" / folder / "metrics" / "ltsf_main5_combined.csv"
        for row in read_csv(source):
            if row["dataset"] == "ETTm2" and int(row["seed"]) == 42:
                rows.append(_normalized_row(row, method, source))

    # Replace historical MVPF aggregates with the final validation-checkpointed
    # model evaluated on every standard test window. Baseline rows are unchanged.
    final_metrics = RESULTS / "pruned_instrumented_820583" / "metrics" / "standard_test_metrics.csv"
    if final_metrics.exists():
        final_lookup = {
            (row["dataset"], int(row["horizon"])): row
            for row in read_csv(final_metrics)
            if int(row["seed"]) == 42
        }
        for row in rows:
            if row["method"] != "MVPF":
                continue
            final = final_lookup.get((str(row["dataset"]), int(row["horizon"])))
            if final is None:
                continue
            row["mse"] = float(final["mse"])
            row["mae"] = float(final["mae"])
            row["source_file"] = str(final_metrics.relative_to(ROOT))
            row["listed_prediction_file"] = ""

    # Replace the earlier capped-window VPNet rows with the fully matched rerun.
    matched_vpnet = RESULTS / "matched_vpnet_final" / "metrics" / "matched_vpnet_standard_test.csv"
    if matched_vpnet.exists():
        vpnet_lookup = {
            (row["dataset"], int(row["horizon"])): row
            for row in read_csv(matched_vpnet)
            if int(row["seed"]) == 42
        }
        for row in rows:
            if row["method"] != "VPNet":
                continue
            final = vpnet_lookup.get((str(row["dataset"]), int(row["horizon"])))
            if final is None:
                continue
            row["mse"] = float(final["mse"])
            row["mae"] = float(final["mae"])
            row["source_file"] = str(matched_vpnet.relative_to(ROOT))
            row["listed_prediction_file"] = ""

    keys = [(row["dataset"], row["horizon"], row["method"]) for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    expected = {(dataset, horizon, method) for dataset in DATASETS for horizon in HORIZONS for method in METHODS}
    actual = set(keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    wrong_protocol = [row for row in rows if row["seed"] != 42 or row["seq_len"] != 96]
    audit = {
        "datasets": list(DATASETS),
        "horizons": list(HORIZONS),
        "methods": list(METHODS),
        "expected_rows": len(expected),
        "actual_rows": len(rows),
        "duplicates": duplicates,
        "missing": missing,
        "extra": extra,
        "wrong_protocol_rows": wrong_protocol,
        "status": "complete" if not (duplicates or missing or extra or wrong_protocol) else "failed",
    }
    write_json(RESULTS / "audit" / "metric_collection.json", audit)
    if audit["status"] != "complete":
        raise RuntimeError(f"Metric collection failed audit: {audit}")
    rows.sort(key=lambda row: (DATASETS.index(str(row["dataset"])), HORIZONS.index(int(row["horizon"])), METHODS.index(str(row["method"]))))
    write_csv(RESULTS / "metrics" / "task_metrics_24.csv", rows)
    print(f"collected {len(rows)} rows: {len(DATASETS)} datasets x {len(HORIZONS)} horizons x {len(METHODS)} methods")


if __name__ == "__main__":
    main()

