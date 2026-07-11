"""Audit whether inference-only and per-window MVPF analyses are locally possible."""

from __future__ import annotations

from pathlib import Path

from common import RESULTS, ROOT, read_csv, write_csv, write_json


def main() -> None:
    metrics = read_csv(RESULTS / "metrics" / "task_metrics_24.csv")
    artifact_roots = (ROOT / "results", RESULTS)
    local_files = {
        path.name: path
        for artifact_root in artifact_roots
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    rows = []
    for row in metrics:
        listed = row.get("listed_prediction_file", "")
        basename = Path(listed).name if listed else ""
        local = local_files.get(basename)
        rows.append(
            {
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "method": row["method"],
                "listed_prediction_file": listed,
                "local_prediction_available": int(local is not None),
                "local_prediction_path": str(local.relative_to(ROOT)) if local else "",
            }
        )
    checkpoints = [
        path
        for run_root in RESULTS.glob("pruned_instrumented_*")
        for path in (run_root / "checkpoints").glob("*")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    ]
    mvpf_rows = [row for row in rows if row["method"] == "MVPF"]
    patch_rows = [row for row in rows if row["method"] == "PatchTST"]
    summary = {
        "mvpf_task_predictions_available": sum(int(row["local_prediction_available"]) for row in mvpf_rows),
        "mvpf_task_predictions_required": len(mvpf_rows),
        "patchtst_task_predictions_available": sum(int(row["local_prediction_available"]) for row in patch_rows),
        "patchtst_task_predictions_required": len(patch_rows),
        "mvpf_checkpoints": [str(path.relative_to(ROOT)) for path in checkpoints],
        "gate_analysis_ready": bool(checkpoints) and all(int(row["local_prediction_available"]) for row in mvpf_rows),
        "paired_window_analysis_ready": all(int(row["local_prediction_available"]) for row in mvpf_rows + patch_rows),
        "blocked_analyses": [
            "learned-versus-fixed gate evaluation",
            "gate-behaviour and scale-specialization figures",
            "moving-block bootstrap confidence intervals",
            "per-window ECDF and tail-error analysis",
            "per-channel Electricity/Traffic analysis",
            "lead-time error curves",
            "qualitative forecast examples",
        ],
        "reason": "Validation-best MVPF checkpoints are available, but matched raw forecast arrays for MVPF and PatchTST are not both present locally.",
    }
    write_csv(RESULTS / "audit" / "prediction_artifact_availability.csv", rows)
    write_json(RESULTS / "audit" / "artifact_availability.json", summary)
    print(
        f"MVPF raw predictions: {summary['mvpf_task_predictions_available']}/{summary['mvpf_task_predictions_required']}; "
        f"PatchTST raw predictions: {summary['patchtst_task_predictions_available']}/{summary['patchtst_task_predictions_required']}; "
        f"MVPF checkpoints: {len(checkpoints)}"
    )


if __name__ == "__main__":
    main()

