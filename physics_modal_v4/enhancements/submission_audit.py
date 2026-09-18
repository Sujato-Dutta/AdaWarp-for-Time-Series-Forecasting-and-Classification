"""Build submission-facing audits from completed LatentFlow experiments.

This module is analysis-only. It never imports torch, opens a checkpoint, or
trains a model. All outputs are deterministic functions of archived CSV/JSON
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "enhancements" / "results" / "submission_audit"
SOURCE_FAMILY = {
    "ETTh1": "ETT station 1",
    "ETTm1": "ETT station 1",
    "ETTh2": "ETT station 2",
    "ETTm2": "ETT station 2",
    "Weather": "Weather",
    "Electricity": "Electricity",
    "Traffic": "Traffic",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=object)
    unique = np.unique(clusters)
    grouped = {key: values[clusters == key] for key in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        samples[index] = np.concatenate([grouped[key] for key in selected]).mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _effect_rows(
    frame: pd.DataFrame,
    *,
    left_prefix: str,
    right_prefix: str,
    label: str,
    repetitions: int,
) -> list[dict[str, object]]:
    frame = frame.copy()
    frame["source_family"] = frame.dataset.map(SOURCE_FAMILY)
    if frame.source_family.isna().any():
        missing = sorted(frame.loc[frame.source_family.isna(), "dataset"].unique())
        raise ValueError(f"Missing source-family mapping for {missing}")
    rows: list[dict[str, object]] = []
    for metric_index, metric in enumerate(("mse", "mae")):
        left = frame[f"{left_prefix}_{metric}"].astype(float).to_numpy()
        right = frame[f"{right_prefix}_{metric}"].astype(float).to_numpy()
        relative = 100.0 * (1.0 - left / right)
        source = frame.source_family.astype(str).to_numpy()
        source_seed = (
            frame.source_family.astype(str) + "::" + frame.seed.astype(int).astype(str)
        ).to_numpy()
        for kind, clusters, offset in (
            ("source_family", source, 0),
            ("source_family_seed", source_seed, 1),
        ):
            low, high = _cluster_interval(
                relative,
                clusters,
                seed=2027 + 17 * metric_index + offset,
                repetitions=repetitions,
            )
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "estimand": "mean paired per-cell relative improvement percent",
                    "cluster_unit": kind,
                    "clusters": int(len(np.unique(clusters))),
                    "cells": int(len(frame)),
                    "mean_relative_improvement_percent": float(relative.mean()),
                    "median_relative_improvement_percent": float(np.median(relative)),
                    "ci95_low_percent": low,
                    "ci95_high_percent": high,
                    "wins": int((left < right - 1e-12).sum()),
                    "ties": int((np.abs(left - right) <= 1e-12).sum()),
                    "losses": int((left > right + 1e-12).sum()),
                }
            )
    return rows


def _degradation_rows(
    frame: pd.DataFrame,
    *,
    control_prefix: str,
    final_prefix: str,
    label: str,
    repetitions: int,
) -> list[dict[str, object]]:
    frame = frame.copy()
    frame["source_family"] = frame.dataset.map(SOURCE_FAMILY)
    if frame.source_family.isna().any():
        missing = sorted(frame.loc[frame.source_family.isna(), "dataset"].unique())
        raise ValueError(f"Missing source-family mapping for {missing}")
    rows: list[dict[str, object]] = []
    for metric_index, metric in enumerate(("mse", "mae")):
        control = frame[f"{control_prefix}_{metric}"].astype(float).to_numpy()
        final = frame[f"{final_prefix}_{metric}"].astype(float).to_numpy()
        relative = 100.0 * (control / final - 1.0)
        source = frame.source_family.astype(str).to_numpy()
        source_seed = (
            frame.source_family.astype(str) + "::" + frame.seed.astype(int).astype(str)
        ).to_numpy()
        for kind, clusters, offset in (
            ("source_family", source, 0),
            ("source_family_seed", source_seed, 1),
        ):
            low, high = _cluster_interval(
                relative,
                clusters,
                seed=4041 + 17 * metric_index + offset,
                repetitions=repetitions,
            )
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "estimand": "mean paired per-cell control degradation percent",
                    "cluster_unit": kind,
                    "clusters": int(len(np.unique(clusters))),
                    "cells": int(len(frame)),
                    "mean_relative_degradation_percent": float(relative.mean()),
                    "median_relative_degradation_percent": float(np.median(relative)),
                    "ci95_low_percent": low,
                    "ci95_high_percent": high,
                    "control_better": int((control < final - 1e-12).sum()),
                    "ties": int((np.abs(control - final) <= 1e-12).sum()),
                    "control_worse": int((control > final + 1e-12).sum()),
                }
            )
    return rows

def _timepro_pairs() -> pd.DataFrame:
    latent = _read_csv(
        ROOT
        / "enhancements"
        / "results"
        / "final_enhancement_statistics"
        / "strongest_comparator_runs.csv"
    )
    latent = latent.loc[latent.model.eq("LatentFlow"), [
        "dataset", "horizon", "seed", "mse", "mae"
    ]].drop_duplicates(["dataset", "horizon", "seed"])
    additions = _read_csv(ROOT / "additions" / "results" / "all_additions_results.csv")
    timepro = additions.loc[
        additions.source_table.eq("summary/timepro_five_seed_runs.csv")
        & additions.model.eq("TimePro"),
        ["dataset", "horizon", "seed", "mse", "mae"],
    ].dropna(subset=["dataset", "horizon", "seed", "mse", "mae"])
    timepro = timepro.drop_duplicates(["dataset", "horizon", "seed"])
    for frame in (latent, timepro):
        frame["horizon"] = frame.horizon.astype(int)
        frame["seed"] = frame.seed.astype(int)
    pairs = latent.merge(
        timepro,
        on=["dataset", "horizon", "seed"],
        suffixes=("_latentflow", "_timepro"),
        validate="one_to_one",
    )
    pairs = pairs.rename(
        columns={
            "mse_latentflow": "latentflow_mse",
            "mae_latentflow": "latentflow_mae",
            "mse_timepro": "timepro_mse",
            "mae_timepro": "timepro_mae",
        }
    )
    if len(pairs) != 140:
        raise RuntimeError(f"Expected 140 LatentFlow/TimePro pairs, found {len(pairs)}")
    return pairs


def analyze_timepro(output: Path, repetitions: int) -> None:
    pairs = _timepro_pairs()
    rows = _effect_rows(
        pairs,
        left_prefix="latentflow",
        right_prefix="timepro",
        label="LatentFlow vs TimePro",
        repetitions=repetitions,
    )
    _write_csv(pd.DataFrame(rows), output / "timepro_source_aware_intervals.csv")

    pairs["source_family"] = pairs.dataset.map(SOURCE_FAMILY)
    leave_rows = []
    for omitted in sorted(pairs.source_family.unique()):
        retained = pairs.loc[pairs.source_family.ne(omitted)]
        for metric in ("mse", "mae"):
            gain = 100.0 * (
                1.0
                - retained[f"latentflow_{metric}"].astype(float)
                / retained[f"timepro_{metric}"].astype(float)
            )
            leave_rows.append(
                {
                    "omitted_source_family": omitted,
                    "metric": metric,
                    "remaining_source_families": int(retained.source_family.nunique()),
                    "remaining_cells": int(len(retained)),
                    "mean_relative_improvement_percent": float(gain.mean()),
                    "median_relative_improvement_percent": float(gain.median()),
                }
            )
    _write_csv(pd.DataFrame(leave_rows), output / "timepro_leave_one_source_out.csv")
    _write_csv(pairs, output / "timepro_five_seed_pairs.csv")


def analyze_seed42_baselines(output: Path, repetitions: int) -> None:
    frame = _read_csv(ROOT / "results" / "final" / "statistics" / "headline_seed_runs.csv")
    frame = frame.loc[frame.seed.astype(int).eq(42)].copy()
    central = _read_csv(
        ROOT / "enhancements" / "results" / "selector_safe_statistics" / "central_runs.csv"
    )
    latent = central.loc[
        central.variant.eq("final_no_exchange") & central.seed.astype(int).eq(42),
        ["dataset", "horizon", "seed", "mse", "mae"],
    ].rename(columns={"mse": "latentflow_mse", "mae": "latentflow_mae"})
    rows: list[dict[str, object]] = []
    for model in sorted(frame.loc[frame.model.ne("LatentFlow"), "model"].unique()):
        comparator = frame.loc[
            frame.model.eq(model),
            ["dataset", "horizon", "seed", "mse", "mae"],
        ].rename(columns={"mse": "comparator_mse", "mae": "comparator_mae"})
        pairs = latent.merge(
            comparator,
            on=["dataset", "horizon", "seed"],
            validate="one_to_one",
        )
        if len(pairs) != 28:
            raise RuntimeError(f"Expected 28 seed-42 pairs for {model}, found {len(pairs)}")
        rows.extend(
            _effect_rows(
                pairs,
                left_prefix="latentflow",
                right_prefix="comparator",
                label=f"LatentFlow vs {model}",
                repetitions=repetitions,
            )
        )
    _write_csv(pd.DataFrame(rows), output / "seed42_baseline_source_aware_intervals.csv")

def analyze_central_mechanisms(output: Path, repetitions: int) -> None:
    frame = _read_csv(
        ROOT
        / "enhancements"
        / "results"
        / "selector_safe_statistics"
        / "central_runs.csv"
    )
    final = frame.loc[
        frame.variant.eq("final_no_exchange"),
        ["dataset", "horizon", "seed", "mse", "mae"],
    ].rename(columns={"mse": "final_mse", "mae": "final_mae"})
    controls = {
        "early_fusion": "process preserving vs early fusion",
        "x_to_z": "process preserving vs causal X-to-Z",
        "no_linear_bank": "process preserving vs no linear bank",
    }
    rows: list[dict[str, object]] = []
    pairs: list[pd.DataFrame] = []
    for variant, label in controls.items():
        control = frame.loc[
            frame.variant.eq(variant),
            ["dataset", "horizon", "seed", "mse", "mae"],
        ].rename(columns={"mse": "control_mse", "mae": "control_mae"})
        joined = final.merge(
            control,
            on=["dataset", "horizon", "seed"],
            validate="one_to_one",
        )
        if len(joined) != 140:
            raise RuntimeError(
                f"Expected 140 final/{variant} pairs, found {len(joined)}"
            )
        joined["control"] = variant
        pairs.append(joined)
        rows.extend(
            _degradation_rows(
                joined,
                control_prefix="control",
                final_prefix="final",
                label=label,
                repetitions=repetitions,
            )
        )
    _write_csv(pd.DataFrame(rows), output / "central_source_aware_intervals.csv")
    _write_csv(pd.concat(pairs, ignore_index=True), output / "central_canonical_pairs.csv")

def analyze_selector(output: Path, repetitions: int) -> None:
    frame = _read_csv(ROOT / "enhancements" / "results" / "final_audits" / "reference_extension_runs.csv")
    frame["test_winner"] = np.where(
        frame.extension_mse.astype(float) < frame.reference_mse.astype(float) - 1e-12,
        "extension",
        np.where(
            frame.reference_mse.astype(float) < frame.extension_mse.astype(float) - 1e-12,
            "reference",
            "tie",
        ),
    )
    confusion = (
        frame.groupby(["validation_selected_stage", "test_winner"], dropna=False)
        .size()
        .rename("fits")
        .reset_index()
    )
    _write_csv(confusion, output / "selector_confusion.csv")

    oracle = frame[["reference_mse", "extension_mse"]].astype(float).min(axis=1)
    frame["relative_regret_percent"] = 100.0 * (
        frame.selected_mse.astype(float) / oracle - 1.0
    )
    quantiles = frame.relative_regret_percent.quantile([0.0, 0.5, 0.9, 0.95, 1.0])
    selector_summary = pd.DataFrame(
        [
            {
                "fits": len(frame),
                "validation_selected_extension": int(frame.validation_selected_stage.eq("extension").sum()),
                "validation_selected_reference": int(frame.validation_selected_stage.eq("reference").sum()),
                "test_extension_better": int(frame.test_winner.eq("extension").sum()),
                "test_reference_better": int(frame.test_winner.eq("reference").sum()),
                "test_ties": int(frame.test_winner.eq("tie").sum()),
                "mean_regret_percent": float(frame.relative_regret_percent.mean()),
                "median_regret_percent": float(quantiles.loc[0.5]),
                "p90_regret_percent": float(quantiles.loc[0.9]),
                "p95_regret_percent": float(quantiles.loc[0.95]),
                "maximum_regret_percent": float(quantiles.loc[1.0]),
            }
        ]
    )
    _write_csv(selector_summary, output / "selector_summary.csv")
    _write_csv(
        frame[[
            "dataset", "horizon", "seed", "validation_selected_stage", "test_winner",
            "reference_mse", "extension_mse", "selected_mse", "relative_regret_percent",
        ]],
        output / "selector_fit_diagnostics.csv",
    )

    renamed = frame.rename(
        columns={
            "selected_mse": "selected_mse",
            "reference_mse": "reference_mse",
            "selected_mae": "selected_mae",
            "reference_mae": "reference_mae",
        }
    )
    effects = _effect_rows(
        renamed,
        left_prefix="selected",
        right_prefix="reference",
        label="validation-selected predictor vs frozen reference",
        repetitions=repetitions,
    )
    _write_csv(pd.DataFrame(effects), output / "selector_source_aware_intervals.csv")


def analyze_priors(output: Path, repetitions: int) -> None:
    controls = _read_csv(
        ROOT / "enhancements" / "results" / "final_audits" / "prior_control_runs.csv"
    ).rename(columns={"mse": "control_mse", "mae": "control_mae"})
    final = _read_csv(
        ROOT
        / "enhancements"
        / "results"
        / "selector_safe_statistics"
        / "central_runs.csv"
    )
    final = final.loc[
        final.variant.eq("final_no_exchange"),
        ["dataset", "horizon", "seed", "mse", "mae"],
    ].rename(columns={"mse": "final_mse", "mae": "final_mae"})
    for frame in (controls, final):
        frame["horizon"] = frame.horizon.astype(int)
        frame["seed"] = frame.seed.astype(int)
    frame = controls.merge(
        final,
        on=["dataset", "horizon", "seed"],
        validate="many_to_one",
    )
    if len(frame) != 252:
        raise RuntimeError(f"Expected 252 matched prior-control rows, found {len(frame)}")
    rows = []
    for variant, group in frame.groupby("variant"):
        rows.extend(
            _effect_rows(
                group,
                left_prefix="final",
                right_prefix="control",
                label=f"final vs {variant}",
                repetitions=repetitions,
            )
        )
    _write_csv(pd.DataFrame(rows), output / "prior_control_source_aware_intervals.csv")
    _write_csv(frame, output / "prior_control_canonical_pairs.csv")
def audit_traffic_protocol(output: Path) -> None:
    headline_path = (
        ROOT
        / "results"
        / "final"
        / "headline"
        / "runs"
        / "LatentFlow"
        / "seed42"
        / "Traffic"
        / "h720.json"
    )
    headline = json.loads(headline_path.read_text(encoding="utf-8"))
    scaling_path = (
        ROOT
        / "enhancements"
        / "results"
        / "final_enhancement_statistics"
        / "real_channel_scaling.csv"
    )
    scaling = _read_csv(scaling_path)
    endpoint = scaling.loc[
        scaling.dataset.eq("Traffic")
        & scaling.model.eq("LatentFlow")
        & scaling.channels.astype(int).eq(862)
    ]
    if len(endpoint) != 1:
        raise RuntimeError(f"Expected one Traffic C=862 endpoint, found {len(endpoint)}")
    endpoint = endpoint.iloc[0]
    rows = pd.DataFrame(
        [
            {
                "result": "canonical_headline",
                "mse": float(headline["mse"]),
                "mae": float(headline["mae"]),
                "best_epoch": int(headline["best_epoch"]),
                "best_stage": headline["best_stage"],
                "checkpoint": headline["checkpoint"],
                "reference_training_micro_batch": 16,
                "extension_training_micro_batch": 4,
                "effective_extension_batch": 16,
                "independent_retraining": True,
                "source_file_sha256": _sha256(headline_path),
            },
            {
                "result": "channel_scaling_endpoint",
                "mse": float(endpoint.mse),
                "mae": float(endpoint.mae),
                "best_epoch": int(endpoint.best_epoch),
                "best_stage": endpoint.best_stage,
                "checkpoint": endpoint.checkpoint,
                "reference_training_micro_batch": 4,
                "extension_training_micro_batch": 4,
                "effective_extension_batch": 16,
                "independent_retraining": True,
                "source_file_sha256": _sha256(scaling_path),
            },
        ]
    )
    _write_csv(rows, output / "traffic_h720_protocol_comparison.csv")
    _write_json(
        {
            "status": "not_a_batch_invariance_comparison",
            "mse_absolute_difference": float(endpoint.mse) - float(headline["mse"]),
            "mse_relative_difference_percent": 100.0
            * (float(endpoint.mse) / float(headline["mse"]) - 1.0),
            "reason": (
                "The channel-scaling endpoint independently retrained the reference with "
                "micro-batch 4; the headline path trained its reference with micro-batch 16. "
                "A canonical checkpoint replay at batch sizes 1 and 4 is still required."
            ),
        },
        output / "traffic_h720_discrepancy.json",
    )


def build_result_manifest(output: Path) -> None:
    rows: list[dict[str, object]] = []
    headline_root = ROOT / "results" / "final" / "headline" / "runs"
    for path in sorted(headline_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "result_file": path.relative_to(ROOT).as_posix(),
                "result_file_sha256": _sha256(path),
                "model": payload.get("model"),
                "dataset": payload.get("dataset"),
                "horizon": payload.get("horizon"),
                "seed": payload.get("seed"),
                "mse": payload.get("mse"),
                "mae": payload.get("mae"),
                "best_epoch": payload.get("best_epoch"),
                "best_stage": payload.get("best_stage"),
                "best_validation_mse": payload.get("best_validation_mse"),
                "candidate": payload.get("candidate"),
                "checkpoint": payload.get("checkpoint"),
                "config_hash": payload.get("config_hash"),
                "selection_protocol": payload.get("selection_protocol"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No headline JSON results were found.")
    _write_csv(frame, output / "headline_result_provenance.csv")
    _write_json(
        {
            "headline_rows": int(len(frame)),
            "models": sorted(frame.model.dropna().unique().tolist()),
            "missing_checkpoint_hashes": int(frame.config_hash.isna().sum()),
            "note": (
                "This local manifest hashes result files. Checkpoint, dataset, split, and "
                "sampled-window hashes require the TACC checkpoint/data audit."
            ),
        },
        output / "provenance_status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    analyze_timepro(args.output, args.bootstrap_repetitions)
    analyze_seed42_baselines(args.output, args.bootstrap_repetitions)
    analyze_central_mechanisms(args.output, args.bootstrap_repetitions)
    analyze_selector(args.output, args.bootstrap_repetitions)
    analyze_priors(args.output, args.bootstrap_repetitions)
    audit_traffic_protocol(args.output)
    build_result_manifest(args.output)
    _write_json(
        {
            "analysis_only": True,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "source_families": SOURCE_FAMILY,
            "status": "complete",
        },
        args.output / "audit_status.json",
    )
    print(f"latentflow-submission-audit complete output={args.output}")


if __name__ == "__main__":
    main()
