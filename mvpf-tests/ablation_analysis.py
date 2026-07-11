"""Paired task-level analysis of the 20-task MVPF ablation evidence."""

from __future__ import annotations

import statistics

from common import ROOT, RESULTS, holm_adjust, paired_bootstrap_ci, read_csv, wilcoxon_signed_rank, write_csv


CORE = ("no_linear_field", "single_scale_16")
ALL = ("no_linear_field", "single_scale_16", "no_adaptive_shifts", "no_trend_residuals")


def main() -> None:
    source = read_csv(ROOT / "results" / "MVPF+ ablations" / "metrics" / "mvpf_final_ablation_deltas.csv")
    task_rows: list[dict[str, object]] = []
    statistics_rows: list[dict[str, object]] = []
    raw_p: dict[str, float] = {}
    cached: dict[str, dict[str, object]] = {}
    for ablation in ALL:
        selected = [row for row in source if row["ablation"] == ablation]
        if len(selected) != 20:
            raise RuntimeError(f"Expected 20 rows for {ablation}, found {len(selected)}")
        for metric in ("mse", "mae"):
            relative = [float(row[f"rel_delta_{metric}_pct"]) for row in selected]
            absolute = [float(row[metric]) - float(row[f"full_{metric}"]) for row in selected]
            test = wilcoxon_signed_rank(absolute)
            lower, upper = paired_bootstrap_ci(relative, statistic="median")
            wins = sum(value > 1e-12 for value in absolute)
            ties = sum(abs(value) <= 1e-12 for value in absolute)
            losses = sum(value < -1e-12 for value in absolute)
            key = f"{ablation}:{metric}"
            raw_p[key] = test["p_value"]
            cached[key] = {
                "ablation": ablation,
                "metric": metric,
                "tasks": len(selected),
                "ablation_worse": wins,
                "ties": ties,
                "ablation_better": losses,
                "mean_relative_degradation_pct": statistics.mean(relative),
                "median_relative_degradation_pct": statistics.median(relative),
                "median_bootstrap_ci_low_pct": lower,
                "median_bootstrap_ci_high_pct": upper,
                "wilcoxon_w": test["statistic"],
                "wilcoxon_p_raw": test["p_value"],
                "rank_biserial_component_benefit": test["rank_biserial"],
                "nonzero_pairs": int(test["n"]),
                "core_test": int(ablation in CORE),
            }
        for row in selected:
            task_rows.append(
                {
                    "ablation": ablation,
                    "dataset": row["dataset"],
                    "horizon": int(row["horizon"]),
                    "mse_relative_degradation_pct": float(row["rel_delta_mse_pct"]),
                    "mae_relative_degradation_pct": float(row["rel_delta_mae_pct"]),
                }
            )

    core_keys = [f"{ablation}:{metric}" for ablation in CORE for metric in ("mse", "mae")]
    core_adjusted = holm_adjust({key: raw_p[key] for key in core_keys})
    all_adjusted = holm_adjust(raw_p)
    for ablation in ALL:
        for metric in ("mse", "mae"):
            key = f"{ablation}:{metric}"
            row = cached[key]
            row["wilcoxon_p_holm_core"] = core_adjusted.get(key, "")
            row["wilcoxon_p_holm_all_diagnostic"] = all_adjusted[key]
            statistics_rows.append(row)
    write_csv(RESULTS / "ablations" / "ablation_task_deltas.csv", task_rows)
    write_csv(RESULTS / "ablations" / "ablation_statistics.csv", statistics_rows)
    for row in statistics_rows:
        print(
            f"{row['ablation']:<20} {str(row['metric']).upper()} "
            f"median={row['median_relative_degradation_pct']:+.3f}% "
            f"W/T/L={row['ablation_worse']}/{row['ties']}/{row['ablation_better']} "
            f"p={row['wilcoxon_p_raw']:.6g} rrb={row['rank_biserial_component_benefit']:+.3f}"
        )


if __name__ == "__main__":
    main()

