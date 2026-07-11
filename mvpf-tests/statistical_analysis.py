"""Task-level ranks and non-parametric tests for the 24 matched LTSF tasks."""

from __future__ import annotations

import statistics

from common import (
    DATASETS,
    HORIZONS,
    METHODS,
    METRICS,
    RESULTS,
    friedman_test,
    holm_adjust,
    rank_values,
    read_csv,
    wilcoxon_signed_rank,
    wins_ties_losses,
    write_csv,
    write_json,
)


def main() -> None:
    source = read_csv(RESULTS / "metrics" / "task_metrics_24.csv")
    matched_methods = tuple(METHODS)
    lookup = {
        (row["dataset"], int(row["horizon"]), row["method"]): row
        for row in source
    }
    rank_rows: list[dict[str, object]] = []
    average_rows = {method: {"method": method} for method in METHODS}
    omnibus: dict[str, object] = {}
    pairwise_rows: list[dict[str, object]] = []
    relative_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []

    for metric in METRICS:
        blocks = []
        per_method_ranks = {method: [] for method in METHODS}
        outright_wins = {method: 0 for method in METHODS}
        tied_best = {method: 0 for method in METHODS}
        for dataset in DATASETS:
            for horizon in HORIZONS:
                values = [float(lookup[(dataset, horizon, method)][metric]) for method in METHODS]
                ranks = rank_values(values)
                blocks.append(values)
                minimum = min(values)
                best = [index for index, value in enumerate(values) if value == minimum]
                for index, method in enumerate(METHODS):
                    per_method_ranks[method].append(ranks[index])
                    if index in best:
                        if len(best) == 1:
                            outright_wins[method] += 1
                        else:
                            tied_best[method] += 1
                    rank_rows.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "metric": metric,
                            "method": method,
                            "value": values[index],
                            "rank": ranks[index],
                            "is_outright_best": int(index in best and len(best) == 1),
                            "is_tied_best": int(index in best and len(best) > 1),
                        }
                    )
        matched_blocks = [
            [float(lookup[(dataset, horizon, method)][metric]) for method in matched_methods]
            for dataset in DATASETS
            for horizon in HORIZONS
        ]
        omnibus[metric] = friedman_test(matched_blocks)
        omnibus[metric]["blocks"] = len(matched_blocks)
        omnibus[metric]["treatments"] = len(matched_methods)
        for method in METHODS:
            average_rows[method][f"{metric}_average_rank"] = statistics.mean(per_method_ranks[method])
            average_rows[method][f"{metric}_outright_wins"] = outright_wins[method]
            average_rows[method][f"{metric}_tied_best"] = tied_best[method]

        raw_p: dict[str, float] = {}
        temporary: dict[str, dict[str, object]] = {}
        mvpf_values = [
            float(lookup[(dataset, horizon, "MVPF")][metric])
            for dataset in DATASETS
            for horizon in HORIZONS
        ]
        for baseline in matched_methods:
            if baseline == "MVPF":
                continue
            baseline_values = [
                float(lookup[(dataset, horizon, baseline)][metric])
                for dataset in DATASETS
                for horizon in HORIZONS
            ]
            differences = [baseline_value - mvpf_value for baseline_value, mvpf_value in zip(baseline_values, mvpf_values)]
            test = wilcoxon_signed_rank(differences)
            wins, ties, losses = wins_ties_losses(mvpf_values, baseline_values)
            relative = [100.0 * difference / baseline_value for difference, baseline_value in zip(differences, baseline_values)]
            raw_p[baseline] = test["p_value"]
            temporary[baseline] = {
                "metric": metric,
                "baseline": baseline,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "median_relative_improvement_pct": statistics.median(relative),
                "mean_relative_improvement_pct": statistics.mean(relative),
                "wilcoxon_w": test["statistic"],
                "wilcoxon_p_raw": test["p_value"],
                "rank_biserial": test["rank_biserial"],
                "nonzero_pairs": int(test["n"]),
            }
        adjusted = holm_adjust(raw_p)
        for baseline in matched_methods:
            if baseline == "MVPF":
                continue
            row = temporary[baseline]
            row["wilcoxon_p_holm"] = adjusted[baseline]
            row["significant_0p05"] = int(adjusted[baseline] < 0.05)
            pairwise_rows.append(row)

        for dataset in DATASETS:
            for horizon in HORIZONS:
                mvpf = float(lookup[(dataset, horizon, "MVPF")][metric])
                patch = float(lookup[(dataset, horizon, "PatchTST")][metric])
                relative_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "metric": metric,
                        "mvpf": mvpf,
                        "patchtst": patch,
                        "relative_improvement_pct": 100.0 * (patch - mvpf) / patch,
                    }
                )
        for horizon in HORIZONS:
            selected = [
                float(row["relative_improvement_pct"])
                for row in relative_rows
                if row["metric"] == metric and int(row["horizon"]) == horizon
            ]
            horizon_rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "mean_relative_improvement_pct": statistics.mean(selected),
                    "median_relative_improvement_pct": statistics.median(selected),
                    "wins": sum(value > 0 for value in selected),
                    "ties": sum(value == 0 for value in selected),
                    "losses": sum(value < 0 for value in selected),
                }
            )

    average_rank_rows = list(average_rows.values())
    average_rank_rows.sort(key=lambda row: float(row["mse_average_rank"]))
    write_csv(RESULTS / "statistics" / "task_ranks.csv", rank_rows)
    write_csv(RESULTS / "statistics" / "average_ranks.csv", average_rank_rows)
    write_csv(RESULTS / "statistics" / "pairwise_wilcoxon_holm.csv", pairwise_rows)
    write_csv(RESULTS / "statistics" / "patchtst_relative_gains.csv", relative_rows)
    write_csv(RESULTS / "statistics" / "horizon_robustness.csv", horizon_rows)
    write_json(RESULTS / "statistics" / "friedman_tests.json", omnibus)
    print("Friedman tests")
    for metric in METRICS:
        result = omnibus[metric]
        print(f"  {metric.upper()}: chi2={result['statistic']:.6f}, df={int(result['treatments']) - 1}, p={result['p_value']:.6g}")
    print("Average ranks")
    for row in average_rank_rows:
        print(f"  {row['method']:<13} MSE={row['mse_average_rank']:.3f} MAE={row['mae_average_rank']:.3f}")


if __name__ == "__main__":
    main()

