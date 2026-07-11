"""Write a concise, evidence-grounded summary for the MVPF-only paper."""

from __future__ import annotations

import json
from pathlib import Path

from common import RESULTS, read_csv


def main() -> None:
    ranks = read_csv(RESULTS / "statistics" / "average_ranks.csv")
    pairwise = read_csv(RESULTS / "statistics" / "pairwise_wilcoxon_holm.csv")
    horizon = read_csv(RESULTS / "statistics" / "horizon_robustness.csv")
    ablations = read_csv(RESULTS / "ablations" / "ablation_statistics.csv")
    friedman = json.loads((RESULTS / "statistics" / "friedman_tests.json").read_text(encoding="utf-8"))
    artifacts = json.loads((RESULTS / "audit" / "artifact_availability.json").read_text(encoding="utf-8"))
    mvpf = next(row for row in ranks if row["method"] == "MVPF")
    patch_mse = next(row for row in pairwise if row["metric"] == "mse" and row["baseline"] == "PatchTST")
    patch_mae = next(row for row in pairwise if row["metric"] == "mae" and row["baseline"] == "PatchTST")
    significant = {
        metric: [row["baseline"] for row in pairwise if row["metric"] == metric and int(row["significant_0p05"]) == 1]
        for metric in ("mse", "mae")
    }
    lines = [
        "# MVPF evidence summary",
        "",
        "## Matched 24-task evidence",
        "",
        f"- MVPF average rank is **{float(mvpf['mse_average_rank']):.3f} MSE** and **{float(mvpf['mae_average_rank']):.3f} MAE** (lower is better).",
        f"- It obtains **{mvpf['mse_outright_wins']}/24 outright MSE wins** and **{mvpf['mae_outright_wins']}/24 outright MAE wins**.",
        f"- Friedman: MSE chi-square={friedman['mse']['statistic']:.3f}, p={friedman['mse']['p_value']:.3g}; MAE chi-square={friedman['mae']['statistic']:.3f}, p={friedman['mae']['p_value']:.3g}.",
        f"- Versus PatchTST, MVPF wins/ties/loses **{patch_mse['wins']}/{patch_mse['ties']}/{patch_mse['losses']}** on MSE and **{patch_mae['wins']}/{patch_mae['ties']}/{patch_mae['losses']}** on MAE.",
        f"- PatchTST comparison is not significant after Holm correction: MSE adjusted p={float(patch_mse['wilcoxon_p_holm']):.3f}; MAE adjusted p={float(patch_mae['wilcoxon_p_holm']):.3f}.",
        f"- Significant Holm-corrected MSE comparisons: {', '.join(significant['mse'])}.",
        f"- Significant Holm-corrected MAE comparisons: {', '.join(significant['mae'])}.",
        "",
        "The defensible claim is: **MVPF achieves the lowest mean MSE and MAE and the most task wins; it has the best MSE average rank and ties the best MAE average rank in the eight-method table, while remaining statistically tied with PatchTST and TimeMixer under task-level Holm-Wilcoxon testing.**",
        "",
        "## Horizon behaviour",
        "",
    ]
    for row in horizon:
        lines.append(
            f"- {row['metric'].upper()} h={row['horizon']}: median PatchTST-relative improvement {float(row['median_relative_improvement_pct']):+.2f}% ({row['wins']}/{row['ties']}/{row['losses']} W/T/L)."
        )
    lines.extend(["", "## Ablations", ""])
    for row in ablations:
        if int(row["core_test"]) == 1:
            lines.append(
                f"- {row['ablation']} {row['metric'].upper()}: median degradation {float(row['median_relative_degradation_pct']):+.3f}%, effect={float(row['rank_biserial_component_benefit']):+.3f}, core-Holm p={float(row['wilcoxon_p_holm_core']):.3f}."
            )
    lines.extend(
        [
            "",
            "The ablations support positive *effect direction* for the multiscale representation and linear field bank, but do not establish corrected task-level significance. Adaptive shifts are operationally inactive in many final-model tasks because prototype memory is disabled, and removing trend/residual decomposition slightly improves the 20-task average; neither should be presented as a demonstrated source of gains.",
            "",
            "## Artifact limitation",
            "",
            f"- Locally available MVPF raw task predictions: {artifacts['mvpf_task_predictions_available']}/{artifacts['mvpf_task_predictions_required']}.",
            f"- Locally available PatchTST raw task predictions: {artifacts['patchtst_task_predictions_available']}/{artifacts['patchtst_task_predictions_required']}.",
            f"- Locally available validation-best MVPF checkpoints: {len(artifacts['mvpf_checkpoints'])}.",
            "- The checkpoints support model re-evaluation and gate interventions. Matched raw forecast arrays for MVPF and PatchTST are still required for paired per-window, per-channel, lead-time, and qualitative forecast comparisons.",
            "",
            "## Scope statement",
            "",
            "All significance tests above measure consistency across 24 dataset-horizon tasks from seed 42. They do not estimate training-seed variability and must not be described as a substitute for multi-seed experiments.",
        ]
    )
    path = RESULTS / "paper_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()

