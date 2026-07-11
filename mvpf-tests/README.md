# MVPF final-model tests and evidence

This directory contains the authoritative training configuration, validation-checkpoint workflow, full-test re-evaluation, statistical analysis, and artifact audits for the final **MVPF** model.

All outputs remain under **mvpf-tests/results/**. Historical model files are imported but not modified.

## Final Code Path

~~~text
../adawarp_mvpf.py                         # multiscale field backbone
../adawarp_mvpf_plus.py                    # final model class and linear field bank
instrumented_pruned_mvpf.py                # authoritative final configuration and training
mvpf_pruned_instrumented_vista.sbatch      # 24-task Vista batch entry point
evaluate_standard_test_windows.py          # full standard-window evaluation
~~~

The final configuration disables prototype memory, frequency gating, adaptive shifts, and backbone trend decomposition. It retains multiscale patches, adaptive local radius mixing, field-component gating, the adaptive linear field bank, and the outer continuation gate.

## Protocol

~~~text
Datasets: ETTh1, ETTh2, ETTm2, Weather, Electricity, Traffic
Horizons: 96, 192, 336, 720
Methods: MVPF, DLinear, PatchTST, TimesNet, iTransformer, TimeMixer, FEDformer, VPNet
Lookback: 96
Seed: 42
~~~

VPNet comes from a custom runner and is treated as a descriptive comparison. The first seven methods form the strictly matched statistical set.

## Run the Final Model on Vista

~~~bash
cd $WORK/motion_code-master
source .venv-adawarp/bin/activate
mkdir -p logs

export ADAWARP_OUTPUT_ROOT=$WORK/motion_code-master/mvpf-tests/results/pruned_instrumented_<run_id>
sbatch mvpf-tests/mvpf_pruned_instrumented_vista.sbatch
~~~

Then evaluate the validation-best checkpoints on every standard test window:

~~~bash
python mvpf-tests/evaluate_standard_test_windows.py \
  --checkpoint-root mvpf-tests/results/pruned_instrumented_<run_id>/checkpoints \
  --output mvpf-tests/results/pruned_instrumented_<run_id>/metrics/standard_test_metrics.csv \
  --data-root TSLibrary/dataset \
  --device cuda \
  --force
~~~

## Matched VPNet Rerun

The original custom VPNet results cap test windows and do not select a validation-best checkpoint. The matched runner uses the final protocol: train-only normalization, 2,048 training windows, 1,024 validation windows, validation-best checkpointing, and every standard test window.

~~~bash
unset ADAWARP_OUTPUT_ROOT
export ADAWARP_REPO_ROOT=$WORK/motion_code-master
sbatch mvpf-tests/matched_vpnet_vista.sbatch
~~~

The output is written to **mvpf-tests/results/matched_vpnet_JOBID/**. After downloading it, merge only the VPNet rows into a new audited task table:

~~~bash
python mvpf-tests/merge_matched_vpnet.py \
  --vpnet mvpf-tests/results/matched_vpnet_JOBID/metrics/matched_vpnet_standard_test.csv
~~~

## Zero-Training Analysis

With the final task table already available:

~~~bash
python mvpf-tests/run_all.py
~~~

This runs:

~~~text
statistical_analysis.py  # average ranks, Friedman, paired Wilcoxon-Holm, effects
ablation_analysis.py     # paired task-level ablation analysis
artifact_audit.py        # checkpoint and prediction-artifact audit
svg_figures.py           # deterministic SVG evidence figures
paper_summary.py         # concise evidence summary
~~~

To reconstruct **results/metrics/task_metrics_24.csv** from the larger source result directories, run:

~~~bash
python mvpf-tests/collect_metrics.py
~~~

When **results/pruned_instrumented_820583/metrics/standard_test_metrics.csv** is present, the collector replaces historical MVPF aggregates with the final standard-window metrics while leaving baseline rows unchanged.

## Main Outputs

~~~text
results/metrics/task_metrics_24.csv
results/statistics/average_ranks.csv
results/statistics/friedman_tests.json
results/statistics/pairwise_wilcoxon_holm.csv
results/statistics/patchtst_relative_gains.csv
results/ablations/ablation_statistics.csv
results/audit/artifact_availability.json
results/paper_summary.md
results/pruned_instrumented_820583/metrics/standard_test_metrics.csv
results/pruned_instrumented_820583/checkpoints/
~~~

## Artifact Scope

Validation-best MVPF checkpoints and per-window intervention errors are available for all 24 tasks in the recorded final run. Matched raw forecast arrays for both MVPF and PatchTST are not both stored locally, so paired raw-window, per-channel, lead-time, and qualitative forecast comparisons require rerunning or retrieving those arrays.

All inferential results use dataset-horizon tasks from seed 42 as blocks. They measure breadth across tasks and do not estimate random-seed variability.