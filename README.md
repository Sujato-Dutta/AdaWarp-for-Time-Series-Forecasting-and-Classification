# MVPF: Multiscale Variate-Patch Fields for Long-Horizon Forecasting

This repository contains the final **MVPF** implementation, its validation-checkpointed long-horizon evaluation workflow, and the evidence scripts used to audit the reported results. The earlier AdaWarp-SGP short-series implementation and the original Motion Code baselines are retained for reproducibility, but MVPF is the primary long-horizon model.

> **Public model name:** MVPF  
> Some Python classes and files retain historical names such as **AdaWarpMVPFPlusForecaster** or **mvpf_plus**. These are implementation identifiers only. Results and external documentation should refer to the final model as **MVPF**.

## Final MVPF Reference

Use the following files for the final model:

| Purpose | Authoritative file |
| --- | --- |
| Multiscale variate-patch backbone | **adawarp_mvpf.py** |
| Final model class and adaptive linear field bank | **adawarp_mvpf_plus.py** |
| Final pruned configuration, training, validation checkpointing, and interventions | **mvpf-tests/instrumented_pruned_mvpf.py** |
| Vista batch entry point for all 24 tasks | **mvpf-tests/mvpf_pruned_instrumented_vista.sbatch** |
| Full standard-test-window re-evaluation | **mvpf-tests/evaluate_standard_test_windows.py** |
| Statistical and artifact audit suite | **mvpf-tests/run_all.py** |
| Fully matched VPNet rerun | **mvpf-tests/matched_vpnet_ltsf.py** |
| Matched VPNet Vista job | **mvpf-tests/matched_vpnet_vista.sbatch** |

The reusable Python class is:

~~~python
from adawarp_mvpf_plus import AdaWarpMVPFPlusForecaster
~~~

The final experiment configuration is defined by **model_config()** in **mvpf-tests/instrumented_pruned_mvpf.py**. It uses:

~~~text
patch lengths                 8, 16, 32
field width                   128
local field blocks            2
dropout                       0.05
adaptive local radii          enabled
field-component gate          enabled
adaptive linear field bank    enabled
prototype memory              disabled
frequency gate                disabled
adaptive patch shifts         disabled
backbone trend decomposition  disabled
forecast loss                 MSE
reconstruction weight         0.03
checkpoint selection          lowest validation MSE
~~~

Do not use the defaults of the historical generic runner as a substitute for this final configuration. In particular, **benchmark_adawarp_mvpf_plus_ltsf.py** remains useful as a generic runner, but **mvpf-tests/instrumented_pruned_mvpf.py** is the authoritative final experiment entry point.

## Model Overview

MVPF takes a regular-grid multivariate prefix **X** with shape **[batch, lookback, variables]** and:

1. normalizes every sample and variable using observed-prefix statistics;
2. embeds non-overlapping temporal patches at scales 8, 16, and 32;
3. constructs local variate-patch fields with input-conditioned radius mixing;
4. decodes a nonlinear continuation from direct and multiscale field components;
5. builds a complementary linear continuation from centered-linear, decomposition-linear, analytic-slope, and persistence experts;
6. combines components and routes through sample-conditioned simplex gates;
7. denormalizes the resulting horizon forecast.

Training minimizes forecast MSE plus an L1 reconstruction regularizer over the multiscale field. Every epoch is evaluated on validation windows, and test evaluation reloads the validation-best checkpoint.

## Benchmark Protocol

The final comparison contains 24 dataset-horizon tasks:

~~~text
Datasets: ETTh1, ETTh2, ETTm2, Weather, Electricity, Traffic
Horizons: 96, 192, 336, 720
Lookback: 96
Seed: 42
~~~

The matched baseline set contains DLinear, PatchTST, TimesNet, iTransformer, TimeMixer, FEDformer, and VPNet. All eight methods use the same six datasets, four horizons, lookback, seed, chronological split boundaries, validation-based checkpoint selection, and complete standard test windows. Model-specific architectures and optimization settings are preserved.

Datasets are expected under:

~~~text
TSLibrary/dataset/ETT-small/
TSLibrary/dataset/weather/
TSLibrary/dataset/electricity/
TSLibrary/dataset/traffic/
~~~

## The mvpf-tests Folder

**mvpf-tests/** is the self-contained final evidence and audit workspace. It does not modify the historical model files.

Important entry points:

~~~text
mvpf-tests/instrumented_pruned_mvpf.py       # final training and checkpoint interventions
mvpf-tests/evaluate_standard_test_windows.py # full standard-window test metrics
mvpf-tests/collect_metrics.py                # audited 8-method task table
mvpf-tests/statistical_analysis.py           # ranks, Friedman, Wilcoxon-Holm, effects
mvpf-tests/artifact_audit.py                 # checkpoint/raw-artifact availability
mvpf-tests/svg_figures.py                    # reproducible evidence figures
mvpf-tests/paper_summary.py                  # concise evidence summary
mvpf-tests/run_all.py                        # zero-training analysis pipeline
~~~

Final evidence is stored under:

~~~text
mvpf-tests/results/metrics/task_metrics_24.csv
mvpf-tests/results/statistics/
mvpf-tests/results/ablations/
mvpf-tests/results/figures/
mvpf-tests/results/audit/
mvpf-tests/results/pruned_instrumented_820583/
~~~

The final standard-window MVPF metrics are:

~~~text
mvpf-tests/results/pruned_instrumented_820583/metrics/standard_test_metrics.csv
~~~

## Local Setup

For CPU analysis on Windows:

~~~bat
python -m venv .venv-awp
.venv-awp\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-tacc-extra.txt
~~~

Run the zero-training evidence suite:

~~~bat
python mvpf-tests\run_all.py
~~~

This regenerates statistical tables, audits, figures, and the evidence summary from the final task table. To rebuild the task table from the larger source result directories first, run **python mvpf-tests/collect_metrics.py** explicitly.

## TACC Vista Setup

The final runs used TACC Vista with NVIDIA GH200 120 GB GPUs. A compatible environment is:

~~~bash
cd $WORK/motion_code-master
module load gcc/14.2.0 cuda/12.6 python3/3.11.8
python3 -m venv --system-site-packages .venv-adawarp
source .venv-adawarp/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tacc-extra.txt
~~~

CUDA should be checked inside an allocated GPU job:

~~~bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
~~~

## Running the Final 24 MVPF Tasks

From the repository root on a Vista login node:

~~~bash
source .venv-adawarp/bin/activate
mkdir -p logs

export ADAWARP_REPO_ROOT=$WORK/motion_code-master
export ADAWARP_OUTPUT_ROOT=$WORK/motion_code-master/mvpf-tests/results/pruned_instrumented_<run_id>

sbatch mvpf-tests/mvpf_pruned_instrumented_vista.sbatch
~~~

The job trains the six datasets at all four horizons, selects checkpoints using validation MSE, and writes checkpoints, per-window intervention metrics, and task summaries under the selected output root.

Re-evaluate those checkpoints on every standard test window:

~~~bash
python mvpf-tests/evaluate_standard_test_windows.py \
  --checkpoint-root mvpf-tests/results/pruned_instrumented_<run_id>/checkpoints \
  --output mvpf-tests/results/pruned_instrumented_<run_id>/metrics/standard_test_metrics.csv \
  --data-root TSLibrary/dataset \
  --device cuda \
  --force
~~~

## Historical and Supporting Code

These files are retained for traceability but are not the final MVPF experiment entry point:

~~~text
benchmark_adawarp_mvpf_plus_ltsf.py
benchmark_adawarp_mvpf_ltsf.py
benchmark_adawarp_mvpf_ablation.py
aggregate_adawarp_mvpf_plus_ablations.py
aggregate_adawarp_mvpf_ablations.py
~~~

Neural baseline runners remain available at:

~~~text
benchmark_tslibrary_neural_forecasting.py
benchmark_custom_neural_ltsf.py
adawarp_neural_baselines.py
scripts/tacc/run_ltsf_single_model.sh
scripts/tacc/submit_ltsf_by_model.sh
~~~

## Retained Short-Series Code

The repository also retains AdaWarp-SGP and the original Motion Code implementation:

~~~text
awp_motion_code.py                 # AdaWarp-SGP model
benchmark_awp_motion_code.py       # short classification/forecasting runner
awp_forecasting_utils.py           # prefix-validated forecasting utilities
motion_code.py                     # original Motion Code baseline
sparse_gp.py                       # sparse GP support
~~~

These modules are independent of the final MVPF long-horizon workflow.

## Reproducibility Checklist

1. Record the Git commit hash.
2. Confirm the six datasets and their standard train/validation/test splits.
3. Use lookback 96, horizons 96/192/336/720, and seed 42.
4. Use the final configuration from **mvpf-tests/instrumented_pruned_mvpf.py**.
5. Fit dataset normalization on the training split only.
6. Select the checkpoint using validation MSE only.
7. Evaluate the selected checkpoint on every standard test window.
8. Confirm that all seven baselines, including VPNet, use validation-best checkpoints and complete standard test windows.
9. Run **mvpf-tests/run_all.py** and inspect the generated audit files.

## License

MIT

## Authors

Sujato Dutta and Chandrajit Bajaj