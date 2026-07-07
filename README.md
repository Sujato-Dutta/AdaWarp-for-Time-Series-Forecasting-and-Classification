# AdaWarp - Adaptive Warped Field models for Short and Long Horizon Time-series Forecasting.

This repository contains the AdaWarp method code. It lives beside the original motion code repository. The family has two models:

| Name | Best regime | Main code |
| --- | --- | --- |
| **AdaWarp-SGP** | short irregular classification and prefix-to-suffix forecasting | `awp_motion_code.py` |
| **AdaWarp-MVPF** | long-term multivariate forecasting | `adawarp_mvpf_plus.py` |

Both models follow the same abstraction: learn adaptive prototype fields from observed trajectories, align the observed prefix, and use the aligned field to score classes or continue the signal. The two implementations differ because short irregular UCR-style trajectories and long regular-grid multivariate forecasting tasks require different inductive biases.

Some files retain historical implementation suffixes such as `plus`. These are code-level labels only.

## AdaWarp-SGP

AdaWarp-SGP is the short-protocol model. It is used for classification retention and short prefix-to-suffix forecasting.

Main implementation:

```text
awp_motion_code.py
```

Key supporting files:

```text
awp_datasets.py                     # short-protocol dataset loading
awp_forecasting_utils.py            # prefix-to-suffix continuation heads and metrics
benchmark_awp_motion_code.py        # AdaWarp-SGP classification/forecasting runner
benchmark_adawarp_protocol_baselines.py  # short-protocol baselines and ablations
aggregate_awp_results.py            # classification aggregation helpers
aggregate_awp_forecasts.py          # forecasting aggregation helpers
```

Method summary:

- sparse Gaussian-process class prototypes;
- sample-adaptive residual codes;
- bounded temporal warping and affine alignment;
- uncertainty-aware class scoring;
- prefix-validated continuation heads for short forecasting.

## AdaWarp-MVPF

AdaWarp-MVPF is the final long-horizon forecasting model. It should be reported as **AdaWarp-MVPF** in papers, figures, and tables.

Main implementation:

```text
adawarp_mvpf_plus.py
benchmark_adawarp_mvpf_plus_ltsf.py
```

The `plus` suffix is only the implementation filename. It denotes the final validation-checkpointed MVPF runner in this repository.

Key supporting files:

```text
benchmark_adawarp_mvpf_plus_ltsf.py      # final AdaWarp-MVPF LTSF runner
benchmark_adawarp_mvpf_ablation.py       # MVPF ablation runner
aggregate_adawarp_mvpf_plus_ablations.py # final MVPF ablation aggregation
benchmark_custom_neural_ltsf.py          # custom neural LTSF baselines
benchmark_tslibrary_neural_forecasting.py # TSLibrary LTSF bridge
adawarp_neural_baselines.py              # repo-native neural baseline wrappers
```

Retained diagnostic files:

```text
adawarp_mvpf.py
benchmark_adawarp_mvpf_ltsf.py
aggregate_adawarp_mvpf_ablations.py
```

These are kept for reproducibility of earlier diagnostics. The final long-horizon model is implemented by `adawarp_mvpf_plus.py`.

## Original Motion Code Baseline

The original Motion Code code remains available for matched baseline reruns:

```text
motion_code.py
motion_code_utils.py
sparse_gp.py
benchmark_motion_code_classification.py
benchmark_motion_code_forecasting.py
```

Use these for baseline reproduction only. New AdaWarp experiments should use the AdaWarp runners above.

## Configs

```text
configs/motioncode_protocol/default.json       # short prefix-to-suffix protocol
configs/classification_retention/default.json  # classification protocol
configs/ltsf_main5/default.json                # ETTh1/ETTh2/Weather/Electricity/Traffic LTSF protocol
configs/ablations/default.json                 # short-protocol ablation plan
configs/baselines/default.json                 # baseline registry
```

## TACC Scripts

Environment and checks:

```text
scripts/tacc/setup_env.sh
scripts/tacc/check_motion_code_env.sh
```

AdaWarp-SGP:

```text
scripts/tacc/run_motioncode_protocol.sh
scripts/tacc/run_classification_retention.sh
scripts/tacc/run_ablation_suite.sh
scripts/tacc/aggregate_all.sh
```

Long-term baselines:

```text
scripts/tacc/run_ltsf_single_model.sh
scripts/tacc/submit_ltsf_by_model.sh
scripts/tacc/aggregate_ltsf_by_model.py
```

Final AdaWarp-MVPF:

```text
scripts/tacc/run_adawarp_mvpf_plus_ltsf.sh
scripts/tacc/submit_adawarp_mvpf_plus_by_dataset.sh
scripts/tacc/run_adawarp_mvpf_plus_ablation.sh
scripts/tacc/submit_adawarp_mvpf_plus_ablations.sh
```

Slurm templates:

```text
slurm/adawarp_mvpf_plus_ltsf_vista.sbatch
slurm/adawarp_mvpf_plus_ablation_vista.sbatch
slurm/ltsf_single_model_vista.sbatch
```

## Data Layout

Datasets are not tracked by Git. Expected layout:

```text
data/                         # short-protocol datasets
TSLibrary/dataset/             # TSLibrary datasets
TSLibrary/dataset/ETT-small/    # ETTh1, ETTh2, ETTm1, ETTm2 when used
TSLibrary/dataset/weather/      # weather.csv
TSLibrary/dataset/LR_Datasets/  # long-range CSV copies used by some runners
```

Large generated artifacts are ignored:

```text
results/
out/
logs/
paper/
TSLibrary/results/
TSLibrary/checkpoints/
```

## Local Setup

For local CPU checks on Windows:

```bat
python -m venv .venv-awp
.venv-awp\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-tacc-extra.txt
```

Local CPU is useful for imports, aggregation, and small smoke tests. Full long-term neural runs are intended for TACC or another GPU cluster.

## TACC Setup

Do not blindly replace cluster-provided CUDA, Torch, NumPy, or JAX packages. On Vista, the development environment used this pattern:

```bash
cd $WORK/motion_code-master
module load gcc/14.2.0 cuda/12.6 python3/3.11.8
python3 -m venv --system-site-packages .venv-adawarp
source .venv-adawarp/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tacc-extra.txt
```

Check CUDA from an allocated GPU job, not only from a login node:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda build', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Running AdaWarp-SGP

Short prefix-to-suffix forecasting:

```bash
export ADAWARP_OUTPUT_ROOT=$WORK/motion_code-master/results/<run_name>
export ADAWARP_DEVICE=cuda
export ADAWARP_EPOCHS=50
export ADAWARP_STEPS_PER_EPOCH=4
export ADAWARP_SEEDS="42 43 44 45 46"
export ADAWARP_PREFIX_FRACTIONS="0.8 0.6"
bash scripts/tacc/run_motioncode_protocol.sh
bash scripts/tacc/aggregate_all.sh
```

Classification:

```bash
export ADAWARP_OUTPUT_ROOT=$WORK/motion_code-master/results/<run_name>
export ADAWARP_DEVICE=cuda
export ADAWARP_SEEDS="42"
bash scripts/tacc/run_classification_retention.sh
bash scripts/tacc/aggregate_all.sh
```

Matched original Motion Code reruns require JAX. If JAX is unavailable or unstable, document the omission and rerun once the environment is fixed.

## Running AdaWarp-MVPF

Long-term forecasting uses ETTh1, ETTh2, Weather, Electricity, and Traffic with horizons 96, 192, 336, and 720.

Run final AdaWarp-MVPF by dataset:

```bash
export ADAWARP_OUTPUT_ROOT=$WORK/motion_code-master/results/<run_name>
export ADAWARP_DEVICE=cuda
bash scripts/tacc/submit_adawarp_mvpf_plus_by_dataset.sh
```

Run matched neural LTSF baselines:

```bash
export ADAWARP_LTSF_MODELS="DLinear PatchTST TimesNet iTransformer TimeMixer FEDformer VPNet"
bash scripts/tacc/submit_ltsf_by_model.sh
```

Aggregate LTSF results:

```bash
bash scripts/tacc/aggregate_all.sh
python scripts/tacc/aggregate_ltsf_by_model.py --root results/ltsf_5
```

## AdaWarp-MVPF Ablations

Run the final AdaWarp-MVPF ablations:

```bash
bash scripts/tacc/submit_adawarp_mvpf_plus_ablations.sh
python aggregate_adawarp_mvpf_plus_ablations.py \
  --root results/vista_ltsf_adawarp_mvpf_final_ablations \
  --full-root results/ltsf_5/MVPF_Cpt
```

Ablations should be tied to the final AdaWarp-MVPF implementation when used in the paper.

## Independent Reproducibility Checklist

1. Confirm the Git commit hash.
2. Confirm the TACC working directory or local workspace.
3. Confirm loaded modules and Python environment.
4. Confirm that required datasets exist in the expected layout.
5. Confirm that short-protocol preprocessing uses only training data and each evaluated trajectory prefix.
6. Confirm that suffix values are not used in any fitted object for prefix-to-suffix forecasting.
7. Confirm that AdaWarp-MVPF checkpoint selection uses validation data only.
8. Confirm that result tables are generated from matched CSV/JSON outputs under `results/`.

## License
MIT

## Author
Sujato Dutta | AI Engineer | Researcher

