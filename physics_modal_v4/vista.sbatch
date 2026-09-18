#!/bin/bash
#SBATCH -J latentflow
#SBATCH -p gh
#SBATCH -A IRI23021
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -o physics_modal_v4/logs/%x.%A_%a.out
#SBATCH -e physics_modal_v4/logs/%x.%A_%a.err

set -euo pipefail

REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
cd "$REPO_ROOT"
module purge
module load TACC gcc/14.2.0 cuda/12.6 python3/3.11.8

ENV_ROOT="${ADAWARP_VENV:-$REPO_ROOT/.venv-adawarp}"
[[ -x "$ENV_ROOT/bin/python" ]] || {
  echo "Missing Python environment: $ENV_ROOT" >&2
  exit 2
}
source "$ENV_ROOT/bin/activate"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python - <<'PY'
import torch
print("preflight", torch.__version__, torch.version.cuda, torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("gpu", torch.cuda.get_device_name(0))
PY

EXPERIMENT="${EXPERIMENT:?Set EXPERIMENT}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/TSLibrary/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/physics_modal_v4/results/final}"
DEVICE="${DEVICE:-cuda}"
SHARD_COUNT="${SHARD_COUNT:-16}"
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic)
HORIZONS=(96 192 336 720)
SEEDS=(42 43 44 45 46)
MODELS=(LatentFlow VPNet xCPD TimePro iTransformer TimeMixer DLinear)
COMPARISON_MODELS=(VPNet xCPD TimePro iTransformer TimeMixer DLinear)
VARIANTS=(single_scale_field multiscale_field adaptive_local_field component_gated_field linear_field_bank x_to_z single_stochastic multi_stochastic learned_inducing_elbo process_preserving fixed_inducing random_process_families no_sensor_priors continuation_frozen continuation_last1 continuation_last2 continuation_full no_nll)

run_latentflow_headline() {
  local index="$1" seed_index rest horizon_index dataset_index
  seed_index=$((index % 5))
  rest=$((index / 5))
  horizon_index=$((rest % 4))
  dataset_index=$((rest / 4))
  python -u -m physics_modal_v4.experiments.run     --mode headline --dataset "${DATASETS[$dataset_index]}"     --horizon "${HORIZONS[$horizon_index]}" --seed "${SEEDS[$seed_index]}"     --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
}

run_baseline() {
  local index="$1" model_index task_index horizon_index dataset_index model
  if [[ -n "${MODEL_ONLY:-}" ]]; then
    model="$MODEL_ONLY"
    task_index="$index"
  else
    model_index=$((index / 28))
    task_index=$((index % 28))
    model="${COMPARISON_MODELS[$model_index]}"
  fi
  horizon_index=$((task_index % 4))
  dataset_index=$((task_index / 4))
  python -u -m physics_modal_v4.experiments.baselines     --mode train --model "$model" --dataset "${DATASETS[$dataset_index]}"     --horizon "${HORIZONS[$horizon_index]}" --seed 42     --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
}

run_ablation() {
  local index="$1" variant_index task_index horizon_index dataset_index
  variant_index=$((index / 28))
  task_index=$((index % 28))
  horizon_index=$((task_index % 4))
  dataset_index=$((task_index / 4))
  python -u -m physics_modal_v4.experiments.run     --mode ablation --variant "${VARIANTS[$variant_index]}"     --dataset "${DATASETS[$dataset_index]}"     --horizon "${HORIZONS[$horizon_index]}" --seed 42     --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
}

run_efficiency() {
  local index="$1" model_index task_index horizon_index dataset_index model
  model_index=$((index / 28))
  task_index=$((index % 28))
  horizon_index=$((task_index % 4))
  dataset_index=$((task_index / 4))
  model="${MODELS[$model_index]}"
  if [[ "$model" == "LatentFlow" ]]; then
    python -u -m physics_modal_v4.experiments.run       --mode efficiency --dataset "${DATASETS[$dataset_index]}"       --horizon "${HORIZONS[$horizon_index]}" --seed 42       --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
  else
    python -u -m physics_modal_v4.experiments.baselines       --mode efficiency --model "$model" --dataset "${DATASETS[$dataset_index]}"       --horizon "${HORIZONS[$horizon_index]}" --seed 42       --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
  fi
}

run_shard() {
  local total="$1" function_name="$2" shard index
  shard="${SLURM_ARRAY_TASK_ID:?Array task ID required}"
  for ((index=shard; index<total; index+=SHARD_COUNT)); do
    echo "task experiment=$EXPERIMENT shard=$shard index=$index/$total"
    "$function_name" "$index"
  done
}

case "$EXPERIMENT" in
  smoke)
    python -u -m physics_modal_v4.experiments.run --mode smoke       --device "$DEVICE" --output-root "$OUTPUT_ROOT"
    python -u -m physics_modal_v4.experiments.baselines --mode smoke       --device "$DEVICE" --output-root "$OUTPUT_ROOT"
    ;;
  headline) run_shard 140 run_latentflow_headline ;;
  baselines) run_shard 168 run_baseline ;;
  baseline) run_shard 28 run_baseline ;;
  ablations) run_shard 504 run_ablation ;;
  efficiency) run_shard 196 run_efficiency ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT" >&2; exit 2 ;;
esac




