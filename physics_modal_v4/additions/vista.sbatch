#!/bin/bash
#SBATCH -J lf_add
#SBATCH -p gh
#SBATCH -A IRI23021
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -o physics_modal_v4/additions/logs/%x.%A_%a.out
#SBATCH -e physics_modal_v4/additions/logs/%x.%A_%a.err

set -euo pipefail

REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
cd "$REPO_ROOT"
module purge
module load TACC gcc/14.2.0 cuda/12.6 python3/3.11.8
ENV_ROOT="${ADAWARP_VENV:-$REPO_ROOT/.venv-adawarp}"
[[ -x "$ENV_ROOT/bin/python" ]] || { echo "Missing environment: $ENV_ROOT" >&2; exit 2; }
source "$ENV_ROOT/bin/activate"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python - <<'PY'
import torch
print("preflight", torch.__version__, torch.version.cuda, torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("gpu", torch.cuda.get_device_name(0))
PY

EXPERIMENT="${EXPERIMENT:?Set EXPERIMENT}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/TSLibrary/dataset}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/physics_modal_v4/results/final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/physics_modal_v4/additions/results}"
DEVICE="${DEVICE:-cuda}"
SHARD_COUNT="${SHARD_COUNT:-16}"
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic)
HORIZONS=(96 192 336 720)
SEEDS=(42 43 44 45 46)
CENTRAL_VARIANTS=(final_no_exchange x_to_z early_fusion no_multiscale no_linear_bank)
SYNTHETIC_SETTINGS=(independent interacting)
SYNTHETIC_VARIANTS=(early_fusion independent_branches process_preserving full)

common=(--data-root "$DATA_ROOT" --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE")

run_timepro() {
  local index="$1" seed_index task_index dataset_index horizon_index
  seed_index=$((index / 28)); task_index=$((index % 28))
  dataset_index=$((task_index / 4)); horizon_index=$((task_index % 4))
  python -u -m physics_modal_v4.additions.run --mode timepro \
    --dataset "${DATASETS[$dataset_index]}" --horizon "${HORIZONS[$horizon_index]}" \
    --seed "$((43 + seed_index))" "${common[@]}"
}

run_synthetic() {
  local index="$1" setting_index rest variant_index seed_index
  seed_index=$((index % 5)); rest=$((index / 5))
  variant_index=$((rest % 4)); setting_index=$((rest / 4))
  python -u -m physics_modal_v4.additions.run --mode synthetic \
    --setting "${SYNTHETIC_SETTINGS[$setting_index]}" \
    --variant "${SYNTHETIC_VARIANTS[$variant_index]}" --seed "${SEEDS[$seed_index]}" "${common[@]}"
}

run_central() {
  local index="$1" dataset_index horizon_index rest variant_index seed_index
  seed_index=$((index % 5)); rest=$((index / 5))
  variant_index=$((rest % 5)); rest=$((rest / 5))
  horizon_index=$((rest % 4)); dataset_index=$((rest / 4))
  python -u -m physics_modal_v4.additions.run --mode central \
    --dataset "${DATASETS[$dataset_index]}" --horizon "${HORIZONS[$horizon_index]}" \
    --variant "${CENTRAL_VARIANTS[$variant_index]}" --seed "${SEEDS[$seed_index]}" "${common[@]}"
}

run_missing() {
  local index="$1" dataset_index horizon_index dataset
  dataset_index=$((index / 4)); horizon_index=$((index % 4))
  dataset=$([[ "$dataset_index" == 0 ]] && echo Weather || echo Traffic)
  python -u -m physics_modal_v4.additions.run --mode missing \
    --dataset "$dataset" --horizon "${HORIZONS[$horizon_index]}" "${common[@]}"
}

run_process() {
  local dataset
  dataset=$([[ "$1" == 0 ]] && echo Weather || echo Traffic)
  python -u -m physics_modal_v4.additions.run --mode process \
    --dataset "$dataset" --horizon 720 "${common[@]}"
}

run_vpnet_control() {
  local index="$1" dataset_index horizon_index
  dataset_index=$((index / 4)); horizon_index=$((index % 4))
  python -u -m physics_modal_v4.additions.run --mode vpnet_control \
    --dataset "${DATASETS[$dataset_index]}" --horizon "${HORIZONS[$horizon_index]}" "${common[@]}"
}

run_shard() {
  local total="$1" function_name="$2" shard index
  shard="${SLURM_ARRAY_TASK_ID:?Array task ID required}"
  for ((index=shard; index<total; index+=SHARD_COUNT)); do
    echo "addition-task experiment=$EXPERIMENT shard=$shard index=$index/$total"
    "$function_name" "$index"
  done
}

case "$EXPERIMENT" in
  smoke) python -u -m physics_modal_v4.additions.run --mode smoke "${common[@]}" ;;
  timepro) run_shard 112 run_timepro ;;
  synthetic) run_shard 40 run_synthetic ;;
  central) run_shard 700 run_central ;;
  missing) run_shard 8 run_missing ;;
  process) run_shard 2 run_process ;;
  vpnet_control) run_shard 28 run_vpnet_control ;;
  scaling) python -u -m physics_modal_v4.additions.run --mode scaling "${common[@]}" ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT" >&2; exit 2 ;;
esac
