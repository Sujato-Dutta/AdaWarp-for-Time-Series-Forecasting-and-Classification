#!/bin/bash
#SBATCH -J lf_enhance
#SBATCH -p gh
#SBATCH -A IRI23021
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -o physics_modal_v4/enhancements/logs/%x.%A_%a.out
#SBATCH -e physics_modal_v4/enhancements/logs/%x.%A_%a.err

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
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/physics_modal_v4/results/final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/physics_modal_v4/enhancements/results}"
DEVICE="${DEVICE:-cuda}"
SHARD_COUNT="${SHARD_COUNT:-16}"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic)
HORIZONS=(96 192 336 720)
SEEDS=(42 43 44 45 46)
FACTORIAL_SEEDS=(42 43 44)
COMPARATOR_SEEDS=(43 44 45 46)
PRIOR_SEEDS=(42 43 44)
PRIOR_VARIANTS=(no_sensor_group_priors permuted_process_families neutral_process_scales)
PERMUTATIONS=(1729 3253)
EXTRA_PERMUTATIONS=(4817 7919 104729)
DATA_BUDGETS=(128 512 2048 all)
TIMEPRO_DATA_BUDGETS=(128 512 all)
DIAGNOSTIC_DATASETS=(ETTh1 ETTh1 ETTm1 ETTm1 Weather Electricity Traffic)
DIAGNOSTIC_HORIZONS=(96 720 336 720 720 336 192)

common=(
  --data-root "$DATA_ROOT"
  --source-root "$SOURCE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
)

decode_task() {
  local task_index="$1"
  DATASET="${DATASETS[$((task_index / 4))]}"
  HORIZON="${HORIZONS[$((task_index % 4))]}"
}

run_central() {
  local index="$1" seed_index task_index
  seed_index=$((index % 5))
  task_index=$((index / 5))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.run --mode central-task \
    --dataset "$DATASET" --horizon "$HORIZON" --seed "${SEEDS[$seed_index]}" \
    "${common[@]}"
}

run_factorial() {
  local index="$1" seed_index task_index
  seed_index=$((index % 3))
  task_index=$((index / 3))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.run --mode factorial-task \
    --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${FACTORIAL_SEEDS[$seed_index]}" "${common[@]}"
}

run_repair_central() {
  local index="$1" seed_index task_index
  seed_index=$((index % 5))
  task_index=$((index / 5))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.run \
    --mode selector-repair-central-task \
    --dataset "$DATASET" --horizon "$HORIZON" --seed "${SEEDS[$seed_index]}" \
    "${common[@]}"
}

run_repair_factorial() {
  local index="$1" seed_index task_index
  seed_index=$((index % 3))
  task_index=$((index / 3))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.run \
    --mode selector-repair-factorial-task \
    --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${FACTORIAL_SEEDS[$seed_index]}" "${common[@]}"
}

run_strongest() {
  local index="$1" seed_index task_index
  seed_index=$((index % 4))
  task_index=$((index / 4))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.run --mode strongest-task \
    --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${COMPARATOR_SEEDS[$seed_index]}" "${common[@]}"
}

run_full_data() {
  local index="$1"
  decode_task "$index"
  python -u -m physics_modal_v4.enhancements.run --mode full-data-task \
    --dataset "$DATASET" --horizon "$HORIZON" --seed 42 "${common[@]}"
}

run_data_curve() {
  local index=$1 budget_index task_index
  budget_index=$((index / 28))
  task_index=$((index % 28))
  decode_task $task_index
  python -u -m physics_modal_v4.enhancements.run --mode data-budget-task \
    --dataset $DATASET --horizon $HORIZON --seed 42 \
    --train-budget ${DATA_BUDGETS[$budget_index]} ${common[@]}
}

run_timepro_data_curve() {
  local index=$1 budget_index task_index
  budget_index=$((index / 28))
  task_index=$((index % 28))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.run --mode timepro-data-budget-task \
    --dataset "$DATASET" --horizon "$HORIZON" --seed 42 \
    --train-budget "${TIMEPRO_DATA_BUDGETS[$budget_index]}" "${common[@]}"
}

run_capacity() {
  local index="$1"
  decode_task "$index"
  python -u -m physics_modal_v4.enhancements.run --mode capacity-task \
    --dataset "$DATASET" --horizon "$HORIZON" --seed 42 "${common[@]}"
}

run_channel_order() {
  local index="$1" dataset_index permutation_index
  permutation_index=$((index % 2))
  dataset_index=$((index / 2))
  DATASET=$([[ "$dataset_index" == 0 ]] && echo Electricity || echo Traffic)
  python -u -m physics_modal_v4.enhancements.run --mode channel-order \
    --dataset "$DATASET" --horizon 720 --seed 42 \
    --permutation-seed "${PERMUTATIONS[$permutation_index]}" "${common[@]}"
}

run_channel_order_expanded() {
  local index="$1" dataset_index permutation_index
  permutation_index=$((index % 3))
  dataset_index=$((index / 3))
  DATASET=$([[ "$dataset_index" == 0 ]] && echo Electricity || echo Traffic)
  python -u -m physics_modal_v4.enhancements.run --mode channel-order-expanded \
    --dataset "$DATASET" --horizon 720 --seed 42 \
    --permutation-seed "${EXTRA_PERMUTATIONS[$permutation_index]}" "${common[@]}"
}

run_diagnostics() {
  local index="$1"
  python -u -m physics_modal_v4.enhancements.run --mode diagnostics-task \
    --dataset "${DIAGNOSTIC_DATASETS[$index]}" \
    --horizon "${DIAGNOSTIC_HORIZONS[$index]}" --seed 42 "${common[@]}"
}

run_synthetic() {
  local index="$1" seed_index scenario_index
  seed_index=$((index % 5))
  scenario_index=$((index / 5))
  local scenarios=(reference interacting period_drift trend_curvature long_ou abrupt_noisy long_horizon)
  python -u -m physics_modal_v4.enhancements.mechanisms --mode synthetic \
    --scenario "${scenarios[$scenario_index]}" --seed "${SEEDS[$seed_index]}" \
    --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
}

run_real_scaling() {
  local index="$1"
  local counts=(32 64 128 256 512 862)
  python -u -m physics_modal_v4.enhancements.mechanisms --mode channel-scaling \
    --channel-count "${counts[$index]}" --seed 42 \
    --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
}

run_prior_control() {
  local index="$1" variant_index seed_index task_index
  variant_index=$((index % 3))
  seed_index=$(((index / 3) % 3))
  task_index=$((index / 9))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.final_audits \
    --mode prior-control-task --variant "${PRIOR_VARIANTS[$variant_index]}" \
    --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${PRIOR_SEEDS[$seed_index]}" "${common[@]}"
}

run_checkpoint_audits() {
  local index="$1" seed_index task_index
  seed_index=$((index % 5))
  task_index=$((index / 5))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.final_audits \
    --mode reference-audit-task --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${SEEDS[$seed_index]}" "${common[@]}"
  python -u -m physics_modal_v4.enhancements.final_audits \
    --mode fusion-audit-task --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${SEEDS[$seed_index]}" "${common[@]}"
}

run_submission_five_seed() {
  local index="$1" mode="$2" seed_index task_index
  seed_index=$((index % 5))
  task_index=$((index / 5))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.submission_checks \
    --mode "$mode" --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${SEEDS[$seed_index]}" "${common[@]}"
}

run_submission_seed42() {
  local index="$1" mode="$2"
  decode_task "$index"
  python -u -m physics_modal_v4.enhancements.submission_checks \
    --mode "$mode" --dataset "$DATASET" --horizon "$HORIZON" \
    --seed 42 "${common[@]}"
}

run_provenance() { run_submission_five_seed "$1" provenance; }
run_selector_check() { run_submission_five_seed "$1" selector; }
run_numerical_check() { run_submission_seed42 "$1" numerical; }
run_fusion_profile() { run_submission_seed42 "$1" fusion-profile; }

run_matched_control() {
  local index="$1"
  decode_task "$index"
  python -u -m physics_modal_v4.enhancements.run --mode matched-control-task \
    --dataset "$DATASET" --horizon "$HORIZON" --seed 42 "${common[@]}"
}


run_matched_control_multiseed() {
  local index="$1" seed_index task_index
  seed_index=$((index % 4))
  task_index=$((index / 4))
  decode_task "$task_index"
  python -u -m physics_modal_v4.enhancements.run \
    --mode matched-control-multiseed-task \
    --dataset "$DATASET" --horizon "$HORIZON" \
    --seed "${COMPARATOR_SEEDS[$seed_index]}" "${common[@]}"
}
run_shard() {
  local total="$1" function_name="$2" shard index
  shard="${SLURM_ARRAY_TASK_ID:?Array task ID required}"
  for ((index=shard; index<total; index+=SHARD_COUNT)); do
    echo "enhancement-task experiment=$EXPERIMENT shard=$shard index=$index/$total"
    "$function_name" "$index"
  done
}

case "$EXPERIMENT" in
  submission_smoke)
    python -u -m physics_modal_v4.enhancements.submission_checks --mode smoke "${common[@]}"
    python -u -m physics_modal_v4.enhancements.run --mode smoke "${common[@]}"
    ;;
  traffic_replay)
    python -u -m physics_modal_v4.enhancements.submission_checks --mode traffic-replay --dataset Traffic --horizon 720 --seed 42 "${common[@]}"
    ;;
  submission_provenance) run_shard 140 run_provenance ;;
  submission_selector) run_shard 140 run_selector_check ;;
  submission_numerical) run_shard 28 run_numerical_check ;;
  submission_fusion_profile) run_shard 28 run_fusion_profile ;;
  matched_controls) run_shard 28 run_matched_control ;;
  matched_controls_multiseed) run_shard 112 run_matched_control_multiseed ;;
  submission_aggregate)
    python -u -m physics_modal_v4.enhancements.submission_checks --mode aggregate "${common[@]}"
    ;;
  remaining_aggregate)
    python -u -m physics_modal_v4.enhancements.submission_checks \
      --mode remaining-aggregate "${common[@]}"
    ;;
  remaining_smoke)
    python -u -m physics_modal_v4.enhancements.run \
      --mode matched-control-multiseed-task \
      --dataset ETTh1 --horizon 96 --seed 43 "${common[@]}"
    python -u -m physics_modal_v4.enhancements.run \
      --mode timepro-data-budget-task \
      --dataset ETTh1 --horizon 96 --seed 42 --train-budget 128 \
      "${common[@]}"
    echo "LATENTFLOW-REMAINING-SMOKE PASSED"
    ;;
  smoke)
    python -u -m physics_modal_v4.enhancements.run --mode data-budget-task \
      --dataset ETTh1 --horizon 96 --seed 42 --train-budget 2048 ${common[@]}
    python -u -m physics_modal_v4.enhancements.run --mode smoke "${common[@]}"
    python -u -m physics_modal_v4.enhancements.mechanisms --mode smoke \
      --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
    ;;
  central) run_shard 140 run_central ;;
  factorial) run_shard 84 run_factorial ;;
  selector_repair_central) run_shard 140 run_repair_central ;;
  selector_repair_factorial) run_shard 84 run_repair_factorial ;;
  selector_repair_tables)
    python -u -m physics_modal_v4.enhancements.aggregate \
      --input-root "$OUTPUT_ROOT" --source-root "$SOURCE_ROOT" \
      --output-root "$OUTPUT_ROOT/selector_safe_statistics" --core-only
    ;;
  strongest) run_shard 112 run_strongest ;;
  capacity) run_shard 28 run_capacity ;;
  full_data) run_shard 28 run_full_data ;;
  data_curve) run_shard 112 run_data_curve ;;
  timepro_data_curve) run_shard 84 run_timepro_data_curve ;;
  channel_order) run_shard 4 run_channel_order ;;
  channel_order_expanded) run_shard 6 run_channel_order_expanded ;;
  diagnostics) run_shard 7 run_diagnostics ;;
  synthetic) run_shard 35 run_synthetic ;;
  real_scaling) run_shard 6 run_real_scaling ;;
  reviewer_smoke)
    python -u -m physics_modal_v4.enhancements.final_audits \
      --mode smoke --check-checkpoints "${common[@]}"
    ;;
  prior_controls) run_shard 252 run_prior_control ;;
  checkpoint_audits) run_shard 140 run_checkpoint_audits ;;
  final_audit_analysis)
    python -u -m physics_modal_v4.enhancements.final_audits \
      --mode analysis --analysis-output "$OUTPUT_ROOT/final_audits" \
      "${common[@]}"
    ;;
  aggregate)
    python -u -m physics_modal_v4.enhancements.aggregate \
      --input-root "$OUTPUT_ROOT" --source-root "$SOURCE_ROOT" \
      --output-root "$OUTPUT_ROOT/statistics"
    ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT" >&2; exit 2 ;;
esac
