#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/mvpf_plus_ablation slurm results

ABLATIONS_DEFAULT="no_linear_field single_scale_16 no_adaptive_shifts no_trend_residual"
DATASETS_DEFAULT="ETTh1 ETTh2 Weather Electricity Traffic"

read -r -a ABLATIONS <<< "${ADAWARP_MVPF_PLUS_ABLATIONS_TO_SUBMIT:-${ADAWARP_MVPF_PLUS_ABLATIONS:-$ABLATIONS_DEFAULT}}"
read -r -a DATASETS <<< "${ADAWARP_MVPF_PLUS_DATASETS_TO_SUBMIT:-${ADAWARP_LTSF_DATASETS:-$DATASETS_DEFAULT}}"

ACCOUNT="${ADAWARP_ALLOCATION:-IRI23021}"
PARTITION="${ADAWARP_PARTITION:-gh}"
WALLTIME="${ADAWARP_WALLTIME:-24:00:00}"
MAX_SUBMITS="${ADAWARP_MAX_SUBMITS:-20}"
JOB_PARENT="${ADAWARP_MVPF_PLUS_ABLATION_JOB_PARENT:-$WORK/motion_code-master/results/vista_ltsf_adawarp_mvpf_final_ablations}"
TOTAL=$(( ${#ABLATIONS[@]} * ${#DATASETS[@]} ))

if (( TOTAL > MAX_SUBMITS )); then
  echo "[submit_mvpf_plus_ablation] refusing to submit $TOTAL jobs because ADAWARP_MAX_SUBMITS=$MAX_SUBMITS" >&2
  echo "[submit_mvpf_plus_ablation] reduce ablations/datasets or raise ADAWARP_MAX_SUBMITS intentionally" >&2
  exit 2
fi

echo "[submit_mvpf_plus_ablation] account=$ACCOUNT partition=$PARTITION walltime=$WALLTIME max_submits=$MAX_SUBMITS"
echo "[submit_mvpf_plus_ablation] parent=$JOB_PARENT"
echo "[submit_mvpf_plus_ablation] ablations=${ABLATIONS[*]}"
echo "[submit_mvpf_plus_ablation] datasets=${DATASETS[*]}"
echo "[submit_mvpf_plus_ablation] horizons=${ADAWARP_LTSF_HORIZONS:-96 192 336 720}"

for ABLATION in "${ABLATIONS[@]}"; do
  ABLATION_SAFE="$(printf "%s" "$ABLATION" | tr -c 'A-Za-z0-9_' '_' | sed 's/_*$//')"
  for DATASET in "${DATASETS[@]}"; do
    DATASET_SAFE="$(printf "%s" "$DATASET" | tr -c 'A-Za-z0-9_' '_' | sed 's/_*$//')"
    OUTPUT_ROOT="$JOB_PARENT/$ABLATION_SAFE/$DATASET_SAFE/AdaWarp_MVPF"
    JOB_NAME="mvpff_${ABLATION_SAFE}_${DATASET_SAFE}"
    sbatch \
      -A "$ACCOUNT" \
      -p "$PARTITION" \
      -t "$WALLTIME" \
      -J "$JOB_NAME" \
      slurm/adawarp_mvpf_plus_ablation_vista.sbatch "$ABLATION" "$DATASET" "$OUTPUT_ROOT"
  done
done
