#!/bin/bash
set -euo pipefail

REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
cd "$REPO_ROOT"
mkdir -p physics_modal_v4/logs
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/physics_modal_v4/results/final}"
MODE="${1:-smoke}"
MODEL="${2:-}"
SHARDS="${LATENTFLOW_SHARDS:-16}"
COMPARISON_MODELS=(VPNet xCPD TimePro iTransformer TimeMixer DLinear)

submit_array() {
  local experiment="$1" total="$2" walltime="$3" name="$4" extra="${5:-}"
  local shards="$SHARDS"
  (( total < shards )) && shards="$total"
  sbatch --array="0-$((shards-1))%$shards" -t "$walltime" -J "$name"     --export="ALL,EXPERIMENT=$experiment,SHARD_COUNT=$shards,OUTPUT_ROOT=$OUTPUT_ROOT$extra"     physics_modal_v4/vista.sbatch
}

is_comparison_model() {
  local candidate="$1" item
  for item in "${COMPARISON_MODELS[@]}"; do
    [[ "$candidate" == "$item" ]] && return 0
  done
  return 1
}

case "$MODE" in
  smoke)
    sbatch -p gh-dev -t 00:30:00 -J lf_smoke       --export="ALL,EXPERIMENT=smoke,OUTPUT_ROOT=$OUTPUT_ROOT"       physics_modal_v4/vista.sbatch
    ;;
  headline|latentflow)
    submit_array headline 140 2-00:00:00 lf_5seed
    ;;
  baselines)
    submit_array baselines 168 2-00:00:00 lf_base42
    ;;
  baseline)
    is_comparison_model "$MODEL" || {
      echo "baseline model must be one of: ${COMPARISON_MODELS[*]}" >&2
      exit 2
    }
    submit_array baseline 28 2-00:00:00 "b42_$MODEL" ",MODEL_ONLY=$MODEL"
    ;;
  ablations)
    submit_array ablations 504 2-00:00:00 lf_ablate
    ;;
  efficiency)
    submit_array efficiency 196 12:00:00 lf_eff7
    ;;
  statistics)
    module purge
    module load TACC gcc/14.2.0 python3/3.11.8
    ENV_ROOT="${ADAWARP_VENV:-$REPO_ROOT/.venv-adawarp}"
    [[ -x "$ENV_ROOT/bin/python" ]] || {
      echo "Missing Python environment: $ENV_ROOT" >&2
      exit 2
    }
    source "$ENV_ROOT/bin/activate"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    python -u -m physics_modal_v4.experiments.statistics       --input-root "$OUTPUT_ROOT" --output-root "$OUTPUT_ROOT/statistics"
    ;;
  *)
    echo "usage: $0 {smoke|headline|baselines|baseline MODEL|ablations|efficiency|statistics}" >&2
    exit 2
    ;;
esac




