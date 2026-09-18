#!/bin/bash
set -euo pipefail

REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
cd "$REPO_ROOT"
mkdir -p physics_modal_v4/additions/logs physics_modal_v4/additions/results
MODE="${1:-smoke}"
SHARDS="${LATENTFLOW_ADDITION_SHARDS:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/physics_modal_v4/additions/results}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/physics_modal_v4/results/final}"

submit_array() {
  local experiment="$1" total="$2" walltime="$3" name="$4"
  local shards="$SHARDS"
  (( total < shards )) && shards="$total"
  sbatch --array="0-$((shards-1))%$shards" -t "$walltime" -J "$name" \
    --export="ALL,EXPERIMENT=$experiment,SHARD_COUNT=$shards,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/additions/vista.sbatch
}

case "$MODE" in
  smoke)
    sbatch -p gh-dev -t 00:30:00 -J lf_add_smoke \
      --export="ALL,EXPERIMENT=smoke,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/additions/vista.sbatch ;;
  timepro) submit_array timepro 112 2-00:00:00 lf_tp5 ;;
  synthetic) submit_array synthetic 40 1-00:00:00 lf_syn ;;
  central) submit_array central 700 2-00:00:00 lf_cabl28 ;;
  missing) submit_array missing 8 06:00:00 lf_miss ;;
  process) submit_array process 2 02:00:00 lf_proc ;;
  vpnet_control) submit_array vpnet_control 28 1-00:00:00 lf_vpctl ;;
  scaling)
    sbatch -t 04:00:00 -J lf_scale \
      --export="ALL,EXPERIMENT=scaling,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/additions/vista.sbatch ;;
  aggregate)
    module purge
    module load TACC gcc/14.2.0 python3/3.11.8
    ENV_ROOT="${ADAWARP_VENV:-$REPO_ROOT/.venv-adawarp}"
    [[ -x "$ENV_ROOT/bin/python" ]] || { echo "Missing environment: $ENV_ROOT" >&2; exit 2; }
    source "$ENV_ROOT/bin/activate"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    python -u -m physics_modal_v4.additions.aggregate \
      --input-root "$OUTPUT_ROOT" --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT/summary" ;;
  *)
    echo "usage: $0 {smoke|timepro|synthetic|central|missing|process|vpnet_control|scaling|aggregate}" >&2
    exit 2 ;;
esac
