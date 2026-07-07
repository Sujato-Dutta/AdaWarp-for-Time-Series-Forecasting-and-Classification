#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ABLATION="${ADAWARP_MVPF_PLUS_ABLATION:-full}"

# Start from the final model defaults. Each ablation changes exactly one design axis.
export ADAWARP_MVPF_PLUS_USE_LINEAR_FIELD="${ADAWARP_MVPF_PLUS_USE_LINEAR_FIELD:-1}"
export ADAWARP_MVPF_PLUS_USE_ADAPTIVE_SHIFTS="${ADAWARP_MVPF_PLUS_USE_ADAPTIVE_SHIFTS:-1}"
export ADAWARP_MVPF_PLUS_USE_ADAPTIVE_RADIUS="${ADAWARP_MVPF_PLUS_USE_ADAPTIVE_RADIUS:-1}"
export ADAWARP_MVPF_PLUS_USE_COMPONENT_GATE="${ADAWARP_MVPF_PLUS_USE_COMPONENT_GATE:-1}"
export ADAWARP_MVPF_PLUS_USE_TREND_DECOMPOSITION="${ADAWARP_MVPF_PLUS_USE_TREND_DECOMPOSITION:-1}"
export ADAWARP_MVPF_PLUS_PATCH_LENS="${ADAWARP_MVPF_PLUS_PATCH_LENS:-8 16 32}"

case "$ABLATION" in
  no_linear_field)
    export ADAWARP_MVPF_PLUS_USE_LINEAR_FIELD=0
    ;;
  single_scale_16)
    export ADAWARP_MVPF_PLUS_PATCH_LENS="16"
    ;;
  no_adaptive_shifts)
    export ADAWARP_MVPF_PLUS_USE_ADAPTIVE_SHIFTS=0
    ;;
  no_trend_residual)
    export ADAWARP_MVPF_PLUS_USE_TREND_DECOMPOSITION=0
    ;;
  full)
    ;;
  *)
    echo "unknown ADAWARP_MVPF_PLUS_ABLATION=$ABLATION" >&2
    echo "valid: no_linear_field single_scale_16 no_adaptive_shifts no_trend_residual full" >&2
    exit 2
    ;;
esac

mkdir -p "${ADAWARP_OUTPUT_ROOT:-results/adawarp_mvpf_final_ablation}/audit"
{
  echo "paper_model=AdaWarp-MVPF"
  echo "implementation=adawarp_mvpf_plus"
  echo "ablation=$ABLATION"
  echo "datasets=${ADAWARP_LTSF_DATASETS:-ETTh1 ETTh2 Weather Electricity Traffic}"
  echo "horizons=${ADAWARP_LTSF_HORIZONS:-96 192 336 720}"
  echo "patch_lens=$ADAWARP_MVPF_PLUS_PATCH_LENS"
  echo "use_linear_field=$ADAWARP_MVPF_PLUS_USE_LINEAR_FIELD"
  echo "use_adaptive_shifts=$ADAWARP_MVPF_PLUS_USE_ADAPTIVE_SHIFTS"
  echo "use_trend_decomposition=$ADAWARP_MVPF_PLUS_USE_TREND_DECOMPOSITION"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${ADAWARP_OUTPUT_ROOT:-results/adawarp_mvpf_final_ablation}/audit/final_mvpf_ablation_config.txt"

bash scripts/tacc/run_adawarp_mvpf_plus_ltsf.sh
