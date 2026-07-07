#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

DATASETS_DEFAULT="ETTh1 ETTh2 Weather Electricity Traffic"
HORIZONS_DEFAULT="96 192 336 720"
SEEDS_DEFAULT="42"
PATCH_LENS_DEFAULT="8 16 32"

read -r -a DATASETS <<< "${ADAWARP_LTSF_DATASETS:-$DATASETS_DEFAULT}"
read -r -a HORIZONS <<< "${ADAWARP_LTSF_HORIZONS:-$HORIZONS_DEFAULT}"
read -r -a SEEDS <<< "${ADAWARP_LTSF_SEEDS:-$SEEDS_DEFAULT}"
read -r -a PATCH_LENS <<< "${ADAWARP_MVPF_PLUS_PATCH_LENS:-$PATCH_LENS_DEFAULT}"

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results/adawarp_mvpf_plus_ltsf}"
mkdir -p "$OUTPUT_ROOT/audit"

{
  echo "model=AdaWarp-MVPF+"
  echo "output_root=$OUTPUT_ROOT"
  echo "datasets=${DATASETS[*]}"
  echo "horizons=${HORIZONS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "patch_lens=${PATCH_LENS[*]}"
  echo "use_linear_field=${ADAWARP_MVPF_PLUS_USE_LINEAR_FIELD:-1}"
  echo "use_prototype_memory=${ADAWARP_MVPF_PLUS_USE_PROTOTYPE_MEMORY:-0}"
  echo "use_frequency_gate=${ADAWARP_MVPF_PLUS_USE_FREQUENCY_GATE:-0}"
  echo "use_adaptive_shifts=${ADAWARP_MVPF_PLUS_USE_ADAPTIVE_SHIFTS:-1}"
  echo "use_adaptive_radius=${ADAWARP_MVPF_PLUS_USE_ADAPTIVE_RADIUS:-1}"
  echo "use_component_gate=${ADAWARP_MVPF_PLUS_USE_COMPONENT_GATE:-1}"
  echo "use_trend_decomposition=${ADAWARP_MVPF_PLUS_USE_TREND_DECOMPOSITION:-1}"
  echo "max_validation_windows=${ADAWARP_LTSF_MAX_VAL_WINDOWS:-1024}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$(command -v python)"
  python - <<'PY'
import sys
print("python_version=" + sys.version.replace("\n", " "))
try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda=" + str(torch.version.cuda))
    print("cuda_available=" + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("cuda_device=" + torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_error=" + repr(exc))
PY
} > "$OUTPUT_ROOT/audit/adawarp_mvpf_plus_config.txt"

EXTRA_ARGS=()
if [[ "${ADAWARP_MVPF_PLUS_USE_LINEAR_FIELD:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-use-linear-field)
fi
if [[ "${ADAWARP_MVPF_PLUS_USE_PROTOTYPE_MEMORY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use-prototype-memory)
fi
if [[ "${ADAWARP_MVPF_PLUS_USE_FREQUENCY_GATE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use-frequency-gate)
fi
if [[ "${ADAWARP_MVPF_PLUS_USE_ADAPTIVE_SHIFTS:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-use-adaptive-shifts)
fi
if [[ "${ADAWARP_MVPF_PLUS_USE_ADAPTIVE_RADIUS:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-use-adaptive-radius)
fi
if [[ "${ADAWARP_MVPF_PLUS_USE_COMPONENT_GATE:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-use-component-gate)
fi
if [[ "${ADAWARP_MVPF_PLUS_USE_TREND_DECOMPOSITION:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-use-trend-decomposition)
fi

python benchmark_adawarp_mvpf_plus_ltsf.py \
  --datasets "${DATASETS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --seeds "${SEEDS[@]}" \
  --seq-len "${ADAWARP_LTSF_SEQ_LEN:-96}" \
  --epochs "${ADAWARP_MVPF_PLUS_EPOCHS:-10}" \
  --batch-size "${ADAWARP_MVPF_PLUS_BATCH_SIZE:-16}" \
  --eval-batch-size "${ADAWARP_MVPF_PLUS_EVAL_BATCH_SIZE:-16}" \
  --learning-rate "${ADAWARP_MVPF_PLUS_LR:-0.0007}" \
  --weight-decay "${ADAWARP_MVPF_PLUS_WEIGHT_DECAY:-0.0001}" \
  --d-model "${ADAWARP_MVPF_PLUS_D_MODEL:-128}" \
  --depth "${ADAWARP_MVPF_PLUS_DEPTH:-2}" \
  --dropout "${ADAWARP_MVPF_PLUS_DROPOUT:-0.05}" \
  --patch-lens "${PATCH_LENS[@]}" \
  --num-prototypes "${ADAWARP_MVPF_PLUS_NUM_PROTOTYPES:-8}" \
  --max-shift "${ADAWARP_MVPF_PLUS_MAX_SHIFT:-2}" \
  --reconstruction-weight "${ADAWARP_MVPF_PLUS_RECONSTRUCTION_WEIGHT:-0.03}" \
  --max-train-windows "${ADAWARP_LTSF_MAX_TRAIN_WINDOWS:-2048}" \
  --max-validation-windows "${ADAWARP_LTSF_MAX_VAL_WINDOWS:-1024}" \
  --max-eval-windows "${ADAWARP_LTSF_MAX_EVAL_WINDOWS:-2048}" \
  --patience "${ADAWARP_MVPF_PLUS_PATIENCE:-0}" \
  --data-root "${ADAWARP_LTSF_DATA_ROOT:-TSLibrary/dataset}" \
  "${EXTRA_ARGS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --device "${ADAWARP_DEVICE:-auto}"

python aggregate_adawarp_experiments.py --output-root "$OUTPUT_ROOT"
echo "[run_adawarp_mvpf_plus_ltsf] complete output_root=$OUTPUT_ROOT"
