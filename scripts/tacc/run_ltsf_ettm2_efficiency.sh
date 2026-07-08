#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

MODELS_DEFAULT="AdaWarp-MVPF DLinear PatchTST TimesNet iTransformer TimeMixer FEDformer VPNet"
HORIZONS_DEFAULT="96 192 336 720"

read -r -a MODELS <<< "${ADAWARP_EFFICIENCY_MODELS:-$MODELS_DEFAULT}"
read -r -a HORIZONS <<< "${ADAWARP_LTSF_HORIZONS:-$HORIZONS_DEFAULT}"

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results/ltsf_efficiency_ettm2}"
mkdir -p "$OUTPUT_ROOT"

python benchmark_ltsf_efficiency.py \
  --dataset "${ADAWARP_EFFICIENCY_DATASET:-ETTm2}" \
  --models "${MODELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --seq-len "${ADAWARP_LTSF_SEQ_LEN:-96}" \
  --label-len "${ADAWARP_TSLIB_LABEL_LEN:-48}" \
  --batch-size "${ADAWARP_EFFICIENCY_BATCH_SIZE:-16}" \
  --warmup-iters "${ADAWARP_EFFICIENCY_WARMUP_ITERS:-3}" \
  --training-iters "${ADAWARP_EFFICIENCY_TRAINING_ITERS:-10}" \
  --inference-iters "${ADAWARP_EFFICIENCY_INFERENCE_ITERS:-20}" \
  --dropout "${ADAWARP_EFFICIENCY_DROPOUT:-0.05}" \
  --tslib-d-model "${ADAWARP_TSLIB_D_MODEL:-128}" \
  --tslib-d-ff "${ADAWARP_TSLIB_D_FF:-256}" \
  --tslib-n-heads "${ADAWARP_TSLIB_N_HEADS:-8}" \
  --tslib-e-layers "${ADAWARP_TSLIB_E_LAYERS:-2}" \
  --tslib-d-layers "${ADAWARP_TSLIB_D_LAYERS:-1}" \
  --custom-d-model "${ADAWARP_CUSTOM_LTSF_D_MODEL:-256}" \
  --custom-depth "${ADAWARP_CUSTOM_LTSF_DEPTH:-2}" \
  --custom-blocks "${ADAWARP_CUSTOM_LTSF_BLOCKS:-4}" \
  --vpnet-patch-len "${ADAWARP_VPNET_PATCH_LEN:-16}" \
  --mvpf-d-model "${ADAWARP_MVPF_PLUS_D_MODEL:-128}" \
  --mvpf-depth "${ADAWARP_MVPF_PLUS_DEPTH:-2}" \
  --mvpf-patch-lens ${ADAWARP_MVPF_PLUS_PATCH_LENS:-8 16 32} \
  --mvpf-num-prototypes "${ADAWARP_MVPF_PLUS_NUM_PROTOTYPES:-8}" \
  --mvpf-max-shift "${ADAWARP_MVPF_PLUS_MAX_SHIFT:-2}" \
  --mvpf-reconstruction-weight "${ADAWARP_MVPF_PLUS_RECONSTRUCTION_WEIGHT:-0.03}" \
  --data-root "${ADAWARP_LTSF_DATA_ROOT:-TSLibrary/dataset}" \
  --tslibrary-root "${ADAWARP_TSLIB_ROOT:-TSLibrary}" \
  --output-root "$OUTPUT_ROOT" \
  --device "${ADAWARP_DEVICE:-auto}" \
  --seed "${ADAWARP_EFFICIENCY_SEED:-42}"

echo "[run_ltsf_ettm2_efficiency] complete output_root=$OUTPUT_ROOT"
