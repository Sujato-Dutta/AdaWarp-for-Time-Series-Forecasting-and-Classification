#!/usr/bin/env bash
#SBATCH -J vpnet_match
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 06:30:00
#SBATCH -A IRI23021
#SBATCH -o logs/vpnet_matched.%j.out
#SBATCH -e logs/vpnet_matched.%j.err

set -euo pipefail

REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-$REPO_ROOT/mvpf-tests/results/matched_vpnet_${SLURM_JOB_ID}}"

module purge
module load TACC
module load gcc/14.2.0
module load cuda/12.6
module load python3/3.11.8

cd "$REPO_ROOT"
source .venv-adawarp/bin/activate
mkdir -p logs "$OUTPUT_ROOT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the matched VPNet run.")
print("gpu", torch.cuda.get_device_name(0))
PY

python -u mvpf-tests/matched_vpnet_ltsf.py \
  --device cuda \
  --datasets Traffic Electricity Weather ETTm2 ETTh1 ETTh2 \
  --horizons 96 192 336 720 \
  --output-root "$OUTPUT_ROOT"

echo "[matched-vpnet] complete: $OUTPUT_ROOT"