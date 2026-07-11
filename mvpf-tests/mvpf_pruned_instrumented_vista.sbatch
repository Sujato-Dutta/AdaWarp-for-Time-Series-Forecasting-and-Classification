#!/usr/bin/env bash
#SBATCH -J mvpf_pruned
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 08:30:00
#SBATCH -A IRI23021
#SBATCH -o logs/mvpf_pruned.%j.out
#SBATCH -e logs/mvpf_pruned.%j.err

set -euo pipefail

REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-$REPO_ROOT/mvpf-tests/results/pruned_instrumented_${SLURM_JOB_ID}}"

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
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

python -u mvpf-tests/instrumented_pruned_mvpf.py \
  --device cuda \
  --datasets ETTm2 ETTh1 ETTh2 Weather Electricity Traffic \
  --horizons 96 192 336 720 \
  --output-root "$OUTPUT_ROOT"

echo "[mvpf-pruned] complete: $OUTPUT_ROOT"
