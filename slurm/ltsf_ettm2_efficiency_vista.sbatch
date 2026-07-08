#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o logs/ltsf_efficiency/%x.%j.out
#SBATCH -e logs/ltsf_efficiency/%x.%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch [sbatch options] slurm/ltsf_ettm2_efficiency_vista.sbatch OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"

module purge
module load TACC
module load gcc/14.2.0
module load cuda/12.6
module load python3/3.11.8

cd "$REPO_ROOT"
source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"

export ADAWARP_OUTPUT_ROOT="$OUTPUT_ROOT"
export ADAWARP_DEVICE="${ADAWARP_DEVICE:-cuda}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

echo "[job] host=$(hostname)"
echo "[job] model_set=ETTm2 efficiency"
echo "[job] output_root=$ADAWARP_OUTPUT_ROOT"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

bash scripts/tacc/run_ltsf_ettm2_efficiency.sh
