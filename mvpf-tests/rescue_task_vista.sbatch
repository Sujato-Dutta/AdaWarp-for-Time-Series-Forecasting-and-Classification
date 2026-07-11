#!/usr/bin/env bash
#SBATCH -J mvpf_rescue
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -A IRI23021
#SBATCH -o logs/mvpf_rescue.%j.out
#SBATCH -e logs/mvpf_rescue.%j.err

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: sbatch [options] mvpf-tests/rescue_task_vista.sbatch EXPERIMENT DATASET SEED" >&2
  exit 2
fi

EXPERIMENT="$1"
DATASET="$2"
SEED="$3"
REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
OUTPUT_ROOT="${MVPF_RESCUE_ROOT:-$REPO_ROOT/mvpf-tests/results/rescue_720}"

module purge
module load TACC
module load gcc/14.2.0
module load cuda/12.6
module load python3/3.11.8

cd "$REPO_ROOT"
source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
mkdir -p logs "$OUTPUT_ROOT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python -u mvpf-tests/rescue_720.py \
  --experiment "$EXPERIMENT" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --device cuda \
  --output-root "$OUTPUT_ROOT"
