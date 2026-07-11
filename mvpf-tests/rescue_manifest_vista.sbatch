#!/usr/bin/env bash
#SBATCH -J mvpf_rescue_all
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 05:30:00
#SBATCH -A IRI23021
#SBATCH -o logs/mvpf_rescue_all.%j.out
#SBATCH -e logs/mvpf_rescue_all.%j.err

set -euo pipefail

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

summarize_partial() {
  python -u mvpf-tests/summarize_rescue.py --output-root "$OUTPUT_ROOT" || true
}
trap summarize_partial EXIT

run_task() {
  python -u mvpf-tests/rescue_720.py \
    --experiment "$1" \
    --dataset "$2" \
    --seed "$3" \
    --device cuda \
    --output-root "$OUTPUT_ROOT"
}

# Priority order. Existing JSON results are skipped automatically.
run_task global_outer Traffic 42
run_task no_variable_mixing Traffic 42
run_task mvpf Traffic 43
run_task patchtst Traffic 43
run_task mvpf Traffic 44
run_task patchtst Traffic 44

run_task global_outer Electricity 42
run_task no_variable_mixing Electricity 42
run_task mvpf Electricity 43
run_task patchtst Electricity 43
run_task mvpf Electricity 44
run_task patchtst Electricity 44

run_task single_scale_8 Traffic 42
run_task single_scale_8 Electricity 42
run_task permute_1 Traffic 42
run_task permute_2 Traffic 42

python -u mvpf-tests/summarize_rescue.py --output-root "$OUTPUT_ROOT"
trap - EXIT
echo "[mvpf-rescue] complete: $OUTPUT_ROOT"
