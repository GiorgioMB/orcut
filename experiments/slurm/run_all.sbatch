#!/bin/bash
# Every step of the experiments on one node, in order, from the repository root with the orcut
# environment active (Slurm passes it on). Resubmit after a time limit: it resumes where it stopped.
#   sbatch experiments/slurm/run_all.sbatch        # options after the script go to main.py
#SBATCH --job-name=orcut-all
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${ORCUT_ROOT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"${ORCUT_PYTHON:-python}" experiments/main.py --threads "$SLURM_CPUS_PER_TASK" \
  --out "${ORCUT_RESULTS:-results}" --data-dir "${ORCUT_DATA:-experiments/data/real}" ${1+"$@"}
