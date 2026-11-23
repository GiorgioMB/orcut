#!/bin/bash
# One step of the experiments per array task, from the repository root with the orcut environment
# active (Slurm passes it on); then the artifacts:
#   sbatch experiments/slurm/array.sbatch          # options after the script go to main.py
#   python experiments/main.py --steps artifacts
# Tasks: 0 e1, 1 e2-controlled, 2 e2-real, 3 e2-dasgupta, 4 e3-real, 5 e3-threads, 6 e3-synth, 7 e4,
# 8 e5-whole, 9 e5-sinkhorn, 10 e6-measures, 11 e6-radius, 12 calib (times two graphs on the type of
# node it lands on: sbatch --array=12 --constraint=<type> for each type). A task resubmitted after a
# time limit resumes where it stopped.
#SBATCH --job-name=orcut-step
#SBATCH --array=0-12
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
cd "${ORCUT_ROOT:-$SLURM_SUBMIT_DIR}"
STEPS=(e1 e2-controlled e2-real e2-dasgupta e3-real e3-threads e3-synth e4 e5-whole e5-sinkhorn e6-measures e6-radius calib)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"${ORCUT_PYTHON:-python}" experiments/main.py --steps "${STEPS[$SLURM_ARRAY_TASK_ID]}" --threads "$SLURM_CPUS_PER_TASK" \
  --out "${ORCUT_RESULTS:-results}" --data-dir "${ORCUT_DATA:-experiments/data/real}" ${1+"$@"}
