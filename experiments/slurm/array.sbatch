#!/bin/bash
# The experiments as a job array, one step per task (independent; run the artifacts afterwards):
#   sbatch experiments/slurm/array.sbatch
#   python experiments/main.py --steps artifacts --out results      # when all tasks are done
# Steps (index: step): see STEPS below; every step resumes where it stopped if resubmitted.
#SBATCH --job-name=orcut-step
#SBATCH --array=0-13
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
STEPS=(e1 e2-controlled e2-real e2-dasgupta e3-real e3-threads e3-synth e4 e5-whole e5-sinkhorn e6-borf e6-community e7-measures e7-radius)
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
: "${ORCUT_DATA:=experiments/data/real}"
: "${ORCUT_RESULTS:=results}"
: "${ORCUT_GRC_PYTHON:=$(conda run -n orcut-grc which python 2>/dev/null || true)}"
export ORCUT_GRC_PYTHON
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python experiments/main.py --steps "${STEP}" --threads "${SLURM_CPUS_PER_TASK}" --out "${ORCUT_RESULTS}" --data-dir "${ORCUT_DATA}"
