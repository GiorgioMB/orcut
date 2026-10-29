#!/bin/bash
# All experiments of the paper on one node, in order (resumable: resubmit after a time-out).
#   sbatch experiments/slurm/run_all.sbatch
# Before: download the data on a node with network access (compute nodes usually have none):
#   python experiments/datasets.py --data-dir "$ORCUT_DATA"
# and create the two environments (see experiments/README.md): orcut (environment.yml) and,
# for the GraphRicciCurvature baseline, orcut-grc (experiments/grc-environment.yml).
#SBATCH --job-name=orcut-all
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${ORCUT_DATA:=experiments/data/real}"
: "${ORCUT_RESULTS:=results}"
: "${ORCUT_GRC_PYTHON:=$(conda run -n orcut-grc which python 2>/dev/null || true)}"
export ORCUT_GRC_PYTHON
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # numba threads only
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python experiments/main.py --threads "${SLURM_CPUS_PER_TASK}" --out "${ORCUT_RESULTS}" --data-dir "${ORCUT_DATA}"
