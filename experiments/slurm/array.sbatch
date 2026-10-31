#!/bin/bash
# The experiments as a job array, one step per task (independent; run the artifacts afterwards):
#   sbatch experiments/slurm/array.sbatch
#   python experiments/main.py --steps artifacts --out results      # when all tasks are done
# Steps (index: step): see STEPS below; every step resumes where it stopped if resubmitted.
# ORCUT_DATA, ORCUT_RESULTS and ORCUT_ROOT as in run_all.sbatch.
#SBATCH --job-name=orcut-step
#SBATCH --array=0-13
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
# sbatch runs a copy of this script from its spool directory: find the repository from
# $ORCUT_ROOT, the directory sbatch was called from, or the path the script was submitted with
ROOT="${ORCUT_ROOT:-}"
if [ -z "$ROOT" ]; then
  cands=("${SLURM_SUBMIT_DIR:-}")
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    sub=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)
    case "$sub" in /*) ;; ?*) sub="${SLURM_SUBMIT_DIR:-.}/$sub" ;; esac
    [ -n "$sub" ] && cands+=("$(dirname "$sub")/../..")
  fi
  cands+=("$(dirname "${BASH_SOURCE[0]}")/../..")
  for c in "${cands[@]}"; do
    if [ -n "$c" ] && [ -f "$c/experiments/main.py" ]; then ROOT="$(cd "$c" && pwd)"; break; fi
  done
fi
[ -n "$ROOT" ] && [ -f "$ROOT/experiments/main.py" ] || {
  echo "cannot find the orcut repository: sbatch from its root, or set ORCUT_ROOT" >&2; exit 2; }
cd "$ROOT"
STEPS=(e1 e2-controlled e2-real e2-dasgupta e3-real e3-threads e3-synth e4 e5-whole e5-sinkhorn e6-borf e6-community e7-measures e7-radius)
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
: "${ORCUT_DATA:=$ROOT/experiments/data/real}"
: "${ORCUT_RESULTS:=$ROOT/results}"
: "${ORCUT_GRC_PYTHON:=$(conda run -n orcut-grc which python 2>/dev/null || true)}"
export ORCUT_GRC_PYTHON
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$THREADS"
python experiments/main.py --steps "${STEP}" --threads "$THREADS" --out "$ORCUT_RESULTS" --data-dir "$ORCUT_DATA"
