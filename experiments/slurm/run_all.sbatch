#!/bin/bash
# All experiments of the paper on one node, in order (resumable: resubmit after a time-out).
#   sbatch experiments/slurm/run_all.sbatch
# Before: download the data on a node with network access (compute nodes usually have none):
#   python experiments/datasets.py --data-dir "$ORCUT_DATA"
# and create the two environments (see experiments/README.md): orcut (environment.yml) and,
# for the GraphRicciCurvature baseline, orcut-grc (experiments/grc-environment.yml).
# ORCUT_DATA and ORCUT_RESULTS default to experiments/data/real and results in the repository;
# relative values are taken relative to the repository root. ORCUT_ROOT sets the repository when
# sbatch is called neither from it nor with the path of this script.
#SBATCH --job-name=orcut-all
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%x-%j.out
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
: "${ORCUT_DATA:=$ROOT/experiments/data/real}"
: "${ORCUT_RESULTS:=$ROOT/results}"
: "${ORCUT_GRC_PYTHON:=$(conda run -n orcut-grc which python 2>/dev/null || true)}"
export ORCUT_GRC_PYTHON
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # numba threads only
export NUMBA_NUM_THREADS="$THREADS"
python experiments/main.py --threads "$THREADS" --out "$ORCUT_RESULTS" --data-dir "$ORCUT_DATA"
