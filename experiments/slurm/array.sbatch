#!/bin/bash
# The experiments as a job array, one step per task (independent; run the artifacts afterwards):
#   sbatch experiments/slurm/array.sbatch                                     # every step of the paper
#   sbatch --array=10,11 experiments/slurm/array.sbatch                       # E6 (not in the paper)
#   sbatch --array=8,10 experiments/slurm/array.sbatch --force               # options after the script go to main.py
#   sbatch --array=9 experiments/slurm/array.sbatch --force -- --budget 3600 # and those after -- to the step
#   python experiments/main.py --steps artifacts --out results               # when all tasks are done
# Steps (index: step): 0 e1, 1 e2-controlled, 2 e2-real, 3 e2-dasgupta, 4 e3-real, 5 e3-threads,
# 6 e3-synth, 7 e4, 8 e5-whole, 9 e5-sinkhorn, 10 e6-borf, 11 e6-community, 12 e7-measures,
# 13 e7-radius; every step resumes where it stopped if resubmitted.
#
# Environments (no need to activate either before sbatch):
#   orcut      runs every step (environment.yml). Found as: $ORCUT_PYTHON; else the active conda
#              environment, if it is orcut; else the environment named orcut in `conda env list`.
#   orcut-grc  runs GraphRicciCurvature for e5-whole (8) and e6-borf (10), which call it for those
#              runs (experiments/grc-environment.yml). Found as: $ORCUT_GRC_PYTHON; else orcut-grc
#              next to orcut (e.g. .../conda_envs/orcut-grc); else `conda env list`. Tasks 8 and 10
#              stop at once if GraphRicciCurvature does not import there.
# ORCUT_DATA and ORCUT_RESULTS as in run_all.sbatch; ORCUT_ROOT sets the repository.
#SBATCH --job-name=orcut-step
#SBATCH --array=0-9,12,13   # 10, 11: E6, not part of the current paper (ask for them with --array)
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
GRC_STEPS=" e5-whole e6-borf "   # the steps that also run GraphRicciCurvature

env_python() {  # env_python <name>: the python of the conda environment called <name>, if conda lists one
  local tool p
  for tool in conda mamba micromamba; do
    command -v "$tool" >/dev/null 2>&1 || continue
    p=$("$tool" env list 2>/dev/null | awk -v n="$1" '!/^#/ && NF { k = split($NF, a, "/"); if ($1 == n || a[k] == n) { print $NF; exit } }')
    if [ -n "$p" ] && [ -x "$p/bin/python" ]; then echo "$p/bin/python"; return; fi
  done
}

# orcut: runs main.py and every step
if [ -z "${ORCUT_PYTHON:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = orcut ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    ORCUT_PYTHON="$CONDA_PREFIX/bin/python"
  else
    ORCUT_PYTHON="$(env_python orcut)"
  fi
fi
if [ -z "$ORCUT_PYTHON" ] || ! "$ORCUT_PYTHON" -c "import numba, numpy, scipy" >/dev/null 2>&1; then
  echo "error: no python of the orcut environment (with numba, numpy, scipy) found (got: ${ORCUT_PYTHON:-none});" \
       "set ORCUT_PYTHON=<orcut env>/bin/python" >&2
  exit 2
fi

# orcut-grc: GraphRicciCurvature, for the GRC steps only
if [ -z "${ORCUT_GRC_PYTHON:-}" ]; then
  sib="$(dirname "$(dirname "$ORCUT_PYTHON")")/../orcut-grc/bin/python"
  if [ -x "$sib" ]; then
    ORCUT_GRC_PYTHON="$(cd "$(dirname "$sib")" && pwd)/python"
  else
    ORCUT_GRC_PYTHON="$(env_python orcut-grc)"
  fi
fi
GRC_ARGS=()
if [[ "$GRC_STEPS" == *" $STEP "* ]]; then
  err=""
  if [ -z "$ORCUT_GRC_PYTHON" ] || ! err=$("$ORCUT_GRC_PYTHON" -c "import GraphRicciCurvature.OllivierRicci" 2>&1); then
    echo "error: $STEP runs GraphRicciCurvature in the orcut-grc environment, but it does not import there" \
         "(${ORCUT_GRC_PYTHON:-orcut-grc not found}${err:+: ${err##*$'\n'}})." >&2
    echo "       Install it (experiments/grc-environment.yml), or set ORCUT_GRC_PYTHON=<env with it>/bin/python." >&2
    exit 3
  fi
  echo "$STEP: steps in orcut ($ORCUT_PYTHON), GraphRicciCurvature in orcut-grc ($ORCUT_GRC_PYTHON)"
  GRC_ARGS=(--grc-python "$ORCUT_GRC_PYTHON")
  export ORCUT_GRC_PYTHON
else
  echo "$STEP: orcut ($ORCUT_PYTHON)"
  unset ORCUT_GRC_PYTHON
fi

: "${ORCUT_DATA:=$ROOT/experiments/data/real}"
: "${ORCUT_RESULTS:=$ROOT/results}"
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$THREADS"
"$ORCUT_PYTHON" experiments/main.py --steps "${STEP}" --threads "$THREADS" --out "$ORCUT_RESULTS" --data-dir "$ORCUT_DATA" \
  ${GRC_ARGS[@]+"${GRC_ARGS[@]}"} ${1+"$@"}
