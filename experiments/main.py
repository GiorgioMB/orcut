"""Run the experiments of the paper (E1--E7) and build its numbers, tables and figures.

Every step is a separate process (a failure stops only that step), logs to
``<out>/logs/<step>.log``, and resumes where it stopped (results are checkpointed row by row), so
the same command can be re-run after a time-out. ``--quick`` runs every step at toy size (minutes
on a laptop) to check the pipeline; its numbers are not the paper's.

  python experiments/main.py --threads 32                       # everything, in order
  python experiments/main.py --steps e3-real e3-threads         # some steps
  python experiments/main.py --quick --out results-quick        # smoke test
  python experiments/main.py --list                             # the steps and what they produce

Steps can also run in parallel on several nodes (``slurm/array.sbatch``); ``artifacts`` reads
whatever results exist.

Relative paths (``--out``, ``--data-dir``, ``--paper-dir``, ``--grc-python``) are relative to the
directory main.py is started from; every step gets them as absolute paths and runs from that same
directory.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import common

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    "e1": (["e1_exactness.py"], "exactness against OR-Tools, POT, the oracle; certificates"),
    "e2-controlled": (["e2_per_edge.py", "--part", "A"], "per-edge cost on controlled neighbourhoods"),
    "e2-real": (["e2_per_edge.py", "--part", "B"], "per-edge cost on real graphs, by degree bin"),
    "e2-dasgupta": (["e2_per_edge.py", "--part", "C"], "the lower-bound instances of DasGupta et al., n up to 2^20"),
    "e3-real": (["e3_scale.py", "--part", "real"], "whole-graph time on real networks up to 1.2e8 edges"),
    "e3-threads": (["e3_scale.py", "--part", "threads"], "strong scaling"),
    "e3-synth": (["e3_scale.py", "--part", "synth"], "scaling on random families up to 1e7 vertices"),
    "e4": (["e4_ablation.py"], "ablation of the engineering steps"),
    "e5-whole": (["e5_approx.py", "--part", "whole"], "GraphRicciCurvature and proxies against exact values"),
    "e5-sinkhorn": (["e5_approx.py", "--part", "sinkhorn"], "entropic OT: time against error"),
    "e6-borf": (["e6_downstream.py", "--part", "borf"], "BORF rewiring: full and incremental exact curvature"),
    "e6-community": (["e6_downstream.py", "--part", "community"], "community detection by edge deletion"),
    "e7-measures": (["e7_generality.py", "--part", "measures"], "other one-step measures"),
    "e7-radius": (["e7_generality.py", "--part", "radius"], "balls and multi-step walks (layered closure)"),
}
THREADED = {"e3-real", "e3-threads", "e3-synth", "e5-whole", "e6-borf", "e7-measures", "e7-radius"}
GRC = {"e5-whole", "e6-borf"}


def _abspath(p):
    return os.path.abspath(os.path.expanduser(p)) if p else p


def _grc_python(explicit):
    """The interpreter of the GRC environment (``--grc-python``, else $ORCUT_GRC_PYTHON) as an
    absolute path, or None. A bare command name is looked up on PATH."""
    p = explicit or os.environ.get("ORCUT_GRC_PYTHON") or None
    if not p:
        return None
    if os.sep in p or p.startswith("~"):
        return _abspath(p)
    return shutil.which(p) or p


def _old_layout(out, data_dir):
    """What main.py up to v1.2.0 left in the wrong place. It ran every step from experiments/ and
    passed the relative --out and --data-dir of the SLURM scripts unchanged, so the steps wrote
    their tables to experiments/<out> (logs and status.json stayed in <out>) and their downloads
    to experiments/<data-dir>, both relative to the repository root. Returns (fatal, messages):
    misplaced tables are fatal, because the steps would not see them and would compute them again
    (and artifacts would not find them); misplaced downloads would only be downloaded again."""
    root = os.path.dirname(HERE)

    def old_place(p):
        rel = os.path.relpath(p, root)
        return None if rel == os.curdir or rel.startswith(os.pardir) else os.path.join(HERE, rel)

    fatal, msgs = False, []
    old = old_place(out)
    if old and glob.glob(os.path.join(old, "e[1-7]", "*.csv")):
        fatal = True
        msgs.append(f"result tables of an earlier run are in {old}, not in {out}; move them there first:\n"
                    f"    for d in {old}/e[1-7]; do mkdir -p {out}/${{d##*/}} && mv -n $d/* {out}/${{d##*/}}/; done")
    data = data_dir or os.path.join(HERE, "data", "real")
    old = old_place(data)
    if old and glob.glob(os.path.join(old, "*")):
        msgs.append(f"downloads of an earlier run are in {old}, not in {data}; to use them instead of downloading "
                    f"again:\n    mkdir -p {data} && mv -n {old}/* {data}/")
    return fatal, msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=None, help=f"subset of: {' '.join(STEPS)} artifacts")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--grc-python", default=None, help="python of the GRC environment (or $ORCUT_GRC_PYTHON)")
    ap.add_argument("--paper-dir", default=None, help="where artifacts writes the numbers/tables/figures")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ignore-old-layout", action="store_true",
                    help="run even if result tables of an earlier run are where main.py up to v1.2.0 put them")
    a = ap.parse_args()
    if a.list:
        for k, (cmd, desc) in STEPS.items():
            print(f"{k:14s} {' '.join(cmd):40s} {desc}")
        print(f"{'artifacts':14s} {'artifacts.py':40s} numbers, tables, figures of the paper")
        return
    steps = a.steps or (list(STEPS) + ["artifacts"])
    bad = [s for s in steps if s not in STEPS and s != "artifacts"]
    if bad:
        ap.error(f"unknown steps: {bad}")
    # every step gets absolute paths (resolved against the directory main.py was started from)
    a.out, a.data_dir, a.paper_dir = _abspath(a.out), _abspath(a.data_dir), _abspath(a.paper_dir)
    a.grc_python = _grc_python(a.grc_python)
    fatal, msgs = _old_layout(a.out, a.data_dir)
    for m in msgs:
        common.log(m)
    if fatal and not a.ignore_old_layout:
        common.log("stopping (--ignore-old-layout runs anyway)")
        sys.exit(2)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    env_path = os.path.join(a.out, "environment.json")
    status_path = os.path.join(a.out, "status.json")
    try:
        common.xio.atomic_json(common.environment(), env_path)
    except Exception as ex:  # environment capture must never block a run
        common.log(f"could not record the environment: {ex}")
    status = common.xio.read_json(status_path, default={}) or {}
    for step in steps:
        if step == "artifacts":
            cmd = [sys.executable, os.path.join(HERE, "artifacts.py"), "--results", a.out]
            if a.paper_dir:
                cmd += ["--paper-dir", a.paper_dir]
            if a.quick:
                cmd += ["--provisional"]
        else:
            script, *args = STEPS[step][0]
            cmd = [sys.executable, os.path.join(HERE, script), *args, "--out", a.out]
            if a.data_dir:
                cmd += ["--data-dir", a.data_dir]
            if step in THREADED:
                cmd += ["--threads", str(a.threads)]
            if step in GRC and a.grc_python:
                cmd += ["--grc-python", a.grc_python]
            if a.quick:
                cmd += ["--quick"]
        log_path = os.path.join(a.out, "logs", f"{step}.log")
        common.log(f"[{step}] {' '.join(cmd[1:])}  (log: {log_path})")
        t0 = time.time()
        with open(log_path, "a") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            fh.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        status[step] = {"returncode": rc, "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "quick": a.quick}
        common.xio.atomic_json(status, status_path)
        common.log(f"[{step}] {'ok' if rc == 0 else f'FAILED (exit {rc})'} in {time.time() - t0:.0f}s")
    failed = [s for s in steps if status.get(s, {}).get("returncode", 0) != 0]
    if failed:
        common.log(f"failed steps: {failed} (see {os.path.join(a.out, 'logs')})")
        sys.exit(1)


if __name__ == "__main__":
    main()
