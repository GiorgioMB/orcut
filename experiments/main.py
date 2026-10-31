"""Run the experiments of the paper (E1--E7) and build its numbers, tables and figures.

Every step is a separate process (a failure stops only that step) and logs to
``<out>/logs/<step>.log``. The same command can be run again after a time-out, a crash or an
out-of-memory kill, and only what is left is computed. There are checkpoints at two levels:

* steps: a step that ends without errors is recorded in ``<out>/status/<step>.json`` and is not
  run again (``--force`` runs it anyway; ``--status`` shows what is done, failed or running). A
  step that failed, was killed, or left rows that ended in an error runs again, and resumes;
* rows: every step saves each result row as soon as it is computed and skips the rows already in
  its tables, except rows that hold no result (an error, or GraphRicciCurvature unavailable),
  which are computed again.

``artifacts`` runs again when the result tables changed since its last run, and a GRC step that
completed without GraphRicciCurvature runs again (for the GRC rows) once GRC is available. While
a step runs, ``<out>/status/<step>.lock`` keeps other jobs from running the same step at once.

  python experiments/main.py --threads 32                       # everything, in order
  python experiments/main.py --steps e3-real e3-threads         # some steps
  python experiments/main.py --status                           # what is done, failed or running
  python experiments/main.py --steps e2-real --force            # a completed step, again
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
import signal
import socket
import subprocess
import sys
import time

import checkpoint as CK
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
HEARTBEAT_S = 60.0  # how often the lock of a running step is touched (CK.STALE_S: when it goes stale)


class Stopped(Exception):
    """main.py got SIGTERM (the job's time limit, scancel) or SIGINT while a step ran."""

    def __init__(self, signum: int):
        super().__init__(f"signal {signum}")
        self.signum = signum


def _stop(signum, frame):
    raise Stopped(signum)


def _signame(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


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


def _command(step, a, steps):
    if step == "artifacts":
        cmd = [sys.executable, os.path.join(HERE, "artifacts.py"), "--results", a.out]
        if a.paper_dir:
            cmd += ["--paper-dir", a.paper_dir]
        if a.quick:
            cmd += ["--provisional"]
        return cmd
    script, *args = steps[step][0]
    cmd = [sys.executable, os.path.join(HERE, script), *args, "--out", a.out]
    if a.data_dir:
        cmd += ["--data-dir", a.data_dir]
    if step in THREADED:
        cmd += ["--threads", str(a.threads)]
    if step in GRC and a.grc_python:
        cmd += ["--grc-python", a.grc_python]
    if a.quick:
        cmd += ["--quick"]
    return cmd


def _config(step, a):
    """What a completed run of ``step`` must have run with to count as done: the same --quick; for
    the GRC steps, the same GRC interpreter (a run without GRC is done until GRC is available);
    for artifacts, the same --paper-dir and the same result tables."""
    cfg = {"quick": bool(a.quick)}
    if step in GRC:
        cfg["grc"] = a.grc_python if a.grc_python and os.path.exists(a.grc_python) else None
    if step == "artifacts":
        cfg["paper_dir"] = a.paper_dir
        cfg["inputs"] = CK.inputs_fingerprint(a.out)
    return cfg


def _rerun_reason(key, got):
    if got is KeyError:
        return f"its record (written by main.py up to v1.2.0) does not say which {key} it ran with"
    if key == "grc":
        return "it ran without GraphRicciCurvature, which is available now" if got is None else \
            f"it ran with another GraphRicciCurvature environment ({got})"
    if key == "inputs":
        return "the result tables changed since"
    if key == "paper_dir":
        return f"it wrote to another --paper-dir ({got})"
    return f"it ran with {key}={got!r}"


def _outcome(rc: int) -> str:
    if rc == 0:
        return "ok"
    if rc == CK.EXIT_INCOMPLETE:
        return "finished, but some rows ended in an error (see the log); they are computed again on the next run"
    if rc < 0:
        hint = " -- from outside the step; usually the memory limit of the job (see MaxRSS in sacct)" if rc == -9 else ""
        return f"FAILED: killed by {_signame(-rc)}{hint}"
    return f"FAILED (exit {rc})"


def _run(cmd, log_path, lock) -> int:
    """Run one step and return its exit code, touching its lock every HEARTBEAT_S seconds. If
    main.py is stopped meanwhile (Stopped, KeyboardInterrupt), the step is terminated first."""
    with open(log_path, "a") as fh:
        fh.write(f"\n===== {CK.now()} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    return p.wait(timeout=HEARTBEAT_S)
                except subprocess.TimeoutExpired:
                    lock.refresh()
        except BaseException:
            for sig in (signal.SIGTERM, signal.SIGINT):  # finish the clean-up whatever comes next
                signal.signal(sig, signal.SIG_IGN)
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise


def main(argv=None, steps=None) -> int:
    """``steps`` replaces the table of steps (tests)."""
    table = STEPS if steps is None else steps
    names = list(table) + ["artifacts"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=None, help=f"subset of: {' '.join(names)}")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--grc-python", default=None, help="python of the GRC environment (or $ORCUT_GRC_PYTHON)")
    ap.add_argument("--paper-dir", default=None, help="where artifacts writes the numbers/tables/figures")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run the selected steps even if they completed before (rows already in the result tables are "
                         "still skipped: delete a table, or some of its rows, to compute them again)")
    ap.add_argument("--status", action="store_true", help="show which steps are done, failed or running, and exit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ignore-old-layout", action="store_true",
                    help="run even if result tables of an earlier run are where main.py up to v1.2.0 put them")
    a = ap.parse_args(argv)
    if a.list:
        for k, (cmd, desc) in table.items():
            print(f"{k:14s} {' '.join(cmd):40s} {desc}")
        print(f"{'artifacts':14s} {'artifacts.py':40s} numbers, tables, figures of the paper")
        return 0
    selected = a.steps or names
    bad = [s for s in selected if s not in names]
    if bad:
        ap.error(f"unknown steps: {bad}")
    # every step gets absolute paths (resolved against the directory main.py was started from)
    a.out, a.data_dir, a.paper_dir = _abspath(a.out), _abspath(a.data_dir), _abspath(a.paper_dir)
    a.grc_python = _grc_python(a.grc_python)
    if a.status:
        print("\n".join(CK.table(a.out, names)))
        return 0
    fatal, msgs = _old_layout(a.out, a.data_dir)
    for m in msgs:
        common.log(m)
    if fatal and not a.ignore_old_layout:
        common.log("stopping (--ignore-old-layout runs anyway)")
        return 2
    other = [s for s in names if (r := CK.load(a.out, s)) is not None and CK.quick_of(r) != bool(a.quick)]
    if other:
        common.log(f"{a.out} holds steps of a {'full' if a.quick else '--quick'} run ({', '.join(other)}): their rows "
                   f"would be taken for this run's. Use another --out (quick.sh uses results-quick).")
        return 2
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    env = {}
    try:
        env = common.environment()
        common.xio.atomic_json(env, os.path.join(a.out, "environment.json"))
    except Exception as ex:  # environment capture must never block a run
        common.log(f"could not record the environment: {ex}")
    handlers = {sig: signal.signal(sig, _stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _pipeline(a, selected, table, env)
    except Stopped as ex:
        return 128 + ex.signum
    finally:
        for sig, h in handlers.items():
            signal.signal(sig, h)


def _pipeline(a, selected, table, env) -> int:
    outcome = {}
    for step in selected:
        cfg = _config(step, a)
        rec = CK.load(a.out, step)
        ok, diff = CK.done(rec, cfg)
        if ok and not a.force:
            common.log(f"[{step}] done on {rec.get('finished', '?')} (exit 0 after {rec.get('seconds', '?')}s); "
                       f"skipped (--force runs it again)")
            outcome[step] = "skipped"
            continue
        if diff and not a.force:
            common.log(f"[{step}] completed before, but {_rerun_reason(*diff)}: running it again")
        cmd = _command(step, a, table)
        log_path = os.path.join(a.out, "logs", f"{step}.log")
        lock = CK.StepLock(a.out, step)
        try:
            lock.acquire()
        except CK.Busy as ex:
            common.log(f"[{step}] {ex}: skipped")
            outcome[step] = "busy"
            continue
        try:
            common.log(f"[{step}] {' '.join(cmd[1:])}  (log: {log_path})")
            rec = dict(step=step, quick=bool(a.quick), config=cfg, started=CK.now(), host=socket.gethostname(),
                       job=CK.job_id(), git=env.get("git"), cmd=cmd, log=log_path)
            t0 = time.time()
            try:
                rc = _run(cmd, log_path, lock)
            except Stopped as ex:
                CK.save(a.out, step, dict(rec, returncode=-ex.signum, state="interrupted",
                                          seconds=round(time.time() - t0, 1), finished=CK.now()))
                common.log(f"[{step}] interrupted by {_signame(ex.signum)} after {time.time() - t0:.0f}s; "
                           f"it resumes on the next run")
                raise
            CK.save(a.out, step, dict(rec, returncode=rc, state=CK.state(rc), seconds=round(time.time() - t0, 1),
                                      finished=CK.now()))
        finally:
            lock.release()
        common.log(f"[{step}] {_outcome(rc)} in {time.time() - t0:.0f}s")
        outcome[step] = "ok" if rc == 0 else "failed"
    failed = [s for s, o in outcome.items() if o == "failed"]
    busy = [s for s, o in outcome.items() if o == "busy"]
    common.log(f"steps: {sum(o == 'ok' for o in outcome.values())} ran ok, "
               f"{sum(o == 'skipped' for o in outcome.values())} done before"
               + (f", {len(busy)} running elsewhere ({' '.join(busy)})" if busy else "")
               + (f", {len(failed)} failed ({' '.join(failed)}; see {os.path.join(a.out, 'logs')}): "
                  f"they run again next time" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
