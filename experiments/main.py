"""Run the experiments of the paper (E1--E7) and build its numbers, tables and figures.

Every step is a separate process (a failure stops only that step) and logs to
``<out>/logs/<step>.log``. The same command can be run again after a time-out, a crash or an
out-of-memory kill, and only what is left is computed. There are checkpoints at two levels:

* steps: a step that ends without errors is recorded in ``<out>/status/<step>.json`` and is not
  run again (``--force`` computes it again from scratch; ``--status`` shows what is done, failed or running). A
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
  python experiments/main.py --steps e2-real --force            # recompute a step from scratch
  python experiments/main.py --steps e5-sinkhorn -- --budget 3600   # options after -- go to the step
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


GRC_ENV = "orcut-grc"  # name of the GRC environment (experiments/grc-environment.yml)


def _grc_python(explicit):
    """(interpreter of the GRC environment as an absolute path, or None; where it came from).
    Looked for in this order: ``--grc-python``; $ORCUT_GRC_PYTHON; an environment named orcut-grc
    next to the one running main.py (conda keeps environments side by side, e.g.
    /scratch/.../conda_envs/orcut and /scratch/.../conda_envs/orcut-grc; this needs no conda in
    the job); ``conda run -n orcut-grc``. A bare command name is looked up on PATH."""
    for p, src in ((explicit, "--grc-python"), (os.environ.get("ORCUT_GRC_PYTHON"), "$ORCUT_GRC_PYTHON")):
        if p:
            if os.sep in p or p.startswith("~"):
                return _abspath(p), src
            return shutil.which(p) or p, src
    sib = os.path.join(os.path.dirname(os.path.abspath(sys.prefix)), GRC_ENV, "bin", "python")
    if os.path.exists(sib):
        return os.path.abspath(sib), f"the {GRC_ENV} environment next to {sys.prefix}"
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("mamba")
    if conda:
        try:
            out = subprocess.run([conda, "run", "-n", GRC_ENV, "python", "-c", "import sys; print(sys.executable)"],
                                 capture_output=True, text=True, timeout=300)
            p = (out.stdout.strip().splitlines() or [""])[-1]
            if out.returncode == 0 and os.path.exists(p):
                return p, f"conda run -n {GRC_ENV}"
        except (OSError, subprocess.SubprocessError):
            pass
    return None, "not found"


def _grc_check(py):
    """(version, None) if GraphRicciCurvature imports in the interpreter ``py``, else (None, why)."""
    if not py:
        return None, "no GRC interpreter (--grc-python, $ORCUT_GRC_PYTHON)"
    if not os.path.exists(py):
        return None, f"{py} does not exist"
    code = ("import GraphRicciCurvature.OllivierRicci; from importlib.metadata import version; "
            "print(version('GraphRicciCurvature'))")
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as ex:
        return None, f"{py} could not be run ({ex})"
    if p.returncode != 0:
        lines = (p.stderr or "").strip().splitlines()
        return None, f"GraphRicciCurvature does not import in {py} ({lines[-1] if lines else f'exit {p.returncode}'})"
    return (p.stdout.strip().splitlines() or ["?"])[-1], None


def _old_place(p):
    """Where main.py up to v1.2.0 put ``p``: it ran every step from experiments/ and passed the
    relative --out and --data-dir of the SLURM scripts unchanged, so the steps wrote their tables
    to experiments/<out> (logs and status.json stayed in <out>) and their downloads to
    experiments/<data-dir>, both relative to the repository root. None if p is not in the repo."""
    root = os.path.dirname(HERE)
    rel = os.path.relpath(p, root)
    return None if rel == os.curdir or rel.startswith(os.pardir) else os.path.join(HERE, rel)


# The row key of every result table, as in the Rows(...) of the experiments (a test checks this).
TABLE_KEYS = {
    "e1/exactness.csv": ["graph", "measure"],
    "e2/controlled.csv": ["sweep", "regime", "s", "seed"],
    "e2/real_bins.csv": ["graph", "bin"],
    "e2/dasgupta.csv": ["kind", "n", "seed"],
    "e3/real.csv": ["graph", "measure"],
    "e3/threads.csv": ["graph", "threads"],
    "e3/synth.csv": ["family", "regime", "n", "seed"],
    "e4/ablation.csv": ["graph", "config"],
    "e5/whole.csv": ["graph", "method"],
    "e5/sinkhorn.csv": ["graph", "method", "reg"],
    "e6/borf.csv": ["graph", "backend"],
    "e6/borf_iterations.csv": ["graph", "backend", "iteration"],
    "e6/community.csv": ["graph", "method"],
    "e7/measures.csv": ["graph", "measure"],
    "e7/radius.csv": ["graph", "config"],
}


# What each step writes into --out: --force moves these aside (to <out>/replaced/) before the
# step runs, so that it computes everything again (a test checks that every table has one step).
STEP_OUTPUTS = {
    "e1": ["e1/exactness.csv"],
    "e2-controlled": ["e2/controlled.csv"],
    "e2-real": ["e2/real_bins.csv"],
    "e2-dasgupta": ["e2/dasgupta.csv"],
    "e3-real": ["e3/real.csv"],             # and its saved chunks, e3/partial/<graph>.*.json
    "e3-threads": ["e3/threads.csv"],
    "e3-synth": ["e3/synth.csv"],           # and its saved chunks, e3/partial/synth.*.json
    "e4": ["e4/ablation.csv"],
    "e5-whole": ["e5/whole.csv", "e5/values/*"],
    "e5-sinkhorn": ["e5/sinkhorn.csv"],
    "e6-borf": ["e6/borf.csv", "e6/borf_iterations.csv"],
    "e6-community": ["e6/community.csv"],
    "e7-measures": ["e7/measures.csv"],
    "e7-radius": ["e7/radius.csv"],
}


def _step_outputs(step, out):
    files = [f for p in STEP_OUTPUTS.get(step, []) for f in glob.glob(os.path.join(out, p))]
    chunks = glob.glob(os.path.join(out, "e3", "partial", "*.json"))
    if step == "e3-real":
        files += [f for f in chunks if not os.path.basename(f).startswith("synth.")]
    elif step == "e3-synth":
        files += [f for f in chunks if os.path.basename(f).startswith("synth.")]
    return sorted(f for f in set(files) if os.path.isfile(f))


def _set_aside(step, out):
    """--force: move the results of ``step`` to <out>/replaced/<step>-<time>/ (same layout), so that
    it starts from empty tables. Returns (folder, ["e5/sinkhorn.csv (70 rows)", ...])."""
    files = _step_outputs(step, out)
    dest = os.path.join(out, "replaced", f"{step}-{time.strftime('%Y%m%d-%H%M%S')}")
    what = []
    for f in files:
        rel = os.path.relpath(f, out)
        n = len(common.read_csv(f)) if f.endswith(".csv") else None
        _move(f, os.path.join(dest, rel))
        what.append(rel + (f" ({n} rows)" if n is not None else ""))
    return dest, what


def _merge_rows(older, newer, key):
    """One row per key from two copies of a table: the newer copy's row, unless only the older one
    holds a result (common.row_ok). Without a key (an unknown table), identical rows once."""
    k = (lambda r: tuple(str(r.get(c)) for c in key)) if key else (lambda r: tuple(sorted(r.items())))
    merged = {}
    for r in older:
        merged[k(r)] = r
    for r in newer:
        old = merged.get(k(r))
        if old is None or common.row_ok(r) or not common.row_ok(old):
            merged[k(r)] = r
    return list(merged.values())


def _move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError:  # another file system
        shutil.move(src, dst)


def _adopt_old_layout(out, data_dir):
    """Move into place what main.py up to v1.2.0 left in the wrong place (_old_place): every file
    under experiments/<out>/e1..e7 goes to <out>/e1..e7; a table in both places is merged row by row
    (_merge_rows, by TABLE_KEYS); another file in both places keeps the newer copy. Downloads in
    experiments/<data-dir> move to <data-dir> unless they are there already. Tasks of a job array
    that start together take turns (a lock in <out>/status). Returns the lines to log."""
    old = _old_place(out)
    data = data_dir or os.path.join(HERE, "data", "real")
    old_data = _old_place(data)
    todo = bool(old and glob.glob(os.path.join(old, "e[1-7]", "**", "*"), recursive=True)) or \
        bool(old_data and os.path.isdir(old_data) and os.listdir(old_data))
    if not todo:
        return []
    lock = CK.StepLock(out, "move-old-layout")
    deadline = time.time() + 1800
    while True:
        try:
            lock.acquire()
            break
        except CK.Busy:
            if time.time() > deadline:
                raise
            time.sleep(5)  # another task is moving them; then there is nothing left to move
    msgs = []
    try:
        moved, merged, kept, writers = [], [], [], []
        files = sorted(f for f in glob.glob(os.path.join(old, "e[1-7]", "**", "*"), recursive=True)
                       if os.path.isfile(f)) if old else []
        for src in files:
            rel = os.path.relpath(src, old)
            dst = os.path.join(out, rel)
            if not os.path.exists(dst):
                _move(src, dst)
                moved.append(rel)
            elif src.endswith(".csv"):
                src_newer = os.path.getmtime(src) > os.path.getmtime(dst)
                a_rows, b_rows = common.read_csv(dst), common.read_csv(src)
                older, newer = (a_rows, b_rows) if src_newer else (b_rows, a_rows)
                rows = _merge_rows(older, newer, TABLE_KEYS.get(rel.replace(os.sep, "/")))
                common.write_csv(rows, dst)
                os.remove(src)
                merged.append(f"{rel} ({len(b_rows)} rows from the old place, {len(a_rows)} here, {len(rows)} now)")
                if src_newer:
                    writers.append(rel)
            else:
                if os.path.getmtime(src) > os.path.getmtime(dst):
                    _move(src, dst)
                else:
                    os.remove(src)
                kept.append(rel)
        if old:
            for d, _, _ in sorted(os.walk(old), key=lambda t: -len(t[0])):  # empty folders, deepest first
                if d != old:
                    try:
                        os.rmdir(d)
                    except OSError:
                        pass
        if moved or merged or kept:
            msgs.append(f"found result files of an earlier run in {old} (main.py up to v1.2.0 put them there): "
                        f"{len(moved)} moved to {out}, {len(merged)} tables merged into their copies there"
                        + (f", {len(kept)} other files replaced by their newer copy" if kept else ""))
            msgs += [f"  merged {m}" for m in merged]
        if writers:
            msgs.append(f"WARNING: {', '.join(writers)} in {old} had rows newer than {out}: a job running the code "
                        f"from before the update is probably still writing there. Cancel it (squeue -u $USER); its "
                        f"rows are kept, they are merged again each time main.py starts")
        if old_data and os.path.isdir(old_data):
            n_moved, n_dup, n_diff = 0, 0, 0
            for name in sorted(os.listdir(old_data)):
                src, dst = os.path.join(old_data, name), os.path.join(data, name)
                if not os.path.exists(dst):
                    _move(src, dst)
                    n_moved += 1
                elif os.path.isfile(src) and os.path.isfile(dst) and os.path.getsize(src) == os.path.getsize(dst):
                    os.remove(src)  # the same download twice
                    n_dup += 1
                else:
                    n_diff += 1
            if n_moved or n_dup:
                msgs.append(f"downloads of an earlier run in {old_data}: {n_moved} moved to {data}, "
                            f"{n_dup} deleted (the same file was there already)")
            if n_diff:
                msgs.append(f"left {n_diff} files in {old_data} that differ from those of the same name in {data}")
            for d in (old_data, os.path.dirname(old_data), os.path.dirname(os.path.dirname(old_data))):
                try:
                    os.rmdir(d)  # experiments/experiments/data/real, if empty now
                except OSError:
                    break
    finally:
        lock.release()
    return msgs


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
    return cmd + list(getattr(a, "step_args", []))


# Steps whose rules changed so that a completed run left rows the step computes now: a record
# with an older (or no) number is not done, and the step runs again (for those rows only).
RULES = {
    "e7-radius": 2,      # 2: large supports run with fewer threads instead of being skipped
    "e2-controlled": 2,  # 2: no time budget; POT and the LP solvers wherever they fit in the job's memory
    "e2-dasgupta": 2,    # 2: the same
    "e3-real": 2,        # 2: Skitter and LiveJournal get the non-lazy and LLY measures too
    "e5-sinkhorn": 2,    # 2: no time budget
    "e5-whole": 2,       # 2: GraphRicciCurvature without a time limit (its timeouts, and what they skipped, run)
    "e6-borf": 2,        # 2: the same
}


def _config(step, a):
    """What a completed run of ``step`` must have run with to count as done: the same --quick and
    rules (RULES); for the GRC steps, the same GRC interpreter (a run without GRC is done until
    GRC is available); for artifacts, the same --paper-dir and the same result tables."""
    cfg = {"quick": bool(a.quick)}
    if step in RULES:
        cfg["rules"] = RULES[step]
    if getattr(a, "step_args", None) and step != "artifacts":
        cfg["step_args"] = list(a.step_args)  # a run with other options for the step is another run
    if step in GRC:
        cfg["grc"] = a.grc_python if a.grc_python and os.path.exists(a.grc_python) else None
    if step == "artifacts":
        cfg["paper_dir"] = a.paper_dir
        cfg["inputs"] = CK.inputs_fingerprint(a.out)
    return cfg


def _rerun_reason(key, got):
    if key == "step_args":
        return "it ran without these options for the step" if got is KeyError else \
            f"it ran with other options for the step ({' '.join(map(str, got))})"
    if key == "rules":
        return "it ran under earlier rules for this step, which skipped rows that are computed now"
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


def _run(cmd, log_path, lock, env=None) -> int:
    """Run one step and return its exit code, touching its lock every HEARTBEAT_S seconds. If
    main.py is stopped meanwhile (Stopped, KeyboardInterrupt), the step is terminated first."""
    with open(log_path, "a") as fh:
        fh.write(f"\n===== {CK.now()} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
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
                    help="compute the selected steps again from scratch, even if they completed before: their result "
                         "tables are moved to <out>/replaced/<step>-<time>/ first")
    ap.add_argument("--status", action="store_true", help="show which steps are done, failed or running, and exit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ignore-old-layout", action="store_true",
                    help="leave the result tables and downloads of an earlier run where main.py up to v1.2.0 put "
                         "them (by default they are moved into --out and --data-dir)")
    raw = list(sys.argv[1:] if argv is None else argv)
    step_args = []
    if "--" in raw:  # everything after -- goes to the step itself (e.g. -- --budget 3600)
        k = raw.index("--")
        raw, step_args = raw[:k], raw[k + 1:]
    a = ap.parse_args(raw)
    a.step_args = step_args
    if not (a.list or a.status):
        common.log("main.py " + " ".join(sys.argv[1:] if argv is None else argv))
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
    a.grc_python, grc_src = _grc_python(a.grc_python)
    if a.status:
        print("\n".join(CK.table(a.out, names)))
        return 0
    if not a.ignore_old_layout:
        for m in _adopt_old_layout(a.out, a.data_dir):
            common.log(m)
    other = [s for s in names if (r := CK.load(a.out, s)) is not None and CK.quick_of(r) != bool(a.quick)]
    if other:
        common.log(f"{a.out} holds steps of a {'full' if a.quick else '--quick'} run ({', '.join(other)}): their rows "
                   f"would be taken for this run's. Use another --out (quick.sh uses results-quick).")
        return 2
    a.env = dict(os.environ)  # the environment of the steps
    grc_steps = [s for s in selected if s in GRC]
    if grc_steps:
        # GRC counts as available only if it imports: a run without it records its rows as
        # unavailable (and runs again once it is), instead of one error per row
        version, why = _grc_check(a.grc_python)
        if version:
            common.log(f"GraphRicciCurvature {version}: {a.grc_python} (from {grc_src}), for {' '.join(grc_steps)}")
            a.env["ORCUT_GRC_PYTHON"] = a.grc_python
        else:
            common.log(f"GraphRicciCurvature UNAVAILABLE ({why}; interpreter {grc_src}): the GRC rows of "
                       f"{' '.join(grc_steps)} stay unavailable until it imports, then these steps run again. "
                       f"Set ORCUT_GRC_PYTHON or --grc-python to the python of the {GRC_ENV} environment.")
            a.grc_python = None
            a.env.pop("ORCUT_GRC_PYTHON", None)  # the steps must not try it either
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
                       f"skipped (--force computes it again from scratch)")
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
        if lock.took_over:
            common.log(f"[{step}] took over the lock left by a run that is gone ({lock.took_over})")
        if a.force and step != "artifacts":
            try:
                dest, what = _set_aside(step, a.out)
            except BaseException:
                lock.release()
                raise
            if what:
                tables = [w for w in what if ".csv" in w]
                common.log(f"[{step}] --force: computing everything again; its results so far are in {dest}: "
                           + (", ".join(tables) if tables else "")
                           + (f" and {len(what) - len(tables)} other files" if len(what) > len(tables) else ""))
            else:
                common.log(f"[{step}] --force: no results of this step yet, computing everything")
        try:
            common.log(f"[{step}] {' '.join(cmd[1:])}  (log: {log_path})")
            rec = dict(step=step, quick=bool(a.quick), config=cfg, started=CK.now(), host=socket.gethostname(),
                       job=CK.job_id(), git=env.get("git"), cmd=cmd, log=log_path)
            t0 = time.time()
            try:
                rc = _run(cmd, log_path, lock, getattr(a, "env", None))
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
