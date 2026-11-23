"""Run the experiments of the paper (E1--E6) and build its numbers, tables and figures.

Every step runs in its own process and logs to ``<out>/logs/<step>.log``. A step that ends without
errors is recorded in ``<out>/status/<step>.json`` and skipped next time (``--force`` recomputes it);
every step also saves each row as soon as it is computed, so a run stopped by a time limit, a crash
or an out-of-memory kill resumes where it stopped when the same command is run again.

  python experiments/main.py --threads 32                       # everything, in order
  python experiments/main.py --steps e3-real e3-threads         # some steps
  python experiments/main.py --status                           # what is done, failed or running
  python experiments/main.py --steps e5-sinkhorn -- --per-bin 20    # options after -- go to the step
  python experiments/main.py --steps e5-whole --worker 2        # a second job sharing e5-whole's GRC runs
  python experiments/main.py --quick --out results-quick        # smoke test
"""
from __future__ import annotations

import argparse
import glob
import re
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import checkpoint as CK
import common
import partial as PT

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
    "e6-measures": (["e6_generality.py", "--part", "measures"], "other one-step measures"),
    "e6-radius": (["e6_generality.py", "--part", "radius"], "balls and multi-step walks (layered closure)"),
    "calib": (["e3_scale.py", "--part", "real"], "calibration: two graphs timed again on this type of node"),
}
THREADED = {"e3-real", "e3-threads", "e3-synth", "e5-whole", "e6-measures", "e6-radius", "calib"}

# Calibration across node types (experiments/hardware.py): the step "calib" times these graphs with
# E3's protocol on the node it runs on, into <out>/calib/<processor>/, so that a run on each type of
# node (array.sbatch: --array=12 --constraint=<type>) adds that type. --quick: bundled graphs.
CALIB_GRAPHS = ["real:DBLP", "real:Epinions"]
CALIB_QUICK_GRAPHS = ["bundled:karate", "bundled:jazz"]
CALIB_ARGS = ["--reps", "3", "--pot-per-bin", "20"]
CALIB_QUICK_ARGS = ["--reps", "1", "--pot-per-bin", "5"]  # e3's own --quick would replace the graphs


def cpu_model() -> str:
    """The processor of this node, as the operating system names it."""
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        import platform
        return platform.processor() or "unknown processor"


def cpu_name(model: str) -> str:
    """'Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz' -> 'Xeon Gold 6248'; 'AMD EPYC 7713 64-Core Processor' -> 'EPYC 7713'."""
    s = re.sub(r"\((R|TM)\)|@.*$|\b\d+-Core\b|\bProcessor\b|\bCPU\b", " ", model)
    s = re.sub(r"^\s*(Intel|AMD)\b", " ", s)
    return " ".join(s.split()) or model.strip()


def cpu_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"


def _calib_dir(out: str) -> str:
    return os.path.join(out, "calib", cpu_slug(cpu_name(cpu_model())))


def record_node(out: str) -> None:
    """Add this node and its processor to <out>/provenance/nodes.txt (read by experiments/hardware.py),
    so that the provenance of the rows needs no list of node types."""
    try:
        path = os.path.join(out, "provenance", "nodes.txt")
        host, cpu = socket.gethostname().split(".")[0], cpu_name(cpu_model())
        known = open(path).read().split("\n") if os.path.exists(path) else []
        if f"{host} {cpu}" not in known:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as fh:
                fh.write(f"{host} {cpu}\n")
    except OSError as ex:  # the record must never block a run
        common.log(f"could not record the node: {ex}")


def _provenance(a) -> None:
    """Before the artifacts: which node produced each row (tools/provenance.py), from the logs and the
    scheduler's records, if Slurm's sacct is there; otherwise nothing (and the table is left out)."""
    logs = os.path.join(a.out, "logs")
    if not shutil.which("sacct") or not os.path.isdir(logs):
        return
    try:
        days = sorted(m.group(1) for f in glob.glob(os.path.join(logs, "*.log"))
                      for m in re.finditer(r"^===== (\d{4}-\d\d-\d\d)", open(f, errors="replace").read(), re.M))
        if not days:
            return
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        res = subprocess.run(["sacct", "-u", user, "-S", days[0], "-X", "-n", "-o",
                              "JobID%25,JobName%25,NodeList%20,Start,End"], capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            common.log(f"[artifacts] provenance skipped: sacct failed ({res.stderr.strip()[:200]})")
            return
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
        import provenance as PV
        d = os.path.join(a.out, "provenance")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "sacct.txt"), "w") as fh:
            fh.write(res.stdout)
        jobs = PV.read_sacct(os.path.join(d, "sacct.txt"))
        rows = []
        for step in PV.STEPS:
            log = os.path.join(logs, f"{step}.log")
            if os.path.exists(log):
                rows += PV.trace(step, log, jobs)
        import io, csv as _csv
        buf = io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=["step", "key", "written", "jobs", "nodes"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        path = os.path.join(d, "rows.csv")
        if not os.path.exists(path) or open(path).read() != buf.getvalue():  # unchanged: the artifacts need not run
            with open(path + ".tmp", "w") as fh:
                fh.write(buf.getvalue())
            os.replace(path + ".tmp", path)
        common.log(f"[artifacts] provenance: {len(rows)} rows traced to their node ({path})")
    except Exception as ex:  # the provenance must never block the artifacts
        common.log(f"[artifacts] provenance skipped: {ex}")
GRC = {"e5-whole"}

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
    "e6-measures": ["e6/measures.csv"],
    "e6-radius": ["e6/radius.csv"],
    "calib": [],  # calib/<this node's processor>/, found by _step_outputs
}


def _step_outputs(step, out):
    files = [f for p in STEP_OUTPUTS.get(step, []) for f in glob.glob(os.path.join(out, p))]
    chunks = glob.glob(os.path.join(out, "e3", "partial", "*.json"))
    if step == "e3-real":
        files += [f for f in chunks if not os.path.basename(f).startswith("synth.")]
    elif step == "e3-synth":
        files += [f for f in chunks if os.path.basename(f).startswith("synth.")]
    elif step == "calib":
        files += glob.glob(os.path.join(_calib_dir(out), "**", "*"), recursive=True)
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


def _move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError:  # another file system
        shutil.move(src, dst)


def _command(step, a, steps):
    if step == "artifacts":
        cmd = [sys.executable, os.path.join(HERE, "artifacts.py"), "--results", a.out]
        if a.paper_dir:
            cmd += ["--paper-dir", a.paper_dir]
        if a.quick:
            cmd += ["--provisional"]
        return cmd
    script, *args = steps[step][0]
    out = a.out
    if step == "calib":
        out = _calib_dir(a.out)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "cpu.txt"), "w") as fh:
            fh.write(cpu_name(cpu_model()) + "\n")
        args = args + (["--graphs", *CALIB_QUICK_GRAPHS, *CALIB_QUICK_ARGS] if a.quick
                       else ["--graphs", *CALIB_GRAPHS, *CALIB_ARGS])
    cmd = [sys.executable, os.path.join(HERE, script), *args, "--out", out]
    if a.data_dir:
        cmd += ["--data-dir", a.data_dir]
    if step in THREADED:
        cmd += ["--threads", str(a.threads)]
    if step in GRC and a.grc_python:
        cmd += ["--grc-python", a.grc_python]
    if a.quick and step != "calib":
        cmd += ["--quick"]
    return cmd + list(getattr(a, "step_args", []))


# Steps that can run in several jobs at once (--worker): they claim each unit of work (e5-whole: each
# GRC run; e3: each chunk of a long solve, each other row) before computing it, and their tables are
# written under a lock (common.Rows.add).
WORKER_STEPS = {"e5-whole", "e3-real", "e3-synth"}  # e3: the chunks of a long row, and the rows themselves


MEMORY_STATUS = ("skipped: needs", "out of memory")  # rows that only a job with more memory can compute


def _memory_gated(step, out) -> int:
    """Rows of the step's tables left for a job with more memory: a status column (``status`` or
    ``<method>_status``) that says it was skipped for memory or killed at the memory limit, or edges
    skipped for memory (``skipped_memory``, E5's Sinkhorn part)."""
    n = 0
    for f in _step_outputs(step, out):
        if f.endswith(".csv"):
            n += sum(1 for r in common.read_csv(f)
                     if common.num(r, "skipped_memory", 0.0) > 0 or
                     any((k == "status" or k.endswith("_status")) and str(v or "").startswith(MEMORY_STATUS)
                         for k, v in r.items()))
    return n


def _more_memory(rec, step, out):
    """Why a step that completed before runs again in this job, for its rows left for more memory:
    there are such rows, and this job has more memory than the one that completed it (or its record
    does not say how much it had). None otherwise; the step itself decides which rows fit now."""
    mem, had = PT.memory_limit(), rec.get("mem_bytes")
    if not mem or (had is not None and not mem > 1.01 * float(had)):
        return None
    n = _memory_gated(step, out)
    if not n:
        return None
    return (f"{n} row(s) were skipped for memory or ran out of it, and this job has {mem / 2**30:.0f} GB"
            + (f" (it completed with {float(had) / 2**30:.0f} GB)" if had is not None else ""))


def _config(step, a):
    """What a completed run of ``step`` must have run with to count as done: the same --quick and
    step options; for the GRC steps, the same GRC interpreter (a run without GRC is done until GRC
    is available); for artifacts, the same --paper-dir and the same result tables."""
    cfg = {"quick": bool(a.quick)}
    if getattr(a, "step_args", None) and step != "artifacts":
        cfg["step_args"] = list(a.step_args)  # a run with other options for the step is another run
    if step in GRC:
        cfg["grc"] = a.grc_python if a.grc_python and os.path.exists(a.grc_python) else None
    if step == "artifacts":
        cfg["paper_dir"] = a.paper_dir
        cfg["inputs"] = CK.inputs_fingerprint(a.out)
    if step == "calib":
        cfg["cpu"] = cpu_slug(cpu_name(cpu_model()))  # each type of node has its own calibration
    return cfg


def _rerun_reason(key, got):
    if key == "step_args":
        return "it ran without these options for the step" if got is KeyError else \
            f"it ran with other options for the step ({' '.join(map(str, got))})"
    if got is KeyError:
        return f"its record does not say which {key} it ran with"
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
        return ("finished, but not everything is done: rows that ended in an error, or work left to other workers "
                "(see the log); it runs again next time")
    if rc < 0:
        hint = " -- from outside the step; usually the memory limit of the job (see MaxRSS in sacct)" if rc == -9 else ""
        return f"FAILED: killed by {_signame(-rc)}{hint}"
    return f"FAILED (exit {rc})"


def _run(cmd, log_path, lock, env=None) -> int:
    """Run one step and return its exit code, touching its lock every HEARTBEAT_S seconds. If
    main.py is stopped meanwhile (Stopped, KeyboardInterrupt), the step is terminated first."""
    with open(log_path, "a") as fh:
        fh.write(f"\n===== {CK.now()} {' '.join(cmd)}  [host {socket.gethostname().split('.')[0]}, {cpu_name(cpu_model())}]\n")
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
    ap.add_argument("--worker", type=int, default=0,
                    help=f"run as worker K (1, 2, ...) of a step that shares its work between several jobs at once "
                         f"({', '.join(sorted(WORKER_STEPS))}): each takes the runs nobody else is computing")
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
    if a.worker:
        if a.force:
            ap.error("--force cannot be used with --worker (it would set aside the tables other workers are writing)")
        if set(selected) - WORKER_STEPS:
            ap.error(f"--worker: only {', '.join(sorted(WORKER_STEPS))} can share their work between jobs")
    # every step gets absolute paths (resolved against the directory main.py was started from)
    a.out, a.data_dir, a.paper_dir = _abspath(a.out), _abspath(a.data_dir), _abspath(a.paper_dir)
    a.grc_python, grc_src = _grc_python(a.grc_python)
    if a.status:
        print("\n".join(CK.table(a.out, names)))
        return 0
    other = [s for s in names if (r := CK.load(a.out, s)) is not None and CK.quick_of(r) != bool(a.quick)]
    if other:
        common.log(f"{a.out} holds steps of a {'full' if a.quick else '--quick'} run ({', '.join(other)}): their rows "
                   f"would be taken for this run's. Use another --out (e.g. results-quick for --quick).")
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
        record_node(a.out)  # every step of this run runs here; before any fingerprint of the artifacts
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
        if step == "artifacts":
            _provenance(a)
        cfg = _config(step, a)
        rec = CK.load(a.out, step)
        ok, diff = CK.done(rec, cfg)
        if ok and not a.force:
            why = _more_memory(rec, step, a.out)
            if why:
                common.log(f"[{step}] completed before, but {why}: running it again (for these rows only)")
                ok = False
        if ok and not a.force:
            common.log(f"[{step}] done on {rec.get('finished', '?')} (exit 0 after {rec.get('seconds', '?')}s); "
                       f"skipped (--force computes it again from scratch)")
            outcome[step] = "skipped"
            continue
        if diff and not a.force:
            common.log(f"[{step}] completed before, but {_rerun_reason(*diff)}: running it again")
        cmd = _command(step, a, table)
        log_path = os.path.join(a.out, "logs", f"{step}.log")
        lock = CK.StepLock(a.out, f"{step}.worker{a.worker}" if getattr(a, "worker", 0) else step)
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
                       job=CK.job_id(), git=env.get("git"), cmd=cmd, log=log_path, mem_bytes=PT.memory_limit())
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
