"""Step-level checkpoints of ``main.py``.

When a step ends, main.py records how it went in ``<out>/status/<step>.json``: exit code, state,
duration, the configuration it ran with (:func:`done` compares it), host, job, git commit, command.
A step recorded with exit code 0 under the same configuration is done and is not run again.
Every step writes only its own record, so steps that run at the same time (``slurm/array.sbatch``)
never overwrite each other's; ``<out>/status.json`` gathers all records and is only a view
(rebuilt after every step). Records of main.py up to v1.2.0, which kept only ``status.json``, are
read as well.

While a step runs, ``<out>/status/<step>.lock`` names the host, process and job that run it, and
main.py touches it every minute. Another main.py that finds the lock skips the step rather than run
it a second time at once (both would rewrite the same result tables), unless the run that holds it
is gone: Slurm says its job has ended (killed, cancelled, out of memory, time limit), or it is this
very job restarted, or (without Slurm) its process is not running on this host, or nobody touched
the lock for ``STALE_S`` seconds. A job that is still completing gets ``ENDING_WAIT_S`` seconds to
release it. The lock is advisory: two runs that find the same stale lock in the same instant may
both start.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from typing import List, Optional, Tuple

import xio

STALE_S = 300.0       # when Slurm cannot be asked: a lock not touched for this long is stale (touched every minute)
ENDING_WAIT_S = 120.0  # a lock whose job is completing (being killed) is given this long to be released
# Slurm states of a job that may still be running main.py; any other state, or a job Slurm no longer
# knows, means the job that holds a lock has ended (killed, cancelled, out of memory, time limit...)
ALIVE = {"RUNNING", "CONFIGURING", "SUSPENDED", "STOPPED", "RESIZING", "SIGNALING", "STAGE_OUT", "COMPLETING"}
EXIT_INCOMPLETE = 3   # exit code of a step that finished with rows that ended in an error (common.finish)


def status_dir(out: str) -> str:
    return os.path.join(out, "status")


def record_path(out: str, step: str) -> str:
    return os.path.join(status_dir(out), f"{step}.json")


def lock_path(out: str, step: str) -> str:
    return os.path.join(status_dir(out), f"{step}.lock")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def job_id() -> Optional[str]:
    """The SLURM job (``<array job>_<task>`` for a job array), or None."""
    arr, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{arr}_{task}" if arr and task else os.environ.get("SLURM_JOB_ID")


def slurm_job_state(job: Optional[str]) -> Optional[str]:
    """The state Slurm reports for ``job`` (e.g. "RUNNING"); "" if Slurm no longer has it (it ended);
    None if Slurm cannot be asked (no squeue, or squeue failed)."""
    sq = shutil.which("squeue") if job else None
    if not sq:
        return None
    try:
        p = subprocess.run([sq, "-h", "-j", str(job), "-o", "%T"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return "" if "invalid job id" in (p.stderr or "").lower() else None
    states = [x.strip() for x in (p.stdout or "").split() if x.strip()]
    return states[0] if states else ""


def state(rc: Optional[int]) -> str:
    """ok, incomplete (finished, but rows ended in an error), killed (by a signal from outside,
    e.g. the out-of-memory killer), failed (any other exit code)."""
    if rc == 0:
        return "ok"
    if rc == EXIT_INCOMPLETE:
        return "incomplete"
    if isinstance(rc, int) and rc < 0:
        return "killed"
    return "failed"


# ------------------------------------------------------------------ records

def load(out: str, step: str) -> Optional[dict]:
    """The record of the last run of ``step``, or None if it never ran."""
    rec = xio.read_json(record_path(out, step), default=None)
    if isinstance(rec, dict):
        return rec
    # status.json mirrors the per-step records (they carry "record"); an entry without it was
    # written by main.py up to v1.2.0, which kept nothing else
    old = (xio.read_json(os.path.join(out, "status.json"), default={}) or {}).get(step)
    return old if isinstance(old, dict) and "record" not in old else None


def save(out: str, step: str, rec: dict) -> None:
    xio.atomic_json(dict(rec, record=os.path.relpath(record_path(out, step), out)), record_path(out, step))
    summary(out)


def summary(out: str) -> None:
    """Rewrite ``<out>/status.json`` from the per-step records (entries of main.py up to v1.2.0
    stay for the steps that have no record of their own)."""
    path = os.path.join(out, "status.json")
    old = xio.read_json(path, default={}) or {}
    merged = {k: v for k, v in old.items() if isinstance(v, dict) and "record" not in v}
    for p in sorted(glob.glob(os.path.join(status_dir(out), "*.json"))):
        rec = xio.read_json(p, default=None)
        if isinstance(rec, dict):
            merged[os.path.basename(p)[:-len(".json")]] = rec
    xio.atomic_json(merged, path)


def done(rec: Optional[dict], config: dict) -> Tuple[bool, Optional[Tuple[str, object]]]:
    """(True, None) if ``rec`` is a run that ended without errors under ``config``. Otherwise
    (False, None), or (False, (key, value it ran with)) when it ended without errors but under
    another value of ``key`` (value ``KeyError`` when its record does not say)."""
    if not rec or rec.get("returncode") != 0:
        return False, None
    have = rec.get("config")
    for k, v in config.items():
        if isinstance(have, dict) and k in have:
            got = have[k]
        elif k == "quick" and "quick" in rec:  # records of main.py up to v1.2.0
            got = rec["quick"]
        else:
            return False, (k, KeyError)
        if got != v:
            return False, (k, got)
    return True, None


def quick_of(rec: dict) -> bool:
    cfg = rec.get("config")
    return bool(cfg.get("quick") if isinstance(cfg, dict) and "quick" in cfg else rec.get("quick", False))


def inputs_fingerprint(out: str) -> str:
    """Fingerprint of what ``artifacts`` reads: the result tables (name, size, modification time)
    and environment.json (contents, without the host and job, which change with every job)."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(out, "e[1-7]", "**", "*.csv"), recursive=True)):
        try:
            st = os.stat(p)
        except OSError:
            continue
        h.update(f"{os.path.relpath(p, out)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    env = xio.read_json(os.path.join(out, "environment.json"), default=None)
    if isinstance(env, dict):
        h.update(json.dumps({k: v for k, v in env.items() if k not in ("host", "slurm_job")}, sort_keys=True).encode())
    return h.hexdigest()[:16]


# -------------------------------------------------------------------- locks

class Busy(RuntimeError):
    """The step is being run by another process."""


class StepLock:
    """``<out>/status/<step>.lock`` while this process runs ``step`` (see the module docstring)."""

    def __init__(self, out: str, step: str, stale_s: float = STALE_S):
        self.path = lock_path(out, step)
        self.stale_s = stale_s
        self.token = uuid.uuid4().hex
        self.held = False
        self.took_over = None  # why the lock of a run that is gone was taken over, if it was

    def info(self) -> Tuple[dict, Optional[float]]:
        """(contents, seconds since the holder last touched it) of the lock, ({}, None) if none."""
        try:
            age = time.time() - os.path.getmtime(self.path)
        except OSError:
            return {}, None
        info = xio.read_json(self.path, default={})
        return (info if isinstance(info, dict) else {}), age

    def holder(self, info: dict, age: float) -> Tuple[str, str]:
        """("alive" | "ending" | "gone", why) for the process that holds the lock. If the lock names a
        Slurm job, Slurm decides (a lock left by a job that was killed is taken over at once); else
        the process, if it ran on this host; else the time since it was last touched."""
        job = info.get("job")
        if job and job == job_id():
            return "gone", f"it was taken by this job ({job}) before it restarted"
        st = slurm_job_state(job) if job else None
        if st is not None:
            if st == "COMPLETING":
                return "ending", f"job {job} is completing"
            if st in ALIVE:
                return "alive", f"job {job} is {st}"
            return "gone", f"job {job} " + (f"is {st}" if st else "has ended")
        pid = info.get("pid")
        if info.get("host") == socket.gethostname() and isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return "gone", f"process {pid} has ended"
            except OSError:
                pass
            else:
                return "alive", f"process {pid} is running"
        if age > self.stale_s:
            return "gone", f"not touched for {age:.0f}s"
        return "alive", (f"Slurm cannot be asked about job {job}; " if job else "") + \
            f"touched {age:.0f}s ago (stale after {self.stale_s:.0f}s)"

    def stale(self, info: dict, age: float) -> bool:
        return self.holder(info, age)[0] != "alive"

    def describe(self, info: dict, age: float, why: str = "") -> str:
        who = f"{info.get('host', '?')} (pid {info.get('pid', '?')}" + (f", job {info['job']}" if info.get("job") else "")
        return (f"running on {who}) since {info.get('started', '?')}" + (f": {why}" if why else "") +
                f"; if that run is gone, delete {self.path}")

    def acquire(self) -> "StepLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        me = dict(token=self.token, host=socket.gethostname(), pid=os.getpid(), job=job_id(), started=now())
        waited = 0.0
        for _ in range(100):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info, age = self.info()
                if age is None:
                    continue  # released meanwhile
                verdict, why = self.holder(info, age)
                if verdict == "alive":
                    raise Busy(self.describe(info, age, why))
                if verdict == "ending" and waited < ENDING_WAIT_S:
                    time.sleep(10)  # its main.py may still release it
                    waited += 10
                    continue
                try:
                    os.remove(self.path)  # left by a run that was killed
                except FileNotFoundError:
                    pass
                self.took_over = why
                continue
            with os.fdopen(fd, "w") as fh:
                json.dump(me, fh)
            if self.info()[0].get("token") == self.token:
                self.held = True
                return self
        info, age = self.info()
        raise Busy(self.describe(info, age or 0.0))

    def refresh(self) -> None:
        if self.held:
            try:
                os.utime(self.path)
            except OSError:
                pass

    def release(self) -> None:
        if self.held:
            self.held = False
            if self.info()[0].get("token") == self.token:
                try:
                    os.remove(self.path)
                except FileNotFoundError:
                    pass

    def __enter__(self) -> "StepLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# ------------------------------------------------------------------- status

def _dur(s) -> str:
    try:
        s = float(s)
    except (TypeError, ValueError):
        return ""
    if s < 60:
        return f"{s:.0f}s"
    m = int(s // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m{int(s % 60):02d}s"


def table(out: str, steps: List[str]) -> List[str]:
    """One line per step: its state (done, incomplete, failed, killed, interrupted, running,
    not run), exit code, duration and date of its last run."""
    lines = [f"{'step':14s} {'state':12s} {'exit':>5s} {'time':>8s}  when"]
    for step in steps:
        rec = load(out, step)
        lock = StepLock(out, step)
        info, age = lock.info()
        if age is not None and not lock.stale(info, age):
            lines.append(f"{step:14s} {'running':12s} {'':>5s} {'':>8s}  {lock.describe(info, age).split(';')[0]}")
            continue
        if rec is None:
            lines.append(f"{step:14s} {'not run':12s}")
            continue
        rc = rec.get("returncode")
        st = rec.get("state") or state(rc)
        st = "done" if st == "ok" else st
        when = str(rec.get("finished", ""))
        if quick_of(rec):
            when += "  (--quick)"
        lines.append(f"{step:14s} {st:12s} {str(rc):>5s} {_dur(rec.get('seconds')):>8s}  {when}")
    return lines
