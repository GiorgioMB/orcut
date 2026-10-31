"""Step-level checkpoints of ``main.py``.

When a step ends, main.py records how it went in ``<out>/status/<step>.json``: exit code, state,
duration, the configuration it ran with (:func:`done` compares it), host, job, git commit, command.
A step recorded with exit code 0 under the same configuration is done and is not run again.
Every step writes only its own record, so steps that run at the same time (``slurm/array.sbatch``)
never overwrite each other's; ``<out>/status.json`` gathers all records and is only a view
(rebuilt after every step). Records of main.py up to v1.2.0, which kept only ``status.json``, are
read as well.

While a step runs, ``<out>/status/<step>.lock`` names the host, process and job that run it, and
main.py touches it every minute. Another main.py that finds a fresh lock skips the step rather
than run it a second time at once (both would rewrite the same result tables). A lock that nobody
touched for ``STALE_S`` seconds, or whose process is gone (same host), was left by a run that was
killed: it is taken over. The lock is advisory: two runs that find the same stale lock in the
same instant may both start.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import socket
import time
import uuid
from typing import List, Optional, Tuple

import xio

STALE_S = 600.0       # a lock not touched for this long is stale
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

    def info(self) -> Tuple[dict, Optional[float]]:
        """(contents, seconds since the holder last touched it) of the lock, ({}, None) if none."""
        try:
            age = time.time() - os.path.getmtime(self.path)
        except OSError:
            return {}, None
        info = xio.read_json(self.path, default={})
        return (info if isinstance(info, dict) else {}), age

    def stale(self, info: dict, age: float) -> bool:
        if age > self.stale_s:
            return True
        pid = info.get("pid")
        if info.get("host") == socket.gethostname() and isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except OSError:
                return False
        return False

    def describe(self, info: dict, age: float) -> str:
        who = f"{info.get('host', '?')} (pid {info.get('pid', '?')}" + (f", job {info['job']}" if info.get("job") else "")
        return (f"running on {who}) since {info.get('started', '?')}, seen {age:.0f}s ago; if that run is gone, "
                f"delete {self.path} or wait {self.stale_s - age:.0f}s")

    def acquire(self) -> "StepLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        me = dict(token=self.token, host=socket.gethostname(), pid=os.getpid(), job=job_id(), started=now())
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info, age = self.info()
                if age is None:
                    continue  # released meanwhile
                if not self.stale(info, age):
                    raise Busy(self.describe(info, age))
                try:
                    os.remove(self.path)  # left by a run that was killed
                except FileNotFoundError:
                    pass
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
