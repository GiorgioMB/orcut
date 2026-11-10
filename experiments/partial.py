"""Checkpoints inside a row. The parts of a row (one method's solve, one sampled edge's timing) are
saved as soon as each is done, so a run stopped in the middle of a row resumes with the parts it
has not done yet.

A part that was started but never finished (the job was killed while computing it) is looked up in
Slurm. If the job hit its memory limit or its time limit, the part would be killed again under the
same limits: it is given up (recorded as such in the row) and tried again only by a job with more
memory, or more time. Any other interruption (cancelled, node failure, ...) is tried again, at most
ATTEMPTS times.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from typing import Dict, Optional, Tuple

import xio

ATTEMPTS = 3
MEM_SHARE = 0.85  # of the job's memory that one computation may plan to use


def _slurm_seconds(s: str) -> Optional[float]:
    """Seconds of a Slurm time ("2-00:00:00", "12:00:00", "30:00"); None if unlimited/unknown."""
    s = (s or "").strip()
    if not s or s.upper() in ("UNLIMITED", "INFINITE", "NOT_SET", "N/A"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(float(x)) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return float(((days * 24 + h) * 60 + m) * 60 + sec)


def memory_limit() -> Optional[float]:
    """Bytes this job may use: Slurm's --mem, else the cgroup's limit, else the machine's memory."""
    env = os.environ
    if env.get("SLURM_MEM_PER_NODE"):
        return float(env["SLURM_MEM_PER_NODE"]) * 2**20
    if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_PER_TASK"):
        return float(env["SLURM_MEM_PER_CPU"]) * float(env["SLURM_CPUS_PER_TASK"]) * 2**20
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and float(v) < 2**60:
                return float(v)
        except OSError:
            pass
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError):
        return None


def memory_available() -> Optional[float]:
    """Bytes one computation may plan to use now: MEM_SHARE of the limit, minus what is in use."""
    lim = memory_limit()
    if lim is None:
        return None
    try:
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        rss = 0
    return max(0.0, MEM_SHARE * lim - rss)


def time_limit() -> Optional[float]:
    """Seconds of this Slurm job's time limit, or None (not under Slurm, or unlimited)."""
    job = os.environ.get("SLURM_JOB_ID")
    sq = shutil.which("squeue") if job else None
    if not sq:
        return None
    try:
        p = subprocess.run([sq, "-h", "-j", job, "-o", "%l"], capture_output=True, text=True, timeout=60)
        return _slurm_seconds((p.stdout or "").split("\n")[0]) if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def end_state(job: Optional[str]) -> Optional[str]:
    """How a Slurm job ended (TIMEOUT, OUT_OF_MEMORY, CANCELLED, NODE_FAIL, ...), or None if unknown."""
    sa = shutil.which("sacct") if job else None
    if not sa:
        return None
    try:
        p = subprocess.run([sa, "-n", "-X", "-P", "-j", str(job), "-o", "State"], capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    words = (p.stdout or "").split()
    return words[0].upper() if p.returncode == 0 and words else None


def _gb(x) -> str:
    return f"{float(x) / 2**30:.0f} GB"


class Parts:
    """The saved parts of the rows of one table: ``<path>`` (JSON), {row key: {part: entry}}."""

    def __init__(self, path: str):
        self.path = path
        data = xio.read_json(path, default=None)
        self.data: Dict[str, Dict[str, dict]] = data if isinstance(data, dict) else {}
        self._touched = set()  # the row keys this process changed (merged into the file on save)

    @staticmethod
    def key(rowkey) -> str:
        return "|".join(str(x) for x in rowkey)

    def _save(self) -> None:
        with xio.locked(self.path):  # parallel workers share the file: keep their rows
            disk = xio.read_json(self.path, default=None)
            disk = disk if isinstance(disk, dict) else {}
            for k in self._touched:
                if k in self.data:
                    disk[k] = self.data[k]
                else:
                    disk.pop(k, None)
            self.data = disk
            xio.atomic_json(self.data, self.path)

    def resume(self, rowkey, part) -> Tuple[str, Optional[dict]]:
        """("todo", None), ("done", result), or ("gave_up", fields for the row: status and the limits
        it was killed under, which ``retry_now`` compares with those of a later job)."""
        e = self.data.get(self.key(rowkey), {}).get(part)
        if e is None:
            return "todo", None
        if e.get("state") == "done":
            return "done", e.get("result") or {}
        last = (e.get("attempts") or [{}])[-1]
        st = end_state(last.get("slurm_job"))
        mem, tl = memory_limit(), time_limit()
        if st == "OUT_OF_MEMORY":
            if mem and last.get("mem") and mem > 1.01 * last["mem"]:
                return "todo", None
            return "gave_up", {"status": f"out of memory with {_gb(last.get('mem') or 0)} (more --mem runs it again)",
                               "mem_bytes": last.get("mem")}
        if st == "TIMEOUT":
            if tl and last.get("time_limit") and tl > 1.01 * last["time_limit"]:
                return "todo", None
            return "gave_up", {"status": "did not finish within the time limit of its job (a longer --time runs it again)",
                               "time_limit_s": last.get("time_limit")}
        if len(e.get("attempts") or []) >= ATTEMPTS:
            return "gave_up", {"status": f"interrupted {len(e['attempts'])} times ({st or 'cause unknown'})"}
        return "todo", None

    def begin(self, rowkey, part) -> None:
        self._touched.add(self.key(rowkey))
        e = self.data.setdefault(self.key(rowkey), {}).setdefault(part, {"attempts": []})
        e["state"] = "running"
        e.setdefault("attempts", []).append(dict(slurm_job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(),
                                                 started=time.time(), mem=memory_limit(), time_limit=time_limit()))
        self._save()

    def finish(self, rowkey, part, result: dict) -> None:
        self._touched.add(self.key(rowkey))
        e = self.data.setdefault(self.key(rowkey), {}).setdefault(part, {"attempts": []})
        e.update(state="done", result=result)
        self._save()

    def drop(self, rowkey) -> None:
        if self.data.pop(self.key(rowkey), None) is not None:
            self._touched.add(self.key(rowkey))
            self._save()


def retry_now(row: dict, prefix: str = "") -> bool:
    """Whether the part ``prefix`` of a row in the table (its columns <prefix>_status, ...; without a
    prefix: status, ...), given up by an earlier job (skipped for memory, or killed at the memory or
    time limit), can run in this job, which has more."""
    import common

    pre = f"{prefix}_" if prefix else ""
    st = str(row.get(f"{pre}status") or "")
    if st.startswith("skipped: needs"):
        need, avail = common.num(row, f"{pre}need_bytes"), memory_available()
        return bool(avail) and need == need and need <= avail
    if st.startswith("out of memory"):
        had, mem = common.num(row, f"{pre}mem_bytes"), memory_limit()
        return bool(mem) and had == had and mem > 1.01 * had
    if st.startswith("did not finish"):
        had, tl = common.num(row, f"{pre}time_limit_s"), time_limit()
        return bool(tl) and had == had and tl > 1.01 * had
    return False
