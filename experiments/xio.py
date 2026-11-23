"""Atomic writes and a short-lived file lock.

Every result file is written to ``<path>.tmp.<host>.<pid>`` and then ``os.replace``-d onto ``<path>``,
so a reader sees either the old file or the complete new one, never a truncated file, also on
Lustre/NFS; a job killed by the scheduler loses at most the unit of work that was running.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import socket
import time
from typing import Any

import numpy as np

__all__ = ["tmp_path", "atomic_write_bytes", "atomic_text", "atomic_json", "read_json", "locked"]


def tmp_path(path: str) -> str:
    """Process-unique temporary name next to ``path``."""
    return f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"


def _replace(tmp: str, path: str) -> None:
    os.replace(tmp, path)
    try:  # make the rename durable where the filesystem allows it
        fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _prepare(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    return tmp_path(path)


@contextlib.contextmanager
def locked(path: str, stale_s: float = 300.0, wait_s: float = 900.0):
    """Hold ``<path>.lock`` while the block runs, so that processes that read, change and rewrite
    the same file (the result tables of parallel workers) take turns. Meant for writes of a few
    milliseconds: a lock older than ``stale_s`` was left by a killed process and is removed."""
    lock = path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock)), exist_ok=True)
    t0 = time.time()
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_s:
                    os.remove(lock)
                    continue
            except OSError:
                continue
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"{lock} has been held for more than {wait_s:.0f} s")
            time.sleep(0.05 + 0.1 * random.random())
    try:
        yield
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def atomic_write_bytes(data: bytes, path: str) -> None:
    tmp = _prepare(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)


def atomic_text(text: str, path: str) -> None:
    atomic_write_bytes(text.encode("utf-8"), path)


def atomic_json(obj: Any, path: str, indent: int = 2) -> None:
    atomic_write_bytes(json.dumps(obj, indent=indent, default=_json_default).encode("utf-8"), path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(f"not JSON serializable: {type(x)}")


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON, returning ``default`` if the file is missing or unreadable."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


