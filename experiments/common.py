"""Shared helpers of the experiments: paths, results, timing, threads, memory, environment."""
from __future__ import annotations

import csv
import io
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import xio  # noqa: E402
from checkpoint import EXIT_INCOMPLETE  # noqa: E402,F401  (re-exported: the exit code of finish)

RESULTS = os.path.join(ROOT, "results")


# ------------------------------------------------------------------ results

def out_dir(results: str, *parts: str) -> str:
    d = os.path.join(results, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def write_csv(rows: Sequence[Dict], path: str) -> None:
    """Atomic CSV write; the header is the union of the keys, in first-seen order."""
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: _fmt(v) for k, v in r.items()})
    xio.atomic_text(buf.getvalue(), path)


def _fmt(v):
    if isinstance(v, (np.floating, float)):
        return repr(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def read_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(row: Dict, key: str, default=float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def row_failed(row: Dict) -> bool:
    """A row that records a failure (status "error: ...", e.g. GraphRicciCurvature crashed)."""
    return str(row.get("status") or "").startswith("error")


def row_ok(row: Dict) -> bool:
    """A row that holds a result: not a failure, and not "unavailable" (GraphRicciCurvature not
    installed). Rows that are not ok are computed again when their step runs again."""
    return not (row_failed(row) or str(row.get("status") or "") == "unavailable")


_TABLES: List["Rows"] = []  # every result table opened by this process (finish)


class Rows:
    """Rows of a result table, checkpointed after every row (a killed job resumes at the next row).
    Each row is keyed by ``key``; rows already present are skipped by :meth:`done`, except rows
    that hold no result (:func:`row_ok`), which are computed again and replaced by :meth:`add`."""

    def __init__(self, path: str, key: Sequence[str]):
        self.path = path
        self.key = tuple(key)
        self.rows = read_csv(path)
        _TABLES.append(self)

    def _k(self, r) -> tuple:
        return tuple(str(r.get(k)) for k in self.key)

    def get(self, **kv) -> Optional[Dict]:
        """The row with this key, whatever its status, or None."""
        want = tuple(str(kv.get(k)) for k in self.key)
        return next((r for r in self.rows if self._k(r) == want), None)

    def done(self, **kv) -> bool:
        r = self.get(**kv)
        return r is not None and row_ok(r)

    def add(self, row: Dict) -> None:
        self.rows = [r for r in self.rows if self._k(r) != self._k(row)] + [row]
        write_csv(self.rows, self.path)


def finish() -> int:
    """Exit code of an experiment script (``sys.exit(common.finish())`` at the end of it): 0, or
    EXIT_INCOMPLETE when rows of its tables record a failure. main.py then does not record the
    step as done, and when the step runs again it computes these rows again (and only these)."""
    bad = [(t, r) for t in _TABLES for r in t.rows if row_failed(r)]
    if not bad:
        return 0
    log(f"{len(bad)} row(s) ended in an error; they are computed again when this step runs again:")
    for t, r in bad[:20]:
        key = ", ".join(f"{k}={r.get(k)}" for k in t.key)
        log(f"  {os.path.basename(t.path)} [{key}]: {str(r.get('status'))[:200]}")
    return EXIT_INCOMPLETE


# ------------------------------------------------------------------- timing

def best_of(f, reps: int = 3):
    """(fastest wall time of ``reps`` calls, last result)."""
    best = float("inf")
    out = None
    for _ in range(max(1, reps)):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


@contextmanager
def numba_threads(k: Optional[int]):
    import numba

    prev = numba.get_num_threads()
    if k is not None:
        numba.set_num_threads(max(1, min(int(k), numba.config.NUMBA_NUM_THREADS)))
    try:
        yield numba.get_num_threads()
    finally:
        numba.set_num_threads(prev)


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far (MB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 if sys.platform != "darwin" else r / 2**20


def warm_jit() -> None:
    """Compile every kernel once (timings exclude JIT compilation)."""
    import orcut
    from orcut import _core as C, _layered as LY, _reference as RF

    g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 0)])
    ids = np.arange(g.num_edges, dtype=np.int64)
    for mode in (C.M_LAZY, C.M_NONLAZY, C.M_LLY):
        for flags in (0, C.F_ALL):
            C.core_ids(g.indptr, g.indices, g.edges, ids, mode, flags=flags, nchunks=1)
    C.core_ids(g.indptr, g.indices, g.edges, ids, C.M_IDLE, 1, 2, nchunks=1)
    orcut.curvature(g, orcut.Walk(weights=np.ones(g.num_edges, np.int64)))
    orcut.exact_ids(g.indptr, g.indices, g.edges, ids, True, True, True, 1)
    RF.exact_ids_reference(g.indptr, g.indices, g.edges, ids, True, False, False, 1)
    for kind in (LY.K_LAZY, LY.K_BALL, LY.K_WALK_LAZY):
        LY.layered_ids(g.indptr, g.indices, g.edges, ids, kind, 2, True, 1)


# ------------------------------------------------------- stratified samples

def stratified_bins(g, per_bin: int, seed: int = 0):
    """Edges binned by floor(log2(d_i + d_j)); up to ``per_bin`` edges sampled per bin.
    Returns a list of (bin, sorted sampled edge ids, number of edges in the bin)."""
    rng = np.random.default_rng(seed)
    dsum = g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]
    bins = np.floor(np.log2(dsum)).astype(int)
    out = []
    for bn in np.unique(bins):
        all_ids = np.flatnonzero(bins == bn)
        s = np.sort(rng.choice(all_ids, min(per_bin, len(all_ids)), replace=False)).astype(np.int64)
        out.append((int(bn), s, len(all_ids)))
    return out


def time_per_edge(f, ids: np.ndarray, block: int = 32, reps: int = 3, deadline: Optional[float] = None) -> np.ndarray:
    """Per-edge wall time of ``f(sub_ids)`` (fastest of ``reps``), measured on blocks of up to ``block``
    edges (the block time is split evenly among its edges); NaN after ``deadline`` (perf_counter)."""
    t = np.full(len(ids), np.nan)
    for lo in range(0, len(ids), block):
        if deadline is not None and time.perf_counter() > deadline:
            break
        sub = ids[lo:lo + block]
        best, _ = best_of(lambda: f(sub), reps=reps)
        t[lo:lo + block] = best / len(sub)
    return t


def estimate_total(strata, times) -> float:
    """Estimated total time on all edges from per-edge times of the sampled edges of each stratum
    (``times[k]`` aligned with the sample of ``strata[k]``); NaN if some stratum has no timed edge."""
    tot = 0.0
    for (bn, s, cnt), t in zip(strata, times):
        ok = np.isfinite(t)
        if not ok.any():
            return float("nan")
        tot += cnt * float(np.mean(t[ok]))
    return tot


# ------------------------------------------------------------------ metrics

def topk_recall(exact: np.ndarray, approx: np.ndarray, k: int, most_negative: bool = True) -> float:
    """Share of the k edges that ``approx`` ranks most negative (or most positive) that are among
    the k most negative (positive) edges of ``exact``, ties at the threshold of ``exact`` counted
    as correct (so ties cannot make an exact method look wrong)."""
    k = int(max(1, min(k, len(exact))))
    s = 1.0 if most_negative else -1.0
    sel = np.argsort(s * approx, kind="stable")[:k]
    thr = np.partition(s * exact, k - 1)[k - 1]
    return float(np.mean(s * exact[sel] <= thr + 1e-12))


def agreement(exact: np.ndarray, approx: np.ndarray, hub: Optional[np.ndarray] = None,
              k_frac: float = 0.01, errors: bool = True) -> Dict[str, float]:
    """Agreement of an approximation with the exact curvature on the same edges: errors (MAE, RMSE,
    max, share above 0.01 and 0.1, MAE on ``hub`` edges), sign disagreement (edges of nonzero exact
    curvature), Spearman correlation, and tie-aware recall of the 1% most negative and most
    positive edges (the edges that curvature-based rewiring acts on). With ``errors=False`` only
    the rank statistics (for proxies on another scale)."""
    from scipy.stats import spearmanr

    ok = np.isfinite(approx) & np.isfinite(exact)
    x, y = exact[ok], approx[ok]
    out: Dict[str, float] = {"edges": int(ok.sum()), "missing": int((~ok).sum())}
    if len(x) == 0:
        return out
    k = max(1, int(round(k_frac * len(x))))
    if errors:
        err = np.abs(y - x)
        out.update(mae=float(err.mean()), rmse=float(np.sqrt(np.mean(err ** 2))), max_err=float(err.max()),
                   share_err_001=float(np.mean(err > 0.01)), share_err_01=float(np.mean(err > 0.1)))
        if hub is not None:
            h = hub[ok]
            out["hub_edges"] = int(h.sum())
            out["hub_mae"] = float(err[h].mean()) if h.any() else float("nan")
            out["hub_max_err"] = float(err[h].max()) if h.any() else float("nan")
    nz = np.abs(x) > 1e-12
    out["sign_disagree"] = float(np.mean(np.sign(y[nz]) != np.sign(x[nz]))) if nz.any() else float("nan")
    out["spearman"] = float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")
    out["recall_neg"] = topk_recall(x, y, k, True)
    out["recall_pos"] = topk_recall(x, y, k, False)
    return out


# -------------------------------------------------------------- environment

def environment() -> Dict:
    import numba
    import numpy
    import scipy

    env = {
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "numpy": numpy.__version__, "scipy": scipy.__version__, "numba": numba.__version__,
        "numba_threads": numba.config.NUMBA_NUM_THREADS, "cpu_count": os.cpu_count(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    try:
        import orcut
        env["orcut"] = orcut.__version__
    except Exception:
        pass
    for mod in ("ot", "ortools", "networkx", "torch", "torch_geometric"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:
            env[mod] = None
    try:
        env["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        env["cpu_model"] = platform.processor()
    try:
        env["git"] = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
                                             stderr=subprocess.DEVNULL).strip()
    except Exception:
        env["git"] = None
    return env


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
