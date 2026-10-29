"""Baselines: exact solvers on explicit local distance matrices, approximations, and proxies.

Local problems. For an edge (i, j), ``local_problem`` builds the two measures on
S = N(i) u N(j) and the |S| x |S| hop-distance matrix with the numba kernels of
:mod:`orcut._reference` (distances 0..3 from adjacency and 2-paths), the fastest way we know to give
a general-purpose solver its input. ``net=True`` cancels the common mass (P x N submatrix), which
POT's network simplex usually solves faster.

Exact solvers on (a, b, M): POT's C++ network simplex (``emd_c``, the routine behind ``ot.emd``) and
its public wrapper ``ot.emd2`` (what GraphRicciCurvature calls), OR-Tools' integer min-cost flow
(cost scaling, C++), and HiGHS through ``scipy.optimize.linprog`` (small problems only).

Distance-free exact baseline: the min-cost flow on the local graph (Beckmann's formulation of W1
on a graph metric), with OR-Tools (:class:`GraphFlowProblems`).

Approximations: entropic OT (POT ``sinkhorn2``, log-domain). Proxies (one pass over the 2-hop
structure): Balanced Forman curvature (Topping et al., 2022), the triangle-augmented Forman curvature
AF3 = 4 - d_i - d_j + 3 t (Fesser and Weber, 2024), and the Jost--Liu bounds and their mean
(the surrogate of Tian, Lubberts and Weber, 2025).

GraphRicciCurvature needs scipy <= 1.13 and runs in its own environment (``grc_runner.py``).
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from numba import njit, prange

from orcut import _reference as RF


# ------------------------------------------------------------ local problems

def measure_weights(kind: str, di: int, dj: int):
    """(self mass, neighbour mass) of m_i and of m_j, as exact integers on the common scale L, and L."""
    if kind == "lazy":
        return (dj + 1, dj + 1), (di + 1, di + 1), (di + 1) * (dj + 1)
    if kind == "nonlazy":
        return (0, dj), (0, di), di * dj
    if kind == "lly":  # uniform idleness 1/(deg_max + 1)
        b = max(di, dj) + 1
        return (di * dj, (b - 1) * dj), (di * dj, (b - 1) * di), b * di * dj
    if kind.startswith("idle:"):  # idle:a/b
        a, b = (int(x) for x in kind.split(":")[1].split("/"))
        return (a * di * dj, (b - a) * dj), (a * di * dj, (b - a) * di), b * di * dj
    raise ValueError(kind)


class LocalProblems:
    """Builds local transport problems for edges of a graph (reuses one O(n) scratch)."""

    def __init__(self, g, kind: str = "lazy"):
        self.g = g
        self.kind = kind
        self.scratch = RF.new_scratch(g.num_nodes)
        self.epoch = 0

    def build(self, e: int, net: bool = True):
        """(a, b, M, L, aint, bint): float marginals summing to 1, the cost matrix (float64), the common
        scale L and the integer marginals (a = aint / L). Empty a means m_i = m_j."""
        g = self.g
        i, j = int(g.edges[e, 0]), int(g.edges[e, 1])
        self.epoch += 1
        S = RF.local_support(g.indptr, g.indices, i, j)
        D = RF.local_distances(g.indptr, g.indices, S, self.scratch, self.epoch)
        di, dj = int(g.deg[i]), int(g.deg[j])
        (si, ni), (sj, nj), L = measure_weights(self.kind, di, dj)
        w = RF._net(S, g.indptr, g.indices, i, j, si, ni, sj, nj)
        if net:
            P, N = np.flatnonzero(w > 0), np.flatnonzero(w < 0)
            aint, bint = w[P].astype(np.int64), (-w[N]).astype(np.int64)
            M = np.ascontiguousarray(D[np.ix_(P, N)], dtype=np.float64)
        else:
            mi = RF._net(S, g.indptr, g.indices, i, j, si, ni, 0, 0)
            mj = -RF._net(S, g.indptr, g.indices, i, j, 0, 0, sj, nj)
            P, N = np.flatnonzero(mi > 0), np.flatnonzero(mj > 0)
            aint, bint = mi[P].astype(np.int64), mj[N].astype(np.int64)
            M = np.ascontiguousarray(D[np.ix_(P, N)], dtype=np.float64)
        return aint / L, bint / L, M, L, aint, bint


# ------------------------------------------------------------- exact solvers

def pot_emd_c(a, b, M) -> float:
    """POT's C++ network simplex (the routine behind ot.emd), single-threaded."""
    from ot.lp.emd_wrap import emd_c

    if a.size == 0:
        return 0.0
    b = b * (a.sum() / b.sum())
    G, cost, u, v, code = emd_c(a, b, M, 10**8, 1)
    if code != 1:
        raise RuntimeError(f"emd_c did not converge (code {code})")
    return float(cost)


def pot_emd2(a, b, M) -> float:
    import ot

    if a.size == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(ot.emd2(a, b * (a.sum() / b.sum()), M, numItermax=10**8))


def ortools_mcf(aint, bint, M) -> int:
    """OR-Tools min-cost flow on the bipartite transportation problem, in integers (exact)."""
    from ortools.graph.python import min_cost_flow

    nP, nN = len(aint), len(bint)
    if nP == 0:
        return 0
    smcf = min_cost_flow.SimpleMinCostFlow()
    st = np.repeat(np.arange(nP, dtype=np.int32), nN)
    en = nP + np.tile(np.arange(nN, dtype=np.int32), nP)
    cap = np.full(nP * nN, int(aint.sum()), dtype=np.int64)
    smcf.add_arcs_with_capacity_and_unit_cost(st, en, cap, M.reshape(-1).astype(np.int64))
    smcf.set_nodes_supplies(np.arange(nP + nN, dtype=np.int32), np.concatenate([aint, -bint]).astype(np.int64))
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"OR-Tools status {status}")
    return int(smcf.optimal_cost())


def highs_lp(a, b, M) -> float:
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    nP, nN = len(a), len(b)
    if nP == 0:
        return 0.0
    rows = np.concatenate([np.repeat(np.arange(nP), nN), nP + np.tile(np.arange(nN), nP)])
    cols = np.concatenate([np.arange(nP * nN), np.arange(nP * nN)])
    A = coo_matrix((np.ones(2 * nP * nN), (rows, cols)), shape=(nP + nN, nP * nN)).tocsr()
    res = linprog(M.reshape(-1), A_eq=A, b_eq=np.concatenate([a, b * (a.sum() / b.sum())]), bounds=(0, None),
                  method="highs")
    return float(res.fun)


def sinkhorn(a, b, M, reg: float, max_iter: int = 100_000) -> float:
    """Entropic OT (POT): standard Sinkhorn for reg >= 0.1, log-domain below (where the kernel
    exp(-M / reg) underflows); stopping threshold 1e-9 on the marginals. Returns the transport cost
    <P, M> of the entropic plan (what ``ot.sinkhorn2`` and GraphRicciCurvature report)."""
    import ot

    if a.size == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = ot.sinkhorn(a, b * (a.sum() / b.sum()), M, reg, method="sinkhorn" if reg >= 0.1 else "sinkhorn_log",
                        numItermax=max_iter, stopThr=1e-9)
    return float(np.sum(G * M))


EXACT = {"pot_simplex": pot_emd_c, "pot_emd2": pot_emd2, "highs": highs_lp}


def _solve(method, a, b, M, L, aint, bint, reg=None) -> float:
    if method == "ortools":
        return ortools_mcf(aint, bint, M) / L
    if method == "sinkhorn":
        return sinkhorn(a, b, M, reg)
    return EXACT[method](a, b, M)


def run_local(g, ids, method: str, kind: str = "lazy", net: bool = True, reps: int = 3, include_build: bool = False,
              reg: Optional[float] = None, budget_s: Optional[float] = None, times: Optional[np.ndarray] = None,
              btimes: Optional[np.ndarray] = None, max_pairs: Optional[float] = None):
    """Per-edge curvature and time of a local-problem method on edges[ids].

    Returns (values (K,), solve seconds per edge, build seconds per edge, number timed). The solve is
    timed alone (inputs built outside the timer, fastest of ``reps`` passes); the build (support,
    distance matrix, marginals) is timed separately once. With ``budget_s`` the loop stops early
    (the remaining values are NaN). ``times`` / ``btimes`` (length K), if given, receive the
    per-edge solve / build times (NaN where not timed). With ``max_pairs``, edges with
    (d_i + 1)(d_j + 1) > max_pairs are not built or solved (value and times NaN): their dense
    problems would need gigabytes; a time estimate that counts them as free is a lower bound."""
    lp = LocalProblems(g, kind)
    K = len(ids)
    vals = np.full(K, np.nan)
    for arr in (times, btimes):
        if arr is not None:
            arr[:] = np.nan
    tbuild = 0.0
    tsolve = 0.0
    done = 0
    if K:  # warm-up: the first call of a solver pays one-time initialization (thread pools, imports)
        e0 = int(ids[int(np.argmin(g.deg[g.edges[ids, 0]] + g.deg[g.edges[ids, 1]]))])
        a, b, M, L, aint, bint = lp.build(e0, net)
        _solve(method, a, b, M, L, aint, bint, reg)
    t_start = time.perf_counter()
    for k, e in enumerate(ids):
        if max_pairs is not None:
            di, dj = int(g.deg[g.edges[e, 0]]), int(g.deg[g.edges[e, 1]])
            if (di + 1) * (dj + 1) > max_pairs:
                continue
        t = time.perf_counter()
        a, b, M, L, aint, bint = lp.build(int(e), net)
        tb_k = time.perf_counter() - t
        tbuild += tb_k
        best = float("inf")
        for _ in range(max(1, reps)):
            t = time.perf_counter()
            w = _solve(method, a, b, M, L, aint, bint, reg)
            best = min(best, time.perf_counter() - t)
        tsolve += best
        if times is not None:
            times[k] = best
        if btimes is not None:
            btimes[k] = tb_k
        vals[k] = 1.0 - w
        done += 1
        if budget_s is not None and time.perf_counter() - t_start > budget_s:
            break
    return vals, tsolve / max(done, 1), tbuild / max(done, 1), done


# ----------------------------------------------- graph min-cost flow (Beckmann)

@njit(cache=True)
def _graph_flow_arcs(indptr, indices, S, stamp, loc, ep):
    """Vertices S first, then the other neighbours of S; arcs: every edge incident to S, in both
    directions (each edge inside S once per direction). Returns (number of vertices, tails, heads)."""
    nS = S.shape[0]
    for a in range(nS):
        stamp[S[a]] = ep
        loc[S[a]] = a
    nv = nS
    tot = 0
    for a in range(nS):
        tot += indptr[S[a] + 1] - indptr[S[a]]
    tl = np.empty(2 * tot, np.int32)
    hd = np.empty(2 * tot, np.int32)
    m = 0
    for a in range(nS):
        x = S[a]
        for t in range(indptr[x], indptr[x + 1]):
            w = indices[t]
            if stamp[w] != ep:
                stamp[w] = ep
                loc[w] = nv
                nv += 1
            b = loc[w]
            if b < nS and b < a:
                continue  # an edge inside S, already added from its other endpoint
            tl[m] = a
            hd[m] = b
            m += 1
            tl[m] = b
            hd[m] = a
            m += 1
    return nv, tl[:m], hd[:m]


class GraphFlowProblems:
    """W1 as a min-cost flow on the graph itself (Beckmann's formulation; e.g. Peyre and Cuturi,
    2019, ch. 6): supplies m_i - m_j on S = N[i] u N[j], uncapacitated arcs of unit cost along every
    edge incident to S. Shortest paths between points of S use only such edges (a common neighbour,
    or the path through i and j), so the optimum is W1 exactly. Like orcut, it needs no distance
    matrix; unlike orcut, it is a min-cost flow on the whole 2-hop neighbourhood. Solved with
    OR-Tools' integer min-cost flow (exact)."""

    def __init__(self, g, kind: str = "lazy"):
        self.g = g
        self.kind = kind
        self.stamp = np.full(g.num_nodes, -1, np.int64)
        self.loc = np.zeros(g.num_nodes, np.int64)
        self.ep = 0

    def curvature(self, e: int):
        """(1 - W1, number of vertices, number of arcs) of edge e (raw 1 - W1 for kind 'lly')."""
        from ortools.graph.python import min_cost_flow

        g = self.g
        i, j = int(g.edges[e, 0]), int(g.edges[e, 1])
        S = RF.local_support(g.indptr, g.indices, i, j)
        di, dj = int(g.deg[i]), int(g.deg[j])
        (si, ni), (sj, nj), L = measure_weights(self.kind, di, dj)
        w = RF._net(S, g.indptr, g.indices, i, j, si, ni, sj, nj)
        self.ep += 1
        nv, tl, hd = _graph_flow_arcs(g.indptr, g.indices, S, self.stamp, self.loc, self.ep)
        sup = np.zeros(nv, np.int64)
        sup[:len(S)] = w
        smcf = min_cost_flow.SimpleMinCostFlow()
        tot = int(np.abs(w).sum())
        smcf.add_arcs_with_capacity_and_unit_cost(tl, hd, np.full(len(tl), max(tot, 1), np.int64),
                                                  np.ones(len(tl), np.int64))
        smcf.set_nodes_supplies(np.arange(nv, dtype=np.int32), sup)
        if smcf.solve() != smcf.OPTIMAL:
            raise RuntimeError("OR-Tools min-cost flow did not reach optimality")
        return 1.0 - smcf.optimal_cost() / L, nv, len(tl)


def run_graph_flow(g, ids, kind: str = "lazy", budget_s: Optional[float] = None,
                   times: Optional[np.ndarray] = None, max_arcs: Optional[float] = 5e7):
    """Per-edge curvature and time (everything included) of the graph min-cost flow on edges[ids].
    Returns (values, seconds per edge, number timed). Edges whose local graph would have more than
    ``max_arcs`` arcs are skipped (NaN)."""
    gp = GraphFlowProblems(g, kind)
    K = len(ids)
    vals = np.full(K, np.nan)
    if times is not None:
        times[:] = np.nan
    if K:
        e0 = int(ids[int(np.argmin(g.deg[g.edges[ids, 0]] + g.deg[g.edges[ids, 1]]))])
        gp.curvature(e0)  # warm-up (imports, first allocation)
    done = 0
    tot = 0.0
    t_start = time.perf_counter()
    for k, e in enumerate(ids):
        i, j = int(g.edges[e, 0]), int(g.edges[e, 1])
        if max_arcs is not None:
            est = 2 * int(np.sum(g.deg[g.neighbors(i)]) + np.sum(g.deg[g.neighbors(j)]))
            if est > max_arcs:
                continue
        t = time.perf_counter()
        v, _, _ = gp.curvature(int(e))
        dt = time.perf_counter() - t
        vals[k] = v
        tot += dt
        if times is not None:
            times[k] = dt
        done += 1
        if budget_s is not None and time.perf_counter() - t_start > budget_s:
            break
    return vals, tot / max(done, 1), done


# ------------------------------------------------------- GraphRicciCurvature

GRC_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grc_runner.py")


def grc_python(explicit: Optional[str] = None) -> Optional[str]:
    """The Python interpreter of the GRC environment: ``explicit``, else $ORCUT_GRC_PYTHON, else None."""
    p = explicit or os.environ.get("ORCUT_GRC_PYTHON")
    return p if p and os.path.exists(p) else None


def grc(g, method: str = "OTDSinkhornMix", alpha: float = 0.5, nbr_topk: int = 3000, proc: int = 1,
        python: Optional[str] = None, timeout: Optional[float] = None, ids: Optional[np.ndarray] = None,
        shortest_path: str = "auto", workdir: Optional[str] = None):
    """GraphRicciCurvature on graph ``g`` (all edges, or ``ids``) through ``grc_runner.py`` in its own
    environment. Returns (values aligned with g.edges[ids], wall seconds, status) with status "ok",
    "timeout", "unavailable" or "error: ..."; values is None unless status is "ok". On timeout the
    whole process group (GRC's worker pool included) is killed."""
    py = grc_python(python)
    if py is None:
        return None, float("nan"), "unavailable"
    with tempfile.TemporaryDirectory(dir=workdir, prefix="grc_") as d:
        np.savez(os.path.join(d, "in.npz"), n=g.num_nodes, edges=g.edges)
        cmd = [py, "-I", GRC_RUNNER, os.path.join(d, "in.npz"), os.path.join(d, "out.npz"), "--alpha", repr(alpha),
               "--method", method, "--nbr-topk", str(nbr_topk), "--proc", str(proc), "--shortest-path", shortest_path]
        if ids is not None:
            np.save(os.path.join(d, "ids.npy"), np.asarray(ids, np.int64))
            cmd += ["--ids", os.path.join(d, "ids.npy")]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            _, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.communicate()
            return None, float("nan"), "timeout"
        if p.returncode != 0:
            return None, float("nan"), "error: " + (err or "")[-400:].replace("\n", " | ")
        z = np.load(os.path.join(d, "out.npz"))
        return np.asarray(z["curvature"], np.float64), float(z["time_s"]), "ok"


# ------------------------------------------------------------------- proxies

@njit(cache=True)
def _proxy_edge(indptr, indices, i, j, stamp, ep, cnt, out):
    di = indptr[i + 1] - indptr[i]
    dj = indptr[j + 1] - indptr[j]
    for e in range(indptr[j], indptr[j + 1]):  # stamp N(j) with 2 ep + 1, then the U_j vertices keep it
        stamp[indices[e]] = 2 * ep + 1
    t = 0
    for e in range(indptr[i], indptr[i + 1]):
        x = indices[e]
        if stamp[x] == 2 * ep + 1:
            t += 1
            stamp[x] = 2 * ep  # common neighbour
    # U_i = N(i) \ (N(j) u {j}), U_j = N(j) \ (N(i) u {i}); count cross adjacency in both directions
    nUi = di - 1 - t
    nUj = dj - 1 - t
    xi_i = 0
    xi_j = 0
    varpi = 0
    # mark U_j (stamp 2 ep + 1 still; i has stamp 2 ep + 1 too since i in N(j): exclude explicitly)
    for e in range(indptr[j], indptr[j + 1]):
        y = indices[e]
        cnt[y] = 0
    for e in range(indptr[i], indptr[i + 1]):
        x = indices[e]
        if x == j or stamp[x] == 2 * ep:
            continue
        c = 0
        for f in range(indptr[x], indptr[x + 1]):
            y = indices[f]
            if y != i and stamp[y] == 2 * ep + 1:
                c += 1
                cnt[y] += 1
        if c > 0:
            xi_i += 1
        if c > varpi:
            varpi = c
    for e in range(indptr[j], indptr[j + 1]):
        y = indices[e]
        if y == i or stamp[y] != 2 * ep + 1:
            continue
        if cnt[y] > 0:
            xi_j += 1
        if cnt[y] > varpi:
            varpi = cnt[y]
    a = min(di, dj)
    b = max(di, dj)
    bf = 0.0
    if a > 1:
        bf = 2.0 / di + 2.0 / dj - 2.0 + (2.0 / b + 1.0 / a) * t
        if varpi > 0:
            bf += (xi_i + xi_j) / (varpi * b)
    out[0] = bf
    out[1] = 4.0 - di - dj + 3.0 * t                       # AF3
    K = 1.0 - 1.0 / di - 1.0 / dj
    jl_lo = -max(0.0, K - t / a) - max(0.0, K - t / b) + t / b
    jl_up = t / b
    out[2] = jl_lo
    out[3] = jl_up
    out[4] = 0.5 * (jl_lo + jl_up)
    out[5] = t


@njit(cache=True, parallel=True)
def _proxies(indptr, indices, edges, nch):
    n = indptr.shape[0] - 1
    M = edges.shape[0]
    out = np.zeros((M, 6))
    for c in prange(nch):
        stamp = np.full(n, -1, np.int64)
        cnt = np.zeros(n, np.int64)
        for k in range(c, M, nch):
            _proxy_edge(indptr, indices, edges[k, 0], edges[k, 1], stamp, k, cnt, out[k])
    return out


@njit(cache=True)
def _proxies_ids(indptr, indices, edges, ids):
    n = indptr.shape[0] - 1
    out = np.zeros((ids.shape[0], 6))
    stamp = np.full(n, -1, np.int64)
    cnt = np.zeros(n, np.int64)
    for k in range(ids.shape[0]):
        e = ids[k]
        _proxy_edge(indptr, indices, edges[e, 0], edges[e, 1], stamp, k, cnt, out[k])
    return out


PROXY_COL = {"bf": 0, "af3": 1, "jl_lower": 2, "jl_upper": 3, "jl_mean": 4, "triangles": 5}


def proxies_ids(indptr, indices, edges, ids, key: str) -> np.ndarray:
    """One proxy (a key of :func:`proxies`) for the edges ``edges[ids]`` of the graph (indptr,
    indices), single-threaded (used for incremental updates on a changing graph)."""
    return _proxies_ids(indptr, indices, edges, np.ascontiguousarray(ids, dtype=np.int64))[:, PROXY_COL[key]]


def proxies(g, nchunks: Optional[int] = None) -> Dict[str, np.ndarray]:
    """BF, AF3, JL lower / upper / mean (non-lazy), and triangle counts for every edge."""
    import numba

    nch = numba.get_num_threads() if nchunks is None else nchunks
    raw = _proxies(g.indptr, g.indices, g.edges, max(1, nch))
    return {"bf": raw[:, 0], "af3": raw[:, 1], "jl_lower": raw[:, 2], "jl_upper": raw[:, 3], "jl_mean": raw[:, 4],
            "triangles": raw[:, 5]}
