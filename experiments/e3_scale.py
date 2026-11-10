"""E3. Exact curvature of every edge of large graphs.

Part ``real``: real networks from 10^5 to 1.2 x 10^8 edges (``registry.MEDIUM``, ``LARGE``,
``XLARGE``). For every graph: wall time of orcut on all edges with ``--threads`` threads (median of
``--reps`` runs; one run above 10^7 edges), peak memory of the solve above the loaded graph, and the
diagnostics (share of edges solved by one bipartite flow, row compression by twin merging on edges
with an endpoint of degree at least 100, network sizes). The time of the POT pipeline (local distance
matrix and network simplex) is estimated from a stratified sample of edges (bins of d_i + d_j),
timed single-threaded, extrapolated to all edges and divided by the number of threads (perfect
parallel scaling, a lower bound on its parallel time).

Part ``threads``: strong scaling (1, 2, 4, ..., ``--threads``) on ``--graphs``.

Part ``synth``: random families (ER, BA, RGG, WS, Chung--Lu 2.1 / 2.5) with average degree 16 and
n = 10^3 .. 10^7, and with average degree 2.1 n^0.4 (as in the companion paper), n = 10^3 .. 10^6;
time against the number of edges and against sum_e (d_i + d_j).

Long solves are saved as they go. A graph of part ``real`` with more than ``--chunk-above`` edges
(10^7), and a row of part ``synth`` that the two previous sizes of its family predict to take more
than ``--chunk-after`` seconds (an hour), is solved in one pass (no warm-up, no repeats; the
diagnostics come from the same pass) by contiguous chunks of edges, about 20 or more, balanced by
the size of their two-hop scans. After each chunk, its time, peak memory and diagnostic counts go
to ``<out>/e3/partial/*.json``, so a run stopped by the time limit resumes at the next chunk. The
row's time is the sum over its chunks and its peak memory the largest over them (the per-edge
outputs are held for one chunk at a time); its ``chunks`` column says how many, ``chunk_hosts`` on
how many machines they ran.

Usage: python e3_scale.py --part real|threads|synth [--graphs ...] [--threads 32] [--out results]
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

import numpy as np

import common
from common import Rows, log, out_dir
import baselines as B
import checkpoint as CK
import registry as R
import xio


class RSSSampler:
    """Peak resident set size (MB) during a block, sampled from /proc every ``dt`` seconds."""

    def __init__(self, dt=0.05):
        self.dt = dt
        self.peak = 0.0
        self.base = self._rss()
        self._stop = threading.Event()

    @staticmethod
    def _rss():
        try:
            with open("/proc/self/statm") as fh:
                return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        except Exception:
            return float("nan")

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.dt)

    def __enter__(self):
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss())


def pot_estimate(g, per_bin=50, seed=0, max_pairs=5e7, ckpt=None):
    """(estimated single-threaded seconds of the POT pipeline on all edges, of its solves alone,
    number of edges timed, share of edges counted as free). Every sampled edge is timed, however long
    it takes, and with ``ckpt`` (a JSON file) saved as soon as it is, so a run stopped in the middle
    resumes at the next edges. Edges whose dense local problem would have more than ``max_pairs``
    entries are not run (their matrix would not fit) and count as free, so the estimate stays a
    lower bound."""
    rng = np.random.default_rng(seed)
    dsum = g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]
    bins = np.floor(np.log2(dsum)).astype(int)
    prod = (g.deg[g.edges[:, 0]] + 1).astype(np.float64) * (g.deg[g.edges[:, 1]] + 1)
    free = int(np.sum(prod > max_pairs))
    plan = []  # every bin's sample first: the same edges, whatever was saved before
    for bn in sorted(np.unique(bins), reverse=True):  # largest first: they dominate
        all_ids = np.flatnonzero((bins == bn) & (prod <= max_pairs))
        if len(all_ids):
            plan.append((len(all_ids), np.sort(rng.choice(all_ids, min(per_bin, len(all_ids)), replace=False))))
    saved = xio.read_json(ckpt, default=None) if ckpt else None
    if not (isinstance(saved, dict) and saved.get("m") == int(g.num_edges) and saved.get("per_bin") == per_bin):
        saved = dict(m=int(g.num_edges), per_bin=per_bin, edges={})
    done = saved["edges"]  # edge id -> [solve s, build s]
    for _, ids in plan:
        todo = np.array([e for e in ids if str(int(e)) not in done], np.int64)
        for k in range(0, len(todo), 10):
            batch = todo[k:k + 10]
            ts, tb = np.full(len(batch), np.nan), np.full(len(batch), np.nan)
            B.run_local(g, batch, "pot_simplex", reps=1, times=ts, btimes=tb)
            done.update({str(int(e)): [float(x), float(y)] for e, x, y in zip(batch, ts, tb)})
            if ckpt:
                xio.atomic_json(saved, ckpt)
    tot_full = tot_solve = 0.0
    timed = 0
    for cnt, ids in plan:
        ts = np.array([done[str(int(e))][0] for e in ids])
        tb = np.array([done[str(int(e))][1] for e in ids])
        ok = np.isfinite(ts) & np.isfinite(tb)
        if not ok.any():
            return float("nan"), float("nan"), timed, free / g.num_edges
        tot_full += cnt * float(np.mean(ts[ok] + tb[ok]))
        tot_solve += cnt * float(np.mean(ts[ok]))
        timed += int(ok.sum())
    return tot_full, tot_solve, timed, free / g.num_edges


def solve_all(g, measure, threads, diagnostics=False):
    import orcut

    return orcut.curvature(g, measure, n_jobs=threads, diagnostics=diagnostics)


# ------------------------------------------------- one pass in saved chunks

def chunk_bounds(g, k):
    """Boundaries 0 = b_0 < b_1 < ... = m of (at most) ``k`` contiguous ranges of edge ids with about
    the same sum of s_i + s_j, where s_v is the sum of the degrees of the neighbours of v: the size
    of the two-hop scans of the local problem of ij. On hubby graphs it predicts the time of a chunk
    far better than its number of edges, d_i + d_j or (d_i + d_j)^2."""
    u, v = g.edges[:, 0], g.edges[:, 1]
    deg = g.deg.astype(np.float64)
    s = np.bincount(u, weights=deg[v], minlength=g.num_nodes) + np.bincount(v, weights=deg[u], minlength=g.num_nodes)
    c = np.cumsum(s[u] + s[v])
    del s
    cuts = np.searchsorted(c, c[-1] * np.arange(1, k) / k, side="right")
    return sorted({0, int(g.num_edges), *(int(x) for x in cuts if 0 < x < g.num_edges)})


def _chunk_stats(val, d, hub):
    """What the row needs from the diagnostics of one chunk, as sums (exact) and maxima."""
    from orcut import _core as C

    path = d[:, C.D_PATH]
    ok = hub & (d[:, C.D_GROUPS] > 0)
    ratio = d[ok, C.D_ROWS] / d[ok, C.D_GROUPS]
    return dict(edges=int(len(val)), nan=int(np.isnan(val).sum()),
                zero=int(np.sum(path == C.PATH_ZERO)),
                fast=int(np.sum((path == C.PATH_FAST) | (path == C.PATH_NOADJ))),
                cut=int(np.sum((path == C.PATH_CUT) | (path == C.PATH_CUTW) | (path == C.PATH_WIT))),
                witness=int(np.sum((path == C.PATH_CUTW) | (path == C.PATH_WIT))),
                nodes=int(d[:, C.D_NODES].sum()), arcs=int(d[:, C.D_ARCS].sum()),
                max_arcs=int(d[:, C.D_ARCS].max()) if len(d) else 0, hub=int(hub.sum()),
                ratio_n=int(ok.sum()), ratio_sum=float(ratio.sum()), ratio_max=float(ratio.max()) if ok.any() else None)


def _combine(chunks):
    tot = dict(edges=0, nan=0, zero=0, fast=0, cut=0, witness=0, nodes=0, arcs=0, hub=0, ratio_n=0, ratio_sum=0.0,
               seconds=0.0, max_arcs=0, ratio_max=None, peak_mb=float("nan"))
    for st in chunks:
        for k in ("edges", "nan", "zero", "fast", "cut", "witness", "nodes", "arcs", "hub", "ratio_n", "ratio_sum",
                  "seconds"):
            tot[k] += st[k]
        tot["max_arcs"] = max(tot["max_arcs"], st["max_arcs"])
        if st["ratio_max"] is not None:
            tot["ratio_max"] = st["ratio_max"] if tot["ratio_max"] is None else max(tot["ratio_max"], st["ratio_max"])
        pk = st["peak_mb"]
        tot["peak_mb"] = pk if tot["peak_mb"] != tot["peak_mb"] else max(tot["peak_mb"], pk)
    return tot


def diag_fields(tot, m):
    """The diagnostic columns of a row (as computed on all edges at once in part_real) from _combine."""
    nan = float("nan")
    return dict(share_zero=tot["zero"] / m, share_fast=tot["fast"] / m, share_cut=tot["cut"] / m,
                share_witness=tot["witness"] / m, mean_nodes=tot["nodes"] / m, mean_arcs=tot["arcs"] / m,
                max_arcs=tot["max_arcs"], hub_edges=tot["hub"],
                twin_ratio_hub=tot["ratio_sum"] / tot["ratio_n"] if tot["ratio_n"] else nan,
                twin_ratio_hub_max=tot["ratio_max"] if tot["ratio_max"] is not None else nan)


def _hms(sec):
    sec = float(sec)
    if sec < 60:
        return f"{sec:.0f}s"
    m = int(sec // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m{int(sec % 60):02d}s"


def _load_plan(path, key):
    state = xio.read_json(path, default=None)
    return state if isinstance(state, dict) and state.get("key") == key and state.get("bounds") else None


def solve_chunked(g, measure, threads, path, chunk_edges, label, after_chunk=None, claims=None):
    """All edges of ``g`` in one pass, by contiguous chunks of about ``chunk_edges`` edges on average
    (balanced by chunk_bounds). After every chunk, its time, peak memory and diagnostic counts are saved to
    ``path`` (JSON), so a run that is stopped (time limit, out of memory, node failure) resumes at the
    next chunk; the chunk boundaries are those of the first run. The time is the sum over the chunks
    (each one call of orcut.curvature with all ``threads``), the peak memory the largest over them.
    With ``claims`` (a folder), several workers (main.py --worker) can solve the same graph at once:
    each chunk is claimed by one of them. The worker that finds every chunk done and gets the row's
    claim returns (state, totals of _combine, that claim), and releases the claim once it has written
    the row; every other worker returns None. Without ``claims`` the third value is None."""
    import orcut

    key = dict(graph=label, measure=str(measure), n=int(g.num_nodes), m=int(g.num_edges), threads=int(threads))
    k = max(1, -(-g.num_edges // max(1, int(chunk_edges))))
    with xio.locked(path):  # workers that start together agree on one plan
        state = _load_plan(path, key)
        if state is None or (not state.get("chunks") and state.get("k") != k):  # nothing done yet: the new size
            state = dict(key=key, k=k, bounds=chunk_bounds(g, k), chunks={})
            xio.atomic_json(state, path)
    base = os.path.basename(path)[:-len(".json")]
    bounds, done = state["bounds"], state["chunks"]
    nch = len(bounds) - 1
    if done and len(done) < nch:
        hosts = sorted({str(c.get("host")) for c in done.values()})
        log(f"E3 {label:16s} {measure!s:8s} resuming at chunk {len(done) + 1}/{nch}, after "
            f"{_hms(sum(c['seconds'] for c in done.values()))} on {', '.join(hosts)} ({path})")
    elif not done:
        log(f"E3 {label:16s} {measure!s:8s} m={g.num_edges:,} in {nch} saved chunks ({path})")
    hub = np.maximum(g.deg[g.edges[:, 0]], g.deg[g.edges[:, 1]]) >= 100
    for c in range(nch):
        if str(c) in done:
            continue
        lock = None
        if claims:
            lock = CK.StepLock(claims, f"{base}.c{c}")
            try:
                lock.acquire()
            except CK.Busy:
                continue  # another worker is computing this chunk
            done = (_load_plan(path, key) or state)["chunks"]
            if str(c) in done:
                lock.release()
                continue
        lo, hi = bounds[c], bounds[c + 1]
        with RSSSampler() as s:
            t0 = time.perf_counter()
            res = orcut.curvature(g, measure, edges=np.arange(lo, hi, dtype=np.int64), n_jobs=threads, diagnostics=True)
            dt = time.perf_counter() - t0
        st = _chunk_stats(res["curvature"], res["diagnostics"], hub[lo:hi])
        del res
        st.update(seconds=dt, peak_mb=s.peak - s.base, host=socket.gethostname(), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with xio.locked(path):  # merge with the chunks other workers saved meanwhile
            state = _load_plan(path, key) or state
            state["chunks"][str(c)] = st
            xio.atomic_json(state, path)
        done = state["chunks"]
        if lock is not None:
            lock.release()
        spent = sum(x["seconds"] for x in done.values())
        log(f"E3 {label:16s} {measure!s:8s} chunk {c + 1}/{nch} ({hi - lo:,} edges) {_hms(dt)}; {len(done)}/{nch} done "
            f"in {_hms(spent)}, about {_hms(spent / len(done) * (nch - len(done)))} left")
        if after_chunk is not None:
            after_chunk(c)
    state = _load_plan(path, key) or state
    if len(state["chunks"]) < nch:
        return None  # chunks still under way on other workers
    finish = None
    if claims:
        finish = CK.StepLock(claims, f"{base}.row")
        try:
            finish.acquire()
        except CK.Busy:
            return None  # another worker is writing the row
    return state, _combine(state["chunks"].values()), finish


def _pot_ckpt(a, name):
    return os.path.join(out_dir(a.out, "e3", "partial"), f"pot.{R.slug(name)}.json")


def _finish_chunked(path):
    """Keep the chunk log of a completed row next to the table, but out of the way of a new run."""
    try:
        os.replace(path, path[:-len(".json")] + ".done.json")
    except OSError:
        pass


def part_real(a):
    from orcut import _core as C

    rows = Rows(f"{out_dir(a.out, 'e3')}/real.csv", ["graph", "measure"])
    claims = os.path.join(out_dir(a.out, "e3", "partial"), "claims")  # chunks and rows taken by workers
    left = []  # rows left to other workers (main.py --worker): the step is done only once they are
    graphs = a.graphs or (R.MEDIUM + R.LARGE + R.XLARGE)
    if a.quick:
        graphs = ["bundled:jazz", "synth:CL2.1:10000:12:0"]
    for spec in graphs:
        name = R.display(spec)
        measures = ["lazy"] if (a.lazy_only or spec in R.LAZY_ONLY) else ["lazy", "nonlazy", "lly"]
        if all(rows.done(graph=name, measure=m) for m in measures):
            continue
        t0 = time.perf_counter()
        name, g, meta = R.load(spec, a.data_dir)
        t_load = time.perf_counter() - t0
        big = g.num_edges > 10**7
        chunked = a.chunk_edges > 0 and g.num_edges > a.chunk_above
        for measure in measures:
            rows.refresh()  # rows other workers wrote meanwhile
            if rows.done(graph=name, measure=measure):
                continue
            if chunked:
                # one pass, saved chunk by chunk: a run stopped by the time limit resumes where it was
                ckpt = os.path.join(out_dir(a.out, "e3", "partial"), f"{R.slug(name)}.{measure}.T{a.threads}.json")
                got = solve_chunked(g, measure, a.threads, ckpt, min(a.chunk_edges, max(1, g.num_edges // 20)), name,
                                    claims=claims)
                if got is None:
                    log(f"E3 {name:16s} {measure:8s} its other chunks are being computed by other workers")
                    left.append((name, measure))
                    continue
                state, tot, finish = got
                peak = tot["peak_mb"]
                row = dict(graph=name, measure=measure, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"],
                           deg_mean=meta["deg_mean"], threads=a.threads, time_s=tot["seconds"], time_min_s=tot["seconds"],
                           runs=1, load_s=t_load, peak_extra_mb=peak,
                           graph_mb=(g.indptr.nbytes + g.indices.nbytes + g.edges.nbytes) / 2**20,
                           nan=tot["nan"], cite=meta.get("cite", ""), chunks=len(state["bounds"]) - 1,
                           chunk_hosts=len({str(c.get("host")) for c in state["chunks"].values()}))
                if measure == "lazy":
                    row.update(diag_fields(tot, g.num_edges))
                    if not a.no_pot:
                        full, solve, timed, free = pot_estimate(g, per_bin=a.pot_per_bin, max_pairs=a.pot_max_pairs,
                                                                ckpt=_pot_ckpt(a, name))
                        row.update(pot_est_1thread_s=full, pot_solve_est_1thread_s=solve, pot_timed=timed,
                                   pot_est_s=full / a.threads, pot_free_share=free)
                rows.add(row)
                _finish_chunked(ckpt)
                _finish_chunked(_pot_ckpt(a, name))
                if finish is not None:
                    finish.release()
                log(f"E3 {name:16s} {measure:8s} m={g.num_edges:>11,} T={a.threads} {row['time_s']:9.2f}s "
                    f"peak+{peak:8.1f}MB" + (f" POT est {row.get('pot_est_s', float('nan')):9.1f}s" if measure == "lazy" else "")
                    + f" ({row['chunks']} chunks)")
                continue
            row_lock = CK.StepLock(claims, f"{R.slug(name)}.{measure}.T{a.threads}.row")  # one worker per row
            try:
                row_lock.acquire()
            except CK.Busy:
                left.append((name, measure))
                continue
            solve_all(g, measure, a.threads)  # warm (and page in the graph)
            times = []
            peak = float("nan")
            for r in range(1 if big else a.reps):
                with RSSSampler() as s:
                    t = time.perf_counter()
                    val = solve_all(g, measure, a.threads)
                    times.append(time.perf_counter() - t)
                peak = max(peak, s.peak - s.base) if peak == peak else s.peak - s.base
            row = dict(graph=name, measure=measure, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"],
                       deg_mean=meta["deg_mean"], threads=a.threads, time_s=float(np.median(times)),
                       time_min_s=float(np.min(times)), runs=len(times), load_s=t_load, peak_extra_mb=peak,
                       graph_mb=(g.indptr.nbytes + g.indices.nbytes + g.edges.nbytes) / 2**20,
                       nan=int(np.isnan(val).sum()), cite=meta.get("cite", ""))
            if measure == "lazy":
                d = solve_all(g, measure, a.threads, diagnostics=True)["diagnostics"]
                path = d[:, C.D_PATH]
                row.update(share_zero=float(np.mean(path == C.PATH_ZERO)),
                           share_fast=float(np.mean((path == C.PATH_FAST) | (path == C.PATH_NOADJ))),
                           share_cut=float(np.mean((path == C.PATH_CUT) | (path == C.PATH_CUTW) | (path == C.PATH_WIT))),
                           share_witness=float(np.mean((path == C.PATH_CUTW) | (path == C.PATH_WIT))),
                           mean_nodes=float(d[:, C.D_NODES].mean()), mean_arcs=float(d[:, C.D_ARCS].mean()),
                           max_arcs=int(d[:, C.D_ARCS].max()))
                hub = np.maximum(g.deg[g.edges[:, 0]], g.deg[g.edges[:, 1]]) >= 100
                ok = hub & (d[:, C.D_GROUPS] > 0)
                row["hub_edges"] = int(hub.sum())
                row["twin_ratio_hub"] = float(np.mean(d[ok, C.D_ROWS] / d[ok, C.D_GROUPS])) if ok.any() else float("nan")
                row["twin_ratio_hub_max"] = float(np.max(d[ok, C.D_ROWS] / d[ok, C.D_GROUPS])) if ok.any() else float("nan")
                if not a.no_pot:
                    full, solve, timed, free = pot_estimate(g, per_bin=a.pot_per_bin, max_pairs=a.pot_max_pairs,
                                                            ckpt=_pot_ckpt(a, name))
                    row.update(pot_est_1thread_s=full, pot_solve_est_1thread_s=solve, pot_timed=timed,
                               pot_est_s=full / a.threads, pot_free_share=free)
            rows.add(row)
            row_lock.release()
            if measure == "lazy" and not a.no_pot:
                _finish_chunked(_pot_ckpt(a, name))
            log(f"E3 {name:16s} {measure:8s} m={g.num_edges:>11,} T={a.threads} {row['time_s']:9.2f}s "
                f"peak+{peak:8.1f}MB" + (f" POT est {row.get('pot_est_s', float('nan')):9.1f}s" if measure == "lazy" else ""))
        del g
    rows.refresh()
    for name, measure in left:  # left to other workers: done only once they are
        if not rows.done(graph=name, measure=measure):
            common.mark_incomplete(f"{name} {measure}")


def part_threads(a):
    rows = Rows(f"{out_dir(a.out, 'e3')}/threads.csv", ["graph", "threads"])
    import numba

    graphs = a.graphs or ["real:GitHub", "real:DBLP", "real:YouTube", "real:LiveJournal"]
    if a.quick:
        graphs = ["synth:CL2.1:10000:12:0"]
    tmax = min(a.threads, numba.config.NUMBA_NUM_THREADS)
    ts = [1]
    while ts[-1] * 2 <= tmax:
        ts.append(ts[-1] * 2)
    if ts[-1] != tmax:
        ts.append(tmax)
    for spec in graphs:
        if all(rows.done(graph=R.display(spec), threads=T) for T in ts):
            continue  # no need to load it or warm up (minutes on the largest graph)
        name, g, meta = R.load(spec, a.data_dir)
        solve_all(g, "lazy", tmax)
        for T in ts:
            if rows.done(graph=name, threads=T):
                continue
            t, _ = common.best_of(lambda: solve_all(g, "lazy", T), reps=a.reps if g.num_edges < 10**7 else 1)
            rows.add(dict(graph=name, threads=T, m=g.num_edges, time_s=t))
            log(f"E3T {name:16s} T={T:3d} {t:9.2f}s")


def predicted_time(rows, fam, regime, seed, m):
    """Seconds per solve of a synthetic graph with m edges, extrapolated from the two largest
    smaller graphs of the same family, regime and seed (time ~ m^k, k from those two, clipped to
    [1, 3]); 0 without two of them."""
    prev = sorted((common.num(r, "m"), common.num(r, "time_s")) for r in rows.rows
                  if r.get("family") == fam and r.get("regime") == regime and str(r.get("seed")) == str(seed))
    prev = [(pm, pt) for pm, pt in prev if pm == pm and pt == pt and 0 < pm < m and pt > 0]
    if len(prev) < 2:
        return 0.0
    (m1, t1), (m2, t2) = prev[-2], prev[-1]
    k = float(np.clip(np.log(t2 / t1) / np.log(m2 / m1), 1.0, 3.0)) if m2 > m1 else 1.0
    return t2 * (m / m2) ** k


def synth_row(rows, fam, regime, n, seed):
    """The row of the synthetic graph requested with n vertices. Rows record the size of its largest
    component (n, smaller than requested for Chung--Lu, and by a few vertices for ER or RGG at 10^7)
    and, from now on, the size requested (n_target); older rows are matched by their size, which
    is at most n and more than n/3 (consecutive sizes of a sweep differ by 3x or more)."""
    for r in rows.rows:
        if r.get("family") != fam or r.get("regime") != regime or str(r.get("seed")) != str(seed):
            continue
        nt = common.num(r, "n_target")
        if nt == nt:
            if int(nt) == n:
                return r
        elif n / 3 < common.num(r, "n") <= n:
            return r
    return None


def synth_done(rows, fam, regime, n, seed):
    r = synth_row(rows, fam, regime, n, seed)
    return r is not None and common.row_ok(r)


def synth_m(rows, fam, regime, n):
    """Edges expected for the requested size n, from the largest graph of the family done so far."""
    done = [(common.num(r, "n"), common.num(r, "m")) for r in rows.rows if r.get("family") == fam and
            r.get("regime") == regime]
    done = [(x, y) for x, y in done if x == x and y == y and x > 0]
    if not done:
        return 0.0
    x, y = max(done)
    return y / x * n


def part_synth(a):
    rows = Rows(f"{out_dir(a.out, 'e3')}/synth.csv", ["family", "regime", "n", "seed"])
    fams = ["ER", "BA", "RGG", "WS", "CL2.1", "CL2.5"]
    sizes_fixed = [10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7]
    sizes_grow = [10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6]
    if a.quick:
        sizes_fixed, sizes_grow, fams = [10**3, 10**4], [10**3, 3 * 10**3], ["ER", "CL2.1"]
    plan = [(f, "fixed16", n, 16.0) for f in fams for n in sizes_fixed] + \
           [(f, "grow0.4", n, 2.1 * n ** 0.4) for f in fams for n in sizes_grow]
    claims = os.path.join(out_dir(a.out, "e3", "partial"), "claims")  # chunks and rows taken by workers
    left = []  # rows left to other workers (main.py --worker): the step is done only once they are
    pending = [(fam, regime, n, d, seed) for fam, regime, n, d in plan for seed in range(a.seeds)
               if not synth_done(rows, fam, regime, n, seed)]
    while pending:
        # in plan order, but a row predicted to take more than --defer-above per solve waits until
        # every other row is done (CL2.1 at 10^7 would otherwise hold the rest up for months)
        item = next((it for it in pending if predicted_time(rows, it[0], it[1], it[4], synth_m(rows, *it[:3])) <=
                     a.defer_above), pending[0])
        pending.remove(item)
        fam, regime, n, d, seed = item
        rows.refresh()  # rows other workers wrote meanwhile
        if synth_done(rows, fam, regime, n, seed):
            continue
        import datasets as D

        nn, e = D.synthetic(fam, n, d, seed)
        g = D.to_graph(nn, e)
        pred = predicted_time(rows, fam, regime, seed, g.num_edges)
        extra = {}
        if a.chunk_edges > 0 and pred > a.chunk_after:
            # predicted to take hours: one pass (no warm-up, no repeats), saved chunk by chunk, in
            # chunks of about --chunk-target seconds (at least 20), so that each fits in a job
            k = max(20, int(np.ceil(pred / a.chunk_target)))
            log(f"E3S {fam:6s} {regime:8s} n={nn:>9,} m={g.num_edges:>11,} predicted {_hms(pred)} per solve, "
                f"{k} chunks")
            ckpt = os.path.join(out_dir(a.out, "e3", "partial"), f"synth.{R.slug(fam)}.{regime}.n{n}.s{seed}.T{a.threads}.json")
            got = solve_chunked(g, "lazy", a.threads, ckpt, max(1, -(-g.num_edges // k)), f"{fam} n={nn}", claims=claims)
            if got is None:
                log(f"E3S {fam:6s} {regime:8s} n={nn:>9,} its other chunks are being computed by other workers")
                left.append(item)
                continue
            state, tot, finish = got
            t = tot["seconds"]
            extra = dict(chunks=len(state["bounds"]) - 1, chunk_hosts=len({str(c.get("host")) for c in state["chunks"].values()}))
        else:
            ckpt = None
            finish = CK.StepLock(claims, f"synth.{R.slug(fam)}.{regime}.n{n}.s{seed}.T{a.threads}.row")  # one worker
            try:
                finish.acquire()
            except CK.Busy:
                left.append(item)
                continue
            solve_all(g, "lazy", a.threads)
            t, _ = common.best_of(lambda: solve_all(g, "lazy", a.threads), reps=a.reps if g.num_edges < 10**7 else 1)
        W = int(np.sum(g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]))
        W2 = float(np.sum((g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]).astype(float) ** 2))
        row = dict(family=fam, regime=regime, n=nn, n_target=n, avg_deg_target=d, seed=seed, m=g.num_edges,
                   deg_max=int(g.deg.max()), sum_dsum=W, sum_dsum2=W2, threads=a.threads, time_s=t, **extra)
        pck = os.path.join(out_dir(a.out, "e3", "partial"), f"synth.pot.{R.slug(fam)}.{regime}.n{n}.s{seed}.json")
        if not a.no_pot:
            full, solve, timed, free = pot_estimate(g, per_bin=a.pot_per_bin, max_pairs=a.pot_max_pairs, ckpt=pck)
            row.update(pot_est_1thread_s=full, pot_est_s=full / a.threads, pot_free_share=free)
        rows.add(row)
        _finish_chunked(pck)
        if ckpt:
            _finish_chunked(ckpt)
        if finish is not None:
            finish.release()
        log(f"E3S {fam:6s} {regime:8s} n={nn:>9,} m={g.num_edges:>11,} {t:8.2f}s")
    rows.refresh()
    for fam, regime, n, d, seed in left:  # left to other workers: done only once they are
        if not synth_done(rows, fam, regime, n, seed):
            common.mark_incomplete(f"{fam} {regime} n={n} seed={seed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["real", "threads", "synth"], required=True)
    ap.add_argument("--graphs", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--lazy-only", action="store_true")
    ap.add_argument("--no-pot", action="store_true", help="skip the POT estimate")
    ap.add_argument("--pot-per-bin", type=int, default=50)
    ap.add_argument("--pot-max-pairs", type=float, default=5e7,
                    help="edges with (d_i+1)(d_j+1) above this count as free in the POT estimate (memory)")
    ap.add_argument("--chunk-edges", type=int, default=2 * 10**6,
                    help="edges per saved chunk of a long solve (0: never in chunks)")
    ap.add_argument("--chunk-above", type=int, default=10**7,
                    help="part real: graphs with more edges are solved once, in saved chunks")
    ap.add_argument("--chunk-after", type=float, default=3600.0,
                    help="part synth: rows predicted to take longer (seconds per solve) are solved once, in saved chunks")
    ap.add_argument("--chunk-target", type=float, default=1800.0,
                    help="part synth: seconds per saved chunk (from the predicted time; at least 20 chunks)")
    ap.add_argument("--defer-above", type=float, default=2 * 86400.0,
                    help="part synth: rows predicted to take longer (seconds per solve) run after all the others")
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.pot_per_bin = 1, 10
    common.warm_jit()
    {"real": part_real, "threads": part_threads, "synth": part_synth}[a.part](a)


if __name__ == "__main__":
    main()
    sys.exit(common.finish())
