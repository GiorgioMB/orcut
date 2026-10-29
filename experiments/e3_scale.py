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

Usage: python e3_scale.py --part real|threads|synth [--graphs ...] [--threads 32] [--out results]
"""
from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np

import common
from common import Rows, log, out_dir
import baselines as B
import registry as R


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


def pot_estimate(g, per_bin=50, budget_s=600.0, seed=0, max_pairs=5e7):
    """(estimated single-threaded seconds of the POT pipeline on all edges, of its solves alone,
    number of edges timed, share of edges counted as free). Edges whose dense local problem would
    have more than ``max_pairs`` entries are not run and count as free, so the estimate stays a
    lower bound."""
    rng = np.random.default_rng(seed)
    dsum = g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]
    bins = np.floor(np.log2(dsum)).astype(int)
    prod = (g.deg[g.edges[:, 0]] + 1).astype(np.float64) * (g.deg[g.edges[:, 1]] + 1)
    tot_full = 0.0
    tot_solve = 0.0
    timed = 0
    free = int(np.sum(prod > max_pairs))
    t0 = time.perf_counter()
    for bn in sorted(np.unique(bins), reverse=True):  # largest first: they dominate
        all_ids = np.flatnonzero((bins == bn) & (prod <= max_pairs))
        if len(all_ids) == 0:
            continue
        ids = np.sort(rng.choice(all_ids, min(per_bin, len(all_ids)), replace=False)).astype(np.int64)
        left = budget_s - (time.perf_counter() - t0)
        if left <= 0:
            return float("nan"), float("nan"), timed, free / g.num_edges
        _, ts, tb, done = B.run_local(g, ids, "pot_simplex", reps=1, budget_s=left)
        if done == 0:
            return float("nan"), float("nan"), timed, free / g.num_edges
        tot_full += len(all_ids) * (ts + tb)
        tot_solve += len(all_ids) * ts
        timed += done
    return tot_full, tot_solve, timed, free / g.num_edges


def solve_all(g, measure, threads, diagnostics=False):
    import orcut

    return orcut.curvature(g, measure, n_jobs=threads, diagnostics=diagnostics)


def part_real(a):
    from orcut import _core as C

    rows = Rows(f"{out_dir(a.out, 'e3')}/real.csv", ["graph", "measure"])
    graphs = a.graphs or (R.MEDIUM + R.LARGE + R.XLARGE)
    if a.quick:
        graphs = ["bundled:jazz", "synth:CL2.1:10000:12:0"]
    for spec in graphs:
        name = R.display(spec)
        measures = ["lazy"] if (a.lazy_only or spec in R.XLARGE) else ["lazy", "nonlazy", "lly"]
        if all(rows.done(graph=name, measure=m) for m in measures):
            continue
        t0 = time.perf_counter()
        name, g, meta = R.load(spec, a.data_dir)
        t_load = time.perf_counter() - t0
        big = g.num_edges > 10**7
        for measure in measures:
            if rows.done(graph=name, measure=measure):
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
                    full, solve, timed, free = pot_estimate(g, per_bin=a.pot_per_bin, budget_s=a.pot_budget,
                                                            max_pairs=a.pot_max_pairs)
                    row.update(pot_est_1thread_s=full, pot_solve_est_1thread_s=solve, pot_timed=timed,
                               pot_est_s=full / a.threads, pot_free_share=free)
            rows.add(row)
            log(f"E3 {name:16s} {measure:8s} m={g.num_edges:>11,} T={a.threads} {row['time_s']:9.2f}s "
                f"peak+{peak:8.1f}MB" + (f" POT est {row.get('pot_est_s', float('nan')):9.1f}s" if measure == "lazy" else ""))
        del g


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
        name, g, meta = R.load(spec, a.data_dir)
        solve_all(g, "lazy", tmax)
        for T in ts:
            if rows.done(graph=name, threads=T):
                continue
            t, _ = common.best_of(lambda: solve_all(g, "lazy", T), reps=a.reps if g.num_edges < 10**7 else 1)
            rows.add(dict(graph=name, threads=T, m=g.num_edges, time_s=t))
            log(f"E3T {name:16s} T={T:3d} {t:9.2f}s")


def part_synth(a):
    rows = Rows(f"{out_dir(a.out, 'e3')}/synth.csv", ["family", "regime", "n", "seed"])
    fams = ["ER", "BA", "RGG", "WS", "CL2.1", "CL2.5"]
    sizes_fixed = [10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7]
    sizes_grow = [10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6]
    if a.quick:
        sizes_fixed, sizes_grow, fams = [10**3, 10**4], [10**3, 3 * 10**3], ["ER", "CL2.1"]
    plan = [(f, "fixed16", n, 16.0) for f in fams for n in sizes_fixed] + \
           [(f, "grow0.4", n, 2.1 * n ** 0.4) for f in fams for n in sizes_grow]
    for fam, regime, n, d in plan:
        for seed in range(a.seeds):
            if rows.done(family=fam, regime=regime, n=n, seed=seed):
                continue
            import datasets as D

            nn, e = D.synthetic(fam, n, d, seed)
            g = D.to_graph(nn, e)
            solve_all(g, "lazy", a.threads)
            t, _ = common.best_of(lambda: solve_all(g, "lazy", a.threads), reps=a.reps if g.num_edges < 10**7 else 1)
            W = int(np.sum(g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]))
            W2 = float(np.sum((g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]).astype(float) ** 2))
            row = dict(family=fam, regime=regime, n=nn, avg_deg_target=d, seed=seed, m=g.num_edges,
                       deg_max=int(g.deg.max()), sum_dsum=W, sum_dsum2=W2, threads=a.threads, time_s=t)
            if not a.no_pot and g.num_edges <= a.pot_max_edges:
                full, solve, timed, free = pot_estimate(g, per_bin=a.pot_per_bin, budget_s=a.pot_budget,
                                                        max_pairs=a.pot_max_pairs)
                row.update(pot_est_1thread_s=full, pot_est_s=full / a.threads, pot_free_share=free)
            rows.add(row)
            log(f"E3S {fam:6s} {regime:8s} n={nn:>9,} m={g.num_edges:>11,} {t:8.2f}s")


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
    ap.add_argument("--pot-budget", type=float, default=1800.0, help="seconds of POT timing per graph")
    ap.add_argument("--pot-max-edges", type=int, default=5 * 10**6)
    ap.add_argument("--pot-max-pairs", type=float, default=5e7,
                    help="edges with (d_i+1)(d_j+1) above this count as free in the POT estimate (memory)")
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.pot_per_bin, a.pot_budget = 1, 10, 60.0
    common.warm_jit()
    {"real": part_real, "threads": part_threads, "synth": part_synth}[a.part](a)


if __name__ == "__main__":
    main()
