"""E7. Beyond the lazy walk: other one-step measures, balls and multi-step walks.

Part ``measures``: whole-graph time of orcut (``--threads`` threads, best of ``--reps``) for the
lazy, non-lazy and Lin--Lu--Yau curvatures, uniform idleness p in {1/4, 1/2, 3/4}, and a weighted
lazy walk with integer edge weights uniform in {1, ..., 10} (all exact), relative to the lazy one.

Part ``radius``: measures with larger supports, solved by the layered closure with D = 2r + 1
(``orcut.ball_curvature``, ``orcut.walk_curvature``; reference implementation with explicit
distances): the uniform measure on the ball of radius r in {2, 3}, and the r-step lazy random walk
for r in {2, 3}. Against POT's network simplex on the same problems (distances by truncated BFS
in numba, the cheapest input we know how to build), on a stratified sample of edges: per-edge time
of both (build included), largest difference of the values, and the share of the sample on which
orcut's plan and potentials are verified exactly (``certify``).

Usage: python e7_generality.py --part measures|radius [--graphs ...] [--threads 32] [--quick]
"""
from __future__ import annotations

import argparse
import os
import time
from fractions import Fraction

import numpy as np
from numba import njit

import common
from common import Rows, log, out_dir
import baselines as B
import registry as R


# ---------------------------------------------------------------- part A

def measure_specs(g, seed=0):
    import orcut

    rng = np.random.default_rng(seed)
    return [("lazy", "lazy"), ("nonlazy", "nonlazy"), ("lly", "lly"), ("idle 1/4", Fraction(1, 4)),
            ("idle 1/2", Fraction(1, 2)), ("idle 3/4", Fraction(3, 4)),
            ("weighted walk", orcut.Walk(weights=rng.integers(1, 11, g.num_edges), idleness="lazy"))]


def part_measures(a):
    import orcut
    from orcut import _core as C

    rows = Rows(f"{out_dir(a.out, 'e7')}/measures.csv", ["graph", "measure"])
    graphs = a.graphs or ["bundled:arxiv", "real:Deezer", "real:Twitch-DE", "real:GitHub", "real:Amazon-Computers",
                          "real:Epinions", "real:DBLP", "real:YouTube", "synth:RGG:100000:20:0",
                          "synth:CL2.1:30000:12:0"]
    if a.quick:
        graphs = ["bundled:jazz", "synth:CL2.1:5000:8:0"]
    for spec in graphs:
        name, g, meta = R.load(spec, a.data_dir)
        t_lazy = None
        for label, m in measure_specs(g):
            if rows.done(graph=name, measure=label):
                continue
            orcut.curvature(g, m, n_jobs=a.threads)  # warm
            t, out = common.best_of(lambda: orcut.curvature(g, m, n_jobs=a.threads, diagnostics=True),
                                    reps=a.reps if g.num_edges < 10**7 else 1)
            d = out["diagnostics"]
            if label == "lazy":
                t_lazy = t
            row = dict(graph=name, measure=label, m=g.num_edges, deg_max=meta["deg_max"], threads=a.threads, time_s=t,
                       rel_lazy=t / t_lazy if t_lazy else float("nan"), nan=int(np.isnan(out["curvature"]).sum()),
                       mean_arcs=float(d[:, C.D_ARCS].mean()),
                       share_cut=float(np.mean(np.isin(d[:, C.D_PATH], (C.PATH_CUT, C.PATH_CUTW, C.PATH_WIT)))))
            rows.add(row)
            log(f"E7M {name:16s} {label:14s} {t:9.3f}s ({row['rel_lazy']:.2f} x lazy)")


# ---------------------------------------------------------------- part B

@njit(cache=True)
def _dist_rows(indptr, indices, P, N, maxd, stamp, dist, order, ep0):
    """Distances (capped at maxd + 1) from every vertex of P to every vertex of N, by BFS."""
    M = np.empty((P.shape[0], N.shape[0]), np.float64)
    from_ep = ep0
    for a in range(P.shape[0]):
        from_ep += 1
        stamp[P[a]] = from_ep
        dist[P[a]] = 0
        order[0] = P[a]
        qh = 0
        qt = 1
        while qh < qt:
            u = order[qh]
            qh += 1
            du = dist[u]
            if du == maxd:
                continue
            for e in range(indptr[u], indptr[u + 1]):
                w = indices[e]
                if stamp[w] != from_ep:
                    stamp[w] = from_ep
                    dist[w] = du + 1
                    order[qt] = w
                    qt += 1
        for b in range(N.shape[0]):
            y = N[b]
            M[a, b] = dist[y] if stamp[y] == from_ep else maxd + 1
    return M, from_ep


class RadiusProblems:
    """Local transport problems for ball and multi-step walk measures (explicit distances)."""

    def __init__(self, g):
        n = g.num_nodes
        self.g = g
        self.stamp = np.full(n, -1, np.int64)
        self.dist = np.zeros(n, np.int64)
        self.order = np.empty(n, np.int64)
        self.loc = np.zeros(n, np.int64)
        self.fbuf = np.zeros(n)
        self.fnxt = np.zeros(n)
        self.ep = 0

    def _measure(self, u, kind, r):
        from orcut import _layered as LY

        g = self.g
        self.ep += 1
        if kind == "ball":
            k = LY._bfs(g.indptr, g.indices, u, r, self.stamp, self.ep, self.dist, self.order)
            return self.order[:k].copy(), np.full(k, 1.0 / k)
        k = LY._walk_measure(g.indptr, g.indices, u, r, True, self.stamp, self.ep, self.loc, self.order, self.fbuf,
                             self.fnxt)
        return self.order[:k].copy(), self.fbuf[:k].copy()

    def build(self, e, kind, r):
        g = self.g
        i, j = int(g.edges[e, 0]), int(g.edges[e, 1])
        Si, mi = self._measure(i, kind, r)
        Sj, mj = self._measure(j, kind, r)
        net = {}
        for x, v in zip(Si.tolist(), mi.tolist()):
            net[x] = net.get(x, 0.0) + v
        for y, v in zip(Sj.tolist(), mj.tolist()):
            net[y] = net.get(y, 0.0) - v
        P = np.array([x for x, v in net.items() if v > 1e-15], np.int64)
        N = np.array([y for y, v in net.items() if v < -1e-15], np.int64)
        a = np.array([net[x] for x in P])
        b = np.array([-net[y] for y in N])
        M, self.ep = _dist_rows(g.indptr, g.indices, P, N, 2 * r, self.stamp, self.dist, self.order, self.ep)
        return a, b, M


def support_pairs(g, ids, r):
    """|B_r(i)| * |B_r(j)| for the edges ids (an upper bound on |P| |N| of balls and r-step walks)."""
    from orcut import _layered as LY

    n = g.num_nodes
    stamp = np.full(n, -1, np.int64)
    dist = np.zeros(n, np.int64)
    order = np.empty(n, np.int64)
    out = np.empty(len(ids), np.int64)
    ep = 0
    for k, e in enumerate(ids):
        i, j = int(g.edges[e, 0]), int(g.edges[e, 1])
        ep += 1
        bi = LY._bfs(g.indptr, g.indices, i, r, stamp, ep, dist, order)
        ep += 1
        bj = LY._bfs(g.indptr, g.indices, j, r, stamp, ep, dist, order)
        out[k] = bi * bj
    return out


def part_radius(a):
    import numba
    import orcut

    rows = Rows(f"{out_dir(a.out, 'e7')}/radius.csv", ["graph", "config"])
    graphs = a.graphs or ["bundled:power_grid", "bundled:yeast", "bundled:jazz", "synth:ER:5000:8:0",
                          "synth:RGG:5000:10:0", "synth:WS:5000:8:0", "synth:BA:5000:6:0", "synth:CL2.5:5000:6:0"]
    if a.quick:
        graphs = ["bundled:yeast", "bundled:jazz"]
    configs = [("ball", 2), ("ball", 3), ("walk", 2), ("walk", 3)]
    for spec in graphs:
        name, g, meta = R.load(spec, a.data_dir)
        strata = common.stratified_bins(g, a.per_bin, seed=2)
        ids_sample = np.concatenate([s_ for _, s_, _ in strata])
        for kind, r in configs:
            cfg = f"{kind} r={r}"
            if rows.done(graph=name, config=cfg):
                continue
            row = dict(graph=name, config=cfg, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"], threads=a.threads)
            # the reference implementation lists every pair of P x N: skip supports that are too large
            # (the sample contains the edges of largest d_i + d_j)
            pairs = support_pairs(g, ids_sample, r)
            row["max_support_pairs"] = int(pairs.max()) if len(pairs) else 0
            if row["max_support_pairs"] > a.max_pairs:
                row["status"] = f"skipped: supports up to {row['max_support_pairs']} pairs (> --max-pairs)"
                rows.add(row)
                log(f"E7R {name:16s} {cfg:9s} skipped ({row['max_support_pairs']:,} pairs)")
                continue
            f_all = (lambda ids=None: orcut.ball_curvature(g, r, edges=ids, n_jobs=a.threads)) if kind == "ball" else \
                (lambda ids=None: orcut.walk_curvature(g, r, edges=ids, n_jobs=a.threads))
            ids_all = np.concatenate([s for _, s, _ in strata])
            f_all(ids_all[:4])  # warm
            # whole graph when affordable, otherwise estimated from the strata
            if g.num_edges <= a.whole_max_edges:
                t_whole, val_all = common.best_of(lambda: f_all(), reps=1)
                row["time_whole_s"] = t_whole
            with common.numba_threads(1):
                times_o = [common.time_per_edge(lambda sub: f_all(sub), s, reps=1) for _, s, _ in strata]
                vals_o = [f_all(s) for _, s, _ in strata]
            row["est_total_1thread_s"] = common.estimate_total(strata, times_o)
            # certificates on the sample (layered closure with certify=True)
            if kind == "ball":
                _, diag = orcut.ball_curvature(g, r, edges=ids_all, n_jobs=a.threads, certify=True)
            else:
                _, diag = orcut.walk_curvature(g, r, edges=ids_all, n_jobs=a.threads, certify=True)
            row["certified_share"] = float(np.mean(diag[:, 5] == 1))
            row["mean_P"] = float(np.mean(diag[:, 0]))
            row["mean_N"] = float(np.mean(diag[:, 1]))
            row["mean_arcs"] = float(np.mean(diag[:, 3]))
            # POT on explicit distances
            lp = RadiusProblems(g)
            deadline = time.perf_counter() + a.budget
            times_p, diffs = [], []
            pa, pb, pM = lp.build(int(ids_all[0]), kind, r)
            B.pot_emd_c(pa, pb, pM)
            for (bn, s, c), vo in zip(strata, vals_o):
                tp = np.full(len(s), np.nan)
                dv = np.full(len(s), np.nan)
                for k_, e in enumerate(s):
                    if time.perf_counter() > deadline:
                        break
                    t0 = time.perf_counter()
                    pa, pb, pM = lp.build(int(e), kind, r)
                    w = B.pot_emd_c(pa, pb, pM)
                    tp[k_] = time.perf_counter() - t0
                    dv[k_] = abs((1.0 - w) - vo[k_])
                times_p.append(tp)
                diffs.append(dv)
            dall = np.concatenate(diffs)
            row.update(pot_est_total_1thread_s=common.estimate_total(strata, times_p),
                       pot_timed=int(np.isfinite(dall).sum()), sample=len(ids_all),
                       max_diff_vs_pot=float(np.nanmax(dall)) if np.isfinite(dall).any() else float("nan"))
            rows.add(row)
            log(f"E7R {name:16s} {cfg:9s} orcut est {row['est_total_1thread_s']:9.2f}s POT est "
                f"{row['pot_est_total_1thread_s']:9.2f}s (1 thread) max|diff| {row['max_diff_vs_pot']:.1e} "
                f"certified {row['certified_share']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["measures", "radius"], required=True)
    ap.add_argument("--graphs", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--per-bin", type=int, default=30)
    ap.add_argument("--whole-max-edges", type=int, default=2 * 10**5)
    ap.add_argument("--max-pairs", type=float, default=5e6,
                    help="skip a graph and support radius whose largest |B_r(i)| |B_r(j)| exceeds this (memory)")
    ap.add_argument("--budget", type=float, default=600.0, help="seconds of POT per graph and configuration")
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.per_bin, a.budget = 1, 8, 30.0
    common.warm_jit()
    (part_measures if a.part == "measures" else part_radius)(a)


if __name__ == "__main__":
    main()
