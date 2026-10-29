"""E1. Exactness and certification.

For every edge of every graph and every measure, the value of orcut is compared, as an exact fraction,
with an integer min-cost flow (OR-Tools) on the explicit local distance matrix; with POT's network
simplex (floating point; we report the largest difference); with the independent oracle of
``orcut._reference`` (full distance matrix, three-level closure, different max-flow code); and with
the layered closure of ``orcut._layered`` (explicit distances, no compression). On a sample of edges,
``orcut.certify`` returns an optimal plan and an optimal potential, verified exactly, and
``Certificate.verify`` re-checks them from scratch.

Usage: python e1_exactness.py [--graphs SPEC ...] [--out results] [--quick]
"""
from __future__ import annotations

import argparse
import time
from fractions import Fraction
from math import gcd

import numpy as np

import common
from common import Rows, log, out_dir
import baselines as B
import registry as R

MEASURES = ["lazy", "nonlazy", "lly", "idle:1/2", "walk"]


def _walk(g, seed=0):
    import orcut

    rng = np.random.default_rng(seed)
    return orcut.Walk(weights=rng.integers(1, 11, g.num_edges), idleness="lazy")


def _spec(measure, g):
    if measure == "walk":
        return _walk(g)
    if measure.startswith("idle:"):
        a, b = measure.split(":")[1].split("/")
        return Fraction(int(a), int(b))
    return measure


def _lp_kind(measure):
    return {"lazy": "lazy", "nonlazy": "nonlazy", "lly": "lly"}.get(measure, measure)


def ortools_fraction(g, e, measure, spec, lp):
    """Exact W1 of edge e as a Fraction, by OR-Tools on the explicit local distances."""
    if measure != "walk":
        a, b, M, L, aint, bint = lp.build(int(e), True)
        return Fraction(B.ortools_mcf(aint, bint, M), L)
    import orcut
    from orcut.certify import _bfs_all, measures_of_edge

    i, j = map(int, g.edges[e])
    mi, mj = measures_of_edge(g, i, j, spec)
    L = 1
    for v in list(mi.values()) + list(mj.values()):
        L = L * v.denominator // gcd(L, v.denominator)
    net = {}
    for x, v in mi.items():
        net[x] = net.get(x, 0) + int(v * L)
    for y, v in mj.items():
        net[y] = net.get(y, 0) - int(v * L)
    P = sorted(x for x, m in net.items() if m > 0)
    N = sorted(y for y, m in net.items() if m < 0)
    if not P:
        return Fraction(0)
    M = np.empty((len(P), len(N)))
    for a_, x in enumerate(P):
        d = _bfs_all(g, x, 3)
        for b_, y in enumerate(N):
            M[a_, b_] = d.get(y, 3)
    return Fraction(B.ortools_mcf(np.array([net[x] for x in P], np.int64), np.array([-net[y] for y in N], np.int64),
                                  M), L)


def run_graph(spec, rows, a):
    import orcut
    from orcut import _core as C, _layered as LY, _reference as RF

    name, g, meta = R.load(spec, a.data_dir)
    ids = np.arange(g.num_edges, dtype=np.int64)
    rng = np.random.default_rng(0)
    frozen = orcut.exact_ids(g.indptr, g.indices, g.edges, ids, True, True, True, 8)
    ref = RF.exact_ids_reference(g.indptr, g.indices, g.edges, ids, True, True, True, 8)
    for measure in MEASURES:
        if rows.done(graph=name, measure=measure):
            continue
        spec_m = _spec(measure, g)
        t0 = time.perf_counter()
        out = orcut.curvature(g, spec_m, exact_fraction=True)
        t_solve = time.perf_counter() - t0
        val, num, den = out["curvature"], out["w1_num"], out["w1_den"]
        row = dict(graph=name, measure=measure, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"],
                   solve_s=t_solve, nan=int(np.isnan(val).sum()))
        # frozen 1.0.0 kernel and independent oracle (lazy / non-lazy / LLY)
        col = {"lazy": 0, "nonlazy": 1, "lly": 2}.get(measure)
        if col is not None:
            row["frozen_mismatch"] = int(np.sum(frozen[:, col] != val))
            row["oracle_mismatch"] = int(np.sum(ref[:, col] != val))
        if measure in ("lazy", "nonlazy"):
            kind = LY.K_LAZY if measure == "lazy" else LY.K_NONLAZY
            lv, _ = LY.layered_ids(g.indptr, g.indices, g.edges, ids, kind, 1, False, 8)
            row["layered_mismatch"] = int(np.sum(lv != val))
        # OR-Tools, exact fractions (all edges, or a sample for the walk and for large graphs)
        cap = a.ortools_cap if measure != "walk" else a.walk_cap
        sub = ids if len(ids) <= cap else np.sort(rng.choice(ids, cap, replace=False))
        lp = B.LocalProblems(g, _lp_kind(measure)) if measure != "walk" else None
        bad = 0
        potd = 0.0
        for e in sub:
            fr = ortools_fraction(g, int(e), measure, spec_m, lp)
            if fr != Fraction(int(num[e]), int(den[e])):
                bad += 1
        row["ortools_checked"] = len(sub)
        row["ortools_mismatch"] = bad
        # POT (floating point), on the same sample
        if measure != "walk":
            psub = sub[: a.pot_cap]
            vp, _, _, done = B.run_local(g, psub, "pot_simplex", kind=_lp_kind(measure), reps=1, max_pairs=5e7)
            if measure == "lly":
                dmax = np.maximum(g.deg[g.edges[psub, 0]], g.deg[g.edges[psub, 1]]).astype(float)
                vp = vp * (dmax + 1) / dmax
            potd = float(np.nanmax(np.abs(vp - val[psub]))) if done else 0.0
            row["pot_checked"] = int(done)
            row["pot_maxdiff"] = potd
        # certificates
        csub = sub[: a.cert_cap]
        t0 = time.perf_counter()
        nfail = 0
        nver = 0
        vfail = 0
        for k, e in enumerate(csub):
            i, j = map(int, g.edges[e])
            c = orcut.certify(g, (i, j), spec_m)
            if not c.verified or abs(float(c.curvature) - val[e]) > 1e-12:
                nfail += 1
            if k < a.verify_cap:
                nver += 1
                if not c.verify(g):
                    vfail += 1
        row.update(cert_checked=len(csub), cert_failed=nfail, cert_s_per_edge=(time.perf_counter() - t0) / max(1, len(csub)),
                   verify_checked=nver, verify_failed=vfail)
        rows.add(row)
        log(f"E1 {name:14s} {measure:9s} m={g.num_edges:>8d} ortools {bad}/{len(sub)} mismatches, "
            f"POT max|diff| {potd:.1e}, certificates {nfail}/{len(csub)} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", nargs="+", default=R.SMALL_REAL + R.SMALL_SYNTH + R.CANON)
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ortools-cap", type=int, default=20000, help="edges checked with OR-Tools per graph and measure")
    ap.add_argument("--walk-cap", type=int, default=500)
    ap.add_argument("--pot-cap", type=int, default=5000)
    ap.add_argument("--cert-cap", type=int, default=1000)
    ap.add_argument("--verify-cap", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.graphs = ["bundled:karate", "bundled:yeast", "synth:BA:300:6:0", "canon:Q:4"]
        a.ortools_cap, a.walk_cap, a.pot_cap, a.cert_cap, a.verify_cap = 300, 30, 200, 60, 10
    common.warm_jit()
    rows = Rows(f"{out_dir(a.out, 'e1')}/exactness.csv", ["graph", "measure"])
    for spec in a.graphs:
        run_graph(spec, rows, a)


if __name__ == "__main__":
    main()
