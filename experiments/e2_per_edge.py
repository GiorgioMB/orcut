"""E2. Per-edge cost of exact curvature, as a function of the neighbourhood size and structure.

Part A (controlled edges, ``--part A``): one edge (0, 1) with |U_0| = a, |U_1| = b, t common
neighbours, in five regimes of the distance structure of U_0 x U_1 (``datasets.edge_instance``):
``tree`` (every pair at distance 3), ``sparse``, ``mixed``, ``dense`` (no pair at distance 3), and
``hubleaf`` (U_1 = leaves of a hub). Two sweeps: symmetric (a = b = s, t = s / 8) and hub (a = 32,
b = s, t = 4). Methods, all single-threaded and timed on the same edge after a warm-up:

  ours        orcut.curvature, everything included (neighbourhoods, bitsets, cut);
  nowitness   the same without witness relays (flag F_WITNESS off);
  witness     the linear-size witness network alone (orcut._witness), everything included;
  graphflow   min-cost flow on the local graph (Beckmann's formulation; OR-Tools), everything included
              (it needs no distance matrix either);
  oracle      orcut._reference: full local distance matrix, three-level closure (the previous solver);
  pot         POT's C++ network simplex on the mass-cancelled problem, given the distance matrix;
  emd2        ot.emd2 (the public API that GraphRicciCurvature calls), given the distance matrix;
  ortools     OR-Tools integer min-cost flow (C++), given the distance matrix;
  highs       HiGHS LP (s <= 256);
  build       the time to build the local distance matrix and marginals (numba), which every
              general-purpose solver needs.

A method whose time exceeds ``--budget`` seconds at some size is not run at larger sizes.

Part B (real graphs, ``--part B``): edges binned by d_i + d_j (powers of two); up to
``--per-bin`` edges per bin; per-edge time of each method per bin, and the time per edge of the
whole graph estimated from the bin shares.

Part C (``--part C``): the instances of the query lower bounds of DasGupta, Grigorescu and Mukherjee
(2023) (``datasets.dasgupta_instance``), with n = |L| = |R| up to 2^20: the local input is O(n), so
the solver should take linear time, while a method that works on the distance matrix needs its n^2
entries. Methods: ours, witness, graphflow, and POT given the distance matrix (with its build time)
while n^2 <= --max-pairs.

Usage: python e2_per_edge.py --part A|B|C [--out results] [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import common
from common import Rows, log, out_dir
import baselines as B
import datasets as D
import registry as R


def _time_repeated(f, min_total=0.2, max_reps=2000, warm=True):
    """Median-free robust per-call time: run f repeatedly until min_total seconds, take the fastest
    of 3 such batches divided by the batch length."""
    if warm:
        f()
    best = float("inf")
    out = None
    for _ in range(3):
        k = 0
        t0 = time.perf_counter()
        while True:
            out = f()
            k += 1
            el = time.perf_counter() - t0
            if el >= min_total / 3 or k >= max_reps:
                break
        best = min(best, el / k)
    return best, out


def part_a(a):
    import orcut
    from orcut import _core as C, _reference as RF

    rows = Rows(f"{out_dir(a.out, 'e2')}/controlled.csv", ["sweep", "regime", "s", "seed"])
    sweeps = []
    sym = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] if not a.quick else [16, 64, 256]
    hub = [64, 256, 1024, 4096, 16384, 65536] if not a.quick else [64, 256]
    for s in sym:
        sweeps.append(("sym", s, s, max(1, s // 8)))
    for s in hub:
        sweeps.append(("hub", 32, s, 4))
    seeds = range(a.seeds)
    methods = ["ours", "nowitness", "witness", "graphflow", "oracle", "pot", "emd2", "ortools", "highs"]
    from orcut import _witness as WT
    dead = {}  # (sweep, regime, method) -> size at which it exceeded the budget
    for r in rows.rows:  # a resumed run keeps off the methods that exceeded it before the restart
        for meth in methods:
            if common.num(r, f"{meth}_s") > a.budget:
                key, s_r = (r.get("sweep"), r.get("regime"), meth), int(common.num(r, "s"))
                dead[key] = min(dead.get(key, s_r), s_r)
    for sweep, ua, ub, t in sweeps:
        s = ub if sweep == "hub" else ua
        for regime in D.REGIMES:
            if regime == "dense" and s > 4096:  # 20% of |U_i| x |U_j| as cross edges: too many to generate
                continue
            for seed in seeds:
                if rows.done(sweep=sweep, regime=regime, s=s, seed=seed):
                    continue
                n, e = D.edge_instance(ua, ub, t, regime, seed)
                g = D.to_graph(n, e)
                eid = np.array([g.edge_id(0, 1)], np.int64)
                row = dict(sweep=sweep, regime=regime, s=s, a=ua, b=ub, t=t, seed=seed, S=ua + ub + t + 2)
                out = orcut.curvature(g, "lazy", edges=eid, diagnostics=True, n_jobs=1)
                d = out["diagnostics"][0]
                ref_val = float(out["curvature"][0])
                row.update(P=int(d[C.D_ROWS]), N=int(d[C.D_COLS]), groups=int(d[C.D_GROUPS]),
                           nodes=int(d[C.D_NODES]), arcs=int(d[C.D_ARCS]), path=int(d[C.D_PATH]))
                wd = WT.witness_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, nchunks=1)[2][0]
                nd = C.core_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, flags=C.F_ALL & ~C.F_WITNESS, nchunks=1)[2][0]
                row.update(nowitness_nodes=int(nd[C.D_NODES]), nowitness_arcs=int(nd[C.D_ARCS]),
                           witness_nodes=int(wd[WT.WD_NODES]), witness_arcs=int(wd[WT.WD_ARCS]),
                           local_input=int(sum(int(g.deg[x]) for x in set(map(int, g.neighbors(0))) |
                                               set(map(int, g.neighbors(1))))))
                gp = B.GraphFlowProblems(g, "lazy")
                big = ua + ub + t + 2 > 20000  # the explicit distance matrix would need > 400 MB
                if not big:
                    lp = B.LocalProblems(g, "lazy")
                    tb, prob = _time_repeated(lambda: lp.build(int(eid[0]), True), min_total=0.1, max_reps=200)
                    pa, pb, M, L, aint, bint = prob
                    row["build_s"] = tb
                    row["pot_input_bytes"] = int(M.size * 8)
                row["dist_matrix_bytes"] = int((ua + ub + t + 2) ** 2)
                for meth in methods:
                    key = (sweep, regime, meth)
                    if key in dead and s > dead[key]:
                        row[f"{meth}_s"] = float("nan")
                        continue
                    if meth == "highs" and s > 256:
                        row[f"{meth}_s"] = float("nan")
                        continue
                    if big and meth not in ("ours", "nowitness", "witness", "graphflow"):
                        row[f"{meth}_s"] = float("nan")
                        continue
                    if meth == "ours":  # the kernel driver directly (no Python wrapper), like the C++ calls below
                        f = lambda: C.core_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, nchunks=1)[0][0]
                    elif meth == "nowitness":  # the default solver without witness relays (orcut <= 1.1 behaviour)
                        f = lambda: C.core_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, flags=C.F_ALL & ~C.F_WITNESS,
                                               nchunks=1)[0][0]
                    elif meth == "witness":
                        f = lambda: WT.witness_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, nchunks=1)[0][0]
                    elif meth == "graphflow":
                        f = lambda: gp.curvature(int(eid[0]))[0]
                    elif meth == "oracle":
                        f = lambda: RF.exact_ids_reference(g.indptr, g.indices, g.edges, eid, True, False, False, 1)[0, 0]
                    elif meth == "pot":
                        f = lambda: 1.0 - B.pot_emd_c(pa, pb, M)
                    elif meth == "emd2":
                        f = lambda: 1.0 - B.pot_emd2(pa, pb, M)
                    elif meth == "ortools":
                        f = lambda: 1.0 - B.ortools_mcf(aint, bint, M) / L
                    else:
                        f = lambda: 1.0 - B.highs_lp(pa, pb, M)
                    t0 = time.perf_counter()
                    v0 = f()  # single call first: decides whether the method is within the budget
                    once = time.perf_counter() - t0
                    if once > a.budget:
                        dead[key] = s
                        tm, val = once, v0
                    else:
                        tm, val = _time_repeated(f, min_total=min(0.5, 10 * once + 0.05), max_reps=1000, warm=False)
                    row[f"{meth}_s"] = tm
                    row[f"{meth}_err"] = abs(float(val) - ref_val)
                rows.add(row)
                log(f"E2A {sweep} {regime:7s} s={s:5d} seed={seed} |S|={row['S']:6d} path={row['path']} "
                    + " ".join(f"{m}={1e6 * row.get(m + '_s', float('nan')):.0f}us" for m in methods))


def part_b(a):
    import orcut

    rows = Rows(f"{out_dir(a.out, 'e2')}/real_bins.csv", ["graph", "bin"])
    graphs = a.graphs or (["bundled:yeast", "bundled:power_grid", "bundled:jazz", "bundled:arxiv"] + R.MEDIUM)
    if a.quick:
        graphs = ["bundled:yeast", "bundled:jazz"]
    rng = np.random.default_rng(0)
    for spec in graphs:
        name, g, meta = R.load(spec, a.data_dir)
        dsum = g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]
        bins = np.floor(np.log2(dsum)).astype(int)
        for bn in np.unique(bins):
            all_ids = np.flatnonzero(bins == bn)
            # drawn also for the bins already done, so that a resumed run times the same edges
            ids = np.sort(rng.choice(all_ids, min(len(all_ids), a.per_bin), replace=False)).astype(np.int64)
            if rows.done(graph=name, bin=int(bn)):
                continue
            row = dict(graph=name, bin=int(bn), lo=2 ** int(bn), hi=2 ** (int(bn) + 1), share=len(all_ids) / g.num_edges,
                       edges_in_bin=len(all_ids), timed=len(ids), m=g.num_edges, deg_max=meta["deg_max"])
            t_ours, v = common.best_of(lambda: orcut.curvature(g, "lazy", edges=ids, n_jobs=1), reps=3)
            row["ours_s"] = t_ours / len(ids)
            budget = a.budget * 4
            gv, gt, gdone = B.run_graph_flow(g, ids, "lazy", budget_s=budget)
            row["graphflow_s"] = gt
            row["graphflow_timed"] = gdone
            row["graphflow_maxdiff"] = float(np.nanmax(np.abs(gv - v))) if gdone else float("nan")
            for meth in ("pot_simplex", "pot_emd2", "ortools"):
                if meth == "ortools" and 2 ** int(bn) > a.ortools_max:
                    continue
                vals, ts, tb, done = B.run_local(g, ids, meth, reps=2, budget_s=budget, max_pairs=5e7)
                row[f"{meth}_s"] = ts
                row["build_s"] = tb
                row[f"{meth}_timed"] = done
                row[f"{meth}_maxdiff"] = float(np.nanmax(np.abs(vals - v))) if done else float("nan")
            rows.add(row)
            log(f"E2B {name:16s} bin [{row['lo']},{row['hi']}) share {row['share']:.3f} ours {1e6 * row['ours_s']:.1f}us "
                f"pot {1e6 * row.get('pot_simplex_s', float('nan')):.1f}us build {1e6 * row.get('build_s', float('nan')):.1f}us")


def part_c(a):
    from orcut import _core as C, _witness as WT

    rows = Rows(f"{out_dir(a.out, 'e2')}/dasgupta.csv", ["kind", "n", "seed"])
    sizes = [2 ** k for k in range(8, 21, 2)] if not a.quick else [2 ** 8, 2 ** 10, 2 ** 12]
    methods = ["ours", "witness", "graphflow", "pot"]
    dead = {}  # (kind, method) -> size at which it exceeded the budget (also before a restart)
    for r in rows.rows:
        for meth in methods:
            if common.num(r, f"{meth}_s") > a.budget:
                key, n_r = (r.get("kind"), meth), int(common.num(r, "n"))
                dead[key] = min(dead.get(key, n_r), n_r)
    for kind in D.DASGUPTA_KINDS:
        for n_ in sizes:
            for seed in range(1 if a.quick else 2):
                if rows.done(kind=kind, n=n_, seed=seed):
                    continue
                nv, e = D.dasgupta_instance(n_, kind, seed)
                g = D.to_graph(nv, e)
                eid = np.array([g.edge_id(0, 1)], np.int64)
                v, num, d = C.core_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, nchunks=1)
                ref = float(v[0])
                row = dict(kind=kind, n=n_, seed=seed, m=g.num_edges, path=int(d[0, C.D_PATH]), nodes=int(d[0, C.D_NODES]),
                           arcs=int(d[0, C.D_ARCS]), value=ref,
                           local_input=int(g.deg[g.neighbors(0)].sum() + g.deg[g.neighbors(1)].sum()))
                pot_ok = (n_ + 2) ** 2 <= a.max_pairs
                if pot_ok:
                    lp = B.LocalProblems(g, "lazy")
                    tb, prob = _time_repeated(lambda: lp.build(int(eid[0]), True), min_total=0.1, max_reps=50)
                    pa, pb, M, L, aint, bint = prob
                    row["build_s"] = tb
                gp = B.GraphFlowProblems(g, "lazy")
                for meth in methods:
                    key = (kind, meth)
                    if (key in dead and n_ > dead[key]) or (meth == "pot" and not pot_ok):
                        row[f"{meth}_s"] = float("nan")
                        continue
                    if meth == "ours":
                        f = lambda: C.core_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, nchunks=1)[0][0]
                    elif meth == "witness":
                        f = lambda: WT.witness_ids(g.indptr, g.indices, g.edges, eid, C.M_LAZY, nchunks=1)[0][0]
                    elif meth == "graphflow":
                        f = lambda: gp.curvature(int(eid[0]))[0]
                    else:
                        f = lambda: 1.0 - B.pot_emd_c(pa, pb, M)
                    t0 = time.perf_counter()
                    v0 = f()
                    once = time.perf_counter() - t0
                    if once > a.budget:
                        dead[key] = n_
                        tm, val = once, v0
                    else:
                        tm, val = _time_repeated(f, min_total=min(0.5, 10 * once + 0.05), max_reps=200, warm=False)
                    row[f"{meth}_s"] = tm
                    row[f"{meth}_err"] = abs(float(val) - ref)
                rows.add(row)
                log(f"E2C {kind:8s} n={n_:8d} m={g.num_edges:9d} path={row['path']} arcs={row['arcs']:8d} "
                    + " ".join(f"{m}={1e3 * row.get(m + '_s', float('nan')):.3f}ms" for m in methods))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["A", "B", "C"], required=True)
    ap.add_argument("--graphs", nargs="+", default=None)
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=30.0, help="seconds per solve above which a method stops growing")
    ap.add_argument("--per-bin", type=int, default=200)
    ap.add_argument("--ortools-max", type=int, default=4096, help="largest d_i + d_j bin for OR-Tools")
    ap.add_argument("--max-pairs", type=float, default=5e7, help="part C: largest distance matrix given to POT")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.budget, a.per_bin = 1, 2.0, 30
    import numba
    numba.set_num_threads(1)
    common.warm_jit()
    {"A": part_a, "B": part_b, "C": part_c}[a.part](a)


if __name__ == "__main__":
    main()
    sys.exit(common.finish())
