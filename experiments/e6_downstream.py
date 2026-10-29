"""E6. Exact curvature inside iterative graph algorithms.

Both algorithms recompute curvature after every change of the graph. By the locality lemma
(:mod:`dynamic`), after changing a set of vertex pairs only the edges with an endpoint in the union
of the closed neighbourhoods N[a] u N[b] of the changed pairs can change curvature; we recompute
exactly those (``incremental``) and check, at regular intervals, that the result equals a full
recomputation (``--check-every``).

Part ``borf``: batch Ollivier--Ricci flow rewiring (BORF; Nguyen et al., 2023) on the graphs of
the rewiring literature. At each of ``--loops`` iterations, the non-lazy curvature (alpha = 0, as
in BORF) of all edges is needed; for each of the ``--add`` most negatively curved edges an optimal
transport plan is computed (``orcut.certify``) and an edge is added between the two distinct
vertices exchanging the largest mass in it (nothing if that mass stays in place or the edge
exists), and the ``--remove`` most positively curved edges are deleted. We time the curvature
step (full and incremental) and the plans per iteration; for reference, GraphRicciCurvature's time
for the curvature of all edges of the input graph (alpha = 0, its default method), a lower bound
on the time of one iteration of the original implementation, which also extracts transport plans.

Part ``community``: community detection by sequential deletion of the most negatively curved edge
(Sia, Jonckheere and Bogdan, 2019): delete the edge of smallest curvature, update the curvature of
the affected edges, repeat until no edge is left (or ``--max-frac`` of the edges); the reported
partition is the component partition of largest modularity along the way (Girvan--Newman
selection). Curvature: exact at idleness 1/2 (GraphRicciCurvature's default; the same order as
Lin--Lu--Yau), or one of the proxies bf, af3, jl, updated with the same locality rule. Graphs: LFR
benchmarks (``--mus``, ``--sizes``, ``--seeds``), planted partitions, email-Eu-core (42
departments), Karate. Scores: NMI and ARI against the ground truth, modularity; Louvain (networkx)
for reference.

Usage: python e6_downstream.py --part borf|community [--graphs ...] [--threads 32] [--quick]
"""
from __future__ import annotations

import argparse
import os
import time
from fractions import Fraction

import numpy as np

import common
from common import Rows, log, out_dir
import baselines as B
import datasets as D
import dynamic as DY
import registry as R

HALF = Fraction(1, 2)


# ------------------------------------------------------------------ BORF

def _keys(g):
    return g.edges[:, 0].astype(np.int64) * g.num_nodes + g.edges[:, 1]


def plan_argmax(cert):
    """The pair (p, q) carrying the largest mass in the optimal plan of a certificate, with the mass
    kept in place (min(m_i(x), m_j(x)) at every x) included; ties broken by (p, q). None when the
    largest mass stays in place."""
    best = None
    for x in set(cert.m_i) & set(cert.m_j):
        cand = (-min(cert.m_i[x], cert.m_j[x]), x, x)
        best = cand if best is None or cand < best else best
    for x, y, f in cert.plan:
        cand = (-f, min(x, y), max(x, y))
        best = cand if best is None or cand < best else best
    if best is None or best[1] == best[2]:
        return None
    return best[1], best[2]


def borf(g0, loops, n_add, n_remove, measure="nonlazy", incremental=True, check_every=0, threads=None):
    """Run BORF; returns (final graph, per-iteration records)."""
    import orcut

    g = g0
    t0 = time.perf_counter()
    kap = orcut.curvature(g, measure, n_jobs=threads)
    recs = [dict(iteration=-1, m=g.num_edges, time_curv_s=time.perf_counter() - t0, affected=g.num_edges)]
    for it in range(loops):
        order = np.argsort(kap, kind="stable")
        neg, pos = order[:n_add], order[::-1][:n_remove]
        t0 = time.perf_counter()
        add = set()
        for e in neg:
            u, v = map(int, g.edges[e])
            pq = plan_argmax(orcut.certify(g, (u, v), measure))
            if pq is not None and not g.has_edge(*pq):
                add.add(pq)
        t_plan = time.perf_counter() - t0
        keep = np.ones(g.num_edges, bool)
        keep[pos] = False
        removed = g.edges[pos]
        added = np.array(sorted(add), np.int64).reshape(-1, 2)
        g_new = orcut.Graph.from_edges(np.concatenate([g.edges[keep], added]), g.num_nodes)
        t0 = time.perf_counter()
        if incremental:
            aff = orcut.affected_edges(g_new, np.concatenate([removed, added]))
            new_kap = np.empty(g_new.num_edges)
            rest = np.setdiff1d(np.arange(g_new.num_edges), aff, assume_unique=True)
            old = np.searchsorted(_keys(g), _keys(g_new)[rest])  # unaffected edges are old edges
            new_kap[rest] = kap[old]
            if len(aff):
                new_kap[aff] = orcut.curvature(g_new, measure, edges=aff, n_jobs=threads)
        else:
            aff = np.arange(g_new.num_edges)
            new_kap = orcut.curvature(g_new, measure, n_jobs=threads)
        t_curv = time.perf_counter() - t0
        rec = dict(iteration=it, m=g_new.num_edges, added=len(added), removed=len(removed), affected=len(aff),
                   time_curv_s=t_curv, time_plan_s=t_plan)
        if check_every and (it + 1) % check_every == 0:
            tc = time.perf_counter()
            rec["check_mismatch"] = int(np.sum(new_kap != orcut.curvature(g_new, measure, n_jobs=threads)))
            rec["time_check_s"] = time.perf_counter() - tc
        recs.append(rec)
        g, kap = g_new, new_kap
    return g, recs


def part_borf(a):
    rows = Rows(f"{out_dir(a.out, 'e6')}/borf.csv", ["graph", "backend"])
    it_rows = Rows(f"{out_dir(a.out, 'e6')}/borf_iterations.csv", ["graph", "backend", "iteration"])
    graphs = a.graphs or (R.GNN + ["real:Squirrel", "real:Crocodile", "real:Amazon-Photo", "real:Amazon-Computers",
                                   "real:Coauthor-Physics", "real:GitHub"])
    if a.quick:
        graphs = ["bundled:jazz", "bundled:yeast", "synth:CL2.1:3000:8:0"]
    for spec in graphs:
        name, g, meta = R.load(spec, a.data_dir)
        base = dict(graph=name, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"], loops=a.loops, add=a.add,
                    remove=a.remove, threads=a.threads, cite=meta.get("cite", ""))
        finals = {}
        for backend, inc in (("orcut-full", False), ("orcut-incremental", True)):
            if rows.done(graph=name, backend=backend):
                continue
            t0 = time.perf_counter()
            gf, recs = borf(g, a.loops, a.add, a.remove, incremental=inc, check_every=a.check_every if inc else 0,
                            threads=a.threads)
            total = time.perf_counter() - t0 - sum(r.get("time_check_s", 0.0) for r in recs)
            finals[backend] = gf
            for r in recs:
                it_rows.add(dict(graph=name, backend=backend, **r))
            loop_recs = recs[1:]
            row = dict(base, backend=backend, total_s=total, initial_curv_s=recs[0]["time_curv_s"],
                       curv_per_iter_s=float(np.mean([r["time_curv_s"] for r in loop_recs])) if loop_recs else 0.0,
                       plan_per_iter_s=float(np.mean([r["time_plan_s"] for r in loop_recs])) if loop_recs else 0.0,
                       affected_share=float(np.mean([r["affected"] / r["m"] for r in loop_recs])) if loop_recs else 0.0,
                       added_total=int(sum(r["added"] for r in loop_recs)), m_final=gf.num_edges,
                       check_mismatch=int(sum(r.get("check_mismatch", 0) for r in loop_recs)))
            rows.add(row)
            log(f"E6B {name:16s} {backend:18s} total {total:8.2f}s curvature/iter {row['curv_per_iter_s']:.3f}s "
                f"plans/iter {row['plan_per_iter_s']:.3f}s affected {row['affected_share']:.3f}")
        if len(finals) == 2:
            same = np.array_equal(finals["orcut-full"].edges, finals["orcut-incremental"].edges)
            rows.add(dict(rows.rows[-1], same_final_graph=bool(same)))
        if not rows.done(graph=name, backend="grc"):
            val, t, status = B.grc(g, "OTDSinkhornMix", 0.0, 3000, a.threads, a.grc_python, a.grc_timeout)
            row = dict(base, backend="grc", status=status, curv_per_iter_s=t)
            if val is not None:
                import orcut

                row["max_diff_vs_exact"] = float(np.max(np.abs(val - orcut.curvature(g, "nonlazy", n_jobs=a.threads))))
            rows.add(row)
            log(f"E6B {name:16s} grc                {status:8s} curvature of all edges {t:9.2f}s")


# ------------------------------------------------------------- community

def sia(g, method="exact", measure=HALF, max_frac=1.0, check_every=0, threads=1, budget=None):
    """Sequential deletion of the most negatively curved edge with incremental updates.
    Returns (deletion order, number of curvature evaluations, seconds (checks excluded), number of
    mismatches between the incremental values and a full recomputation every ``check_every``
    deletions)."""
    import orcut
    from orcut import _core as C
    from orcut.measures import parse

    indptr, indices, eid = DY.csr_with_ids(g)
    edges = g.edges
    m = g.num_edges
    if method == "exact":
        mode, pn, pd, tabs = parse(measure, g)
        f = lambda ids: C.core_ids(indptr, indices, edges, ids, mode, pn, pd, C.F_ALL, 1, tabs)[0]
    else:
        key = {"bf": "bf", "af3": "af3", "jl": "jl_mean"}[method]
        f = lambda ids: B.proxies_ids(indptr, indices, edges, ids, key)
    t0 = time.perf_counter()
    t_check = 0.0
    with common.numba_threads(threads):
        kap = f(np.arange(m, dtype=np.int64)).astype(np.float64)
        stamp_v = np.full(g.num_nodes, -1, np.int64)
        stamp_e = np.full(m, -1, np.int64)
        order = np.empty(m, np.int64)
        evals = m
        mismatch = 0
        T = int(np.floor(max_frac * m))
        s = 0
        while s < T:
            e = int(np.argmin(kap))
            order[s] = e
            a_, b_ = int(edges[e, 0]), int(edges[e, 1])
            DY.delete_edge(indptr, indices, eid, a_, b_)
            kap[e] = np.inf
            aff = DY.affected_edges(indptr, indices, eid, np.array([a_, b_], np.int64), stamp_v, stamp_e, s)
            if len(aff):
                kap[aff] = f(aff)
            evals += len(aff)
            s += 1
            if check_every and method == "exact" and s % check_every == 0:
                tc = time.perf_counter()
                alive = np.flatnonzero(np.isfinite(kap))
                mismatch += int(np.sum(f(alive) != kap[alive]))
                t_check += time.perf_counter() - tc
            if budget is not None and time.perf_counter() - t0 - t_check > budget:
                break
    return order[:s], evals, time.perf_counter() - t0 - t_check, mismatch


def part_community(a):
    import networkx as nx
    import orcut

    rows = Rows(f"{out_dir(a.out, 'e6')}/community.csv", ["graph", "method"])
    specs = a.graphs or ([f"lfr:{n}:{mu}:{s}" for n in a.sizes for mu in a.mus for s in range(a.seeds)]
                         + ["email-Eu-core", "karate"])
    if a.quick:
        specs = ["karate", "lfr:500:0.1:0", "lfr:500:0.3:0", "pp:600:6:12:2:0"]
    for spec in specs:
        name, n, e, lab = D.with_labels(spec, a.data_dir)
        g = D.to_graph(n, e)
        base = dict(graph=name, spec=spec, n=g.num_nodes, m=g.num_edges, deg_max=int(g.deg.max()),
                    communities_true=int(len(np.unique(lab))))
        t0 = time.perf_counter()
        orcut.curvature(g, HALF, n_jobs=1)
        t_full = time.perf_counter() - t0
        for method in ["exact", "bf", "af3", "jl"]:
            if rows.done(graph=name, method=method):
                continue
            order, evals, secs, mism = sia(g, method, max_frac=a.max_frac, check_every=a.check_every, budget=a.budget)
            T = len(order)
            full_equiv = int((T + 1) * g.num_edges - T * (T + 1) // 2)  # all alive edges after every deletion
            Q = DY.modularity_along(g.num_nodes, g.edges, g.indptr, g.indices, order, T)
            s_best = int(np.argmax(Q))
            pred = DY.components_after(g.num_nodes, g.edges, order, s_best)
            row = dict(base, method=method, nmi=DY.nmi(lab, pred), ari=DY.ari(lab, pred), modularity=float(Q[s_best]),
                       communities=int(pred.max() + 1), deletions=len(order), best_at=s_best, time_s=secs,
                       evals=int(evals), evals_full_equiv=full_equiv,
                       est_full_recompute_s=t_full * full_equiv / g.num_edges if method == "exact" else float("nan"),
                       check_mismatch=mism)
            rows.add(row)
            log(f"E6C {name:28s} {method:6s} NMI {row['nmi']:.3f} ARI {row['ari']:.3f} Q {row['modularity']:.3f} "
                f"k={row['communities']:4d} {secs:8.2f}s evals {evals / max(1, g.num_edges):.1f} m")
        if not rows.done(graph=name, method="louvain"):
            G = nx.Graph()
            G.add_nodes_from(range(g.num_nodes))
            G.add_edges_from(map(tuple, g.edges.tolist()))
            t0 = time.perf_counter()
            parts = nx.community.louvain_communities(G, seed=0)
            secs = time.perf_counter() - t0
            pred = np.empty(g.num_nodes, np.int64)
            for c, p in enumerate(parts):
                pred[list(p)] = c
            rows.add(dict(base, method="louvain", nmi=DY.nmi(lab, pred), ari=DY.ari(lab, pred),
                          modularity=DY.modularity(g, pred), communities=len(parts), time_s=secs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["borf", "community"], required=True)
    ap.add_argument("--graphs", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--add", type=int, default=20)
    ap.add_argument("--remove", type=int, default=10)
    ap.add_argument("--check-every", type=int, default=None,
                    help="compare with a full recomputation every k iterations (default: borf 5, community 500)")
    ap.add_argument("--grc-python", default=None)
    ap.add_argument("--grc-timeout", type=float, default=2 * 3600.0, help="seconds per GRC run")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000])
    ap.add_argument("--mus", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-frac", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=None, help="seconds per community-detection run")
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.check_every is None:
        a.check_every = 5 if a.part == "borf" else 500
    if a.quick:
        a.loops, a.check_every, a.grc_timeout = 4, (1 if a.part == "borf" else 100), 120.0
    common.warm_jit()
    (part_borf if a.part == "borf" else part_community)(a)


if __name__ == "__main__":
    main()
