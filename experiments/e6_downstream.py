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
Both are saved as they go: BORF after every iteration, and a GRC run that did not finish in one call
before, or on a graph above ``--grc-chunk-above`` edges, in chunks of edges of about ``--grc-chunk``
seconds (its time is then the sum over the chunks). The GRC reference gets ``--grc-budget`` seconds
(24 h) per graph: if it runs out, the row is ``partial``, with the edges done, the time spent (a lower
bound on GRC's time per iteration, ``curv_lower_s``) and the extrapolation to all edges
(``curv_est_s``); a run with a larger budget continues from the saved chunks.

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
import json
import os
import shutil
import sys
import time
from fractions import Fraction

import numpy as np

import common
from common import Rows, log, out_dir
import baselines as B
import datasets as D
import dynamic as DY
import registry as R
import xio

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


def _borf_save(ckpt, meta, g, kap, recs):
    if ckpt:
        xio.atomic_npz(ckpt, edges=g.edges, kap=kap, meta=np.array(json.dumps(dict(meta, recs=recs))))


def _borf_load(ckpt, meta):
    """(graph, curvature, records) saved after the last iteration done, or None."""
    import orcut

    if not (ckpt and os.path.exists(ckpt)):
        return None
    try:
        z = np.load(ckpt)
        saved = json.loads(str(z["meta"]))
    except (OSError, ValueError, KeyError):
        return None
    recs = saved.pop("recs")
    if saved != meta:
        return None
    return orcut.Graph.from_edges(z["edges"], meta["n"]), z["kap"], recs


def borf(g0, loops, n_add, n_remove, measure="nonlazy", incremental=True, check_every=0, threads=None, ckpt=None):
    """Run BORF; returns (final graph, per-iteration records). With ``ckpt`` (an .npz path), the graph
    and curvature are saved after every iteration, and a run stopped in the middle resumes after the
    last iteration done; ``time_iter_s`` records each iteration's time (checks excluded)."""
    import orcut

    meta = dict(n=int(g0.num_nodes), m=int(g0.num_edges), loops=int(loops), add=int(n_add), remove=int(n_remove),
                measure=str(measure), incremental=bool(incremental))
    got = _borf_load(ckpt, meta)
    if got is not None:
        g, kap, recs = got
    else:
        g = g0
        t0 = time.perf_counter()
        kap = orcut.curvature(g, measure, n_jobs=threads)
        recs = [dict(iteration=-1, m=g.num_edges, time_curv_s=time.perf_counter() - t0, affected=g.num_edges)]
        _borf_save(ckpt, meta, g, kap, recs)
    for it in range(len(recs) - 1, loops):
        t_it = time.perf_counter()
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
        rec["time_iter_s"] = time.perf_counter() - t_it - rec.get("time_check_s", 0.0)
        recs.append(rec)
        g, kap = g_new, new_kap
        _borf_save(ckpt, meta, g, kap, recs)
    return g, recs


def part_borf(a):
    rows = Rows(f"{out_dir(a.out, 'e6')}/borf.csv", ["graph", "backend"])
    it_rows = Rows(f"{out_dir(a.out, 'e6')}/borf_iterations.csv", ["graph", "backend", "iteration"])
    graphs = a.graphs or (R.GNN + ["real:Squirrel", "real:Crocodile", "real:Amazon-Photo", "real:Amazon-Computers",
                                   "real:Coauthor-Physics", "real:GitHub"])
    if a.quick:
        graphs = ["bundled:jazz", "bundled:yeast", "synth:CL2.1:3000:8:0"]
    import partial as PT

    grc_ok = B.grc_python(a.grc_python) is not None
    grc_runs = PT.Parts(os.path.join(out_dir(a.out, "e6", "partial"), "grc.json"))  # the GRC runs under way
    kept = 0
    log(f"E6 GraphRicciCurvature: {B.grc_python(a.grc_python)}" if grc_ok else
        "E6 GraphRicciCurvature unavailable: its rows are recorded as unavailable (see main.py's log)")
    for spec in graphs:
        name = R.display(spec)
        r = rows.get(graph=name, backend="grc")
        # without GRC, a row that already says "unavailable" stays as it is
        grc_left = not rows.done(graph=name, backend="grc") and (grc_ok or r is None or common.row_failed(r))
        # a timeout of the old 2-hour limit runs now (there is no limit any more); so does a run given up
        # by a job whose time or memory this one exceeds
        grc_left = grc_left or (r is not None and grc_ok and (str(r.get("status") or "") == "timeout" or PT.retry_now(r)))
        # a partial reference (its budget ran out) continues from its chunks when the budget is raised
        grc_left = grc_left or (r is not None and grc_ok and str(r.get("status") or "") == "partial" and
                                (a.grc_budget is None or a.grc_budget > 1.01 * common.num(r, "grc_budget_s")))
        if grc_left and r is not None:
            log(f"E6 {name:16s} grc computed again (was: {str(r.get('status'))[:60]})")
        kept += int(not grc_left and r is not None and not rows.done(graph=name, backend="grc"))
        if not grc_left and all(rows.done(graph=name, backend=b) for b in ("orcut-full", "orcut-incremental")):
            continue
        name, g, meta = R.load(spec, a.data_dir)
        base = dict(graph=name, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"], loops=a.loops, add=a.add,
                    remove=a.remove, threads=a.threads, cite=meta.get("cite", ""))
        finals = {}
        for backend, inc in (("orcut-full", False), ("orcut-incremental", True)):
            if rows.done(graph=name, backend=backend):
                continue
            ckpt = os.path.join(out_dir(a.out, "e6", "partial"), f"borf.{R.slug(name)}.{backend}.npz")
            gf, recs = borf(g, a.loops, a.add, a.remove, incremental=inc, check_every=a.check_every if inc else 0,
                            threads=a.threads, ckpt=ckpt)
            total = recs[0]["time_curv_s"] + sum(r["time_iter_s"] for r in recs[1:])  # also across a restart
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
            try:
                os.remove(ckpt)
            except OSError:
                pass
            log(f"E6B {name:16s} {backend:18s} total {total:8.2f}s curvature/iter {row['curv_per_iter_s']:.3f}s "
                f"plans/iter {row['plan_per_iter_s']:.3f}s affected {row['affected_share']:.3f}")
        if len(finals) == 2:
            same = np.array_equal(finals["orcut-full"].edges, finals["orcut-incremental"].edges)
            rows.add(dict(rows.rows[-1], same_final_graph=bool(same)))
        if grc_left:
            extra = {}
            state_dir = os.path.join(out_dir(a.out, "e6", "partial"), f"grc.{R.slug(name)}")
            if grc_ok:
                st, gave = grc_runs.resume((name,), "grc")
                prev = str(r.get("status") or "") if r is not None else ""
                # in saved chunks: a run that did not finish in one call before (a timeout, killed by its
                # job's time limit), one under way in chunks, or a large graph
                chunked = os.path.isdir(state_dir) or g.num_edges > a.grc_chunk_above or prev in ("timeout", "partial") \
                    or prev.startswith("did not finish") or (st == "gave_up" and gave["status"].startswith("did not finish"))
                if chunked:
                    val, t, status, nch, ndone = B.grc_chunked(g, "OTDSinkhornMix", 0.0, 3000, a.threads, a.grc_python,
                                                               state_dir, chunk_s=a.grc_chunk, budget_s=a.grc_budget,
                                                               log=lambda msg: log(f"E6B {name:16s} {msg}"))
                    extra = dict(chunks=nch, grc_budget_s=a.grc_budget)
                    if status == "partial":  # the budget ran out: GRC's time so far is a lower bound
                        extra.update(grc_edges_done=ndone, curv_lower_s=t, curv_est_s=t * g.num_edges / max(1, ndone))
                        val, t = None, float("nan")
                elif st == "gave_up":  # killed by its job's memory limit: only a bigger job runs it again
                    val, t, status = None, float("nan"), gave["status"]
                    extra = {k: v for k, v in gave.items() if k != "status"}
                else:
                    grc_runs.begin((name,), "grc")
                    val, t, status = B.grc(g, "OTDSinkhornMix", 0.0, 3000, a.threads, a.grc_python)
                for k, v in B.grc_row_fields(status).items():
                    extra.setdefault(k, v)  # killed at the memory limit: what a job with more memory compares with
            else:
                val, t, status = None, float("nan"), "unavailable"
            row = dict(base, backend="grc", status=status, curv_per_iter_s=t, **extra)
            if val is not None:
                import orcut

                row["max_diff_vs_exact"] = float(np.max(np.abs(val - orcut.curvature(g, "nonlazy", n_jobs=a.threads))))
            rows.add(row)
            grc_runs.drop((name,))
            if status == "ok":
                shutil.rmtree(state_dir, ignore_errors=True)
            if status == "partial":
                log(f"E6B {name:16s} grc                partial  {row['grc_edges_done']:,} of {g.num_edges:,} edges in "
                    f"{row['curv_lower_s'] / 3600:.1f} h (the budget): more than that per iteration, about "
                    f"{row['curv_est_s'] / 3600:.0f} h extrapolated")
            else:
                log(f"E6B {name:16s} grc                {status:8s} curvature of all edges {t:9.2f}s")
    if kept:
        log(f"E6 {kept} GRC rows left as unavailable: GraphRicciCurvature is not available in this run")


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
    methods = ["exact", "bf", "af3", "jl"]
    for spec in specs:
        if all(rows.done(graph=D.labels_name(spec), method=m) for m in methods + ["louvain"]):
            continue  # not even generated (LFR graphs take a while)
        name, n, e, lab = D.with_labels(spec, a.data_dir)
        g = D.to_graph(n, e)
        base = dict(graph=name, spec=spec, n=g.num_nodes, m=g.num_edges, deg_max=int(g.deg.max()),
                    communities_true=int(len(np.unique(lab))))
        t_full = None  # one full computation, for the estimate of the exact method's row
        for method in methods:
            if rows.done(graph=name, method=method):
                continue
            if method == "exact":
                t0 = time.perf_counter()
                orcut.curvature(g, HALF, n_jobs=1)
                t_full = time.perf_counter() - t0
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
    ap.add_argument("--grc-chunk", type=float, default=3600.0, help="part borf: seconds per saved chunk of a long GRC run")
    ap.add_argument("--grc-budget", type=float, default=86400.0,
                    help="part borf: seconds of GRC time per graph for the reference; when it runs out, the row is "
                         "'partial' with the time spent (a lower bound) and an extrapolation (0: no budget)")
    ap.add_argument("--grc-chunk-above", type=int, default=10**6,
                    help="part borf: GRC runs in saved chunks on graphs with more edges (and wherever one call did "
                         "not finish before)")
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
    a.grc_budget = a.grc_budget or None  # 0: no budget
    if a.quick:
        a.loops, a.check_every = 4, (1 if a.part == "borf" else 100)
    common.warm_jit()
    (part_borf if a.part == "borf" else part_community)(a)


if __name__ == "__main__":
    main()
    sys.exit(common.finish())
