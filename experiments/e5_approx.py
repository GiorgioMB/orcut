"""E5. What approximate pipelines get wrong, and what they cost.

Part ``whole``: on every graph, the exact curvature at idleness 1/2 (GraphRicciCurvature's default
alpha = 0.5) and the non-lazy curvature (alpha = 0, as in BORF) are computed with orcut; against
them we score

  grc-mix       GraphRicciCurvature 0.6.1 with its defaults (OTDSinkhornMix, nbr_topk = 3000,
                ot.emd2 with numItermax = 100000), alpha = 0.5;
  grc-otd       method OTD (ot.emd2 everywhere), nbr_topk = 3000;
  grc-otd-full  method OTD with no neighbour truncation (only when some degree exceeds 3000);
  grc-sinkhorn  method Sinkhorn (regularization 0.1);
  grc-atd       method ATD (average transport distance, an upper bound of W1);
  bf, af3, jl   one-pass proxies (Balanced Forman, triangle-augmented Forman, mean of the
                Jost--Liu bounds), scored by rank (Spearman, sign, recall of the 1% most negative /
                positive edges) against the non-lazy and the Lin--Lu--Yau curvature, and jl also by
                its error against the non-lazy curvature, which it estimates. On every edge we
                check that the Jost--Liu bounds bracket the non-lazy curvature and that BF is below
                the Lin--Lu--Yau curvature (Topping et al., 2022).

Errors are reported on all edges and on hub edges (an endpoint of degree >= ``--hub``); every
method gets the same number of threads (``--threads``) and is timed end to end (GRC includes its
shortest paths and its process pool). GRC runs in its own environment (``--grc-python`` or
$ORCUT_GRC_PYTHON); a run longer than ``--grc-timeout`` is reported as a timeout.

Part ``sinkhorn``: time-accuracy trade-off on a stratified sample of edges (bins of d_i + d_j), on
the local problems at idleness 1/2: entropic OT at regularization ``--regs`` against the exact value,
per-edge solve time (given the distance matrix) against the per-edge time of orcut and of POT's
network simplex.

Usage: python e5_approx.py --part whole|sinkhorn [--graphs SPEC ...] [--threads 32] [--grc-python PY]
"""
from __future__ import annotations

import argparse
import os
import time
from fractions import Fraction

import numpy as np

import common
from common import Rows, agreement, log, out_dir
import baselines as B
import registry as R

GRAPHS = ["bundled:yeast", "bundled:arxiv", "real:Deezer", "real:Twitch-DE", "real:Facebook", "real:Squirrel",
          "real:Crocodile", "real:Amazon-Computers", "real:Enron", "real:AstroPh", "real:GitHub", "real:Epinions",
          "synth:ER:20000:20:0", "synth:CL2.1:30000:12:0"]
GRC_METHODS = [("grc-mix", "OTDSinkhornMix", 3000), ("grc-otd", "OTD", 3000), ("grc-otd-full", "OTD", 10**9),
               ("grc-sinkhorn", "Sinkhorn", 3000), ("grc-atd", "ATD", 3000)]
HALF = Fraction(1, 2)


def part_whole(a):
    import orcut

    rows = Rows(f"{out_dir(a.out, 'e5')}/whole.csv", ["graph", "method"])
    cache = out_dir(a.out, "e5", "values")
    for spec in a.graphs:
        name, g, meta = R.load(spec, a.data_dir)
        hub = np.maximum(g.deg[g.edges[:, 0]], g.deg[g.edges[:, 1]]) >= a.hub
        base = dict(graph=name, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"], threads=a.threads)
        orcut.curvature(g, HALF, n_jobs=a.threads)  # warm
        t0 = time.perf_counter()
        k_half = orcut.curvature(g, HALF, n_jobs=a.threads)
        t_half = time.perf_counter() - t0
        t0 = time.perf_counter()
        k_zero = orcut.curvature(g, "nonlazy", n_jobs=a.threads)
        t_zero = time.perf_counter() - t0
        if not rows.done(graph=name, method="orcut"):
            rows.add(dict(base, method="orcut", target="idle 1/2", time_s=t_half, time_nonlazy_s=t_zero,
                          hub_edges=int(hub.sum()), share_negative=float(np.mean(k_half < 0))))
        np.savez_compressed(os.path.join(cache, f"{R.slug(name)}.npz"), half=k_half, zero=k_zero, edges=g.edges)
        if not rows.done(graph=name, method="jl"):
            B.proxies(g)  # compile
            t0 = time.perf_counter()
            px = B.proxies(g)
            t_px = time.perf_counter() - t0
            # the Jost--Liu bounds bracket the non-lazy curvature, and BF bounds the Lin--Lu--Yau
            # curvature from below (= 2 x the curvature at idleness 1/2); checked on every edge
            k_lly = orcut.curvature(g, "lly", n_jobs=a.threads)
            jl_ok = bool(np.all(px["jl_lower"] <= k_zero + 1e-12) and np.all(k_zero <= px["jl_upper"] + 1e-12))
            bf_ok = bool(np.all(px["bf"] <= k_lly + 1e-12))
            for meth, key in (("bf", "bf"), ("af3", "af3"), ("jl", "jl_mean")):
                for target, ref in (("nonlazy", k_zero), ("lly", k_lly)):
                    row = dict(base, method=meth if target == "nonlazy" else f"{meth}@lly", target=target, time_s=t_px,
                               jl_bounds_hold=jl_ok, bf_bound_holds=bf_ok)
                    row.update(agreement(ref, px[key], hub, errors=(meth == "jl" and target == "nonlazy")))
                    rows.add(row)
        mix_timed_out = any(r.get("graph") == name and r.get("method") == "grc-mix" and r.get("status") == "timeout"
                            for r in rows.rows)
        for meth, grc_method, topk in GRC_METHODS:
            if rows.done(graph=name, method=meth):
                continue
            if meth == "grc-otd-full" and meta["deg_max"] <= 3000:
                continue
            if mix_timed_out:  # the default (fastest exact-ish) method did not finish: skip the others
                rows.add(dict(base, method=meth, target="idle 1/2", status="skipped", time_s=float("nan")))
                continue
            val, t, status = B.grc(g, grc_method, 0.5, topk, a.threads, a.grc_python, a.grc_timeout)
            if meth == "grc-mix" and status == "timeout":
                mix_timed_out = True
            row = dict(base, method=meth, target="idle 1/2", status=status, time_s=t)
            if val is not None:
                row.update(agreement(k_half, val, hub))
                np.save(os.path.join(cache, f"{R.slug(name)}.{meth}.npy"), val)
            rows.add(row)
            log(f"E5 {name:16s} {meth:12s} {status:8s} {t:9.1f}s "
                + (f"MAE {row.get('mae', float('nan')):.2e} max {row.get('max_err', float('nan')):.3f} "
                   f"recall- {row.get('recall_neg', float('nan')):.3f}" if val is not None else ""))
        log(f"E5 {name:16s} orcut idle 1/2 {t_half:.2f}s, non-lazy {t_zero:.2f}s ({a.threads} threads)")


def part_sinkhorn(a):
    """Per stratum and method: per-edge time (solve given the distance matrix; for orcut everything)
    and error against the exact value; the graph-level time and mean error are weighted by the bin
    counts (what the method would cost, and how wrong it would be, on average over all edges)."""
    import numba
    import orcut

    numba.set_num_threads(1)
    rows = Rows(f"{out_dir(a.out, 'e5')}/sinkhorn.csv", ["graph", "method", "reg"])
    for spec in a.graphs:
        name, g, meta = R.load(spec, a.data_dir)
        strata = common.stratified_bins(g, a.per_bin, seed=1)
        exact = [orcut.curvature(g, HALF, edges=s, n_jobs=1) for _, s, _ in strata]
        cnt = np.array([c for _, _, c in strata], float)
        hubs = [np.maximum(g.deg[g.edges[s, 0]], g.deg[g.edges[s, 1]]) >= a.hub for _, s, _ in strata]
        methods = [("orcut", 0.0), ("pot", 0.0)] + [("sinkhorn", r) for r in a.regs]
        for meth, reg in methods:
            if rows.done(graph=name, method=meth, reg=reg):
                continue
            deadline = time.perf_counter() + a.budget
            times, errs = [], []
            for (bn, s, c), ex in zip(strata, exact):
                if meth == "orcut":
                    t = common.time_per_edge(lambda sub: orcut.curvature(g, HALF, edges=sub, n_jobs=1), s, reps=3,
                                             deadline=deadline)
                    errs.append(np.where(np.isfinite(t), 0.0, np.nan))
                else:
                    t = np.empty(len(s))
                    left = deadline - time.perf_counter()
                    if left <= 0:
                        times.append(np.full(len(s), np.nan))
                        errs.append(np.full(len(s), np.nan))
                        continue
                    v, _, _, done = B.run_local(g, s, "pot_simplex" if meth == "pot" else "sinkhorn", kind="idle:1/2",
                                                reps=1, reg=reg or None, budget_s=left, times=t)
                    errs.append(np.abs(v - ex))
                times.append(t)
            ok_bins = np.array([np.isfinite(e).any() for e in errs])
            mean_err_bin = np.array([np.nanmean(e) if np.isfinite(e).any() else np.nan for e in errs])
            hub_err = np.concatenate([e[h] for e, h in zip(errs, hubs)])
            all_err = np.concatenate(errs)
            row = dict(graph=name, method=meth, reg=reg, m=g.num_edges, sample=int(sum(len(s) for _, s, _ in strata)),
                       timed=int(np.isfinite(all_err).sum()), bins_complete=int(ok_bins.sum()), bins=len(strata),
                       est_total_1thread_s=common.estimate_total(strata, times),
                       mae_weighted=float(np.sum(cnt[ok_bins] * mean_err_bin[ok_bins]) / np.sum(cnt[ok_bins])),
                       max_err=float(np.nanmax(all_err)) if np.isfinite(all_err).any() else float("nan"),
                       hub_mae=float(np.nanmean(hub_err)) if np.isfinite(hub_err).any() else float("nan"))
            row["time_edge_weighted_s"] = row["est_total_1thread_s"] / g.num_edges
            rows.add(row)
            log(f"E5S {name:16s} {meth:8s} reg={reg:<6g} {1e6 * row['time_edge_weighted_s']:10.1f}us/edge "
                f"MAE {row['mae_weighted']:.2e} max {row['max_err']:.3f} ({row['timed']}/{row['sample']} timed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["whole", "sinkhorn"], required=True)
    ap.add_argument("--graphs", nargs="+", default=GRAPHS)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--hub", type=int, default=1000, help="hub edges: an endpoint of at least this degree")
    ap.add_argument("--grc-python", default=None, help="python of the GRC environment (default $ORCUT_GRC_PYTHON)")
    ap.add_argument("--grc-timeout", type=float, default=2 * 3600.0, help="seconds per GRC run")
    ap.add_argument("--regs", nargs="+", type=float, default=[1.0, 0.3, 0.1, 0.03, 0.01])
    ap.add_argument("--per-bin", type=int, default=50)
    ap.add_argument("--budget", type=float, default=900.0, help="seconds per method and graph (part sinkhorn)")
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.graphs = ["bundled:jazz", "synth:CL2.1:5000:8:0"]
        a.hub, a.grc_timeout, a.per_bin, a.budget, a.regs = 200, 120.0, 10, 30.0, [1.0, 0.1, 0.01]
    common.warm_jit()
    (part_whole if a.part == "whole" else part_sinkhorn)(a)


if __name__ == "__main__":
    main()
