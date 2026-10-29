"""E4. Ablation: where the time goes.

On a stratified sample of edges (bins of d_i + d_j, up to ``--per-bin`` edges per bin) every
configuration is timed single-threaded on the same edges, and its time on the whole graph is
estimated from the bin counts. Configurations (``flags`` of :func:`orcut._core.core_ids`):

  cut        distance-free bitsets and the plain two-layer cut (no relays, twins, fast path, warm start)
  +relays    + implications of whole blocks through hub vertices (relays)
  +twins     + merging of rows with identical patterns
  +fast      + one bipartite flow when no pair is at distance 3
  full       + warm start of the two-layer cut (what ``orcut.curvature`` runs)
  -relays, -twins, -fast   full without one step ("-warm" is "+fast")
  oracle     explicit local distance matrix and three-level closure (``orcut._reference``,
             the previous implementation)
  pot        explicit local distance matrix (numba) and POT's network simplex (build included)

All orcut configurations return identical values (checked: ``value_mismatch``); for each, the mean
number of nodes and arcs of the solved flow networks is reported.

Usage: python e4_ablation.py [--graphs SPEC ...] [--per-bin 100] [--out results] [--quick]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common
from common import Rows, estimate_total, log, out_dir, stratified_bins, time_per_edge
import baselines as B
import registry as R

CONFIGS = {"cut": 0, "+relays": 2, "+twins": 3, "+fast": 7, "full": 15, "-relays": 13, "-twins": 14, "-fast": 11}
# flags: F_TWINS = 1, F_RELAYS = 2, F_FAST = 4, F_WARM = 8 (orcut._core)
ORDER = ["full", "cut", "+relays", "+twins", "+fast", "-relays", "-twins", "-fast", "oracle", "pot"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", nargs="+", default=["synth:ER:20000:20:0", "synth:BA:20000:10:0", "synth:RGG:20000:20:0",
                                                    "synth:CL2.1:50000:10:0", "bundled:arxiv", "real:GitHub",
                                                    "real:Twitch-DE", "real:Squirrel", "real:Epinions", "real:DBLP"])
    ap.add_argument("--per-bin", type=int, default=100)
    ap.add_argument("--budget", type=float, default=900.0, help="seconds per configuration and graph")
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.graphs, a.per_bin, a.budget = ["bundled:jazz", "synth:CL2.1:5000:8:0"], 20, 60.0
    import numba

    numba.set_num_threads(1)
    common.warm_jit()
    from orcut import _core as C, _reference as RF

    rows = Rows(f"{out_dir(a.out, 'e4')}/ablation.csv", ["graph", "config"])
    for spec in a.graphs:
        name, g, meta = R.load(spec, a.data_dir)
        strata = stratified_bins(g, a.per_bin)
        ids = np.concatenate([s for _, s, _ in strata])
        base = None
        for cfg in ORDER:
            if rows.done(graph=name, config=cfg):
                continue
            row = dict(graph=name, config=cfg, m=g.num_edges, deg_max=meta["deg_max"], sample=len(ids))
            deadline = time.perf_counter() + a.budget
            if cfg in CONFIGS:
                fl = CONFIGS[cfg]
                f = lambda sub: C.core_ids(g.indptr, g.indices, g.edges, sub, C.M_LAZY, flags=fl, nchunks=1)
                times = [time_per_edge(f, s, reps=3, deadline=deadline) for _, s, _ in strata]
                v, num, d = f(ids)
                row.update(mean_nodes=float(np.mean(d[:, C.D_NODES])), mean_arcs=float(np.mean(d[:, C.D_ARCS])),
                           max_arcs=int(np.max(d[:, C.D_ARCS])), mean_groups=float(np.mean(d[:, C.D_GROUPS])))
                if base is None:
                    base = v
                row["value_mismatch"] = int(np.sum(v != base))
            elif cfg == "oracle":
                f = lambda sub: RF.exact_ids_reference(g.indptr, g.indices, g.edges, sub, True, False, False, 1)
                times = [time_per_edge(f, s, reps=1, deadline=deadline) for _, s, _ in strata]
            else:  # pot: build + solve per edge
                times = []
                for _, s, _ in strata:
                    ts, tb = np.empty(len(s)), np.empty(len(s))
                    left = deadline - time.perf_counter()
                    if left <= 0:
                        times.append(np.full(len(s), np.nan))
                        continue
                    B.run_local(g, s, "pot_simplex", reps=1, budget_s=left, times=ts, btimes=tb)
                    times.append(ts + tb)
            timed = sum(int(np.isfinite(t).sum()) for t in times)
            row.update(timed=timed, est_total_s=estimate_total(strata, times),
                       mean_sample_edge_s=float(np.nanmean(np.concatenate(times))))
            rows.add(row)
            log(f"E4 {name:16s} {cfg:8s} est. {row['est_total_s']:10.2f}s single-threaded (timed {timed}/{len(ids)})"
                + (f" arcs {row['mean_arcs']:9.1f}" if cfg in CONFIGS else ""))


if __name__ == "__main__":
    main()
