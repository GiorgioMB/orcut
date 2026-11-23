"""Run GraphRicciCurvature (GRC) 0.6.1 on one graph, as a user would; called by E5 in GRC's environment.

Input: an ``.npz`` with ``n`` and ``edges``; output: an ``.npz`` with ``curvature`` (aligned with the
edges, or with ``--ids``) and ``time_s``, the wall time of ``compute_ricci_curvature_edges``
(including GRC's shortest paths and its process pool). GRC's defaults: alpha = 0.5,
``OTDSinkhornMix`` (``ot.emd2``, or Sinkhorn when both endpoints have more than 2000 neighbours),
``nbr_topk = 3000``; all-pairs shortest paths up to ``--apsp-max-nodes`` vertices, pairwise beyond.

Usage: python grc_runner.py IN.npz OUT.npz [--alpha 0.5] [--method OTDSinkhornMix] [--nbr-topk 3000]
                           [--proc 32] [--shortest-path auto|all_pairs|pairwise] [--ids IDS.npy]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--method", default="OTDSinkhornMix", choices=["OTD", "ATD", "Sinkhorn", "OTDSinkhornMix"])
    ap.add_argument("--nbr-topk", type=int, default=3000)
    ap.add_argument("--proc", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--shortest-path", default="auto", choices=["auto", "all_pairs", "pairwise"])
    ap.add_argument("--apsp-max-nodes", type=int, default=20_000,
                    help="with auto: all_pairs up to this many vertices, pairwise beyond (GRC builds the n x n "
                         "matrix from n^2 Python floats, about 48 n^2 bytes at peak)")
    ap.add_argument("--ids", default=None, help=".npy of edge ids (rows of edges) to compute; default all")
    a = ap.parse_args()

    import networkx as nx
    import GraphRicciCurvature
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    z = np.load(a.inp)
    n = int(z["n"])
    edges = np.asarray(z["edges"], dtype=np.int64)
    ids = np.load(a.ids) if a.ids else np.arange(len(edges))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges.tolist()))
    sp = a.shortest_path
    if sp == "auto":
        sp = "all_pairs" if n <= a.apsp_max_nodes else "pairwise"
    edge_list = [tuple(map(int, edges[e])) for e in ids]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orc = OllivierRicci(G, alpha=a.alpha, method=a.method, proc=a.proc, shortest_path=sp, nbr_topk=a.nbr_topk,
                            verbose="ERROR")
        t = time.perf_counter()
        res = orc.compute_ricci_curvature_edges(edge_list)
        dt = time.perf_counter() - t
    val = np.array([res[(u, v)] if (u, v) in res else res[(v, u)] for u, v in edge_list], dtype=np.float64)
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, curvature=val, ids=ids, time_s=dt, alpha=a.alpha, method=a.method, nbr_topk=a.nbr_topk,
             shortest_path=sp, proc=a.proc, grc_version=getattr(GraphRicciCurvature, "__version__", "?"))
    os.replace(tmp, a.out)
    print(f"GRC {a.method} alpha={a.alpha} topk={a.nbr_topk} sp={sp} proc={a.proc}: {len(ids)} edges in {dt:.2f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
