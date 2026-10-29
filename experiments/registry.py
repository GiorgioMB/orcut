"""Graph specifications used on the command lines of the experiments.

A spec is a string:
  bundled:<name>                    karate, jazz, yeast, power_grid, arxiv (experiments/data)
  real:<Name>                       a network of ``datasets.REAL`` (downloaded and cached)
  synth:<family>:<n>:<avg_deg>:<seed>   ER, BA, RGG, WS, CL2.1, CL2.5
  canon:<kind>:<k>                  K (complete K_k), Q (hypercube Q_k), grid (k x k), tree (binary, depth k),
                                    cycle (C_k), star (k leaves), wheel (k rim vertices), kbip (K_{k, 3k})
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import datasets as D


def _canon(kind: str, k: int) -> Tuple[int, np.ndarray]:
    import networkx as nx

    G = {"K": lambda: nx.complete_graph(k), "Q": lambda: nx.hypercube_graph(k), "grid": lambda: nx.grid_2d_graph(k, k),
         "tree": lambda: nx.balanced_tree(2, k), "cycle": lambda: nx.cycle_graph(k), "star": lambda: nx.star_graph(k),
         "wheel": lambda: nx.wheel_graph(k + 1), "kbip": lambda: nx.complete_bipartite_graph(k, 3 * k)}[kind]()
    G = nx.convert_node_labels_to_integers(G)
    return G.number_of_nodes(), np.array(G.edges(), np.int64).reshape(-1, 2)


CANON_NAMES = {"K": "K_{%d}", "Q": "Q_{%d}", "grid": "Grid(%d)", "tree": "Tree(2,%d)", "cycle": "C_{%d}",
               "star": "Star(%d)", "wheel": "Wheel(%d)", "kbip": "K_{%d,3k}"}


def display(spec: str) -> str:
    p = spec.split(":")
    if p[0] == "bundled":
        return D.BUNDLED[p[1]]
    if p[0] == "real":
        return p[1]
    if p[0] == "synth":
        n = int(float(p[2]))
        return f"{p[1]}({_short(n)}, {p[3]})"
    if p[0] == "canon":
        return CANON_NAMES[p[1]].replace("%d", p[2]).replace("3k", str(3 * int(p[2])))
    return spec


def slug(name: str) -> str:
    """A file-name-safe version of a display name."""
    import re

    return re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_").lower()


def _short(n: int) -> str:
    for div, suf in ((10**6, "M"), (10**3, "k")):
        if n >= div and n % (div // 10) == 0:
            x = n / div
            return f"{x:g}{suf}"
    return str(n)


def load(spec: str, data_dir: Optional[str] = None):
    """(display name, orcut.Graph, meta) for a spec."""
    p = spec.split(":")
    meta: Dict = {"spec": spec}
    if p[0] == "bundled":
        n, e = D.bundled(p[1])
    elif p[0] == "real":
        n, e, info = D.real(p[1], data_dir)
        meta.update(info)
        meta["cite"] = D.REAL_BY_NAME[p[1]].cite
    elif p[0] == "synth":
        n, e = D.synthetic(p[1], int(float(p[2])), float(p[3]), int(p[4]))
    elif p[0] == "canon":
        n, e = _canon(p[1], int(p[2]))
    else:
        raise ValueError(f"unknown graph spec {spec!r}")
    g = D.to_graph(n, e)
    meta.update(n=g.num_nodes, m=g.num_edges, deg_max=int(g.deg.max()) if g.num_nodes else 0,
                deg_mean=float(g.deg.mean()) if g.num_nodes else 0.0)
    return display(spec), g, meta


SMALL_REAL = ["bundled:karate", "bundled:jazz", "bundled:yeast", "bundled:power_grid", "bundled:arxiv"]
SMALL_SYNTH = ["synth:ER:2000:10:0", "synth:BA:2000:10:0", "synth:RGG:2000:12:0", "synth:WS:2000:10:0",
               "synth:CL2.1:3000:8:0", "synth:CL2.5:3000:8:0"]
CANON = ["canon:K:20", "canon:Q:6", "canon:grid:20", "canon:tree:8", "canon:cycle:100", "canon:star:200",
         "canon:wheel:50", "canon:kbip:10"]
MEDIUM = ["real:" + s.name for s in D.REAL if s.group == "medium"]
LARGE = ["real:" + s.name for s in D.REAL if s.group == "large"]
XLARGE = ["real:" + s.name for s in D.REAL if s.group == "xlarge"]
HUBBY = ["real:GitHub", "real:Twitch-DE", "real:Squirrel", "real:Amazon-Computers", "real:Facebook"]
GNN = ["real:" + s.name for s in D.REAL if s.group == "gnn"]
