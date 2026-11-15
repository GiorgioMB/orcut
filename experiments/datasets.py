"""Graphs of the experiments.

* ``bundled``: small real networks shipped in ``experiments/data`` (Karate, Jazz, Yeast, Power Grid,
  arXiv HepPh), as in the companion paper.
* ``REAL``: real networks downloaded on first use (SNAP, GitHub mirrors of MUSAE / gnn-benchmark),
  from 10^5 to 1.2 x 10^8 edges, cached as ``<data_dir>/<tag>.npz`` (largest connected component).
* ``synthetic``: seeded random graphs with numpy/numba generators that scale to 10^7 vertices
  (Erdos--Renyi G(n, m), Barabasi--Albert, random geometric, Watts--Strogatz, Chung--Lu power law).
* ``edge_instance``: one edge with prescribed neighbourhood sizes and distance structure (the
  controlled per-edge experiment, E2), and ``dasgupta_instance``: the lower-bound instances of
  DasGupta et al. (E2, part C).

Every graph is simple, undirected, relabeled 0..n-1, and restricted to its largest connected
component.
"""
from __future__ import annotations

import gzip
import io
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from common import HERE, log
import xio

DATA = os.path.join(HERE, "data")
SNAP = "https://snap.stanford.edu/data/"
GH = "https://raw.githubusercontent.com/"


# ---------------------------------------------------------------- utilities

def largest_component(n: int, edges) -> Tuple[int, np.ndarray]:
    """Largest connected component, relabeled 0..n'-1, edges (u < v) unique."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    if len(e) == 0:
        return 0, e
    e = np.unique(np.sort(e, axis=1), axis=0)
    A = sp.coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    keep = np.flatnonzero(lab == big)
    remap = -np.ones(n, np.int64)
    remap[keep] = np.arange(len(keep))
    e = remap[e]
    e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0)]
    return len(keep), e


def relabel(e: np.ndarray) -> Tuple[int, np.ndarray]:
    e = np.asarray(e, dtype=np.int64).reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    uniq, inv = np.unique(e, return_inverse=True)
    return int(len(uniq)), inv.reshape(-1, 2).astype(np.int64)


def to_graph(n: int, e: np.ndarray):
    import orcut

    return orcut.Graph.from_edges(np.asarray(e, dtype=np.int64).reshape(-1, 2), n)


# ------------------------------------------------------------------ bundled

BUNDLED = {"karate": "Karate", "jazz": "Jazz", "yeast": "Yeast", "power_grid": "Power Grid", "arxiv": "arXiv HepPh"}


def bundled(name: str) -> Tuple[int, np.ndarray]:
    e = np.loadtxt(os.path.join(DATA, f"{name}.csv"), delimiter=",", dtype=np.int64)
    n, e = relabel(e)
    return largest_component(n, e)


# --------------------------------------------------------- downloaded graphs

@dataclass(frozen=True)
class RealSpec:
    name: str
    tag: str
    url: str
    fmt: str          # "snap" (whitespace edge list, '#' comments, .gz), "csv" (header), "npz" (gnn-benchmark)
    min_edges: int    # sanity check on the parsed raw graph
    cite: str         # bibliography key of the source
    group: str        # "medium" (10^5-10^6 edges), "large" (10^6-10^7), "xlarge" (> 10^7)


REAL: List[RealSpec] = [
    RealSpec("Deezer", "deezer_europe", GH + "benedekrozemberczki/karateclub/master/dataset/node_level/deezer/edges.csv",
             "csv", 90_000, "rozemberczki_characteristic_2020", "medium"),
    RealSpec("Amazon-Photo", "amazon_photo", GH + "shchur/gnn-benchmark/master/data/npz/amazon_electronics_photo.npz",
             "npz", 115_000, "shchur_pitfalls_2018", "medium"),
    RealSpec("Twitch-DE", "twitch_de", GH + "benedekrozemberczki/MUSAE/master/input/edges/DE_edges.csv",
             "csv", 150_000, "rozemberczki_multiscale_2021", "medium"),
    RealSpec("Facebook", "facebook_pages", GH + "benedekrozemberczki/MUSAE/master/input/edges/facebook_edges.csv",
             "csv", 165_000, "rozemberczki_multiscale_2021", "medium"),
    RealSpec("Crocodile", "wiki_crocodile", GH + "benedekrozemberczki/MUSAE/master/input/edges/crocodile_edges.csv",
             "csv", 165_000, "rozemberczki_multiscale_2021", "medium"),
    RealSpec("Enron", "email_enron", SNAP + "email-Enron.txt.gz", "snap", 180_000, "leskovec_community_2009", "medium"),
    RealSpec("AstroPh", "ca_astroph", SNAP + "ca-AstroPh.txt.gz", "snap", 190_000, "leskovec_graph_2007", "medium"),
    RealSpec("Squirrel", "wiki_squirrel", GH + "benedekrozemberczki/MUSAE/master/input/edges/squirrel_edges.csv",
             "csv", 195_000, "rozemberczki_multiscale_2021", "medium"),
    RealSpec("Brightkite", "loc_brightkite", SNAP + "loc-brightkite_edges.txt.gz", "snap", 210_000,
             "cho_friendship_2011", "medium"),
    RealSpec("Amazon-Computers", "amazon_computers",
             GH + "shchur/gnn-benchmark/master/data/npz/amazon_electronics_computers.npz", "npz", 240_000,
             "shchur_pitfalls_2018", "medium"),
    RealSpec("Coauthor-Physics", "coauthor_physics", GH + "shchur/gnn-benchmark/master/data/npz/ms_academic_phy.npz",
             "npz", 240_000, "shchur_pitfalls_2018", "medium"),
    RealSpec("GitHub", "musae_github", GH + "benedekrozemberczki/MUSAE/master/input/edges/git_edges.csv",
             "csv", 285_000, "rozemberczki_multiscale_2021", "medium"),
    RealSpec("Epinions", "soc_epinions1", SNAP + "soc-Epinions1.txt.gz", "snap", 400_000, "richardson_trust_2003",
             "medium"),
    RealSpec("Amazon", "com_amazon", SNAP + "bigdata/communities/com-amazon.ungraph.txt.gz", "snap", 900_000,
             "yang_defining_2012", "medium"),
    RealSpec("DBLP", "com_dblp", SNAP + "bigdata/communities/com-dblp.ungraph.txt.gz", "snap", 1_000_000,
             "yang_defining_2012", "large"),
    RealSpec("roadNet-CA", "roadnet_ca", SNAP + "roadNet-CA.txt.gz", "snap", 2_700_000, "leskovec_community_2009",
             "large"),
    RealSpec("YouTube", "com_youtube", SNAP + "bigdata/communities/com-youtube.ungraph.txt.gz", "snap", 2_900_000,
             "yang_defining_2012", "large"),
    RealSpec("Skitter", "as_skitter", SNAP + "as-skitter.txt.gz", "snap", 11_000_000, "leskovec_graphs_2005",
             "xlarge"),
    RealSpec("LiveJournal", "com_lj", SNAP + "bigdata/communities/com-lj.ungraph.txt.gz", "snap", 34_000_000,
             "yang_defining_2012", "xlarge"),
    RealSpec("Orkut", "com_orkut", SNAP + "bigdata/communities/com-orkut.ungraph.txt.gz", "snap", 117_000_000,
             "yang_defining_2012", "xlarge"),
]
REAL_BY_NAME = {s.name: s for s in REAL}


def _download(url: str, path: str, retries: int = 4) -> None:
    tmp = xio.tmp_path(path)  # process-unique: steps of a job array may download the same file at once
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "orcut-experiments/1.1"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
            os.replace(tmp, path)
            return
        except Exception as ex:  # network errors: retry with backoff
            last = ex
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


def _parse_edges_text(raw: bytes, sep: str, skip: int = 0) -> np.ndarray:
    """Edge list from text (pandas C parser when available; ``sep`` is ",", "\\t", or r"\\s+")."""
    try:
        import pandas as pd

        df = pd.read_csv(io.BytesIO(raw), sep=sep, comment="#", header=None, skiprows=skip, usecols=[0, 1],
                         dtype=np.int64, engine="c")
        return df.to_numpy(np.int64)
    except ImportError:
        return np.loadtxt(io.BytesIO(raw), comments="#", delimiter=None if sep != "," else ",", skiprows=skip,
                          dtype=np.int64, usecols=(0, 1))


def _parse(spec: RealSpec, raw: bytes) -> Tuple[int, np.ndarray]:
    if spec.fmt == "snap":
        txt = gzip.decompress(raw) if spec.url.endswith(".gz") else raw
        e = _parse_edges_text(txt, sep=r"\s+")
    elif spec.fmt == "csv":
        e = _parse_edges_text(raw, sep=",", skip=1)
    elif spec.fmt == "npz":
        import scipy.sparse as sp

        z = np.load(io.BytesIO(raw), allow_pickle=True)
        A = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"])).tocoo()
        e = np.stack([A.row, A.col], axis=1)
    else:
        raise ValueError(spec.fmt)
    return relabel(e)


def real(name: str, data_dir: Optional[str] = None) -> Tuple[int, np.ndarray, Dict]:
    """(n, edges, info) of the largest connected component of a downloaded network (cached)."""
    spec = REAL_BY_NAME[name]
    d = os.path.abspath(data_dir or os.path.join(DATA, "real"))
    os.makedirs(d, exist_ok=True)
    cache = os.path.join(d, f"{spec.tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return int(z["n"]), z["edges"], {"raw_nodes": int(z["raw_nodes"]), "raw_edges": int(z["raw_edges"])}
    base = os.path.basename(spec.url)
    ext = ".txt.gz" if base.endswith(".txt.gz") else os.path.splitext(base)[1]
    src = os.path.join(d, f"{spec.tag}.raw{ext}")
    if not os.path.exists(src):
        log(f"downloading {spec.url}")
        _download(spec.url, src)
    n0, e0 = _parse(spec, open(src, "rb").read())
    if len(e0) < spec.min_edges:
        raise RuntimeError(f"{name}: parsed {len(e0)} edges, expected at least {spec.min_edges}; delete {src}")
    n, e = largest_component(n0, e0)
    tmp = xio.tmp_path(cache)  # process-unique, as in _download
    with open(tmp, "wb") as fh:  # a file object: numpy must not append ".npz" to the name
        np.savez_compressed(fh, n=n, edges=e, raw_nodes=n0, raw_edges=len(e0))
    os.replace(tmp, cache)
    return n, e, {"raw_nodes": n0, "raw_edges": int(len(e0))}


# ---------------------------------------------------------------- synthetic

def _unique_pairs(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    keep = a != b
    return np.unique(np.stack([a[keep], b[keep]], axis=1), axis=0)


def er(n: int, avg_deg: float, seed: int) -> np.ndarray:
    """G(n, m) with m = n avg_deg / 2 (sampled with replacement, duplicates merged)."""
    rng = np.random.default_rng(seed)
    m = int(round(n * avg_deg / 2))
    return _unique_pairs(rng.integers(0, n, m), rng.integers(0, n, m))


@njit(cache=True)
def _ba_kernel(n, m, seed):
    np.random.seed(seed)
    rep = np.empty(2 * n * m + m, np.int64)
    E = np.empty((n * m, 2), np.int64)
    nr = 0
    ne = 0
    for v in range(m):  # the first new vertex connects to 0..m-1
        rep[nr] = v
        nr += 1
    chosen = np.empty(m, np.int64)
    for v in range(m, n):
        k = 0
        while k < m:
            t = rep[np.random.randint(0, nr)]
            dup = False
            for z in range(k):
                if chosen[z] == t:
                    dup = True
                    break
            if not dup:
                chosen[k] = t
                k += 1
        for z in range(m):
            E[ne, 0] = v
            E[ne, 1] = chosen[z]
            ne += 1
            rep[nr] = chosen[z]
            rep[nr + 1] = v
            nr += 2
    return E[:ne]


def ba(n: int, m: int, seed: int) -> np.ndarray:
    """Barabasi--Albert preferential attachment (m edges per new vertex), linear time."""
    E = _ba_kernel(n, m, seed)
    return _unique_pairs(E[:, 0], E[:, 1])


def rgg(n: int, avg_deg: float, seed: int) -> np.ndarray:
    """Random geometric graph in the unit square with expected average degree avg_deg."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    x = rng.random((n, 2))
    r = np.sqrt(avg_deg / (np.pi * (n - 1)))
    pairs = cKDTree(x).query_pairs(r, output_type="ndarray")
    return _unique_pairs(pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64))


def ws(n: int, k: int, beta: float, seed: int) -> np.ndarray:
    """Watts--Strogatz: ring lattice of even degree k, each edge rewired with probability beta."""
    rng = np.random.default_rng(seed)
    half = k // 2
    u = np.repeat(np.arange(n, dtype=np.int64), half)
    v = (u + np.tile(np.arange(1, half + 1, dtype=np.int64), n)) % n
    flip = rng.random(len(u)) < beta
    v = v.copy()
    v[flip] = rng.integers(0, n, int(flip.sum()))
    return _unique_pairs(u, v)


def chung_lu(n: int, avg_deg: float, gamma: float, seed: int, max_deg: Optional[float] = None) -> np.ndarray:
    """Chung--Lu graph with power-law expected degrees (exponent gamma), m = n avg_deg / 2 edges
    sampled with endpoint probabilities proportional to the weights (duplicates merged)."""
    rng = np.random.default_rng(seed)
    i = np.arange(1, n + 1, dtype=np.float64)
    w = i ** (-1.0 / (gamma - 1.0))
    w *= avg_deg * n / w.sum()
    if max_deg is not None:
        w = np.minimum(w, max_deg)
    p = w / w.sum()
    m = int(round(n * avg_deg / 2))
    return _unique_pairs(rng.choice(n, m, p=p), rng.choice(n, m, p=p))


SYNTH = {
    "ER": lambda n, d, s: er(n, d, s),
    "BA": lambda n, d, s: ba(n, max(1, int(round(d / 2))), s),
    "RGG": lambda n, d, s: rgg(n, d, s),
    "WS": lambda n, d, s: ws(n, 2 * max(1, int(round(d / 2))), 0.2, s),
    "CL2.1": lambda n, d, s: chung_lu(n, d, 2.1, s),
    "CL2.5": lambda n, d, s: chung_lu(n, d, 2.5, s),
}


def synthetic(family: str, n: int, avg_deg: float, seed: int = 0) -> Tuple[int, np.ndarray]:
    e = SYNTH[family](n, avg_deg, seed)
    return largest_component(n, e)


# --------------------------------------------------- controlled single edges

REGIMES = ("tree", "sparse", "mixed", "dense", "hubleaf")
_COVER = {"tree": 0.0, "sparse": 0.05, "mixed": 0.5, "dense": 1.0, "hubleaf": 0.5}


def edge_instance(a: int, b: int, t: int, regime: str, seed: int = 0) -> Tuple[int, np.ndarray]:
    """A graph around the edge (0, 1) with |U_0| = a, |U_1| = b, and t common neighbours.

    Regimes (distance structure of U_0 x U_1):
      tree    no cross edges and no common neighbours outside S: every pair at distance 3;
      sparse  about 2 cross edges per vertex of U_0, and about 5% of the pairs at distance 2;
      mixed   about 2 cross edges per vertex of U_0, and about half of the pairs at distance 2;
      dense   cross-edge density 0.2 and a common neighbour for every pair (no distance 3);
      hubleaf U_1 is made of leaves of vertex 1 (twins), except 2% of it, which carries the
              cross structure of ``mixed``.
    Distance-2 pairs come from connectors, extra vertices adjacent to random fractions f of U_0
    and of U_1; with k connectors about 1 - (1 - f^2)^k of the pairs are covered.
    """
    rng = np.random.default_rng(seed)
    i, j = 0, 1
    C = np.arange(2, 2 + t)
    Ui = np.arange(2 + t, 2 + t + a)
    Uj = np.arange(2 + t + a, 2 + t + a + b)
    nxt = 2 + t + a + b
    parts = [np.array([(i, j)], np.int64)]
    parts += [np.stack([np.full(t, i), C], 1), np.stack([np.full(t, j), C], 1),
              np.stack([np.full(a, i), Ui], 1), np.stack([np.full(b, j), Uj], 1)]
    Ujc = Uj if regime != "hubleaf" else Uj[: max(1, b // 50)]  # the non-leaf part of U_1
    bc = len(Ujc)
    if regime != "tree" and a and bc:
        p = 0.2 if regime == "dense" else min(1.0, 2.0 / bc)
        ncr = rng.binomial(a * bc, p)
        if ncr:
            parts.append(np.stack([rng.choice(Ui, ncr), rng.choice(Ujc, ncr)], 1))
    cover = _COVER[regime]
    if cover >= 1.0 and a and bc:
        z = nxt
        nxt += 1
        parts.append(np.stack([np.full(a + bc, z), np.concatenate([Ui, Ujc])], 1))
    elif cover > 0 and a and bc:
        k = 8
        f = np.sqrt(1.0 - (1.0 - cover) ** (1.0 / k))
        for _ in range(k):
            z = nxt
            nxt += 1
            sa = rng.choice(Ui, max(1, int(round(f * a))), replace=False)
            sb = rng.choice(Ujc, max(1, int(round(f * bc))), replace=False)
            parts.append(np.stack([np.full(len(sa) + len(sb), z), np.concatenate([sa, sb])], 1))
    E = np.concatenate([x.reshape(-1, 2) for x in parts if x.size])
    n = int(E.max()) + 1
    return n, _unique_pairs(E[:, 0], E[:, 1])


DASGUPTA_KINDS = ("single", "matching", "bounded", "blocks")


def dasgupta_instance(n: int, kind: str, seed: int = 0) -> Tuple[int, np.ndarray]:
    """The graphs of DasGupta, Grigorescu and Mukherjee (2023, Proposition 2) around the edge (0, 1):
    L = N(0) minus 1 and R = N(1) minus 0, both of size n, no common neighbour; a pair of L x R at
    distance 1 is an edge, a pair at distance 2 gets a middle vertex adjacent to both, and every other
    pair is at distance 3 (path through 0 and 1). Equal degrees, so the lazy curvature is a perfect
    matching problem with costs in {1, 2, 3} (their Fact 1). Kinds:

      single    one pair at distance 1 (the instances of their Theorem 4(a));
      matching  a random perfect matching at distance 1 (Theorem 4(b));
      bounded   two random partners at distance 1 and two at distance 2 per vertex of L, each
                distance-2 pair with its own middle vertex (bounded weighted degrees, Theorem 5(b));
      blocks    about sqrt(n) middle vertices, each adjacent to sqrt(n) random vertices of L and of
                R: about n^{3/2} pairs at distance 2 from O(n) edges.
    """
    rng = np.random.default_rng(seed)
    L = np.arange(2, 2 + n)
    R = np.arange(2 + n, 2 + 2 * n)
    parts = [np.array([(0, 1)], np.int64), np.stack([np.zeros(n, np.int64), L], 1), np.stack([np.ones(n, np.int64), R], 1)]
    nxt = 2 + 2 * n
    if kind == "single":
        parts.append(np.array([(int(rng.choice(L)), int(rng.choice(R)))], np.int64))
    elif kind == "matching":
        parts.append(np.stack([L, rng.permutation(R)], 1))
    elif kind == "bounded":
        for _ in range(2):
            parts.append(np.stack([L, rng.permutation(R)], 1))
        for _ in range(2):
            mid = np.arange(nxt, nxt + n)
            nxt += n
            parts.append(np.stack([mid, L], 1))
            parts.append(np.stack([mid, rng.permutation(R)], 1))
    elif kind == "blocks":
        k = max(1, int(round(np.sqrt(n))))
        for _ in range(k):
            parts.append(np.stack([np.full(k, nxt), rng.choice(L, k, replace=False)], 1))
            parts.append(np.stack([np.full(k, nxt), rng.choice(R, k, replace=False)], 1))
            nxt += 1
    else:
        raise ValueError(kind)
    E = np.concatenate(parts)
    return int(E.max()) + 1, _unique_pairs(E[:, 0], E[:, 1])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and cache the real networks of the experiments (run once, on a "
                                             "machine with network access; compute nodes then read the cache).")
    ap.add_argument("--data-dir", default=None, help="cache directory (default experiments/data/real)")
    ap.add_argument("--groups", nargs="+", default=["medium", "large", "xlarge"])
    a = ap.parse_args()
    for s in REAL:
        if s.group in a.groups:
            n, e, info = real(s.name, a.data_dir)
            log(f"{s.name:18s} n={n:>11,} m={len(e):>12,} (raw {info['raw_nodes']:,} nodes, {info['raw_edges']:,} edges)")
