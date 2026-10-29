"""Certificates: an optimal transport plan and an optimal Kantorovich potential for one edge.

``certify(graph, (i, j), measure)`` solves the edge with the layered closure of
:mod:`orcut._layered` (explicit distances, every implication listed), reads integer potentials off
the source side of a minimum cut, recovers a plan by a maximum flow on the tight pairs, and checks
exactly that the plan moves the two measures onto each other, that the potentials are feasible
(u_x - v_y <= d(x, y) on every pair), and that both have the same cost. Anyone can re-check the
returned objects with :meth:`Certificate.verify`, which needs only shortest-path distances.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

from . import _layered as LY
from .api import as_graph
from .graph import Graph
from .measures import Walk, _frac, parse, walk_tables

__all__ = ["Certificate", "certify", "measures_of_edge"]


@dataclass
class Certificate:
    edge: Tuple[int, int]
    curvature: Fraction
    w1: Fraction
    potential: Dict[int, int]                    # f on P u N with f(x) - f(y) <= d(x, y)
    plan: List[Tuple[int, int, Fraction]]        # (x, y, mass) moved at cost d(x, y)
    m_i: Dict[int, Fraction] = field(repr=False, default_factory=dict)
    m_j: Dict[int, Fraction] = field(repr=False, default_factory=dict)
    verified: bool = False
    scale: Fraction = Fraction(1)                # curvature = (1 - w1) * scale (LLY: (dmax + 1) / dmax)

    def verify(self, graph) -> bool:
        """Re-check the certificate from scratch (BFS distances, exact rationals)."""
        g = as_graph(graph)
        net = {}
        for x, v in self.m_i.items():
            net[x] = net.get(x, Fraction(0)) + v
        for y, v in self.m_j.items():
            net[y] = net.get(y, Fraction(0)) - v
        out, inn = {}, {}
        cost = Fraction(0)
        dist = {}
        for x, y, f in self.plan:
            if f <= 0:
                return False
            if x not in dist:
                dist[x] = _bfs_all(g, x, 10**9)
            cost += f * dist[x][y]
            out[x] = out.get(x, Fraction(0)) + f
            inn[y] = inn.get(y, Fraction(0)) + f
        for v, m in net.items():
            if m > 0 and out.get(v, 0) != m:
                return False
            if m < 0 and inn.get(v, 0) != -m:
                return False
        dual = Fraction(0)
        P = [v for v, m in net.items() if m > 0]
        N = [v for v, m in net.items() if m < 0]
        for x in P:
            if x not in dist:
                dist[x] = _bfs_all(g, x, 10**9)
            for y in N:
                if self.potential[x] - self.potential[y] > dist[x][y]:
                    return False
            dual += net[x] * self.potential[x]
        for y in N:
            dual += net[y] * self.potential[y]
        return cost == dual == self.w1 and self.curvature == (1 - self.w1) * self.scale


def _bfs_all(g: Graph, s: int, maxd: int) -> Dict[int, int]:
    d = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        if d[u] == maxd:
            continue
        for w in g.neighbors(u):
            w = int(w)
            if w not in d:
                d[w] = d[u] + 1
                q.append(w)
    return d


def measures_of_edge(g: Graph, i: int, j: int, measure="lazy", radius: int = 1):
    """The two measures of edge (i, j) as {vertex: Fraction}. ``measure`` as in
    :func:`orcut.curvature`, or ``"ball"`` (uniform on the ball of radius ``radius``)."""
    if isinstance(measure, str) and measure.lower() == "ball":
        Bi = _bfs_all(g, i, radius)
        Bj = _bfs_all(g, j, radius)
        return ({x: Fraction(1, len(Bi)) for x in Bi}, {y: Fraction(1, len(Bj)) for y in Bj})
    mode, pn, pd, tabs = parse(measure, g)
    from . import _core as C

    def one(u, other):
        d = int(g.deg[u])
        nb = [int(x) for x in g.neighbors(u)]
        if mode == C.M_LAZY:
            m = {x: Fraction(1, d + 1) for x in nb}
            m[u] = Fraction(1, d + 1)
        elif mode == C.M_NONLAZY:
            m = {x: Fraction(1, d) for x in nb}
        elif mode in (C.M_IDLE, C.M_LLY):
            p = Fraction(pn, pd) if mode == C.M_IDLE else Fraction(1, max(int(g.deg[i]), int(g.deg[j])) + 1)
            m = {x: (1 - p) / d for x in nb}
            m[u] = p
        else:
            wts, Dp, nself, coef, gdv = tabs
            ws = wts[g.indptr[u]:g.indptr[u + 1]]
            m = {x: Fraction(int(coef[u]) * int(w) // int(gdv[u]), int(Dp[u])) for x, w in zip(nb, ws)}
            m[u] = Fraction(int(nself[u]), int(Dp[u]))
        return {x: v for x, v in m.items() if v != 0}

    return one(i, j), one(j, i)


def certify(graph, edge, measure="lazy", radius: int = 1) -> Certificate:
    """A verified certificate for the curvature of ``edge`` = (i, j) (see the module docstring)."""
    g = as_graph(graph)
    i, j = int(edge[0]), int(edge[1])
    if not g.has_edge(i, j):
        raise ValueError(f"({i}, {j}) is not an edge")
    mi, mj = measures_of_edge(g, i, j, measure, radius)
    L = 1
    for v in list(mi.values()) + list(mj.values()):
        L = L * v.denominator // np.gcd(L, v.denominator)
    net = {}
    for x, v in mi.items():
        net[x] = net.get(x, 0) + int(v * L)
    for y, v in mj.items():
        net[y] = net.get(y, 0) - int(v * L)
    P = sorted(v for v, m in net.items() if m > 0)
    N = sorted(v for v, m in net.items() if m < 0)
    r = radius if (isinstance(measure, str) and measure.lower() == "ball") else 1
    D = 2 * r + 1
    scale = Fraction(1)
    if isinstance(measure, str) and measure.lower().replace("-", "").replace("_", "") in ("lly", "linluyau", "clly"):
        dmax = max(int(g.deg[i]), int(g.deg[j]))
        scale = Fraction(dmax + 1, dmax)
    if not P:
        return Certificate((i, j), scale, Fraction(0), {}, [], mi, mj, True, scale)
    Dm = np.empty(len(P) * len(N), np.int64)
    for a, x in enumerate(P):
        dx = _bfs_all(g, x, D - 1)
        for b, y in enumerate(N):
            Dm[a * len(N) + b] = dx.get(y, D)
    wP = np.array([net[x] for x in P], np.int64)
    wN = np.array([-net[y] for y in N], np.int64)
    W, nV, nA, cert, u, v, px, py, pf, _, _ = LY._layered_cut(len(P), len(N), Dm, D, wP, wN, True,
                                                              np.empty(1 << 12, np.int64), 0)
    pot = {x: int(u[a]) for a, x in enumerate(P)}
    pot.update({y: int(v[b]) for b, y in enumerate(N)})
    plan = [(P[int(a)], N[int(b)], Fraction(int(f), L)) for a, b, f in zip(px, py, pf)]
    w1 = Fraction(int(W), L)
    return Certificate((i, j), (1 - w1) * scale, w1, pot, plan, mi, mj, cert == 1, scale)
