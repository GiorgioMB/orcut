"""Incremental curvature on a changing graph, and partition scoring (E6).

Locality. For curvatures whose measures m_u are supported on N[u] (all measures of
:func:`orcut.curvature`), the curvature of an edge (i, j) depends on the measures of i and j and on
the hop distances between vertices of S = N[i] u N[j], which are 0 on the diagonal, 1 on edges,
2 on pairs with a common neighbour, and 3 otherwise. Inserting or deleting the edge (a, b) changes
the measures of a and b only, and changes the distance of a pair {x, y} only if x or y is a or b
(the pair {a, b} itself, or a 2-path through a or b). Hence the curvature of (i, j) can change only
if a or b lies in S, i.e. only if i or j lies in N[a] u N[b]. After a batch of changes it suffices
to recompute the edges of the final graph with an endpoint in the union of N[a] u N[b] over the
changed pairs (neighbourhoods in the final graph; the endpoints themselves are always included).

This module provides a CSR graph with in-place edge deletion (rows stay sorted; a parallel array
maps CSR entries to the ids of the original edges), the affected edge set of a deletion, and the
scoring of the component partitions met along a deletion sequence (modularity on the original
graph, by a reverse union-find), plus NMI and ARI.
"""
from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------- dynamic CSR

def csr_with_ids(g):
    """(indptr, indices, eid) copies of g's CSR with, for every entry, the id of its edge in g.edges."""
    n = g.num_nodes
    indptr = g.indptr.astype(np.int64).copy()
    indices = g.indices.astype(np.int64).copy()
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    u = np.minimum(rows, indices)
    v = np.maximum(rows, indices)
    keys = g.edges[:, 0].astype(np.int64) * n + g.edges[:, 1]
    eid = np.searchsorted(keys, u * n + v).astype(np.int64)
    return indptr, indices, eid


@njit(cache=True)
def _remove_entry(indptr, indices, eid, u, v):
    lo = indptr[u]
    hi = indptr[u + 1]
    p = lo
    while p < hi and indices[p] != v:
        p += 1
    if p == hi:
        return False
    end = indptr[indptr.shape[0] - 1]
    for q in range(p, end - 1):
        indices[q] = indices[q + 1]
        eid[q] = eid[q + 1]
    for r in range(u + 1, indptr.shape[0]):
        indptr[r] -= 1
    return True


@njit(cache=True)
def delete_edge(indptr, indices, eid, a, b):
    """Delete the undirected edge (a, b) in place (both CSR entries); the arrays keep their length,
    the tail beyond indptr[-1] is garbage."""
    ok1 = _remove_entry(indptr, indices, eid, a, b)
    ok2 = _remove_entry(indptr, indices, eid, b, a)
    return ok1 and ok2


@njit(cache=True)
def affected_edges(indptr, indices, eid, verts, stamp_v, stamp_e, ep):
    """Ids of the edges with an endpoint in N[verts] (closed neighbourhoods in the current graph).
    ``stamp_v`` (n) and ``stamp_e`` (number of original edges) are scratch arrays stamped with
    ``ep`` (a fresh value per call)."""
    nv = 0
    tot = 0
    buf = np.empty(16, np.int64)
    for k in range(verts.shape[0]):
        x = verts[k]
        for t in range(indptr[x] - 1, indptr[x + 1]):
            y = x if t < indptr[x] else indices[t]
            if stamp_v[y] != ep:
                stamp_v[y] = ep
                if nv == buf.shape[0]:
                    nb = np.empty(2 * nv, np.int64)
                    nb[:nv] = buf
                    buf = nb
                buf[nv] = y
                nv += 1
                tot += indptr[y + 1] - indptr[y]
    out = np.empty(tot, np.int64)
    m = 0
    for k in range(nv):
        y = buf[k]
        for t in range(indptr[y], indptr[y + 1]):
            e = eid[t]
            if stamp_e[e] != ep:
                stamp_e[e] = ep
                out[m] = e
                m += 1
    return np.sort(out[:m])


# --------------------------------------------- partitions along a deletion

@njit(cache=True)
def _find(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        nx_ = parent[x]
        parent[x] = r
        x = nx_
    return r


@njit(cache=True)
def modularity_along(n, edges, indptr0, indices0, order, T):
    """Modularity (on the original graph) of the component partition of G - {order[0..s-1]} for
    s = 0..T, where ``order`` lists edge ids in deletion order. Returns Q (T + 1,). Reverse
    union-find: the graph after T deletions is built first, then the deleted edges are added back;
    when components A and B merge, the original edges between them are counted by scanning the
    smaller one (small-to-large, O(m log n) in total)."""
    m = edges.shape[0]
    two_m = 2.0 * m
    deg = np.empty(n, np.float64)
    for x in range(n):
        deg[x] = indptr0[x + 1] - indptr0[x]
    parent = np.arange(n)
    size = np.ones(n, np.int64)
    head = np.arange(n)          # member lists (circular linked lists via nxt)
    nxt = np.arange(n)
    Dc = deg.copy()
    Lc = np.zeros(n, np.float64)
    alive = np.ones(m, np.bool_)
    for s in range(T):
        alive[order[s]] = False
    Q = np.empty(T + 1, np.float64)

    def merge(a, b):
        ra = _find(parent, a)
        rb = _find(parent, b)
        if ra == rb:
            return 0.0, False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        # count original edges between the smaller component rb and ra
        e_ab = 0
        x = rb
        while True:
            for t in range(indptr0[x], indptr0[x + 1]):
                if _find(parent, indices0[t]) == ra:
                    e_ab += 1
            x = nxt[x]
            if x == rb:
                break
        dq = e_ab / m - 2.0 * Dc[ra] * Dc[rb] / (two_m * two_m)
        parent[rb] = ra
        size[ra] += size[rb]
        Dc[ra] += Dc[rb]
        Lc[ra] += Lc[rb] + e_ab
        tmp = nxt[ra]
        nxt[ra] = nxt[rb]
        nxt[rb] = tmp
        return dq, True

    q = 0.0
    for x in range(n):
        q -= (deg[x] / two_m) ** 2
    for e in range(m):
        if alive[e]:
            dq, _ = merge(edges[e, 0], edges[e, 1])
            q += dq
    Q[T] = q
    for s in range(T - 1, -1, -1):
        e = order[s]
        dq, _ = merge(edges[e, 0], edges[e, 1])
        q += dq
        Q[s] = q
    return Q


@njit(cache=True)
def components_after(n, edges, order, s):
    """Component labels (0..c-1) of G - {order[0..s-1]}."""
    m = edges.shape[0]
    alive = np.ones(m, np.bool_)
    for k in range(s):
        alive[order[k]] = False
    parent = np.arange(n)
    for e in range(m):
        if alive[e]:
            ra = _find(parent, edges[e, 0])
            rb = _find(parent, edges[e, 1])
            if ra != rb:
                parent[rb] = ra
    lab = -np.ones(n, np.int64)
    out = np.empty(n, np.int64)
    c = 0
    for x in range(n):
        r = _find(parent, x)
        if lab[r] < 0:
            lab[r] = c
            c += 1
        out[x] = lab[r]
    return out


def modularity(g, labels) -> float:
    """Newman's modularity of a partition of g."""
    m = g.num_edges
    same = labels[g.edges[:, 0]] == labels[g.edges[:, 1]]
    D = np.bincount(labels, weights=g.deg.astype(float))
    return float(same.sum() / m - np.sum((D / (2.0 * m)) ** 2))


# ------------------------------------------------------ partition scores

def _contingency(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1), np.int64)
    np.add.at(C, (ai, bi), 1)
    return C


def nmi(a, b) -> float:
    """Normalized mutual information (arithmetic-mean normalization)."""
    C = _contingency(a, b).astype(float)
    N = C.sum()
    pa, pb = C.sum(1) / N, C.sum(0) / N
    nz = C > 0
    I = float(np.sum(C[nz] / N * np.log((C[nz] / N) / np.outer(pa, pb)[nz])))
    Ha = -float(np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    Hb = -float(np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    if Ha + Hb == 0:
        return 1.0
    return 2.0 * I / (Ha + Hb)


def ari(a, b) -> float:
    """Adjusted Rand index."""
    C = _contingency(a, b).astype(float)
    comb = lambda x: x * (x - 1) / 2.0
    s_ij = comb(C).sum()
    s_a = comb(C.sum(1)).sum()
    s_b = comb(C.sum(0)).sum()
    N = comb(C.sum())
    exp = s_a * s_b / N if N else 0.0
    mx = 0.5 * (s_a + s_b)
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0
