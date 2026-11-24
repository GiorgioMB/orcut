"""Primitives shared by the exact solvers: integer constants, per-thread arenas, bitsets, the CSR
network and Dinic's maximum flow (iterative DFS), duplicate detection of row patterns, and the
helpers of the parallel drivers.

The solver of the one-step measures is :mod:`orcut._core`, with the witness network of
:mod:`orcut._witness`; the layered closure of Theorem 1 is :mod:`orcut._layered`. All arithmetic is
exact (integer masses), and every temporary comes from a per-thread arena, so the steady state
allocates nothing.
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF = np.int64(1) << np.int64(60)

# ----------------------------------------------------------- bitsets, arenas

U0 = np.uint64(0)
U1 = np.uint64(1)
_BLOCK_BUDGET = 1 << 22  # words of middle-vertex bitsets held at once (32 MB)
_WARM_MIN_ARCS = 300  # token networks at least this large get the G1 warm start


@njit(cache=True, inline="always")
def _take_i(buf, off, n):
    """Carve n int64 from the arena ``buf`` at ``off`` (grow it if needed;
    views carved earlier stay valid)."""
    if off + n > buf.shape[0]:
        buf = np.empty(max(2 * buf.shape[0], 2 * n + 64), np.int64)
        off = 0
    return buf, off + n, buf[off:off + n]


@njit(cache=True, inline="always")
def _take_u(buf, off, n):
    if off + n > buf.shape[0]:
        buf = np.empty(max(2 * buf.shape[0], 2 * n + 64), np.uint64)
        off = 0
    return buf, off + n, buf[off:off + n]


@njit(cache=True, inline="always")
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, inline="always")
def _ctz(x):
    """Index of the lowest set bit of x != 0."""
    return _popcount((x & (~x + U1)) - U1)


@njit(cache=True, inline="always")
def _setbit(arr, base, k):
    arr[base + (k >> 6)] |= U1 << np.uint64(k & 63)


@njit(cache=True, inline="always")
def _getbit(arr, base, k):
    return ((arr[base + (k >> 6)] >> np.uint64(k & 63)) & U1) != U0


# ------------------------------------------------------------- max flow


@njit(cache=True)
def _csr(nV, nA, tail, head, capv, bi, oi):
    """CSR residual network of the arcs (tail[k], head[k], capv[k]), k < nA:
    arcs of a node contiguous, paired reverse arcs; pos[k] is the position of
    arc k. Returns (start, to, cap, rev, pos, arena)."""
    bi, oi, start = _take_i(bi, oi, nV + 1)
    bi, oi, fill = _take_i(bi, oi, nV)
    bi, oi, to = _take_i(bi, oi, 2 * nA)
    bi, oi, cap = _take_i(bi, oi, 2 * nA)
    bi, oi, rev = _take_i(bi, oi, 2 * nA)
    bi, oi, pos = _take_i(bi, oi, nA)
    for v in range(nV + 1):
        start[v] = 0
    for k in range(nA):
        start[tail[k] + 1] += 1
        start[head[k] + 1] += 1
    for v in range(nV):
        start[v + 1] += start[v]
    for v in range(nV):
        fill[v] = start[v]
    for k in range(nA):
        u = tail[k]
        v = head[k]
        a = fill[u]
        fill[u] = a + 1
        b = fill[v]
        fill[v] = b + 1
        to[a] = v
        cap[a] = capv[k]
        rev[a] = b
        to[b] = u
        cap[b] = 0
        rev[b] = a
        pos[k] = a
    return start, to, cap, rev, pos, bi, oi


@njit(cache=True)
def _dinic(nV, start, to, cap, rev, s, t, bi, oi):
    """Dinic from the current residual capacities (any feasible flow), with
    an iterative DFS and current-arc pointers. Returns (added flow, arena)."""
    bi, oi, level = _take_i(bi, oi, nV)
    bi, oi, it = _take_i(bi, oi, nV)
    bi, oi, q = _take_i(bi, oi, nV)
    bi, oi, path = _take_i(bi, oi, nV)
    flow = np.int64(0)
    while True:
        for v in range(nV):
            level[v] = -1
        level[s] = 0
        q[0] = s
        qh = 0
        qt = 1
        while qh < qt:
            u = q[qh]
            qh += 1
            lu = level[u] + 1
            for a in range(start[u], start[u + 1]):
                if cap[a] > 0:
                    v = to[a]
                    if level[v] < 0:
                        level[v] = lu
                        q[qt] = v
                        qt += 1
            if level[t] >= 0:
                break  # BFS is level by level: everything below t's level is labeled
        if level[t] < 0:
            break
        lt = level[t]
        for v in range(nV):
            it[v] = start[v]
            if level[v] >= lt and v != t:
                level[v] = -1  # nodes at or beyond t's level cannot reach t
        top = 0
        u = s
        while True:
            if u == t:
                f = INF
                for p in range(top):
                    c = cap[path[p]]
                    if c < f:
                        f = c
                for p in range(top):
                    a = path[p]
                    cap[a] -= f
                    cap[rev[a]] += f
                flow += f
                p0 = 0
                while cap[path[p0]] > 0:
                    p0 += 1
                top = p0
                u = s if top == 0 else to[path[top - 1]]
                continue
            a = it[u]
            end = start[u + 1]
            lu = level[u] + 1
            while a < end:
                if cap[a] > 0 and level[to[a]] == lu:
                    break
                a += 1
            it[u] = a
            if a < end:
                path[top] = a
                top += 1
                u = to[a]
            else:
                level[u] = -1
                if top == 0:
                    break
                top -= 1
                u = to[rev[path[top]]]
                it[u] += 1
    return flow, bi, oi


@njit(cache=True)
def _maxflow(nV, nA, tail, head, capv, s, t, bi, oi):
    """Max flow of the network with arcs (tail[k], head[k], capv[k])."""
    start, to, cap, rev, pos, bi, oi = _csr(nV, nA, tail, head, capv, bi, oi)
    return _dinic(nV, start, to, cap, rev, s, t, bi, oi)


# ------------------------------------------------------ twin merging


@njit(cache=True)
def _dedupe(keys, nrows, KW, bi, oi):
    """Group equal rows of the (nrows, KW) uint64 key matrix (flat) by
    hashing with open addressing. Returns (group id per row, number of
    groups, leader row of each group, arena)."""
    tsize = 4
    while tsize < 2 * nrows:
        tsize <<= 1
    mask = np.uint64(tsize - 1)
    bi, oi, table = _take_i(bi, oi, tsize)
    bi, oi, gid = _take_i(bi, oi, nrows)
    bi, oi, leader = _take_i(bi, oi, nrows)
    for x in range(tsize):
        table[x] = -1
    ng = 0
    for r in range(nrows):
        base = r * KW
        h = np.uint64(1469598103934665603)
        for x in range(KW):
            h = (h ^ keys[base + x]) * np.uint64(1099511628211)
        h ^= h >> np.uint64(29)
        pos = np.int64(h & mask)
        while True:
            g = table[pos]
            if g < 0:
                table[pos] = ng
                leader[ng] = r
                gid[r] = ng
                ng += 1
                break
            lb = leader[g] * KW
            same = True
            for x in range(KW):
                if keys[lb + x] != keys[base + x]:
                    same = False
                    break
            if same:
                gid[r] = g
                break
            pos = (pos + 1) & (tsize - 1)
    return gid, ng, leader, bi, oi


# The public drivers are Python functions on purpose: Numba launches its
# thread pool when a ``parallel=True`` function is dispatched from Python, and
# a parallel function first entered from inside compiled code runs on an
# uninitialized pool. These helpers also fix the argument types, so that every
# caller reuses one compiled specialization of each driver.

def _ids(ids):
    return np.ascontiguousarray(ids, dtype=np.int64)


def _nchunks(nchunks):
    """Number of chunks actually used: at most one per thread. Every chunk
    already samples the whole id range (blocks are dealt cyclically), and each
    one allocates O(n) stamps, so more chunks than threads only add work."""
    import numba
    k = numba.get_num_threads() if nchunks is None else int(nchunks)
    return max(1, min(k, numba.get_num_threads()))
