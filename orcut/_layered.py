"""Exact W1 on bounded integer metrics by one minimum cut, and its certificates.

If all distances between the points of positive and negative net mass are integers in
{1, ..., D}, the Kantorovich dual has an integral optimum with u in {1, ..., D} on P and
v in {0, ..., D - 1} on N. In unary, u_x = 1 + sum_{t=2}^{D} [u_x >= t] and
v_y = sum_{s=1}^{D-1} [v_y >= s], the constraint u_x - v_y <= d(x, y) is the set of
implications [u_x >= t] => [v_y >= t - d(x, y)] for t = d(x, y) + 1, ..., D, so the dual is a
maximum-weight closure on (D - 1)(|P| + |N|) nodes, and

    W1 = D * A - maxflow,

with A the transported mass. For one-step measures D = 3 (two layers, the network of
:mod:`orcut._core` before its compressions); for measures supported on balls of radius r
around the endpoints of an edge, D = 2r + 1.

This module is the reference implementation of that statement: it builds the distances between
P and N explicitly (truncated BFS), lists every implication, and also returns certificates: the
potentials read off the source side of a minimum cut (a dual optimum) and a transport plan on the
tight pairs (a primal optimum), with an exact check that both have the same cost.

Measures (``kind``): 0 one-step lazy (uniform on N[u]), 1 one-step non-lazy, 2 uniform on the ball
B(u, r), 3 r-step lazy random walk, 4 r-step non-lazy random walk. Kinds 0-2 are exact rationals;
the r-step walks are rounded to fixed point with 2^40 units of mass (the error of W1 is at most
D (|B_i| + |B_j|) 2^-40).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._kernels import INF, _csr, _dinic, _take_i

K_LAZY = 0
K_NONLAZY = 1
K_BALL = 2
K_WALK_LAZY = 3
K_WALK = 4
FIX = np.int64(1) << np.int64(40)

N_LDIAG = 6  # |P|, |N|, nodes, arcs, D, certified (1 = plan and potentials verified, 0 = not requested)


@njit(cache=True, inline="always")
def _gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


@njit(cache=True, inline="always")
def _tick(ctr):
    """A fresh stamp value (per-thread counter; stamps start at -1, so values never repeat)."""
    ctr[0] += 1
    return ctr[0]


@njit(cache=True)
def _bfs(indptr, indices, src, maxd, stamp, ep, dist, order):
    """Vertices within distance maxd of src (BFS order in ``order``); dist valid where stamp == ep."""
    stamp[src] = ep
    dist[src] = 0
    order[0] = src
    qh = 0
    qt = 1
    while qh < qt:
        u = order[qh]
        qh += 1
        du = dist[u]
        if du == maxd:
            continue
        for e in range(indptr[u], indptr[u + 1]):
            w = indices[e]
            if stamp[w] != ep:
                stamp[w] = ep
                dist[w] = du + 1
                order[qt] = w
                qt += 1
    return qt


@njit(cache=True)
def _walk_measure(indptr, indices, u, r, lazy, stamp, ep, loc, order, buf, nxt):
    """r-step random-walk distribution from u (lazy: uniform on N[v] at every step), as float64
    on the vertices order[:n] of the ball B(u, r) (loc maps a vertex to its position)."""
    n = 0
    stamp[u] = ep
    loc[u] = 0
    order[0] = u
    n = 1
    buf[0] = 1.0
    front_lo = 0
    for step in range(r):
        front_hi = n
        for a in range(front_lo, front_hi):  # extend the support by one hop
            x = order[a]
            for e in range(indptr[x], indptr[x + 1]):
                w = indices[e]
                if stamp[w] != ep:
                    stamp[w] = ep
                    loc[w] = n
                    order[n] = w
                    buf[n] = 0.0
                    n += 1
        front_lo = front_hi
        for a in range(n):
            nxt[a] = 0.0
        for a in range(n):
            mass = buf[a]
            if mass == 0.0:
                continue
            x = order[a]
            d = indptr[x + 1] - indptr[x]
            if lazy:
                share = mass / (d + 1)
                nxt[a] += share
            else:
                share = mass / d
            for e in range(indptr[x], indptr[x + 1]):
                nxt[loc[indices[e]]] += share
        for a in range(n):
            buf[a] = nxt[a]
    return n


@njit(cache=True)
def _fixed(vals, n, out):
    """Round masses (summing to 1) to integers summing to FIX."""
    tot = np.int64(0)
    best = 0
    for a in range(n):
        out[a] = np.int64(vals[a] * FIX + 0.5)
        tot += out[a]
        if out[a] > out[best]:
            best = a
    out[best] += FIX - tot


@njit(cache=True)
def _layered_cut(nP, nN, Dm, D, wP, wN, want_cert, bi, oi):
    """W1 * scale for masses wP > 0 on P and wN > 0 (demands) on N, distances Dm (nP * nN, int64,
    entries in 1..D). Returns (W, nodes, arcs, certified, u, v, plan_x, plan_y, plan_mass, arena);
    certified = 1 if the plan on the tight pairs and the potentials of the minimum cut were
    verified (want_cert), 2 if verification failed, 0 if not requested. u (on P) and v (on N) are
    the integer potentials (u_x - v_y <= d(x, y), with equality on the support of the plan), and
    the plan moves plan_mass[k] from P[plan_x[k]] to N[plan_y[k]] (scaled masses)."""
    empty = np.empty(0, np.int64)
    A = np.int64(0)
    for x in range(nP):
        A += wP[x]
    L = D - 1
    nV = 2 + L * (nP + nN)
    # arcs: s -> P nodes, N nodes -> t, chains on both sides, implications
    nA = L * nP + L * nN + (L - 1) * (nP + nN)
    for x in range(nP):
        for y in range(nN):
            nA += D - Dm[x * nN + y]
    bi, oi, tl = _take_i(bi, oi, nA)
    bi, oi, hd = _take_i(bi, oi, nA)
    bi, oi, cp = _take_i(bi, oi, nA)
    # node ids: P_x level t (t = 2..D) -> 2 + x * L + (t - 2); N_y level s (s = 1..D-1) -> 2 + nP*L + y * L + (s - 1)
    m = 0
    for x in range(nP):
        for t in range(L):
            tl[m] = 0; hd[m] = 2 + x * L + t; cp[m] = wP[x]; m += 1
        for t in range(L - 1):
            tl[m] = 2 + x * L + t + 1; hd[m] = 2 + x * L + t; cp[m] = INF; m += 1
    nb = 2 + nP * L
    for y in range(nN):
        for s in range(L):
            tl[m] = nb + y * L + s; hd[m] = 1; cp[m] = wN[y]; m += 1
        for s in range(L - 1):
            tl[m] = nb + y * L + s + 1; hd[m] = nb + y * L + s; cp[m] = INF; m += 1
    for x in range(nP):
        for y in range(nN):
            d = Dm[x * nN + y]
            for t in range(d + 1, D + 1):  # [u_x >= t] => [v_y >= t - d]
                tl[m] = 2 + x * L + (t - 2); hd[m] = nb + y * L + (t - d - 1); cp[m] = INF; m += 1
    if m != nA:
        raise RuntimeError("internal error: layered network arc count")
    start, to, cap, rev, pos, bi, oi = _csr(nV, nA, tl, hd, cp, bi, oi)
    F, bi, oi = _dinic(nV, start, to, cap, rev, 0, 1, bi, oi)
    W = D * A - F
    if not want_cert:
        return W, nV, nA, 0, empty, empty, empty, empty, empty, bi, oi
    # ---- dual certificate: the source side of the minimum cut (residual reachability)
    bi, oi, seen = _take_i(bi, oi, nV)
    bi, oi, q = _take_i(bi, oi, nV)
    for v in range(nV):
        seen[v] = 0
    seen[0] = 1
    q[0] = 0
    qh = 0
    qt = 1
    while qh < qt:
        v = q[qh]
        qh += 1
        for a in range(start[v], start[v + 1]):
            if cap[a] > 0 and seen[to[a]] == 0:
                seen[to[a]] = 1
                q[qt] = to[a]
                qt += 1
    bi, oi, u = _take_i(bi, oi, nP)
    bi, oi, vv = _take_i(bi, oi, nN)
    for x in range(nP):
        u[x] = 1
        for t in range(L):
            u[x] += seen[2 + x * L + t]
    for y in range(nN):
        vv[y] = 0
        for s in range(L):
            vv[y] += seen[nb + y * L + s]
    ok = True
    dual = np.int64(0)
    for x in range(nP):
        dual += wP[x] * u[x]
        for y in range(nN):
            if u[x] - vv[y] > Dm[x * nN + y]:
                ok = False
    for y in range(nN):
        dual -= wN[y] * vv[y]
    if dual != W:
        ok = False
    # ---- primal certificate: a maximum flow on the tight pairs must carry all the mass
    nT = 0
    for x in range(nP):
        for y in range(nN):
            if u[x] - vv[y] == Dm[x * nN + y]:
                nT += 1
    nV2 = 2 + nP + nN
    nA2 = nP + nN + nT
    bi, oi, tl2 = _take_i(bi, oi, nA2)
    bi, oi, hd2 = _take_i(bi, oi, nA2)
    bi, oi, cp2 = _take_i(bi, oi, nA2)
    bi, oi, px = _take_i(bi, oi, nT)
    bi, oi, py = _take_i(bi, oi, nT)
    m = 0
    for x in range(nP):
        tl2[m] = 0; hd2[m] = 2 + x; cp2[m] = wP[x]; m += 1
    for y in range(nN):
        tl2[m] = 2 + nP + y; hd2[m] = 1; cp2[m] = wN[y]; m += 1
    k = 0
    for x in range(nP):
        for y in range(nN):
            if u[x] - vv[y] == Dm[x * nN + y]:
                tl2[m] = 2 + x; hd2[m] = 2 + nP + y; cp2[m] = INF; m += 1
                px[k] = x
                py[k] = y
                k += 1
    st2, to2, ca2, rv2, ps2, bi, oi = _csr(nV2, nA2, tl2, hd2, cp2, bi, oi)
    F2, bi, oi = _dinic(nV2, st2, to2, ca2, rv2, 0, 1, bi, oi)
    if F2 != A:
        ok = False
    # plan cost and marginals, exactly
    cost = np.int64(0)
    bi, oi, outP = _take_i(bi, oi, nP)
    bi, oi, inN = _take_i(bi, oi, nN)
    for x in range(nP):
        outP[x] = 0
    for y in range(nN):
        inN[y] = 0
    npl = 0
    for kk in range(nT):
        if ca2[rv2[ps2[nP + nN + kk]]] > 0:
            npl += 1
    plx = np.empty(npl, np.int64)
    ply = np.empty(npl, np.int64)
    plf = np.empty(npl, np.int64)
    npl = 0
    for kk in range(nT):
        f = ca2[rv2[ps2[nP + nN + kk]]]  # flow on the arc = residual capacity of its reverse
        if f > 0:
            cost += f * Dm[px[kk] * nN + py[kk]]
            outP[px[kk]] += f
            inN[py[kk]] += f
            plx[npl] = px[kk]
            ply[npl] = py[kk]
            plf[npl] = f
            npl += 1
    for x in range(nP):
        if outP[x] != wP[x]:
            ok = False
    for y in range(nN):
        if inN[y] != wN[y]:
            ok = False
    if cost != W:
        ok = False
    return W, nV, nA, 1 if ok else 2, u.copy(), vv.copy(), plx, ply, plf, bi, oi


@njit(cache=True)
def _edge_layered(indptr, indices, i, j, kind, r, want_cert, stamp, dist, order, loc, fbuf, fnxt,
                  ctr, bi, k, out, diag):
    """Curvature of edge (i, j) for measure ``kind`` with support radius r (r = 1 for kinds 0, 1)."""
    oi = 0
    # ---- the two measures on S = B(i, r) u B(j, r), as integers on a common scale
    if kind == K_WALK_LAZY or kind == K_WALK:
        lazy = kind == K_WALK_LAZY
        ni = _walk_measure(indptr, indices, i, r, lazy, stamp, _tick(ctr), loc, order, fbuf, fnxt)
        bi, oi, Si = _take_i(bi, oi, ni)
        bi, oi, mi = _take_i(bi, oi, ni)
        for a in range(ni):
            Si[a] = order[a]
        bi, oi, tmp = _take_i(bi, oi, ni)
        _fixed(fbuf, ni, tmp)
        for a in range(ni):
            mi[a] = tmp[a]
        nj = _walk_measure(indptr, indices, j, r, lazy, stamp, _tick(ctr), loc, order, fbuf, fnxt)
        bi, oi, Sj = _take_i(bi, oi, nj)
        bi, oi, mj = _take_i(bi, oi, nj)
        for a in range(nj):
            Sj[a] = order[a]
        bi, oi, tmp2 = _take_i(bi, oi, nj)
        _fixed(fbuf, nj, tmp2)
        for a in range(nj):
            mj[a] = tmp2[a]
        scale = FIX
    else:
        rr = 1 if kind != K_BALL else r
        ni = _bfs(indptr, indices, i, rr, stamp, _tick(ctr), dist, order)
        bi, oi, Si = _take_i(bi, oi, ni)
        for a in range(ni):
            Si[a] = order[a]
        nj = _bfs(indptr, indices, j, rr, stamp, _tick(ctr), dist, order)
        bi, oi, Sj = _take_i(bi, oi, nj)
        for a in range(nj):
            Sj[a] = order[a]
        ci = ni - 1 if kind == K_NONLAZY else ni
        cj = nj - 1 if kind == K_NONLAZY else nj
        g = _gcd(ci, cj)
        scale = (ci // g) * cj
        bi, oi, mi = _take_i(bi, oi, ni)
        bi, oi, mj = _take_i(bi, oi, nj)
        for a in range(ni):
            mi[a] = 0 if (kind == K_NONLAZY and Si[a] == i) else scale // ci
        for a in range(nj):
            mj[a] = 0 if (kind == K_NONLAZY and Sj[a] == j) else scale // cj
    # net mass mu = m_i - m_j on the union (loc maps vertices of S_i; stamp ep2 marks S_i)
    ep2 = _tick(ctr)
    for a in range(ni):
        stamp[Si[a]] = ep2
        loc[Si[a]] = a
    bi, oi, net_i = _take_i(bi, oi, ni)
    for a in range(ni):
        net_i[a] = mi[a]
    bi, oi, onlyj = _take_i(bi, oi, nj)
    nonly = 0
    for b in range(nj):
        y = Sj[b]
        if stamp[y] == ep2:
            net_i[loc[y]] -= mj[b]
        else:
            onlyj[nonly] = b
            nonly += 1
    nP = 0
    nN = 0
    bi, oi, P = _take_i(bi, oi, ni + nj)
    bi, oi, wP = _take_i(bi, oi, ni + nj)
    bi, oi, Nn = _take_i(bi, oi, ni + nj)
    bi, oi, wN = _take_i(bi, oi, ni + nj)
    for a in range(ni):
        if net_i[a] > 0:
            P[nP] = Si[a]; wP[nP] = net_i[a]; nP += 1
        elif net_i[a] < 0:
            Nn[nN] = Si[a]; wN[nN] = -net_i[a]; nN += 1
    for c in range(nonly):
        b = onlyj[c]
        if mj[b] > 0:
            Nn[nN] = Sj[b]; wN[nN] = mj[b]; nN += 1
    diag[k, 0] = nP
    diag[k, 1] = nN
    rr = 1 if (kind == K_LAZY or kind == K_NONLAZY) else r
    D = 2 * rr + 1
    diag[k, 4] = D
    if nP == 0:  # m_i = m_j: W1 = 0, certified by the empty plan and the zero potential
        out[k] = 1.0
        diag[k, 5] = 1 if want_cert else 0
        return bi
    # ---- distances P x N, by BFS from each vertex of P truncated at depth D - 1
    bi, oi, Dm = _take_i(bi, oi, nP * nN)
    for x in range(nP):
        epx = _tick(ctr)
        _bfs(indptr, indices, P[x], D - 1, stamp, epx, dist, order)
        for y in range(nN):
            yv = Nn[y]
            Dm[x * nN + y] = dist[yv] if stamp[yv] == epx else D
    Wd, nV, nA, cert, _u, _v, _px, _py, _pf, bi, oi = _layered_cut(nP, nN, Dm, D, wP, wN, want_cert, bi, oi)
    diag[k, 2] = nV
    diag[k, 3] = nA
    diag[k, 5] = cert
    out[k] = 1.0 - Wd / scale
    return bi


@njit(cache=True)
def _layered_chunk(indptr, indices, edges, ids, c, nch, kind, r, want_cert, out, diag):
    n = indptr.shape[0] - 1
    stamp = np.full(n, -1, np.int64)
    dist = np.zeros(n, np.int64)
    order = np.empty(n, np.int64)
    loc = np.zeros(n, np.int64)
    fbuf = np.zeros(n, np.float64)
    fnxt = np.zeros(n, np.float64)
    bi = np.empty(1 << 12, np.int64)
    ctr = np.zeros(1, np.int64)
    K = ids.shape[0]
    for k in range(c, K, nch):
        e = ids[k]
        bi = _edge_layered(indptr, indices, edges[e, 0], edges[e, 1], kind, r, want_cert, stamp, dist, order,
                           loc, fbuf, fnxt, ctr, bi, k, out, diag)


@njit(cache=True, parallel=True)
def _layered_run(indptr, indices, edges, ids, kind, r, want_cert, nchunks):
    K = ids.shape[0]
    out = np.full(K, np.nan)
    diag = np.zeros((K, N_LDIAG), np.int64)
    if K == 0:
        return out, diag
    nchunks = max(1, min(nchunks, K))
    for c in prange(nchunks):
        _layered_chunk(indptr, indices, edges, ids, c, nchunks, kind, r, want_cert, out, diag)
    return out, diag


def layered_ids(indptr, indices, edges, ids, kind, r=1, certify=False, nchunks=None):
    """Curvatures of edges[ids] by the layered closure (reference implementation; see the module
    docstring for ``kind``). Returns (values, diagnostics (K, N_LDIAG))."""
    from ._kernels import _ids, _nchunks
    if kind in (K_LAZY, K_NONLAZY):
        r = 1
    if r < 1:
        raise ValueError("radius must be at least 1")
    return _layered_run(indptr, indices, edges, _ids(ids), int(kind), int(r), bool(certify), _nchunks(nchunks))
