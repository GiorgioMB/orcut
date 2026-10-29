"""Generalized one-step kernel (orcut >= 1.1).

The same reduction as :mod:`orcut._kernels` (frozen at 1.0.0), for every pair of
one-step measures m_i on N[i] and m_j on N[j] given by integer masses:

* ``M_LAZY``    m_u uniform on the closed neighbourhood (idleness 1/(deg+1));
* ``M_NONLAZY`` m_u uniform on the open neighbourhood;
* ``M_IDLE``    uniform idleness p = pn/pd: mass p at u, (1-p)/deg on each neighbour;
* ``M_LLY``     ``M_IDLE`` at p = 1/(deg_max + 1), rescaled by 1/(1-p): the
                Lin--Lu--Yau curvature (idleness linearity of Bourne et al., 2018);
* ``M_WALK``    a weighted lazy walk: rational idleness a_u/b_u at u and the rest
                split over N(u) in proportion to positive integer edge weights.

The cost is the hop distance of the graph, so every pair of S = N[i] u N[j] is
at distance at most 2 except pairs of U_i x U_j, which are at distance 1, 2, or
3; the measures only enter through the capacities of the closure network.

Differences from the 1.0.0 kernel: masses are per vertex (arrays), the TLW
certificate is not computed, every engineering step can be switched off
(``F_TWINS``, ``F_RELAYS``, ``F_FAST``, ``F_WARM``, for ablations; values do not
depend on the flags), and every edge reports diagnostics (``N_DIAG`` columns)
and its exact value as an integer fraction W1 = num / den.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._kernels import (INF, U0, U1, _BLOCK_BUDGET, _WARM_MIN_ARCS, _csr, _ctz, _dedupe, _dinic, _maxflow,
                       _popcount, _setbit, _take_i, _take_u)

M_LAZY = 0
M_NONLAZY = 1
M_IDLE = 2
M_LLY = 3
M_WALK = 4

F_TWINS = 1    # merge rows with identical adjacency and coverage patterns
F_RELAYS = 2   # route implications of whole blocks of pairs within distance 2 through hub nodes
F_FAST = 4     # W1 = 2A - maxflow(G1) when no pair is at distance 3
F_WARM = 8     # warm-start the two-layer network with a maximum flow of G1
F_ALL = 15

N_DIAG = 8
D_PATH = 0      # PATH_* below
D_ROWS = 1      # vertices of positive net mass (P)
D_GROUPS = 2    # row nodes after twin merging
D_COLS = 3      # vertices of negative net mass (N)
D_NODES = 4     # nodes of the network that was solved
D_ARCS = 5      # arcs of the network that was solved
D_WARM = 6      # value of the warm-start flow (scaled masses)
D_STATUS = 7    # 0 ok, 1 masses do not fit in 64-bit integers (value is NaN)

PATH_ZERO = 0   # m_i = m_j
PATH_NOADJ = 1  # no pair of P x N is adjacent and none is at distance 3: W1 = 2A
PATH_FAST = 2   # one bipartite max flow (no pair at distance 3)
PATH_CUT = 3    # two-layer closure network

_LMAX = np.int64(1) << np.int64(50)   # common denominators up to 2^50 keep W1 / den exact in float64


@njit(cache=True, inline="always")
def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b != 0:
        a, b = b, a % b
    return a


# ------------------------------------------------- one measure on one edge


@njit(cache=True)
def _measure_arr(Adj, Cov, fullr, colfull, nUi, nUj, nT, W, rm, cm, flags, bi, oi, bu, ou, diag):
    """W1 (scaled by the common denominator of the masses) for net masses rm on the
    rows (U_i, then i, j, triangle vertices) and demands cm = -(net mass) on the
    columns (U_j, then i, j, triangle vertices); a vertex is in P where its row
    mass is positive and in N where its column demand is positive. Returns
    (W1, arenas); fills ``diag``."""
    nC = 2 + nT
    nr = nUi + nC
    nc = nUj + nC
    twins = (flags & F_TWINS) != 0
    relays = (flags & F_RELAYS) != 0
    fast = (flags & F_FAST) != 0
    warm_on = (flags & F_WARM) != 0
    for z in range(N_DIAG - 1):
        diag[z] = 0
    A = np.int64(0)
    nP = 0
    for r in range(nr):
        if rm[r] > 0:
            A += rm[r]
            nP += 1
    diag[D_ROWS] = nP
    if A == 0:
        diag[D_PATH] = PATH_ZERO
        return np.int64(0), bi, bu
    bu, ou, Nm = _take_u(bu, ou, W)  # columns in N
    for x in range(W):
        Nm[x] = U0
    nN = 0
    nNC = 0
    for k in range(nc):
        if cm[k] > 0:
            _setbit(Nm, 0, k)
            nN += 1
            if k >= nUj:
                nNC += 1
    nPC = 0
    for r in range(nUi, nr):
        if rm[r] > 0:
            nPC += 1
    diag[D_COLS] = nN

    nfull = 0
    for r in range(nUi):
        nfull += fullr[r]

    if fast and nfull == nUi:
        # -------- fast path: every P x N pair is within distance 2
        bu, ou, keys = _take_u(bu, ou, nr * W)
        bi, oi, rid = _take_i(bi, oi, nr)
        ncand = 0
        for r in range(nr):
            if rm[r] <= 0:
                continue
            nz = False
            for x in range(W):
                z = Adj[r * W + x] & Nm[x]
                keys[ncand * W + x] = z
                if z != U0:
                    nz = True
            if nz:
                rid[ncand] = r
                ncand += 1
        if ncand == 0:
            diag[D_PATH] = PATH_NOADJ
            return 2 * A, bi, bu
        if twins:
            gid, ng, leader, bi, oi = _dedupe(keys, ncand, W, bi, oi)
        else:
            bi, oi, gid = _take_i(bi, oi, ncand)
            bi, oi, leader = _take_i(bi, oi, ncand)
            for c in range(ncand):
                gid[c] = c
                leader[c] = c
            ng = ncand
        bi, oi, gm = _take_i(bi, oi, ng)
        for g in range(ng):
            gm[g] = 0
        for c in range(ncand):
            gm[gid[c]] += rm[rid[c]]
        bi, oi, cnode = _take_i(bi, oi, nc)
        for k in range(nc):
            cnode[k] = -1
        nV = 2 + ng
        nA = ng
        for g in range(ng):
            lb = leader[g] * W
            for x in range(W):
                z = keys[lb + x]
                while z != U0:
                    k = x * 64 + _ctz(z)
                    if cnode[k] < 0:
                        cnode[k] = nV
                        nV += 1
                        nA += 1
                    nA += 1
                    z &= z - U1
        bi, oi, tl = _take_i(bi, oi, nA)
        bi, oi, hd = _take_i(bi, oi, nA)
        bi, oi, cp = _take_i(bi, oi, nA)
        m = 0
        for g in range(ng):
            tl[m] = 0
            hd[m] = 2 + g
            cp[m] = gm[g]
            m += 1
            lb = leader[g] * W
            for x in range(W):
                z = keys[lb + x]
                while z != U0:
                    k = x * 64 + _ctz(z)
                    tl[m] = 2 + g
                    hd[m] = cnode[k]
                    cp[m] = INF
                    m += 1
                    z &= z - U1
        for k in range(nc):
            if cnode[k] >= 0:
                tl[m] = cnode[k]
                hd[m] = 1
                cp[m] = cm[k]
                m += 1
        if m != nA:
            raise RuntimeError("internal error: bipartite network arc count")
        F, bi, oi = _maxflow(nV, nA, tl, hd, cp, 0, 1, bi, oi)
        diag[D_PATH] = PATH_FAST
        diag[D_GROUPS] = ng
        diag[D_NODES] = nV
        diag[D_ARCS] = nA
        return 2 * A - F, bi, bu

    # -------- general path: token network (p, q per row; r, t per column)
    KW = 2 * W + 1
    bu, ou, keys = _take_u(bu, ou, nr * KW)
    bi, oi, rid = _take_i(bi, oi, nr)
    ncand = 0
    for r in range(nr):
        if rm[r] <= 0:
            continue
        b0 = ncand * KW
        for x in range(W):
            keys[b0 + x] = Adj[r * W + x] & Nm[x]
        if r < nUi:
            for x in range(W):
                keys[b0 + W + x] = Cov[r * W + x]
            keys[b0 + 2 * W] = U0
        else:
            for x in range(W):
                keys[b0 + W + x] = U0
            keys[b0 + 2 * W] = U1
        rid[ncand] = r
        ncand += 1
    if twins:
        gid, ng, leader, bi, oi = _dedupe(keys, ncand, KW, bi, oi)
    else:
        bi, oi, gid = _take_i(bi, oi, ncand)
        bi, oi, leader = _take_i(bi, oi, ncand)
        for c in range(ncand):
            gid[c] = c
            leader[c] = c
        ng = ncand
    bi, oi, gm = _take_i(bi, oi, ng)
    bi, oi, gkind = _take_i(bi, oi, ng)  # 0 U_i row, 1 full U_i row, 2 C row
    for g in range(ng):
        gm[g] = 0
        L = leader[g]
        if keys[L * KW + 2 * W] == U1:
            gkind[g] = 2
        else:
            gkind[g] = 1 if fullr[rid[L]] == 1 else 0
    for c in range(ncand):
        gm[gid[c]] += rm[rid[c]]
    bi, oi, cidx = _take_i(bi, oi, nc)
    nN = 0
    for k in range(nc):
        if cm[k] > 0:
            cidx[k] = nN
            nN += 1
        else:
            cidx[k] = -1
    n_rowfull = 0
    n_rowpart = 0
    n_rowC = 0
    for g in range(ng):
        if gkind[g] == 1:
            n_rowfull += 1
        elif gkind[g] == 0:
            n_rowpart += 1
        else:
            n_rowC += 1
    n_colfull = 0
    for x in range(W):
        n_colfull += _popcount(colfull[x])
    use_h4 = n_colfull > 0 and n_rowpart > 0
    # nodes: s=0, t=1, hubs 2..5, rows (p, q), columns (r, t)
    H1 = 2  # every P row -> columns of N in C
    H2 = 3  # rows of P in C -> every column
    H3 = 4  # fully covered U_i rows -> every U_j column
    H4 = 5  # every partial U_i row -> fully covered U_j columns
    cbase = 6 + 2 * ng
    nV = cbase + 2 * nN
    nA = 3 * ng + 2 * nN
    for g in range(ng):
        lb = leader[g] * KW
        for x in range(W):
            nA += 2 * _popcount(keys[lb + x])
        if relays:
            if gkind[g] == 0:
                for x in range(W):
                    nA += _popcount(keys[lb + W + x] & ~keys[lb + x] & ~colfull[x])
        else:
            for k in range(nc):  # one arc q -> r per non-adjacent pair within distance 2
                if cidx[k] < 0 or ((keys[lb + (k >> 6)] >> np.uint64(k & 63)) & U1) != U0:
                    continue
                if (gkind[g] != 0 or k >= nUj
                        or ((keys[lb + W + (k >> 6)] >> np.uint64(k & 63)) & U1) != U0):
                    nA += 1
    if relays:
        if nNC > 0:
            nA += ng + nNC
        if nPC > 0:
            nA += n_rowC + nN
        if n_rowfull > 0:
            nA += n_rowfull + nUj
        if use_h4:
            nA += n_rowpart + n_colfull
    # warm start: a maximum flow on the adjacency G1 alone (one layer, small),
    # loaded on both layers (p -> r and q -> t), is a feasible token flow of
    # value 2 F1; on small networks the extra flow costs more than it saves.
    warm = warm_on and nA >= _WARM_MIN_ARCS
    nG1 = 0
    for g in range(ng):
        lb = leader[g] * KW
        for x in range(W):
            nG1 += _popcount(keys[lb + x])
    nV1 = 2 + ng + nN if warm else 2  # an empty stub when not warm
    nA1 = ng + nG1 + nN if warm else 0
    bi, oi, tl1 = _take_i(bi, oi, nA1)
    bi, oi, hd1 = _take_i(bi, oi, nA1)
    bi, oi, cp1 = _take_i(bi, oi, nA1)
    if warm:
        m = 0
        for g in range(ng):
            tl1[m] = 0; hd1[m] = 2 + g; cp1[m] = gm[g]; m += 1
        for g in range(ng):
            lb = leader[g] * KW
            for x in range(W):
                z = keys[lb + x]
                while z != U0:
                    tl1[m] = 2 + g; hd1[m] = 2 + ng + cidx[x * 64 + _ctz(z)]; cp1[m] = INF; m += 1
                    z &= z - U1
        for k in range(nc):
            if cidx[k] >= 0:
                tl1[m] = 2 + ng + cidx[k]; hd1[m] = 1; cp1[m] = cm[k]; m += 1
    st1, to1, ca1, rv1, ps1, bi, oi = _csr(nV1, nA1, tl1, hd1, cp1, bi, oi)
    F1 = np.int64(0)
    if warm:
        F1, bi, oi = _dinic(nV1, st1, to1, ca1, rv1, 0, 1, bi, oi)

    bi, oi, tl = _take_i(bi, oi, nA)
    bi, oi, hd = _take_i(bi, oi, nA)
    bi, oi, cp = _take_i(bi, oi, nA)
    bi, oi, tsp = _take_i(bi, oi, ng)    # token index of s -> p_g (s -> q_g follows)
    bi, oi, tg1 = _take_i(bi, oi, nG1)   # token index of p_g -> r_c (q_g -> t_c follows)
    bi, oi, tcol = _take_i(bi, oi, nN)   # token index of r_c -> t (t_c -> t follows)
    m = 0
    jj = 0
    for g in range(ng):
        pn_ = 6 + 2 * g
        qn = pn_ + 1
        tsp[g] = m
        tl[m] = 0; hd[m] = pn_; cp[m] = gm[g]; m += 1
        tl[m] = 0; hd[m] = qn; cp[m] = gm[g]; m += 1
        tl[m] = qn; hd[m] = pn_; cp[m] = INF; m += 1
        lb = leader[g] * KW
        for x in range(W):
            z = keys[lb + x]
            while z != U0:  # adjacent pairs: both implications
                c = cidx[x * 64 + _ctz(z)]
                tg1[jj] = m
                jj += 1
                tl[m] = pn_; hd[m] = cbase + 2 * c; cp[m] = INF; m += 1
                tl[m] = qn; hd[m] = cbase + 2 * c + 1; cp[m] = INF; m += 1
                z &= z - U1
            if relays and gkind[g] == 0:  # covered, non-adjacent, column not fully covered
                z = keys[lb + W + x] & ~keys[lb + x] & ~colfull[x]
                while z != U0:
                    c = cidx[x * 64 + _ctz(z)]
                    tl[m] = qn; hd[m] = cbase + 2 * c; cp[m] = INF; m += 1
                    z &= z - U1
        if relays:
            if nNC > 0:
                tl[m] = qn; hd[m] = H1; cp[m] = INF; m += 1
            if gkind[g] == 2 and nPC > 0:
                tl[m] = qn; hd[m] = H2; cp[m] = INF; m += 1
            if gkind[g] == 1:
                tl[m] = qn; hd[m] = H3; cp[m] = INF; m += 1
            if gkind[g] == 0 and use_h4:
                tl[m] = qn; hd[m] = H4; cp[m] = INF; m += 1
        else:
            for k in range(nc):
                if cidx[k] < 0 or ((keys[lb + (k >> 6)] >> np.uint64(k & 63)) & U1) != U0:
                    continue
                if (gkind[g] != 0 or k >= nUj
                        or ((keys[lb + W + (k >> 6)] >> np.uint64(k & 63)) & U1) != U0):
                    tl[m] = qn; hd[m] = cbase + 2 * cidx[k]; cp[m] = INF; m += 1
    for k in range(nc):
        c = cidx[k]
        if c < 0:
            continue
        rn = cbase + 2 * c
        tn = rn + 1
        mk = cm[k]
        tcol[c] = m
        tl[m] = rn; hd[m] = 1; cp[m] = mk; m += 1
        tl[m] = tn; hd[m] = 1; cp[m] = mk; m += 1
        # no t -> r arc: flow q -> t -> r can always be rerouted q -> p -> r
        if relays:
            if nNC > 0 and k >= nUj:
                tl[m] = H1; hd[m] = rn; cp[m] = INF; m += 1
            if nPC > 0:
                tl[m] = H2; hd[m] = rn; cp[m] = INF; m += 1
            if n_rowfull > 0 and k < nUj:
                tl[m] = H3; hd[m] = rn; cp[m] = INF; m += 1
            if use_h4 and k < nUj and ((colfull[k >> 6] >> np.uint64(k & 63)) & U1) != U0:
                tl[m] = H4; hd[m] = rn; cp[m] = INF; m += 1
    if m != nA or jj != nG1:
        raise RuntimeError("internal error: token network arc count")
    start, to, cap, rev, pos, bi, oi = _csr(nV, nA, tl, hd, cp, bi, oi)
    for g in range(ng if warm else 0):  # load the G1 flow on both layers
        f = ca1[rv1[ps1[g]]]
        if f > 0:
            for d in range(2):
                e = pos[tsp[g] + d]
                cap[e] -= f
                cap[rev[e]] += f
    for x in range(nG1 if warm else 0):
        f = ca1[rv1[ps1[ng + x]]]
        if f > 0:
            for d in range(2):
                e = pos[tg1[x] + d]
                cap[e] -= f
                cap[rev[e]] += f
    for c in range(nN if warm else 0):
        f = ca1[rv1[ps1[ng + nG1 + c]]]
        if f > 0:
            for d in range(2):
                e = pos[tcol[c] + d]
                cap[e] -= f
                cap[rev[e]] += f
    F2, bi, oi = _dinic(nV, start, to, cap, rev, 0, 1, bi, oi)
    diag[D_PATH] = PATH_CUT
    diag[D_GROUPS] = ng
    diag[D_NODES] = nV
    diag[D_ARCS] = nA
    diag[D_WARM] = 2 * F1
    return 3 * A - (2 * F1 + F2), bi, bu


# --------------------------------------------------------- one edge


@njit(cache=True)
def _edge_core(indptr, indices, wts, Dp, nself, coef, gdv, i0, j0, mode, pn, pd, flags,
               slot, sstamp, epoch, bi, bu, k, out, num, diag, budget):
    """The curvature of edge (i0, j0) for measure ``mode``, written into out[k]
    (float), num[k] = (W1 numerator, denominator) and diag[k]. ``wts`` are the
    integer weights of the directed CSR entries (``M_WALK`` only); ``Dp``,
    ``nself``, ``coef``, ``gdv`` the per-vertex reduced denominators, self
    numerators, idleness complements, and divisors of the walk (see
    :func:`orcut.measures.walk_tables`). Returns the (possibly grown) arenas."""
    # orient so that the larger neighbourhood is on the row side
    if indptr[i0 + 1] - indptr[i0] >= indptr[j0 + 1] - indptr[j0]:
        i = i0
        j = j0
    else:
        i = j0
        j = i0
    di = np.int64(indptr[i + 1] - indptr[i])
    dj = np.int64(indptr[j + 1] - indptr[j])
    walk = mode == M_WALK
    oi = 0
    ou = 0
    bi, oi, Ui = _take_i(bi, oi, di)
    bi, oi, Uj = _take_i(bi, oi, dj)
    bi, oi, Tt = _take_i(bi, oi, dj)
    bi, oi, wUi = _take_i(bi, oi, di if walk else 0)
    bi, oi, wUj = _take_i(bi, oi, dj if walk else 0)
    bi, oi, wTi = _take_i(bi, oi, dj if walk else 0)
    bi, oi, wTj = _take_i(bi, oi, dj if walk else 0)
    w_ij = np.int64(0)
    w_ji = np.int64(0)
    nUi = 0
    nUj = 0
    nT = 0
    p = indptr[i]
    pe = indptr[i + 1]
    q = indptr[j]
    qe = indptr[j + 1]
    while p < pe or q < qe:
        if q >= qe or (p < pe and indices[p] < indices[q]):
            x = indices[p]
            if x != j:
                Ui[nUi] = x
                if walk:
                    wUi[nUi] = wts[p]
                nUi += 1
            elif walk:
                w_ij = wts[p]
            p += 1
        elif p >= pe or indices[q] < indices[p]:
            y = indices[q]
            if y != i:
                Uj[nUj] = y
                if walk:
                    wUj[nUj] = wts[q]
                nUj += 1
            elif walk:
                w_ji = wts[q]
            q += 1
        else:
            Tt[nT] = indices[p]
            if walk:
                wTi[nT] = wts[p]
                wTj[nT] = wts[q]
            nT += 1
            p += 1
            q += 1
    nC = 2 + nT
    nr = nUi + nC
    nc = nUj + nC
    W = (nc + 63) >> 6
    bi, oi, colv = _take_i(bi, oi, nc)
    bi, oi, rowv = _take_i(bi, oi, nr)
    for t in range(nUj):
        colv[t] = Uj[t]
    for r in range(nUi):
        rowv[r] = Ui[r]
    colv[nUj] = i
    colv[nUj + 1] = j
    rowv[nUi] = i
    rowv[nUi + 1] = j
    for t in range(nT):
        colv[nUj + 2 + t] = Tt[t]
        rowv[nUi + 2 + t] = Tt[t]

    # ---- the common denominator and the integer net masses
    bi, oi, rm = _take_i(bi, oi, nr)
    bi, oi, cm = _take_i(bi, oi, nc)
    dmax = di if di > dj else dj
    status = 0
    if mode == M_LAZY:
        L = (di + 1) * (dj + 1)
        mi_s = dj + 1
        mi_n = dj + 1
        mj_s = di + 1
        mj_n = di + 1
    elif mode == M_NONLAZY:
        L = di * dj
        mi_s = np.int64(0)
        mi_n = dj
        mj_s = np.int64(0)
        mj_n = di
    elif mode == M_IDLE or mode == M_LLY:
        a = np.int64(pn)
        b = np.int64(pd)
        if mode == M_LLY:
            a = np.int64(1)
            b = dmax + 1
        if b > _LMAX // max(di * dj, 1):
            status = 1
            L = np.int64(1)
        else:
            L = b * di * dj
        mi_s = a * di * dj
        mi_n = (b - a) * dj
        mj_s = a * di * dj
        mj_n = (b - a) * di
    else:
        Di = Dp[i]
        Dj = Dp[j]
        g = _gcd(Di, Dj)
        if Di // g > _LMAX // Dj:
            status = 1
            L = np.int64(1)
        else:
            L = (Di // g) * Dj
        mi_s = np.int64(0)
        mi_n = np.int64(0)
        mj_s = np.int64(0)
        mj_n = np.int64(0)
    if status != 0:
        diag[k, D_STATUS] = 1
        out[k] = np.nan
        num[k, 0] = -1
        num[k, 1] = 0
        return bi, bu
    if not walk:
        for r in range(nUi):
            rm[r] = mi_n
        for t in range(nUj):
            cm[t] = mj_n
        rm[nUi] = mi_s - mj_n          # vertex i: m_i(i) - m_j(i)
        rm[nUi + 1] = mi_n - mj_s      # vertex j: m_i(j) - m_j(j)
        for t in range(nT):
            rm[nUi + 2 + t] = mi_n - mj_n
    else:
        si = L // Dp[i]
        sj = L // Dp[j]
        ci = coef[i]
        cj = coef[j]
        gi = gdv[i]
        gj = gdv[j]
        for r in range(nUi):
            rm[r] = (ci * wUi[r] // gi) * si
        for t in range(nUj):
            cm[t] = (cj * wUj[t] // gj) * sj
        rm[nUi] = nself[i] * si - (cj * w_ji // gj) * sj
        rm[nUi + 1] = (ci * w_ij // gi) * si - nself[j] * sj
        for t in range(nT):
            rm[nUi + 2 + t] = (ci * wTi[t] // gi) * si - (cj * wTj[t] // gj) * sj
    for t in range(nC):
        cm[nUj + t] = -rm[nUi + t]

    # ---- distance-free local structure (as in orcut 1.0.0)
    ns = 0
    for kk in range(nc):
        bb = colv[kk]
        for e in range(indptr[bb], indptr[bb + 1]):
            w = indices[e]
            if sstamp[w] != epoch:
                sstamp[w] = epoch
                slot[w] = ns
                ns += 1
    WB = W
    if ns * W > budget:
        WB = max(1, budget // max(ns, 1))

    bu, ou, UjMask = _take_u(bu, ou, W)
    for x in range(W):
        lo = x * 64
        if lo + 64 <= nUj:
            UjMask[x] = ~U0
        elif lo < nUj:
            UjMask[x] = (U1 << np.uint64(nUj - lo)) - U1
        else:
            UjMask[x] = U0
    bu, ou, Adj = _take_u(bu, ou, nr * W)  # columns adjacent to each row
    bu, ou, Cov = _take_u(bu, ou, nUi * W)  # U_j columns within distance 2
    bu, ou, R = _take_u(bu, ou, ns * WB)  # columns adjacent to each middle vertex
    for b0 in range(0, W, WB):
        wb = min(WB, W - b0)
        for z in range(ns * WB):
            R[z] = U0
        k_lo = b0 * 64
        k_hi = min(nc, (b0 + wb) * 64)
        for kk in range(k_lo, k_hi):
            k2 = kk - k_lo
            wq = k2 >> 6
            bit = U1 << np.uint64(k2 & 63)
            bb = colv[kk]
            for e in range(indptr[bb], indptr[bb + 1]):
                R[slot[indices[e]] * WB + wq] |= bit
        for r in range(nr):
            av = rowv[r]
            base = r * W + b0
            if sstamp[av] == epoch:
                sb = slot[av] * WB
                for x in range(wb):
                    Adj[base + x] = R[sb + x]
            else:
                for x in range(wb):
                    Adj[base + x] = U0
        if k_lo >= nUj:
            for r in range(nUi):
                for x in range(wb):
                    Cov[r * W + b0 + x] = U0
            continue
        for r in range(nUi):
            cb = r * W + b0
            done = True
            for x in range(wb):
                Cov[cb + x] = Adj[cb + x]
                if (Cov[cb + x] & UjMask[b0 + x]) != UjMask[b0 + x]:
                    done = False
            if done:
                continue
            xv = rowv[r]
            for e in range(indptr[xv], indptr[xv + 1]):
                w = indices[e]
                if sstamp[w] != epoch:
                    continue
                sb = slot[w] * WB
                done = True
                for x in range(wb):
                    c = Cov[cb + x] | R[sb + x]
                    Cov[cb + x] = c
                    if (c & UjMask[b0 + x]) != UjMask[b0 + x]:
                        done = False
                if done:
                    break
    bi, oi, fullr = _take_i(bi, oi, nUi)
    bu, ou, colfull = _take_u(bu, ou, W)
    for x in range(W):
        colfull[x] = UjMask[x] if nUi > 0 else U0
    for r in range(nUi):
        f = 1
        for x in range(W):
            c = Cov[r * W + x] & UjMask[x]
            Cov[r * W + x] = c
            colfull[x] &= c
            if c != UjMask[x]:
                f = 0
        fullr[r] = f

    Wd, bi, bu = _measure_arr(Adj, Cov, fullr, colfull, nUi, nUj, nT, W, rm, cm, flags,
                              bi, oi, bu, ou, diag[k])
    num[k, 0] = Wd
    num[k, 1] = L
    if mode == M_LLY:
        kp = 1.0 - Wd / L
        out[k] = kp * (dmax + 1) / dmax
    else:
        out[k] = 1.0 - Wd / L
    return bi, bu


# ------------------------------------------------------------- drivers

_RUN_BLOCK = 16


@njit(cache=True)
def _core_chunk(indptr, indices, wts, Dp, nself, coef, gdv, edges, ids, c, nch, mode, pn, pd, flags,
                out, num, diag, budget):
    n = indptr.shape[0] - 1
    slot = np.empty(n, np.int64)
    sstamp = np.full(n, -1, np.int64)
    bi = np.empty(1 << 12, np.int64)
    bu = np.empty(1 << 10, np.uint64)
    K = ids.shape[0]
    nb = (K + _RUN_BLOCK - 1) // _RUN_BLOCK
    for bk in range(c, nb, nch):
        lo = bk * _RUN_BLOCK
        hi = min(K, lo + _RUN_BLOCK)
        for k in range(lo, hi):
            e = ids[k]
            bi, bu = _edge_core(indptr, indices, wts, Dp, nself, coef, gdv, edges[e, 0], edges[e, 1], mode,
                                pn, pd, flags, slot, sstamp, k, bi, bu, k, out, num, diag, budget)


@njit(cache=True, parallel=True)
def _core_run(indptr, indices, wts, Dp, nself, coef, gdv, edges, ids, mode, pn, pd, flags, nchunks,
              budget=_BLOCK_BUDGET):
    K = ids.shape[0]
    out = np.full(K, np.nan)
    num = np.zeros((K, 2), np.int64)
    diag = np.zeros((K, N_DIAG), np.int64)
    if K == 0:
        return out, num, diag
    nb = (K + _RUN_BLOCK - 1) // _RUN_BLOCK
    nchunks = max(1, min(nchunks, nb))
    for c in prange(nchunks):
        _core_chunk(indptr, indices, wts, Dp, nself, coef, gdv, edges, ids, c, nchunks, mode, pn, pd, flags,
                    out, num, diag, budget)
    return out, num, diag


_DUMMY_I = np.zeros(1, np.int64)


def core_ids(indptr, indices, edges, ids, mode, pn=0, pd=1, flags=F_ALL, nchunks=None, walk=None,
             budget=_BLOCK_BUDGET):
    """Curvatures of edges[ids] for measure ``mode`` (``M_*``): returns
    (values (K,), exact W1 as (numerator, denominator) (K, 2), diagnostics (K, N_DIAG)).
    ``walk`` = (wts, Dp, nself, coef, gdv) for ``M_WALK`` (see :func:`orcut.measures.walk_tables`)."""
    from ._kernels import _ids, _nchunks
    if mode == M_WALK:
        if walk is None:
            raise ValueError("M_WALK needs the walk tables")
        wts, Dp, nself, coef, gdv = (np.ascontiguousarray(a, dtype=np.int64) for a in walk)
    else:
        wts = Dp = nself = coef = gdv = _DUMMY_I
    return _core_run(indptr, indices, wts, Dp, nself, coef, gdv, edges, _ids(ids), int(mode), int(pn), int(pd),
                     int(flags), _nchunks(nchunks), budget)
