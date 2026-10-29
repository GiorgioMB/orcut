"""Exact one-step curvature on a linear-size network: one relay per common neighbour.

The closure network of :mod:`orcut._core` needs, for every pair (x, y) of P x N at distance 2,
the implication [u_x >= 3] => [v_y >= 1]. A pair is at distance 2 exactly when it is not adjacent
and has a common neighbour w, and all pairs in (N(w) & P) x (N(w) & N) are within distance 2. So
the implications can be routed through one zero-weight relay node H_w per *witness* w (a vertex
adjacent to some vertex of P and some vertex of N), with arcs a_{x,3} -> H_w for x in N(w) & P and
H_w -> b_{y,1} for y in N(w) & N (Proposition "relays" of the paper, applied once per witness).
Adjacent pairs keep their two arcs a_{x,2} -> b_{y,1} and a_{x,3} -> b_{y,2}.

The network then has 2(|P| + |N|) + |witnesses| + 2 nodes and at most
3|P| + 2|N| + 2 e_1 + sum_{x in P u N} deg(x) arcs, and it is built in one pass over the adjacency
lists of P u N, with no bitsets, no coverage test and no list of pairs at distance 2: its size is
linear in the part of the graph that the curvature depends on. W1 = 3 A - maxflow, as in
:mod:`orcut._core`.

Rows (vertices of P) with the same adjacent columns and the same witnesses are merged (their
keys are the sorted codes 2 v + [v is a witness] of their relevant neighbours, which come out of
the sorted adjacency lists already in order); a witness that reaches a single row class or a single
column is replaced by direct arcs. Masses and orientation are those of :mod:`orcut._core`, so the
exact fractions are identical (tested).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._kernels import INF, _csr, _dinic, _take_i
from ._base import M_IDLE, M_LAZY, M_LLY, M_NONLAZY, M_WALK, _gcd, _LMAX

W_TWINS = 1    # merge rows with the same relevant neighbourhood
W_DIRECT = 2   # direct arcs for witnesses that reach one row class or one column
W_WARM = 4     # warm start with a maximum flow of the adjacency layer
W_ALL = 7

N_WDIAG = 7
WD_ROWS = 0     # |P|
WD_GROUPS = 1   # row classes after merging
WD_COLS = 2     # |N|
WD_WIT = 3      # witnesses (relay nodes + witnesses turned into direct arcs)
WD_NODES = 4
WD_ARCS = 5
WD_STATUS = 6   # 0 ok, 1 masses do not fit in 64-bit integers


@njit(cache=True)
def _edge_witness(indptr, indices, wts, Dp, nself, coef, gdv, i0, j0, mode, pn, pd, flags,
                  stP, idP, stN, idN, stR, stW, idW, cntR, cntC, ep, bi, k, out, num, diag):
    # ---- orientation, neighbourhoods and integer masses: as in orcut._core._edge_core
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
        diag[k, WD_STATUS] = 1
        out[k] = np.nan
        num[k, 0] = -1
        num[k, 1] = 0
        return bi
    if not walk:
        for r in range(nUi):
            rm[r] = mi_n
        for t in range(nUj):
            cm[t] = mj_n
        rm[nUi] = mi_s - mj_n
        rm[nUi + 1] = mi_n - mj_s
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

    # ---- P (positive rows) and N (positive column demands), marked by vertex
    A = np.int64(0)
    bi, oi, Pv = _take_i(bi, oi, nr)
    bi, oi, Pm = _take_i(bi, oi, nr)
    nP = 0
    for r in range(nr):
        if rm[r] > 0:
            Pv[nP] = rowv[r]
            Pm[nP] = rm[r]
            A += rm[r]
            nP += 1
    bi, oi, Nv = _take_i(bi, oi, nc)
    bi, oi, Nm = _take_i(bi, oi, nc)
    nN = 0
    for c in range(nc):
        if cm[c] > 0:
            Nv[nN] = colv[c]
            Nm[nN] = cm[c]
            nN += 1
    diag[k, WD_ROWS] = nP
    diag[k, WD_COLS] = nN
    num[k, 1] = L
    if A == 0:
        num[k, 0] = 0
        out[k] = (dmax + 1) / dmax if mode == M_LLY else 1.0
        return bi
    for c in range(nN):
        stN[Nv[c]] = ep
        idN[Nv[c]] = c
    for r in range(nP):
        stP[Pv[r]] = ep
        idP[Pv[r]] = r

    # ---- witnesses: vertices adjacent to some row of P and some column of N
    for r in range(nP):
        x = Pv[r]
        for e in range(indptr[x], indptr[x + 1]):
            w = indices[e]
            if stR[w] != ep:
                stR[w] = ep
                cntR[w] = 0
            cntR[w] += 1
    nW = 0
    nWC = 0  # witness-column incidences
    for c in range(nN):
        y = Nv[c]
        for e in range(indptr[y], indptr[y + 1]):
            w = indices[e]
            if stR[w] == ep:
                if stW[w] != ep:
                    stW[w] = ep
                    idW[w] = nW
                    cntC[nW] = 0
                    nW += 1
                cntC[idW[w]] += 1
                nWC += 1
    diag[k, WD_WIT] = nW
    # column lists of the witnesses (CSR by witness id)
    bi, oi, wcs = _take_i(bi, oi, nW + 1)
    wcs[0] = 0
    for h in range(nW):
        wcs[h + 1] = wcs[h] + cntC[h]
    bi, oi, wfill = _take_i(bi, oi, nW)
    for h in range(nW):
        wfill[h] = wcs[h]
    bi, oi, wcol = _take_i(bi, oi, max(nWC, 1))
    for c in range(nN):
        y = Nv[c]
        for e in range(indptr[y], indptr[y + 1]):
            w = indices[e]
            if stR[w] == ep and stW[w] == ep:
                h = idW[w]
                wcol[wfill[h]] = c
                wfill[h] += 1

    # ---- row keys: sorted codes 2 v (v an adjacent column) and 2 v + 1 (v a witness)
    tot = 0
    for r in range(nP):
        tot += indptr[Pv[r] + 1] - indptr[Pv[r]]
    bi, oi, ks = _take_i(bi, oi, nP + 1)
    bi, oi, keys = _take_i(bi, oi, 2 * tot + 1)
    m = 0
    ks[0] = 0
    for r in range(nP):
        x = Pv[r]
        for e in range(indptr[x], indptr[x + 1]):
            w = indices[e]
            if stN[w] == ep:
                keys[m] = 2 * w
                m += 1
            if stW[w] == ep:
                keys[m] = 2 * w + 1
                m += 1
        ks[r + 1] = m
    # ---- merge rows with equal keys (open addressing on a hash of the key)
    bi, oi, gid = _take_i(bi, oi, nP)
    bi, oi, leader = _take_i(bi, oi, nP)
    ng = 0
    if (flags & W_TWINS) != 0:
        tsize = 4
        while tsize < 2 * nP:
            tsize <<= 1
        bi, oi, table = _take_i(bi, oi, tsize)
        for z in range(tsize):
            table[z] = -1
        for r in range(nP):
            h = np.uint64(1469598103934665603)
            for z in range(ks[r], ks[r + 1]):
                h = (h ^ np.uint64(keys[z])) * np.uint64(1099511628211)
            h ^= h >> np.uint64(29)
            pos = np.int64(h & np.uint64(tsize - 1))
            while True:
                gg = table[pos]
                if gg < 0:
                    table[pos] = ng
                    leader[ng] = r
                    gid[r] = ng
                    ng += 1
                    break
                L0 = leader[gg]
                same = (ks[L0 + 1] - ks[L0]) == (ks[r + 1] - ks[r])
                if same:
                    for z in range(ks[r + 1] - ks[r]):
                        if keys[ks[L0] + z] != keys[ks[r] + z]:
                            same = False
                            break
                if same:
                    gid[r] = gg
                    break
                pos = (pos + 1) & (tsize - 1)
    else:
        for r in range(nP):
            gid[r] = r
            leader[r] = r
        ng = nP
    bi, oi, gm = _take_i(bi, oi, ng)
    for g in range(ng):
        gm[g] = 0
    for r in range(nP):
        gm[gid[r]] += Pm[r]
    diag[k, WD_GROUPS] = ng

    # ---- witness row incidences, counted on row classes (leaders)
    bi, oi, wr = _take_i(bi, oi, max(nW, 1))
    for h in range(nW):
        wr[h] = 0
    for g in range(ng):
        r = leader[g]
        for z in range(ks[r], ks[r + 1]):
            code = keys[z]
            if (code & 1) == 1:
                wr[idW[code >> 1]] += 1
    direct = (flags & W_DIRECT) != 0
    # relay node ids
    bi, oi, hnode = _take_i(bi, oi, max(nW, 1))
    base_b = 2 + 2 * ng
    base_h = base_b + 2 * nN
    nV = base_h
    nA = 3 * ng + 2 * nN
    for h in range(nW):
        nr_h = wr[h]
        nc_h = wcs[h + 1] - wcs[h]
        if nr_h == 0 or nc_h == 0:
            hnode[h] = -2
        elif direct and (nr_h == 1 or nc_h == 1):
            hnode[h] = -1
            nA += nr_h * nc_h
        else:
            hnode[h] = nV
            nV += 1
            nA += nr_h + nc_h
    e1 = 0
    for g in range(ng):
        r = leader[g]
        for z in range(ks[r], ks[r + 1]):
            if (keys[z] & 1) == 0:
                e1 += 1
    nA += 2 * e1
    diag[k, WD_NODES] = nV
    diag[k, WD_ARCS] = nA
    bi, oi, tl = _take_i(bi, oi, nA)
    bi, oi, hd = _take_i(bi, oi, nA)
    bi, oi, cp = _take_i(bi, oi, nA)
    mm = 0
    for g in range(ng):
        pa = 2 + 2 * g       # a_{g,2}
        qa = pa + 1          # a_{g,3}
        tl[mm] = 0; hd[mm] = pa; cp[mm] = gm[g]; mm += 1
        tl[mm] = 0; hd[mm] = qa; cp[mm] = gm[g]; mm += 1
        tl[mm] = qa; hd[mm] = pa; cp[mm] = INF; mm += 1
        r = leader[g]
        for z in range(ks[r], ks[r + 1]):
            code = keys[z]
            v = code >> 1
            if (code & 1) == 0:  # adjacent column: both layers
                c = idN[v]
                tl[mm] = pa; hd[mm] = base_b + 2 * c; cp[mm] = INF; mm += 1
                tl[mm] = qa; hd[mm] = base_b + 2 * c + 1; cp[mm] = INF; mm += 1
            else:
                h = idW[v]
                if hnode[h] >= 0:
                    tl[mm] = qa; hd[mm] = hnode[h]; cp[mm] = INF; mm += 1
                elif hnode[h] == -1:
                    for z2 in range(wcs[h], wcs[h + 1]):
                        tl[mm] = qa; hd[mm] = base_b + 2 * wcol[z2]; cp[mm] = INF; mm += 1
    for h in range(nW):
        if hnode[h] >= 0:
            for z2 in range(wcs[h], wcs[h + 1]):
                tl[mm] = hnode[h]; hd[mm] = base_b + 2 * wcol[z2]; cp[mm] = INF; mm += 1
    for c in range(nN):
        tl[mm] = base_b + 2 * c; hd[mm] = 1; cp[mm] = Nm[c]; mm += 1
        tl[mm] = base_b + 2 * c + 1; hd[mm] = 1; cp[mm] = Nm[c]; mm += 1
    if mm != nA:
        raise RuntimeError("internal error: witness network arc count")
    start, to, cap, rev, pos_, bi, oi = _csr(nV, nA, tl, hd, cp, bi, oi)
    F = np.int64(0)
    if (flags & W_WARM) != 0 and e1 > 0:
        # one layer of adjacency (a_{g,2} -> b_{c,1}) first: its flow, mirrored on a_{g,3} -> b_{c,2},
        # is feasible in the full network; Dinic then continues from it
        nV1 = 2 + ng + nN
        nA1 = ng + e1 + nN
        bi, oi, tl1 = _take_i(bi, oi, nA1)
        bi, oi, hd1 = _take_i(bi, oi, nA1)
        bi, oi, cp1 = _take_i(bi, oi, nA1)
        m1 = 0
        for g in range(ng):
            tl1[m1] = 0; hd1[m1] = 2 + g; cp1[m1] = gm[g]; m1 += 1
        for g in range(ng):
            r = leader[g]
            for z in range(ks[r], ks[r + 1]):
                if (keys[z] & 1) == 0:
                    tl1[m1] = 2 + g; hd1[m1] = 2 + ng + idN[keys[z] >> 1]; cp1[m1] = INF; m1 += 1
        for c in range(nN):
            tl1[m1] = 2 + ng + c; hd1[m1] = 1; cp1[m1] = Nm[c]; m1 += 1
        st1, to1, ca1, rv1, ps1, bi, oi = _csr(nV1, nA1, tl1, hd1, cp1, bi, oi)
        F1, bi, oi = _dinic(nV1, st1, to1, ca1, rv1, 0, 1, bi, oi)
        # load: the arcs of the full network were emitted in the order s->a2, s->a3, a3->a2, (adjacent: a2->b1,
        # a3->b2 | witness arcs), so walk the same order
        mm = 0
        m1 = 0
        for g in range(ng):
            f = ca1[rv1[ps1[g]]]
            if f > 0:
                for dd in range(2):
                    e_ = pos_[mm + dd]
                    cap[e_] -= f
                    cap[rev[e_]] += f
            mm += 3
            r = leader[g]
            for z in range(ks[r], ks[r + 1]):
                code = keys[z]
                if (code & 1) == 0:
                    f = ca1[rv1[ps1[ng + m1]]]
                    m1 += 1
                    if f > 0:
                        for dd in range(2):
                            e_ = pos_[mm + dd]
                            cap[e_] -= f
                            cap[rev[e_]] += f
                    mm += 2
                else:
                    h = idW[code >> 1]
                    if hnode[h] >= 0:
                        mm += 1
                    elif hnode[h] == -1:
                        mm += wcs[h + 1] - wcs[h]
        for h in range(nW):
            if hnode[h] >= 0:
                mm += wcs[h + 1] - wcs[h]
        for c in range(nN):
            f = ca1[rv1[ps1[ng + e1 + c]]]
            if f > 0:
                for dd in range(2):
                    e_ = pos_[mm + dd]
                    cap[e_] -= f
                    cap[rev[e_]] += f
            mm += 2
        F = 2 * F1
    F2, bi, oi = _dinic(nV, start, to, cap, rev, 0, 1, bi, oi)
    Wd = 3 * A - (F + F2)
    num[k, 0] = Wd
    if mode == M_LLY:
        out[k] = (1.0 - Wd / L) * (dmax + 1) / dmax
    else:
        out[k] = 1.0 - Wd / L
    return bi


@njit(cache=True)
def _witness_chunk(indptr, indices, wts, Dp, nself, coef, gdv, edges, ids, c, nch, mode, pn, pd, flags,
                   out, num, diag):
    n = indptr.shape[0] - 1
    stP = np.full(n, -1, np.int64)
    idP = np.empty(n, np.int64)
    stN = np.full(n, -1, np.int64)
    idN = np.empty(n, np.int64)
    stR = np.full(n, -1, np.int64)
    stW = np.full(n, -1, np.int64)
    idW = np.empty(n, np.int64)
    cntR = np.empty(n, np.int64)
    cntC = np.empty(n, np.int64)
    bi = np.empty(1 << 12, np.int64)
    K = ids.shape[0]
    blk = 16
    nb = (K + blk - 1) // blk
    for bk in range(c, nb, nch):
        for k in range(bk * blk, min(K, bk * blk + blk)):
            e = ids[k]
            bi = _edge_witness(indptr, indices, wts, Dp, nself, coef, gdv, edges[e, 0], edges[e, 1], mode, pn, pd,
                               flags, stP, idP, stN, idN, stR, stW, idW, cntR, cntC, k, bi, k, out, num, diag)


@njit(cache=True, parallel=True)
def _witness_run(indptr, indices, wts, Dp, nself, coef, gdv, edges, ids, mode, pn, pd, flags, nchunks):
    K = ids.shape[0]
    out = np.full(K, np.nan)
    num = np.zeros((K, 2), np.int64)
    diag = np.zeros((K, N_WDIAG), np.int64)
    if K == 0:
        return out, num, diag
    nb = (K + 15) // 16
    nchunks = max(1, min(nchunks, nb))
    for c in prange(nchunks):
        _witness_chunk(indptr, indices, wts, Dp, nself, coef, gdv, edges, ids, c, nchunks, mode, pn, pd, flags,
                       out, num, diag)
    return out, num, diag


_DUMMY_I = np.zeros(1, np.int64)


def witness_ids(indptr, indices, edges, ids, mode, pn=0, pd=1, flags=W_ALL, nchunks=None, walk=None):
    """Curvatures of edges[ids] on the witness network: (values, exact W1 (num, den), diagnostics)."""
    from ._kernels import _ids, _nchunks

    if mode == M_WALK:
        if walk is None:
            raise ValueError("M_WALK needs the walk tables")
        wts, Dp, nself, coef, gdv = (np.ascontiguousarray(a, dtype=np.int64) for a in walk)
    else:
        wts = Dp = nself = coef = gdv = _DUMMY_I
    return _witness_run(indptr, indices, wts, Dp, nself, coef, gdv, edges, _ids(ids), int(mode), int(pn), int(pd),
                        int(flags), _nchunks(nchunks))
