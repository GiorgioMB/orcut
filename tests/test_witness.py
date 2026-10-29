"""Witness relays: the linear-size network of orcut._witness and the hybrid of orcut._core
(flag F_WITNESS: relays per common neighbour, and the whole witness network when the bitsets would
cost too much) return exactly the values of the default kernel."""
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

import orcut
from conftest import random_graphs, small_graphs
from orcut import _core as C
from orcut import _witness as WT
from orcut.measures import parse


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_edges() and not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return orcut.Graph.from_edges(np.array(G.edges()).reshape(-1, 2), G.number_of_nodes())


def connector_edge(a, b, k, f, seed):
    """Edge (0, 1) with |U_0| = a, |U_1| = b, two cross edges per vertex of U_0, and k connector
    vertices each adjacent to a fraction f of U_0 and of U_1 (many pairs at distance 2)."""
    rng = np.random.default_rng(seed)
    Ui = np.arange(2, 2 + a)
    Uj = np.arange(2 + a, 2 + a + b)
    E = [(0, 1)] + [(0, x) for x in Ui] + [(1, y) for y in Uj]
    for x in Ui:
        for y in rng.choice(Uj, 2, replace=False):
            E.append((int(x), int(y)))
    nxt = 2 + a + b
    for _ in range(k):
        for x in rng.choice(Ui, max(1, int(f * a)), replace=False):
            E.append((nxt, int(x)))
        for y in rng.choice(Uj, max(1, int(f * b)), replace=False):
            E.append((nxt, int(y)))
        nxt += 1
    return orcut.Graph.from_edges(np.array(E))


GRAPHS = small_graphs() + random_graphs(2) + [nx.barabasi_albert_graph(300, 4, seed=7),
                                              nx.powerlaw_cluster_graph(250, 3, 0.7, seed=8)]


def _measures(g):
    rng = np.random.default_rng(1)
    return ["lazy", "nonlazy", "lly", Fraction(1, 3), Fraction(1, 2),
            orcut.Walk(weights=rng.integers(1, 6, g.num_edges), idleness=Fraction(1, 4))]


@pytest.mark.parametrize("G", GRAPHS, ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_witness_kernel_matches_core(G):
    g = _g(G)
    ids = np.arange(g.num_edges, dtype=np.int64)
    for meas in _measures(g):
        mode, pn, pd, tabs = parse(meas, g)
        v, num, _ = C.core_ids(g.indptr, g.indices, g.edges, ids, mode, pn, pd, C.F_ALL, None, tabs)
        for flags in range(8):
            w, wnum, wd = WT.witness_ids(g.indptr, g.indices, g.edges, ids, mode, pn, pd, flags, None, tabs)
            assert np.array_equal(w, v)
            assert np.all(num[:, 0].astype(object) * wnum[:, 1] == wnum[:, 0].astype(object) * num[:, 1])


@pytest.mark.parametrize("seed", range(4))
def test_witness_relays_are_used_and_exact(seed):
    g = connector_edge(300, 300, 8, 0.4, seed)
    ids = np.arange(g.num_edges, dtype=np.int64)
    eid = np.array([g.edge_id(0, 1)], np.int64)
    for meas in ("lazy", "nonlazy", Fraction(1, 2)):
        mode, pn, pd, tabs = parse(meas, g)
        a = C.core_ids(g.indptr, g.indices, g.edges, eid, mode, pn, pd, C.F_ALL & ~C.F_WITNESS, 1, tabs)
        b = C.core_ids(g.indptr, g.indices, g.edges, eid, mode, pn, pd, C.F_ALL, 1, tabs, dispatch=0)
        assert b[2][0, C.D_PATH] == C.PATH_CUTW          # the relays were chosen
        assert b[2][0, C.D_ARCS] < a[2][0, C.D_ARCS]     # and the network is smaller
        assert np.array_equal(a[1], b[1]) and np.array_equal(a[0], b[0])
        w = WT.witness_ids(g.indptr, g.indices, g.edges, eid, mode, pn, pd, WT.W_ALL, 1, tabs)
        assert np.array_equal(w[0], a[0])
    # every edge of the instance, all flags
    ref = C.core_ids(g.indptr, g.indices, g.edges, ids, C.M_LAZY, flags=15)[0]
    for flags in (16, 19, 23, 31):
        assert np.array_equal(C.core_ids(g.indptr, g.indices, g.edges, ids, C.M_LAZY, flags=flags)[0], ref)


def test_witness_network_is_linear_in_the_local_graph():
    """Arcs of the witness network <= 3|P| + 2|N| + 2 e_1 + sum of the degrees of P u N."""
    g = _g(nx.powerlaw_cluster_graph(400, 5, 0.5, seed=3))
    ids = np.arange(g.num_edges, dtype=np.int64)
    _, _, d = WT.witness_ids(g.indptr, g.indices, g.edges, ids, C.M_LAZY, flags=0)
    for e in range(g.num_edges):
        i, j = map(int, g.edges[e])
        S = set(map(int, g.neighbors(i))) | set(map(int, g.neighbors(j))) | {i, j}
        bound = 5 * len(S) + 3 * sum(int(g.deg[x]) for x in S)
        assert d[e, WT.WD_ARCS] <= bound


WIT_FLAGS = [f for f in range(32) if f & C.F_WITNESS and f & C.F_RELAYS]


@pytest.mark.parametrize("seed", range(3))
def test_dispatch_to_the_witness_network(seed):
    """Large neighbourhoods with few pairs within distance 2: the bitsets would cost far more than the
    local input, the edge is solved on the witness network (PATH_WIT), with the same exact value."""
    g = connector_edge(1500, 1500, 4, 0.01, seed)
    eid = np.array([g.edge_id(0, 1)], np.int64)
    for meas in _measures(g):
        mode, pn, pd, tabs = parse(meas, g)
        ref = C.core_ids(g.indptr, g.indices, g.edges, eid, mode, pn, pd, C.F_ALL & ~C.F_WITNESS, 1, tabs)
        for flags in WIT_FLAGS:
            v, num, d = C.core_ids(g.indptr, g.indices, g.edges, eid, mode, pn, pd, flags, 1, tabs)
            assert d[0, C.D_PATH] == C.PATH_WIT
            assert np.array_equal(num, ref[1]) and np.array_equal(v, ref[0])
            assert d[0, C.D_NODES] > 0 and d[0, C.D_ARCS] > 0


def middle_edge(a, b, mid, per_row, per_mid, seed):
    """Edge (0, 1), a > b: every vertex of U_0 (size a, the row side) is adjacent to ``per_row`` of ``mid`` middle vertices,
    each adjacent to ``per_mid`` vertices of U_1 (size b). The bitsets of the middle vertices are small
    in number, so the edge passes the a-priori test, but no row fills up and the coverage work grows
    as a * per_row * b / 64: the kernel switches to the witness network during the coverage."""
    rng = np.random.default_rng(seed)
    Ui = np.arange(2, 2 + a)
    Uj = np.arange(2 + a, 2 + a + b)
    M = np.arange(2 + a + b, 2 + a + b + mid)
    E = [(0, 1)] + [(0, int(x)) for x in Ui] + [(1, int(y)) for y in Uj]
    for x in Ui:
        E += [(int(x), int(w)) for w in rng.choice(M, per_row, replace=False)]
    for w in M:
        E += [(int(w), int(y)) for y in rng.choice(Uj, per_mid, replace=False)]
    return orcut.Graph.from_edges(np.array(E))


@pytest.mark.parametrize("seed", range(3))
def test_dispatch_during_the_coverage(seed):
    g = middle_edge(700, 640, 50, 40, 1, seed)
    eid = np.array([g.edge_id(0, 1)], np.int64)
    for meas in _measures(g):
        mode, pn, pd, tabs = parse(meas, g)
        ref = C.core_ids(g.indptr, g.indices, g.edges, eid, mode, pn, pd, C.F_ALL & ~C.F_WITNESS, 1, tabs)
        for flags in WIT_FLAGS:
            v, num, d = C.core_ids(g.indptr, g.indices, g.edges, eid, mode, pn, pd, flags, 1, tabs)
            assert d[0, C.D_PATH] == C.PATH_WIT
            assert np.array_equal(num, ref[1]) and np.array_equal(v, ref[0])
    # all edges of the instance, with and without the switch
    ids = np.arange(g.num_edges, dtype=np.int64)
    a = C.core_ids(g.indptr, g.indices, g.edges, ids, C.M_LAZY, dispatch=0)
    b = C.core_ids(g.indptr, g.indices, g.edges, ids, C.M_LAZY)
    assert np.array_equal(a[1], b[1])


@pytest.mark.parametrize("G", GRAPHS, ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_dispatch_never_changes_values(G):
    g = _g(G)
    ids = np.arange(g.num_edges, dtype=np.int64)
    for meas in _measures(g):
        mode, pn, pd, tabs = parse(meas, g)
        ref = C.core_ids(g.indptr, g.indices, g.edges, ids, mode, pn, pd, C.F_ALL, None, tabs, dispatch=0)
        for flags in WIT_FLAGS:
            v, num, d = C.core_ids(g.indptr, g.indices, g.edges, ids, mode, pn, pd, flags, None, tabs, dispatch=1)
            assert np.array_equal(num, ref[1]) and np.array_equal(v, ref[0])
            assert not np.any(d[:, C.D_PATH] < 0)
