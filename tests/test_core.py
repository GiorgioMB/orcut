"""The generalized one-step kernel (orcut._core): bit-identical to the 1.0.0 kernel on the
lazy, non-lazy and LLY measures under every ablation flag, and exact (against linear
programming) for uniform idleness and weighted walks."""
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

import orcut
from conftest import _w1, random_graphs, small_graphs
from orcut import _core as C
from orcut import _kernels as K
from orcut import measures as Ms


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_edges() and not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G, orcut.Graph.from_edges(np.array(G.edges()).reshape(-1, 2), G.number_of_nodes())


GRAPHS = small_graphs() + random_graphs(2) + [nx.barabasi_albert_graph(300, 4, seed=7),
                                              nx.powerlaw_cluster_graph(250, 3, 0.7, seed=8)]


@pytest.mark.parametrize("G", GRAPHS, ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_core_matches_frozen_kernel_under_every_flag(G):
    _, g = _g(G)
    ids = np.arange(g.num_edges, dtype=np.int64)
    ref = K.exact_ids(g.indptr, g.indices, g.edges, ids, True, True, True, 3)
    for mode, col in ((C.M_LAZY, 0), (C.M_NONLAZY, 1), (C.M_LLY, 2)):
        for flags in range(16):
            v, num, diag = C.core_ids(g.indptr, g.indices, g.edges, ids, mode, flags=flags, nchunks=3)
            assert np.array_equal(v, ref[:, col]), (mode, flags)
            # exact fraction: curvature = 1 - num/den (LLY rescaled)
            if mode != C.M_LLY:
                assert np.array_equal(v, 1.0 - num[:, 0] / num[:, 1])


def test_curvature_api_matches_exact():
    G, g = _g(nx.karate_club_graph())
    ex = orcut.exact(g)
    assert np.array_equal(orcut.curvature(g), ex["c_OR"])
    assert np.array_equal(orcut.curvature(g, "nonlazy"), ex["c_OR0"])
    assert np.array_equal(orcut.curvature(g, "lly"), ex["c_LLY"])
    assert np.array_equal(orcut.curvature(g, 0), ex["c_OR0"])


@pytest.mark.parametrize("p", [Fraction(1, 2), Fraction(1, 3), Fraction(2, 7), 0.25])
def test_uniform_idleness_matches_lp(p):
    for G0 in small_graphs()[:6]:
        G, g = _g(G0)
        v = orcut.curvature(g, p)
        q = Fraction(p).limit_denominator(10_000)
        for e, (i, j) in enumerate(g.edges):
            i, j = int(i), int(j)
            mu = {u: float((1 - q) / G.degree(i)) for u in G[i]}
            mu[i] = float(q)
            nu = {u: float((1 - q) / G.degree(j)) for u in G[j]}
            nu[j] = float(q)
            assert v[e] == pytest.approx(1 - _w1(G, mu, nu), abs=1e-9)


def test_lly_is_idleness_linear():
    """c_OR-p = (1 - p) c_LLY for p >= 1/(deg_max + 1) (Bourne et al.), checked at p = 1/2 on regular graphs."""
    G, g = _g(nx.random_regular_graph(3, 40, seed=1))  # deg_max + 1 = 4: p = 1/2 is in the linear range
    assert np.allclose(orcut.curvature(g, Fraction(1, 2)), 0.5 * orcut.curvature(g, "lly"), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_walk_matches_lp(seed):
    rng = np.random.default_rng(seed)
    G, g = _g(nx.powerlaw_cluster_graph(45, 3, 0.5, seed=seed))
    w = rng.integers(1, 7, g.num_edges)
    idl = np.array([Fraction(int(rng.integers(0, 3)), int(rng.integers(3, 8))) for _ in range(g.num_nodes)], dtype=object)
    walk = Ms.Walk(weights=w, idleness=idl)
    v = orcut.curvature(g, walk)
    wd = {}
    for e, (a, b) in enumerate(g.edges):
        wd[(int(a), int(b))] = wd[(int(b), int(a))] = int(w[e])

    def meas(u):
        Wu = sum(wd[(u, x)] for x in G[u])
        m = {x: float((1 - idl[u]) * Fraction(wd[(u, x)], Wu)) for x in G[u]}
        m[u] = m.get(u, 0.0) + float(idl[u])
        return {k: val for k, val in m.items() if val > 0}

    for e, (i, j) in enumerate(g.edges):
        assert v[e] == pytest.approx(1 - _w1(G, meas(int(i)), meas(int(j))), abs=1e-9)
    for flags in (0, 5, 10, 15):
        assert np.array_equal(orcut.curvature(g, walk, flags=flags), v)


def test_walk_with_lazy_idleness_and_unit_weights_is_lazy():
    G, g = _g(nx.barabasi_albert_graph(120, 3, seed=3))
    assert np.array_equal(orcut.curvature(g, Ms.Walk()), orcut.curvature(g))


def test_diagnostics_are_consistent():
    G, g = _g(nx.barabasi_albert_graph(400, 3, seed=4))
    out = orcut.curvature(g, diagnostics=True, exact_fraction=True)
    d = out["diagnostics"]
    assert np.all(d[:, C.D_STATUS] == 0)
    assert np.all(d[:, C.D_GROUPS] <= d[:, C.D_ROWS])
    assert set(np.unique(d[:, C.D_PATH])) <= {C.PATH_ZERO, C.PATH_NOADJ, C.PATH_FAST, C.PATH_CUT}
    # twin merging only merges: without it there are at least as many row nodes
    d0 = orcut.curvature(g, diagnostics=True, flags=C.F_ALL & ~C.F_TWINS)["diagnostics"]
    assert np.all(d0[:, C.D_GROUPS] >= d[:, C.D_GROUPS])


def test_overflow_is_flagged_not_wrong():
    G, g = _g(nx.star_graph(6))
    big = Fraction(1, (1 << 49) + 1)  # pd * deg_i * deg_j > 2^50
    out = orcut.curvature(g, big, diagnostics=True)
    assert np.all(np.isnan(out["curvature"])) and np.all(out["diagnostics"][:, C.D_STATUS] == 1)
