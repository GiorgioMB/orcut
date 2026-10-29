"""Certificates: exact, verified, and re-checkable from scratch."""
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

import orcut
from orcut import measures as Ms


@pytest.mark.parametrize("measure", ["lazy", "nonlazy", "lly", Fraction(1, 2)])
def test_certificates_match_solver_and_verify(measure):
    G = nx.convert_node_labels_to_integers(nx.powerlaw_cluster_graph(60, 3, 0.6, seed=2))
    g = orcut.as_graph(G)
    vals = orcut.curvature(g, measure)
    for e, (i, j) in enumerate(g.edges):
        c = orcut.certify(g, (int(i), int(j)), measure)
        assert c.verified and c.verify(g)
        assert float(c.curvature) == pytest.approx(vals[e], abs=1e-12)


def test_certificate_for_weighted_walk_and_ball():
    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(50, 3, seed=1))
    g = orcut.as_graph(G)
    walk = Ms.Walk(weights=np.arange(1, g.num_edges + 1) % 5 + 1, idleness=Fraction(1, 3))
    vals = orcut.curvature(g, walk)
    vb = orcut.ball_curvature(g, 2)
    for e in range(0, g.num_edges, 7):
        i, j = map(int, g.edges[e])
        c = orcut.certify(g, (i, j), walk)
        assert c.verified and c.verify(g) and float(c.curvature) == pytest.approx(vals[e], abs=1e-12)
        cb = orcut.certify(g, (i, j), "ball", radius=2)
        assert cb.verified and cb.verify(g) and float(cb.curvature) == pytest.approx(vb[e], abs=1e-12)


def test_tampered_certificate_fails():
    G = nx.convert_node_labels_to_integers(nx.karate_club_graph())
    g = orcut.as_graph(G)
    c = orcut.certify(g, (0, 1))
    assert c.verify(g)
    x = next(iter(c.potential))
    c.potential[x] += 1
    assert not c.verify(g)


def test_pn_distances_match_bfs():
    """The numba P x N distances of certify equal truncated BFS distances (hubs included)."""
    from orcut.certify import _bfs_all, _pn_distances, _scratch

    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(300, 4, seed=3))
    g = orcut.as_graph(G)
    rng = np.random.default_rng(0)
    for e in rng.choice(g.num_edges, 40, replace=False):
        i, j = map(int, g.edges[e])
        S = sorted(set(map(int, g.neighbors(i))) | set(map(int, g.neighbors(j))) | {i, j})
        rng.shuffle(S)
        P = np.array(S[: len(S) // 2], np.int64)
        N = np.array(S[len(S) // 2:], np.int64)
        Dm = _pn_distances(g.indptr, g.indices, P, N, *_scratch(g.num_nodes))
        for a, x in enumerate(P):
            dx = _bfs_all(g, int(x), 2)
            for b, y in enumerate(N):
                assert Dm[a, b] == dx.get(int(y), 3)


def test_identical_measures_are_certified():
    """An edge whose two balls coincide (W1 = 0) is reported as certified."""
    g = orcut.as_graph(nx.complete_graph(6))
    v, d = orcut.ball_curvature(g, 2, certify=True)
    assert np.all(v == 1.0) and np.all(d[:, 5] == 1)
    c = orcut.certify(g, (0, 1), "ball", radius=2)
    assert c.verified and c.verify(g) and c.curvature == 1


def test_idleness_half_is_half_lly():
    """kappa_p = (1 - p) kappa_LLY for p >= 1 / (max degree + 1) (Bourne et al., 2018)."""
    G = nx.convert_node_labels_to_integers(nx.powerlaw_cluster_graph(200, 4, 0.3, seed=5))
    g = orcut.as_graph(G)
    lly = orcut.curvature(g, "lly", exact_fraction=True)
    for p in (Fraction(1, 2), Fraction(2, 3)):
        out = orcut.curvature(g, p, exact_fraction=True)
        for e in range(g.num_edges):
            kp = 1 - Fraction(int(out["w1_num"][e]), int(out["w1_den"][e]))
            dmax = max(int(g.deg[g.edges[e, 0]]), int(g.deg[g.edges[e, 1]]))
            kl = (1 - Fraction(int(lly["w1_num"][e]), int(lly["w1_den"][e]))) * Fraction(dmax + 1, dmax)
            assert kp == (1 - p) * kl
