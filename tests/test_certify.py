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
