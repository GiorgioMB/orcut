"""Locality: after inserting or deleting vertex pairs, only affected_edges can change curvature."""
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

import orcut


@pytest.mark.parametrize("measure", ["lazy", "nonlazy", "lly", Fraction(1, 3)])
def test_only_affected_edges_change(measure):
    rng = np.random.default_rng(7)
    G = nx.convert_node_labels_to_integers(nx.powerlaw_cluster_graph(150, 3, 0.4, seed=4))
    g = orcut.as_graph(G)
    for trial in range(8):
        E = {tuple(e) for e in g.edges.tolist()}
        pairs = []
        for _ in range(rng.integers(1, 5)):
            if rng.random() < 0.5 and E:  # delete an edge
                e = sorted(E)[rng.integers(len(E))]
                E.discard(e)
                pairs.append(e)
            else:  # insert a non-edge
                while True:
                    a, b = sorted(map(int, rng.choice(g.num_nodes, 2, replace=False)))
                    if (a, b) not in E:
                        E.add((a, b))
                        pairs.append((a, b))
                        break
        g2 = orcut.Graph.from_edges(np.array(sorted(E)), g.num_nodes)
        old = dict(zip(map(tuple, g.edges.tolist()), orcut.curvature(g, measure)))
        new = orcut.curvature(g2, measure)
        aff = set(orcut.affected_edges(g2, pairs).tolist())
        for k, e in enumerate(map(tuple, g2.edges.tolist())):
            if k not in aff:
                assert e in old and old[e] == new[k], (trial, e)
        g = g2
