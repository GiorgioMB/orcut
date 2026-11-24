"""The production exact solver (orcut._core, through orcut.curvature) against the oracle
(orcut._oracle): different formulation, different distance computation, different
max-flow code, so bit-identical outputs on varied graphs are a strong check.
The oracle itself is checked against linear programming in test_exact.py."""
import networkx as nx
import numpy as np
import pytest

import orcut
from orcut import Graph
from orcut import _core as C
from orcut import _oracle as R


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    return Graph.from_edges(np.array(G.edges()), G.number_of_nodes())


def _star_of_cliques(seed):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    hub = 10_000
    for _ in range(4):
        k = int(rng.integers(3, 9))
        base = G.number_of_nodes()
        G.add_edges_from((base + a, base + b) for a in range(k) for b in range(a + 1, k))
        G.add_edge(base, hub)
        G.add_edges_from((base + a, hub + 1 + base + a) for a in range(k))  # pendants
    return G


FAMILIES = {
    "er_sparse": lambda s: nx.gnp_random_graph(150, 0.04, seed=s),
    "er_dense": lambda s: nx.gnp_random_graph(50, 0.5, seed=s),
    "ba_hubs": lambda s: nx.barabasi_albert_graph(200, 3, seed=s),
    "powerlaw_cluster": lambda s: nx.powerlaw_cluster_graph(150, 3, 0.7, seed=s),
    "geometric": lambda s: nx.random_geometric_graph(150, 0.15, seed=s),
    "regular": lambda s: nx.random_regular_graph(4, 60, seed=s),  # degree ties: zero-mass triangle vertices
    "sbm": lambda s: nx.stochastic_block_model([25, 25], [[0.4, 0.05], [0.05, 0.4]], seed=s),
    "star_of_cliques": _star_of_cliques,
}
FIXED = {
    "complete": nx.complete_graph(8), "star": nx.star_graph(7), "single_edge": nx.path_graph(2),
    "path": nx.path_graph(5), "cycle": nx.cycle_graph(7), "petersen": nx.petersen_graph(),
    "complete_bipartite": nx.complete_bipartite_graph(4, 6), "wheel": nx.wheel_graph(8),
    "barbell": nx.barbell_graph(5, 2), "grid": nx.grid_2d_graph(4, 5), "karate": nx.karate_club_graph(),
}


def _all(g):
    return np.arange(g.num_edges, dtype=np.int64)


def _production(g, nchunks=None):
    """Lazy, non-lazy and LLY curvatures of the production solver, as the oracle's first three columns."""
    ids = _all(g)
    return np.column_stack([C.core_ids(g.indptr, g.indices, g.edges, ids, mode, nchunks=nchunks)[0]
                            for mode in (C.M_LAZY, C.M_NONLAZY, C.M_LLY)])


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_oracle_on_random_families(family, seed):
    g = _g(FAMILIES[family](seed))
    ref = R.exact_ids_oracle(g.indptr, g.indices, g.edges, _all(g), True, True, True, 3)
    assert np.array_equal(_production(g, 3), ref[:, :3], equal_nan=True)


@pytest.mark.parametrize("name", sorted(FIXED))
def test_matches_oracle_on_fixed_graphs(name):
    g = _g(FIXED[name])
    ref = R.exact_ids_oracle(g.indptr, g.indices, g.edges, _all(g), True, True, True, 2)
    assert np.array_equal(_production(g, 2), ref[:, :3], equal_nan=True)


def test_closed_forms():
    g = _g(nx.complete_graph(6))  # identical lazy laws; non-lazy: move 1/d from j to i
    assert np.allclose(orcut.curvature(g), 1.0) and np.allclose(orcut.curvature(g, "nonlazy"), 1 - 1 / 5)
    g = _g(nx.path_graph(2))
    assert orcut.curvature(g)[0] == 1.0 and orcut.curvature(g, "nonlazy")[0] == 0.0
    g = _g(nx.star_graph(3))  # center-leaf, lazy: W1 = 3/4 -> 1/4
    assert np.allclose(orcut.curvature(g), 0.25)


def test_column_blocking_is_exact():
    """Tiny bitset budget: columns are processed in many blocks."""
    g = _g(nx.barabasi_albert_graph(300, 4, seed=5))
    ids = _all(g)
    for mode in (C.M_LAZY, C.M_NONLAZY, C.M_LLY):
        full = C.core_ids(g.indptr, g.indices, g.edges, ids, mode, nchunks=2)[0]
        blocked = C.core_ids(g.indptr, g.indices, g.edges, ids, mode, nchunks=2, budget=32)[0]
        assert np.array_equal(full, blocked, equal_nan=True)


def test_chunking_and_subsets_do_not_change_values():
    g = _g(nx.random_geometric_graph(250, 0.12, seed=2))
    ids = _all(g)
    base = _production(g, 1)
    for nch in (2, 7, 64):
        assert np.array_equal(_production(g, nch), base, equal_nan=True)
    sub = np.random.default_rng(0).permutation(ids)[: len(ids) // 3]
    v = C.core_ids(g.indptr, g.indices, g.edges, sub, C.M_LAZY, nchunks=3)[0]
    assert np.array_equal(v, base[sub, 0], equal_nan=True)
    assert C.core_ids(g.indptr, g.indices, g.edges, ids[:0], C.M_LAZY, nchunks=3)[0].shape == (0,)
