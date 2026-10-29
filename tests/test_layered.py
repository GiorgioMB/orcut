"""The layered closure (orcut._layered): equal to the one-step kernel for one-step measures, and
exact against linear programming for ball measures and multi-step walks."""
import networkx as nx
import numpy as np
import pytest

import orcut
from scipy.optimize import linprog
from orcut import _core as C
from orcut import _layered as LY


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G, orcut.Graph.from_edges(np.array(G.edges()).reshape(-1, 2), G.number_of_nodes())


GRAPHS = [nx.karate_club_graph(), nx.barabasi_albert_graph(60, 2, seed=1), nx.gnp_random_graph(45, 0.09, seed=3),
          nx.random_geometric_graph(50, 0.22, seed=4), nx.cycle_graph(11), nx.grid_2d_graph(4, 5),
          nx.balanced_tree(2, 4), nx.petersen_graph()]


def _ball(G, u, r):
    return nx.single_source_shortest_path_length(G, u, cutoff=r)


def _walk(G, u, r, lazy):
    m = {u: 1.0}
    for _ in range(r):
        nm = {}
        for x, v in m.items():
            d = G.degree(x)
            if lazy:
                nm[x] = nm.get(x, 0) + v / (d + 1)
                for w in G[x]:
                    nm[w] = nm.get(w, 0) + v / (d + 1)
            else:
                for w in G[x]:
                    nm[w] = nm.get(w, 0) + v / d
        m = nm
    return {k: v for k, v in m.items() if v > 0}


def _w1(G, mu, nu):
    """LP reference with exact shortest-path distances (no cutoff)."""
    S, T = list(mu), list(nu)
    dist = {s: nx.single_source_shortest_path_length(G, s) for s in S}
    c = np.array([dist[s][t] for s in S for t in T], float)
    m, n = len(S), len(T)
    A = np.zeros((m + n, m * n))
    b = np.zeros(m + n)
    for a in range(m):
        A[a, a * n:(a + 1) * n] = 1
        b[a] = mu[S[a]]
    for k in range(n):
        A[m + k, k::n] = 1
        b[m + k] = nu[T[k]]
    return linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun


@pytest.mark.parametrize("G0", GRAPHS, ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_one_step_equals_core_and_is_certified(G0):
    G, g = _g(G0)
    ids = np.arange(g.num_edges, dtype=np.int64)
    for kind, mode in ((LY.K_LAZY, C.M_LAZY), (LY.K_NONLAZY, C.M_NONLAZY)):
        v, d = LY.layered_ids(g.indptr, g.indices, g.edges, ids, kind, 1, True, 2)
        assert np.array_equal(v, orcut.curvature(g, "lazy" if kind == LY.K_LAZY else "nonlazy"))
        assert np.all(d[d[:, 0] > 0, 5] == 1)
    assert np.array_equal(orcut.ball_curvature(g, 1), orcut.curvature(g))


@pytest.mark.parametrize("G0", GRAPHS[:5], ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_ball_radius_two_matches_lp(G0):
    G, g = _g(G0)
    v, d = orcut.ball_curvature(g, 2, certify=True)
    assert np.all(d[d[:, 0] > 0, 5] == 1) and np.all(d[:, 4] == 5)
    for e, (i, j) in enumerate(g.edges[:30]):
        Bi, Bj = _ball(G, int(i), 2), _ball(G, int(j), 2)
        lp = 1 - _w1(G, {x: 1 / len(Bi) for x in Bi}, {y: 1 / len(Bj) for y in Bj})
        assert v[e] == pytest.approx(lp, abs=1e-9)


@pytest.mark.parametrize("steps,lazy", [(1, True), (2, True), (2, False), (3, False)])
def test_multistep_walks_match_lp(steps, lazy):
    G, g = _g(nx.barabasi_albert_graph(40, 2, seed=5))
    v = orcut.walk_curvature(g, steps, lazy)
    for e, (i, j) in enumerate(g.edges[:25]):
        lp = 1 - _w1(G, _walk(G, int(i), steps, lazy), _walk(G, int(j), steps, lazy))
        assert v[e] == pytest.approx(lp, abs=1e-8)
