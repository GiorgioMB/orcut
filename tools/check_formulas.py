"""Brute-force checks, against orcut's exact fractions, of two results of the paper:
 (1) the matching identity  W1 = (3n - 2 nu(B)) / (n + 2)  on the graph G_B (lazy walk; the theorem
     that curvature is at least as hard as bipartite matching), and
 (2) the min-max formula (the corollary on level sets of the potential)
     W1 = A + max_{X3 c X2 c P} [ mu(X2) - nu(N<=1(X2) u N<=2(X3)) + mu(X3) - nu(N<=1(X3)) ]
     for several one-step measures on random small graphs.

Usage: python tools/check_formulas.py [MATCHING_TRIALS] [EDGES]   (default 3000 4000, a few minutes);
tests/test_theory.py runs a small version."""
import itertools, random, sys
from fractions import Fraction as Fr
import numpy as np, networkx as nx
import orcut

def w1_exact(g, edge_id, measure):
    r = orcut.curvature(g, measure, edges=np.array([edge_id]), exact_fraction=True, n_jobs=1)
    return Fr(int(r["w1_num"][0]), int(r["w1_den"][0]))

# ---------- (1) matching identity
def check_matching(trials, rng):
    bad = 0
    for _ in range(trials):
        n = rng.randint(1, 25)
        p = rng.choice([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0])
        G = nx.Graph()
        i, j = 0, 1
        L = list(range(2, 2 + n)); R = list(range(2 + n, 2 + 2 * n))
        G.add_edge(i, j)
        G.add_edges_from((i, x) for x in L); G.add_edges_from((j, y) for y in R)
        B = nx.Graph(); B.add_nodes_from(L, bipartite=0); B.add_nodes_from(R, bipartite=1)
        for x in L:
            for y in R:
                if rng.random() < p:
                    G.add_edge(x, y); B.add_edge(x, y)
        nu = len(nx.bipartite.hopcroft_karp_matching(B, top_nodes=L)) // 2
        g = orcut.Graph.from_edges(list(G.edges()), num_nodes=G.number_of_nodes())
        eid = int(np.where((g.edges[:, 0] == 0) & (g.edges[:, 1] == 1))[0][0])
        if w1_exact(g, eid, "lazy") != Fr(3 * n - 2 * nu, n + 2):
            bad += 1
    return bad

# ---------- (2) Hall formula
def masses(G, u, measure, wts):
    """The one-step measure m_u as a dict of Fractions."""
    nb = sorted(G[u]); d = len(nb)
    if measure == "lazy":
        p, w = Fr(1, d + 1), {x: Fr(1) for x in nb}
    elif measure == "nonlazy":
        p, w = Fr(0), {x: Fr(1) for x in nb}
    elif measure == "half":
        p, w = Fr(1, 2), {x: Fr(1) for x in nb}
    elif measure == "weighted":
        p, w = Fr(1, d + 1), {x: Fr(wts[frozenset((u, x))]) for x in nb}
    tot = sum(w.values())
    m = {x: (1 - p) * w[x] / tot for x in nb}
    if p:
        m[u] = p
    return m

def hall_w1(G, i, j, measure, wts):
    mi, mj = masses(G, i, measure, wts), masses(G, j, measure, wts)
    mu = {x: mi.get(x, 0) - mj.get(x, 0) for x in set(mi) | set(mj)}
    P = [x for x, v in mu.items() if v > 0]; Nn = [y for y, v in mu.items() if v < 0]
    nu = {y: -mu[y] for y in Nn}
    A = sum(mu[x] for x in P)
    dist = {x: nx.single_source_shortest_path_length(G, x, cutoff=3) for x in P}
    best = Fr(0)
    for lab in itertools.product((0, 2, 3), repeat=len(P)):  # 0: not in X2; 2: X2 only; 3: X3 (c X2)
        X2 = [x for x, l in zip(P, lab) if l >= 2]; X3 = [x for x, l in zip(P, lab) if l == 3]
        n1X2 = {y for y in Nn if any(dist[x].get(y, 9) <= 1 for x in X2)}
        n2X3 = {y for y in Nn if any(dist[x].get(y, 9) <= 2 for x in X3)}
        n1X3 = {y for y in Nn if any(dist[x].get(y, 9) <= 1 for x in X3)}
        val = (sum(mu[x] for x in X2) - sum(nu[y] for y in n1X2 | n2X3)
               + sum(mu[x] for x in X3) - sum(nu[y] for y in n1X3))
        best = max(best, val)
    return A + best, len(P)

def check_hall(trials, rng):
    bad, done, maxP = 0, 0, 0
    for t in range(trials):
        n = rng.randint(4, 16)
        kind = rng.choice(["gnp", "ba", "ws", "tree_plus"])
        if kind == "gnp":
            G = nx.gnp_random_graph(n, rng.choice([0.2, 0.35, 0.5]), seed=rng.randint(0, 10**9))
        elif kind == "ba":
            G = nx.barabasi_albert_graph(n, rng.choice([1, 2, 3]), seed=rng.randint(0, 10**9))
        elif kind == "ws":
            G = nx.watts_strogatz_graph(n, 4, 0.3, seed=rng.randint(0, 10**9))
        else:
            G = nx.random_labeled_tree(n, seed=rng.randint(0, 10**9)); G.add_edges_from(
                (rng.randrange(n), rng.randrange(n)) for _ in range(n // 3))
        G.remove_edges_from(nx.selfloop_edges(G))
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        G = nx.convert_node_labels_to_integers(G)
        g = orcut.Graph.from_edges(list(G.edges()), num_nodes=G.number_of_nodes())
        E = [tuple(map(int, e)) for e in g.edges]
        wts = {frozenset(e): rng.randint(1, 5) for e in E}
        eid = rng.randrange(len(E)); i, j = E[eid]
        measure = rng.choice(["lazy", "nonlazy", "half", "weighted"])
        if measure == "lazy":
            ref = w1_exact(g, eid, "lazy")
        elif measure == "nonlazy":
            ref = w1_exact(g, eid, "nonlazy")
        elif measure == "half":
            ref = w1_exact(g, eid, Fr(1, 2))
        else:
            ref = w1_exact(g, eid, orcut.measures.Walk(weights=np.array([wts[frozenset(e)] for e in E]),
                                                        idleness="lazy"))
        got, nP = hall_w1(G, i, j, measure, wts)
        maxP = max(maxP, nP)
        if nP > 11:
            continue
        done += 1
        if got != ref:
            bad += 1
            print("MISMATCH", measure, (i, j), got, ref, file=sys.stderr)
    return bad, done, maxP

if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    edges = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    rng = random.Random(20261113)
    print("matching identity: mismatches", check_matching(trials, rng), f"of {trials}")
    bad, done, maxP = check_hall(edges, rng)
    print(f"min-max formula: mismatches {bad} of {done} edges checked (|P| up to {maxP})")
