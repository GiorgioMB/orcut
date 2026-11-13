"""Executable check of the paper's bound on Dinic's phases (the proposition "Phases") on every
kind of network it covers.

For random edges and four one-step measures, builds in plain Python: G3, the witness network G^w,
and the solver's network (relays H1-H4, explicit arcs or witness relays for the remaining pairs),
each with and without twin merging of rows and with and without the warm start. Runs a line-by-line
port of orcut._kernels._dinic with shuffled arc orders, and checks
  (a) max flow F = 3A - W1 * scale, against orcut's exact W1 (the network is right);
  (b) at every phase, the distance l <= 24 A / M~ + 1, M~ the flow still missing (the key step);
  (c) phases <= 2 sqrt(24 A) + 1 (the statement);
  (d) no relay is adjacent to a relay, and every flow obeys the throughput bounds T(v);
  (e) at every phase, the missing flow f* - f enters each node v with at most 2 T(v) units.

Usage: python tools/check_phases.py [EDGES]   (default 400; 600 take a few minutes);
tests/test_theory.py runs a small version."""
import math, random, sys, itertools
from collections import defaultdict
from fractions import Fraction as Fr
import numpy as np, networkx as nx
import orcut

INF = 1 << 60

# ------------------------------------------------------------------ network
class Net:
    def __init__(self):
        self.n = 0; self.kind = []; self.tail = []; self.head = []; self.cap = []; self.T = {}
    def node(self, kind, T=None):
        self.kind.append(kind); self.n += 1
        if T is not None:
            self.T[self.n - 1] = T
        return self.n - 1
    def arc(self, u, v, c):
        self.tail.append(u); self.head.append(v); self.cap.append(c)

def build_csr(net, rng):
    """Arcs and reverse arcs in CSR with shuffled order per node (the order Dinic visits them)."""
    m = len(net.tail)
    pairs = []
    for k in range(m):
        pairs.append((net.tail[k], net.head[k], net.cap[k], k, 0))
        pairs.append((net.head[k], net.tail[k], 0, k, 1))
    rng.shuffle(pairs)
    pairs.sort(key=lambda p: p[0])
    start = [0] * (net.n + 1)
    for p in pairs:
        start[p[0] + 1] += 1
    for v in range(net.n):
        start[v + 1] += start[v]
    to = [p[1] for p in pairs]; cap = [p[2] for p in pairs]
    idx = {(p[3], p[4]): q for q, p in enumerate(pairs)}
    rev = [idx[(p[3], 1 - p[4])] for p in pairs]
    fwd = [idx[(k, 0)] for k in range(m)]
    return start, to, cap, rev, fwd

def dinic(nV, start, to, cap, rev, s, t, trace):
    """Port of orcut._kernels._dinic; appends (level of t, flow at phase start) per phase to trace."""
    flow = 0
    level = [-1] * nV; it = [0] * nV; path = [0] * nV
    while True:
        for v in range(nV):
            level[v] = -1
        level[s] = 0; q = [s]; qh = 0
        while qh < len(q):
            u = q[qh]; qh += 1; lu = level[u] + 1
            for a in range(start[u], start[u + 1]):
                if cap[a] > 0 and level[to[a]] < 0:
                    level[to[a]] = lu; q.append(to[a])
            if level[t] >= 0:
                break
        if level[t] < 0:
            break
        lt = level[t]; trace.append((lt, flow, list(cap)))
        for v in range(nV):
            it[v] = start[v]
            if level[v] >= lt and v != t:
                level[v] = -1
        top = 0; u = s
        while True:
            if u == t:
                f = min(cap[path[p]] for p in range(top))
                for p in range(top):
                    a = path[p]; cap[a] -= f; cap[rev[a]] += f
                flow += f
                p0 = 0
                while cap[path[p0]] > 0:
                    p0 += 1
                top = p0; u = s if top == 0 else to[path[top - 1]]
                continue
            a = it[u]; end = start[u + 1]; lu = level[u] + 1
            while a < end:
                if cap[a] > 0 and level[to[a]] == lu:
                    break
                a += 1
            it[u] = a
            if a < end:
                path[top] = a; top += 1; u = to[a]
            else:
                level[u] = -1
                if top == 0:
                    break
                top -= 1; u = to[rev[path[top]]]; it[u] += 1
    return flow

# ------------------------------------------------------------------ measures and distances
def masses(G, u, measure, wts):
    nb = sorted(G[u]); d = len(nb)
    p = {"lazy": Fr(1, d + 1), "nonlazy": Fr(0), "half": Fr(1, 2), "weighted": Fr(1, d + 1)}[measure]
    w = {x: Fr(wts[frozenset((u, x))]) if measure == "weighted" else Fr(1) for x in nb}
    tot = sum(w.values())
    m = {x: (1 - p) * w[x] / tot for x in nb}
    if p:
        m[u] = p
    return m

def instance(G, i, j, measure, wts):
    mi, mj = masses(G, i, measure, wts), masses(G, j, measure, wts)
    mu = {x: mi.get(x, 0) - mj.get(x, 0) for x in set(mi) | set(mj)}
    lcm = 1
    for v in mu.values():
        lcm = lcm * v.denominator // math.gcd(lcm, v.denominator)
    ints = {x: int(v * lcm) for x, v in mu.items() if v != 0}
    g = 0
    for v in ints.values():
        g = math.gcd(g, abs(v))
    scale = Fr(lcm, g)                              # mu * scale = smallest integer masses
    mu_i = {x: v // g for x, v in ints.items()}
    P = sorted(x for x, v in mu_i.items() if v > 0); N = sorted(y for y, v in mu_i.items() if v < 0)
    dist = {x: nx.single_source_shortest_path_length(G, x, cutoff=3) for x in P}
    D = {(x, y): dist[x].get(y, 3) for x in P for y in N}
    return mu_i, P, N, D, scale

# ------------------------------------------------------------------ the networks
def make_net(G, i, j, inst, variant, merge):
    """variant: 'G3' | 'witness' | 'solver' (H1-H4 + explicit remaining) | 'solverw' (H1-H4 + witness relays)."""
    mu, P, N, D, _ = inst
    net = Net(); s = net.node("s"); t = net.node("t")
    Ui = set(G[i]) - set(G[j]) - {j}; Uj = set(G[j]) - set(G[i]) - {i}
    core = {i, j} | (set(G[i]) & set(G[j]))
    full_row = {x for x in P if x in Ui and all(D[(x, y)] <= 2 for y in N if y in Uj)}
    full_col = {y for y in N if y in Uj and all(D[(x, y)] <= 2 for x in P if x in Ui)}
    # successor description of every row, used for twin merging
    def succ(x):
        s2 = frozenset(y for y in N if D[(x, y)] == 1)
        s3 = frozenset(y for y in N if D[(x, y)] == 2)
        return (s2, s3, x in full_row, x in core, frozenset(w for w in G[x] if w not in (i, j)) if variant in ("witness", "solverw") else None)
    groups = defaultdict(list)
    for x in P:
        groups[succ(x) if merge else ("solo", x)].append(x)
    b1 = {y: None for y in N}; b2 = {y: None for y in N}
    for y in N:
        b1[y] = net.node("b", -mu[y]); b2[y] = net.node("b", 2 * -mu[y])
        net.arc(b1[y], t, -mu[y]); net.arc(b2[y], t, -mu[y])
    T = {s: None, t: None}
    row_nodes = []
    for key, xs in groups.items():
        cap = sum(mu[x] for x in xs)
        a3 = net.node("a3", cap); a2 = net.node("a2", 2 * cap)
        net.arc(s, a3, cap); net.arc(s, a2, cap); net.arc(a3, a2, INF)
        row_nodes.append((xs, a2, a3, cap))
    # implication arcs
    relays = {}
    def relay(name):
        if name not in relays:
            relays[name] = net.node("relay")
        return relays[name]
    for xs, a2, a3, cap in row_nodes:
        x = xs[0]
        for y in N:
            d = D[(x, y)]
            if d == 1:
                net.arc(a2, b1[y], INF); net.arc(a3, b2[y], INF)
        if variant == "G3":
            for y in N:
                if D[(x, y)] == 2:
                    net.arc(a3, b1[y], INF)
        elif variant == "witness":
            for w in G[x]:
                if any(w in G[y] for y in N):
                    net.arc(a3, relay(("w", w)), INF)
        else:  # solver: H1 all rows -> core cols; H2 core rows -> all cols; H3 full rows -> Uj cols; H4 non-full Ui rows -> full cols
            net.arc(a3, relay("H1"), INF)
            if x in core:
                net.arc(a3, relay("H2"), INF)
            if x in full_row:
                net.arc(a3, relay("H3"), INF)
            elif x in Ui:
                net.arc(a3, relay("H4"), INF)
                rest = [y for y in N if y in Uj and y not in full_col and D[(x, y)] == 2]
                if variant == "solver":
                    for y in rest:
                        net.arc(a3, b1[y], INF)
                else:
                    for w in G[x]:
                        if any(w in G[y] and y in rest for y in N):
                            net.arc(a3, relay(("w", w)), INF)
    if variant == "witness":
        for name, h in list(relays.items()):
            for y in N:
                if name[1] in G[y]:
                    net.arc(h, b1[y], INF)
    elif variant in ("solver", "solverw"):
        for y in N:
            if y in core and "H1" in relays:
                net.arc(relays["H1"], b1[y], INF)
            if "H2" in relays:
                net.arc(relays["H2"], b1[y], INF)
            if y in Uj and "H3" in relays:
                net.arc(relays["H3"], b1[y], INF)
            if y in full_col and "H4" in relays:
                net.arc(relays["H4"], b1[y], INF)
        if variant == "solverw":
            for name, h in relays.items():
                if isinstance(name, tuple):
                    for y in N:
                        if y in Uj and y not in full_col and name[1] in G[y]:
                            net.arc(h, b1[y], INF)
    return net, s, t, row_nodes, b1, b2

def warm_flow(G, inst, row_nodes, b1, b2, net, s, t, rng, csr):
    """Warm start: a max flow of the one-layer adjacency network, loaded on both layers."""
    mu, P, N, D, _ = inst
    one = Net(); s1 = one.node("s"); t1 = one.node("t")
    xs_n = [one.node("x") for _ in row_nodes]; ys_n = {y: one.node("y") for y in N}
    for k, (xs, a2, a3, cap) in enumerate(row_nodes):
        one.arc(s1, xs_n[k], cap)
        for y in N:
            if D[(xs[0], y)] == 1:
                one.arc(xs_n[k], ys_n[y], INF)
    for y in N:
        one.arc(ys_n[y], t1, -mu[y])
    st, to, cp, rv, fw = build_csr(one, rng)
    dinic(one.n, st, to, cp, rv, s1, t1, [])
    f1 = {(one.tail[k], one.head[k]): one.cap[k] - cp[fw[k]] for k in range(len(one.tail))}
    start, to2, cap2, rev2, fwd2 = csr
    pos = defaultdict(list)
    for k in range(len(net.tail)):
        pos[(net.tail[k], net.head[k])].append(fwd2[k])
    def push(u, v, f):
        a = pos[(u, v)][0]; cap2[a] -= f; cap2[rev2[a]] += f
    for k, (xs, a2, a3, cap) in enumerate(row_nodes):
        fx = f1.get((s1, xs_n[k]), 0)
        if fx:
            push(s, a2, fx); push(s, a3, fx)
        for y in N:
            f = f1.get((xs_n[k], ys_n[y]), 0)
            if f:
                push(a2, b1[y], f); push(a3, b2[y], f)
    for y in N:
        f = f1.get((ys_n[y], t1), 0)
        if f:
            push(b1[y], t, f); push(b2[y], t, f)
    return sum(f1.get((s1, x), 0) for x in xs_n) * 2

def check(net, s, t, inst, w1_ref, rng, warm, G):
    mu, P, N, D, scale = inst
    A = sum(v for v in mu.values() if v > 0)
    kind = net.kind
    for u, v in zip(net.tail, net.head):                      # (d) no relay next to a relay
        assert not (kind[u] == "relay" and kind[v] == "relay")
    csr = build_csr(net, rng)
    start, to, cap, rev, fwd = csr
    F0 = warm_flow(G, inst, *warm, net, s, t, rng, csr) if warm else 0
    trace = []
    F = F0 + dinic(net.n, start, to, cap, rev, s, t, trace)
    assert Fr(3 * A - F) / scale == w1_ref, ("wrong network", Fr(3 * A - F) / scale, w1_ref)   # (a)
    assert sum(net.T.values()) <= 6 * A                       # the bounds T(v) sum to at most 6A
    m = len(net.tail)
    fstar = [net.cap[k] - cap[fwd[k]] for k in range(m)]
    for v, Tv in net.T.items():                               # (d) every feasible flow obeys T(v)
        assert sum(fstar[k] for k in range(m) if net.head[k] == v) <= Tv
    for lt, f_at, snap in trace:                              # (e) g = f* - f enters v with <= 2T(v)
        f = [net.cap[k] - snap[fwd[k]] for k in range(m)]
        gin = defaultdict(int)
        for k in range(m):
            d = fstar[k] - f[k]
            if d > 0:
                gin[net.head[k]] += d
            elif d < 0:
                gin[net.tail[k]] += -d
        for v, Tv in net.T.items():
            assert gin[v] <= 2 * Tv, ("residual multiplicity", gin[v], Tv)
    worst = 0.0
    for lt, f_at, snap in trace:                              # (b) l <= 24A/M~ + 1
        Mt = F - f_at
        assert Mt > 0 and lt <= 24 * A / Mt + 1, ("phase bound", lt, A, Mt)
        worst = max(worst, (lt - 1) * Mt / (24 * A))
    phases = len(trace)
    assert phases <= 2 * math.sqrt(24 * A) + 1, ("phases", phases, A)                       # (c)
    return phases, A, worst

def w1_orcut(G, i, j, measure, wts):
    g = orcut.Graph.from_edges(list(G.edges()), num_nodes=G.number_of_nodes())
    E = [tuple(map(int, e)) for e in g.edges]
    eid = E.index((min(i, j), max(i, j)))
    m = {"lazy": "lazy", "nonlazy": "nonlazy", "half": Fr(1, 2)}.get(measure)
    if measure == "weighted":
        m = orcut.measures.Walk(weights=np.array([wts[frozenset(e)] for e in E]), idleness="lazy")
    r = orcut.curvature(g, m, edges=np.array([eid]), exact_fraction=True, n_jobs=1)
    return Fr(int(r["w1_num"][0]), int(r["w1_den"][0]))

def random_graph(rng):
    n = rng.randint(8, 60)
    kind = rng.choice(["gnp", "ba", "ws", "hubs"])
    if kind == "gnp":
        G = nx.gnp_random_graph(n, rng.choice([0.1, 0.2, 0.35]), seed=rng.randint(0, 10**9))
    elif kind == "ba":
        G = nx.barabasi_albert_graph(n, rng.choice([1, 2, 4]), seed=rng.randint(0, 10**9))
    elif kind == "ws":
        G = nx.watts_strogatz_graph(n, rng.choice([4, 6]), 0.3, seed=rng.randint(0, 10**9))
    else:  # two hubs with leaves, cross edges and connectors (like E2's controlled edges)
        a, b = rng.randint(3, 25), rng.randint(3, 25)
        G = nx.Graph(); G.add_edge(0, 1)
        L = list(range(2, 2 + a)); R = list(range(2 + a, 2 + a + b))
        G.add_edges_from((0, x) for x in L); G.add_edges_from((1, y) for y in R)
        for x in L:
            for y in R:
                if rng.random() < 0.05:
                    G.add_edge(x, y)
        c0 = 2 + a + b
        for c in range(rng.randint(0, 6)):
            G.add_edges_from((c0 + c, v) for v in rng.sample(L, min(len(L), 3)) + rng.sample(R, min(len(R), 3)))
        for k in range(rng.randint(0, 3)):
            G.add_edge(0, R[k % len(R)])  # common neighbours
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_edges() == 0 or not nx.is_connected(G):
        return None
    return nx.convert_node_labels_to_integers(G)

def matching_family(m):
    """Even-Tarjan's matching family G_m (their Section 4), embedded as an edge by Theorem 9."""
    G = nx.Graph(); G.add_edge(0, 1)
    L = {(i, j): 2 + len([1 for _ in range(0)]) for i in range(1, m + 1) for j in range(1, i + 1)}
    idx = {}
    for (i, j) in [(i, j) for i in range(1, m + 1) for j in range(1, i + 1)]:
        idx[("a", i, j)] = len(idx) + 2
    for (i, j) in [(i, j) for i in range(1, m + 1) for j in range(1, i + 1)]:
        idx[("b", i, j)] = len(idx) + 2
    for key, v in idx.items():
        G.add_edge(0 if key[0] == "a" else 1, v)
    for i in range(1, m + 1):
        for j in range(1, i + 1):
            G.add_edge(idx[("a", i, j)], idx[("b", i, j)])
            if j < i:
                G.add_edge(idx[("a", i, j)], idx[("b", i, j + 1)])
    return G

if __name__ == "__main__":
    rng = random.Random(7)
    stats = defaultdict(int); worst = 0.0; max_ratio = 0.0
    n_edges = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    done = 0
    while done < n_edges:
        G = random_graph(rng)
        if G is None:
            continue
        i, j = rng.choice(list(G.edges()))
        wts = {frozenset(e): rng.randint(1, 4) for e in G.edges()}
        measure = rng.choice(["lazy", "nonlazy", "half", "weighted"])
        inst = instance(G, i, j, measure, wts)
        if not inst[1]:
            continue
        ref = w1_orcut(G, i, j, measure, wts)
        for variant in ("G3", "witness", "solver", "solverw"):
            for merge in (False, True):
                for warm in (False, True):
                    net, s, t, rows, b1, b2 = make_net(G, i, j, inst, variant, merge)
                    ph, A, w = check(net, s, t, inst, ref, rng, (rows, b1, b2) if warm else None, G)
                    stats["networks"] += 1; worst = max(worst, w)
                    max_ratio = max(max_ratio, ph / (2 * math.sqrt(24 * A) + 1))
        done += 1
    print(f"random edges: {done}, networks: {stats['networks']}, all checks passed")
    print(f"  tightest phase step: (l-1) M~ / 24A = {worst:.3f} (must be <= 1)")
    print(f"  most phases relative to the bound: {max_ratio:.3f} (must be <= 1)")
    for m in (4, 6, 8, 10, 12):
        G = matching_family(m)
        inst = instance(G, 0, 1, "lazy", {})
        ref = w1_orcut(G, 0, 1, "lazy", {})
        mx = 0
        for rep in range(40):
            for variant in ("G3", "witness"):
                net, s, t, rows, b1, b2 = make_net(G, 0, 1, inst, variant, False)
                ph, A, w = check(net, s, t, inst, ref, rng, None, G)
                mx = max(mx, ph)
        print(f"  Even-Tarjan family m={m}: A={A}, most phases over 80 runs {mx}, bound {2*math.sqrt(24*A)+1:.1f}")
