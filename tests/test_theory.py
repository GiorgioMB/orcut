"""The paper's new results, checked numerically on small random instances: the matching identity
(curvature is at least as hard as bipartite matching), the min-max formula, and the bound on the
phases of Dinic's algorithm on every kind of network the solver builds. The full checks are
tools/check_formulas.py and tools/check_phases.py."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import check_formulas as CF  # noqa: E402
import check_phases as CP  # noqa: E402


def test_matching_identity():
    assert CF.check_matching(150, random.Random(1)) == 0


def test_min_max_formula():
    bad, done, _ = CF.check_hall(250, random.Random(2))
    assert bad == 0 and done >= 100


@pytest.mark.parametrize("seed", [3, 4])
def test_dinic_phases_on_every_network(seed):
    """G3, the witness network and the solver's networks, with and without twin merging and the
    warm start: the maximum flow gives orcut's W1, and every step of the phase bound holds."""
    rng, done = random.Random(seed), 0
    while done < 12:
        G = CP.random_graph(rng)
        if G is None:
            continue
        i, j = rng.choice(list(G.edges()))
        wts = {frozenset(e): rng.randint(1, 4) for e in G.edges()}
        measure = rng.choice(["lazy", "nonlazy", "half", "weighted"])
        inst = CP.instance(G, i, j, measure, wts)
        if not inst[1]:
            continue
        ref = CP.w1_orcut(G, i, j, measure, wts)
        for variant in ("G3", "witness", "solver", "solverw"):
            for merge in (False, True):
                for warm in (False, True):
                    net, s, t, rows, b1, b2 = CP.make_net(G, i, j, inst, variant, merge)
                    CP.check(net, s, t, inst, ref, rng, (rows, b1, b2) if warm else None, G)  # asserts
        done += 1


def test_dinic_phases_on_the_hard_matching_family():
    """Even and Tarjan's family needs about sqrt(2A) phases: within the bound."""
    G = CP.matching_family(8)
    inst = CP.instance(G, 0, 1, "lazy", {})
    ref = CP.w1_orcut(G, 0, 1, "lazy", {})
    rng = random.Random(5)
    for _ in range(10):
        net, s, t, rows, b1, b2 = CP.make_net(G, 0, 1, inst, "witness", False)
        phases, A, _ = CP.check(net, s, t, inst, ref, rng, None, G)
        assert phases >= 3
