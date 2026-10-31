# Transport-free Ollivier-Ricci curvature

Ollivier–Ricci curvature of every edge of an unweighted graph, with one
minimum cut per edge, in exact integer arithmetic and without a distance matrix.

Any two vertices of the closed neighbourhoods of an edge are at distance at most
three, so the Kantorovich dual of the edgewise transport problem has an integral
optimum with potentials in {0, 1, 2, 3}. Written in unary, it is a maximum-weight
closure, solved by one s–t minimum cut. Its arcs come from adjacency and common
neighbours only (pairs at distance three give none), and with one relay node per
common neighbour the network is linear in the adjacency lists of the vertices that
carry net mass. The same holds for any one-step measures (lazy, non-lazy, any idleness,
weighted walks), and, with 2r layers, for measures supported on balls of radius r.

```python
import orcut
g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a networkx graph / PyG Data
orcut.curvature(g)                      # lazy (idleness 1/(deg+1)), aligned with g.edges (u < v)
orcut.curvature(g, "lly")               # Lin–Lu–Yau; also "nonlazy", a number p (idleness p),
orcut.curvature(g, orcut.Walk(weights=w, idleness=0.5))   # or a weighted walk (integer weights)
orcut.curvature(g, exact_fraction=True) # also W1 as an exact integer fraction per edge
c = orcut.certify(g, (0, 1))            # an optimal plan and an optimal 1-Lipschitz potential
c.verified, c.verify(g)                 # checked exactly when built, and re-checkable from scratch
orcut.ball_curvature(g, 2)              # Ollivier's ball measures, radius 2 (layered closure)
orcut.walk_curvature(g, 2)              # two-step lazy random walk
orcut.affected_edges(g2, changed_pairs) # after inserting/deleting pairs: the only edges to recompute
```

## Installation

`orcut` is on PyPI as [`faster-or-curvature`](https://pypi.org/project/faster-or-curvature/), but that package is the solver alone; for the experiments and everything else beyond the solver, install from this repository. Without any package index (for example on a cluster without PyPI access), use the conda-forge environment:
```bash
conda env create -f environment.yml && conda activate orcut   # numpy, scipy, numba, + test/experiment deps
pip install --no-index --no-deps --no-build-isolation -e .
pytest -q
```

## Versions
* **1.2.1**: fixed the unpickler for downloaded Planetoid datasets to support loading Python 2 `__builtin__` containers.
* **1.2.0**: witness relays (one zero-weight relay per common neighbour) for the pairs at distance 2
  that the bitset network would list one by one, chosen per edge when the network gets smaller; a
  bound on the bitset work (four words per adjacency entry) beyond which an edge is solved on the
  linear-size witness network (`orcut._witness`, also usable on its own); graph min-cost flow
  (Beckmann) and the lower-bound instances of DasGupta et al. (2023) in the experiments. Values
  unchanged (tested).
* **1.1.0**: `orcut.curvature` (any one-step measure, exact fractions, diagnostics, ablation
  flags), certificates (`orcut.certify`), the layered closure for supports of radius r
  (`ball_curvature`, `walk_curvature`), exact incremental updates (`affected_edges`), and the
  experiments of the paper (`experiments/`). The kernel of 1.0.0 is kept unchanged in
  `orcut/_kernels.py`, and `orcut.curvature` returns the same values bit for bit (tested).
* **1.0.0** is the solver used for every exact curvature in Micaletto and Nigrelli,
  [*From Proxy to Certificate: Two-Sided Ollivier--Ricci Bounds from Balanced Forman Curvature*](https://arxiv.org/abs/2603.13535): 
 `curvature-transfer` depends on `orcut==1.0.0` (tag `v1.0.0`) to have values and timings reproduce that paper. 

## Layout

| Path | Contents |
| --- | --- |
| `orcut/graph.py` | CSR graph, canonical undirected edges, networkx / PyG / edge-list constructors |
| `orcut/_kernels.py` | the 1.0.0 kernel (frozen): lazy, non-lazy, LLY, and the TLW certificate |
| `orcut/_core.py` | the general one-step kernel: per-vertex masses, bitsets, relays, witness relays, ablation flags, diagnostics |
| `orcut/_witness.py` | the linear-size witness network on its own (one relay per common neighbour; no bitsets, no distances) |
| `orcut/_layered.py` | W1 on bounded integer metrics by a layered closure; ball and multi-step measures; certificates |
| `orcut/measures.py` | measure specifications (`"lazy"`, `"nonlazy"`, `"lly"`, idleness p, `Walk`) |
| `orcut/certify.py` | per-edge certificates (plan and potential) and their exact verification |
| `orcut/_reference.py` | independent oracle: full local distance matrix, three-level closure, recursive Dinic |
| `orcut/api.py` | `curvature`, `ball_curvature`, `walk_curvature`, `affected_edges`, `exact` (1.0.0 API), thread control |
| `tests/` | kernels against each other (bit for bit), against the oracle, and against linear programming |
| `experiments/` | the experiments of the paper (see `experiments/README.md`) |
