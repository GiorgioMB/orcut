# orcut

Exact Ollivier–Ricci curvature of every edge of an unweighted graph, with one
minimum cut per edge, in exact integer arithmetic and without a distance matrix.

Any two vertices of the closed neighbourhoods of an edge are at distance at most
three, so the Kantorovich dual of the edgewise transport problem has an integral
optimum with potentials in {0, 1, 2, 3}. Written in unary, it is a maximum-weight
closure, solved by one s–t minimum cut on 2|P| + 2|N| + 2 nodes.

```python
import orcut
g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a networkx graph
out = orcut.exact(g)          # {"c_OR": lazy, "c_OR0": non-lazy, "c_LLY": Lin–Lu–Yau}
out["c_OR"]                   # aligned with g.edges (u < v)
```

## Installation

```bash
pip install -e .            # numpy, scipy, numba
pip install -e ".[dev]"     # + pytest, hypothesis, networkx
pytest -q
```

## Versions

* **1.0.0** is the solver used for every exact curvature in Micaletto and Nigrelli,
  *What Balanced Forman Curvature Certifies About Ollivier–Ricci Curvature*: its
  compiled kernels (`orcut/_kernels.py`) are those of `curvature_transfer` v11,
  unchanged, so values and timings reproduce that paper. `curvature-transfer`
  depends on `orcut==1.0.0`.

## Layout

| Path | Contents |
| --- | --- |
| `orcut/graph.py` | CSR graph, canonical undirected edges, networkx / PyG / edge-list constructors |
| `orcut/_kernels.py` | Numba kernels: distance-free bitsets, twin merging, block relays, Dinic, exact W1 |
| `orcut/_reference.py` | independent oracle: full local distance matrix, three-level closure, recursive Dinic |
| `orcut/api.py` | `exact(...)`, thread control |
| `tests/` | the kernel against the oracle (bit for bit) and against linear programming |
