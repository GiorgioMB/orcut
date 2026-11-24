# orcut

Exact Ollivier–Ricci curvature of every edge of a graph, with one minimum cut per edge: exact
integer arithmetic, no distance matrix, no linear program. The repository holds the solver
(`orcut/`), its tests, and the experiments of the paper (`experiments/`); `v1.4.4` is the version of
the paper.

```python
import orcut
g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a networkx graph
orcut.curvature(g)                        # lazy walk, aligned with g.edges (u < v)
orcut.curvature(g, "lly")                 # also "nonlazy", an idleness p, or orcut.Walk(weights, idleness)
orcut.curvature(g, exact_fraction=True)   # and W1 as an exact fraction per edge
orcut.certify(g, (0, 1))                  # an optimal plan and potential, verified exactly
orcut.ball_curvature(g, 2)                # uniform measures on balls of radius 2
orcut.walk_curvature(g, 2)                # two-step lazy walk
orcut.affected_edges(g, pairs)            # after changing these pairs of g: the only edges to recompute
```

## Install

```bash
conda env create -f environment.yml && conda activate orcut
pip install --no-index --no-deps --no-build-isolation -e .
pytest -q
```

The solver alone is on PyPI as [`faster-or-curvature`](https://pypi.org/project/faster-or-curvature/).

## Layout

| Path | Contents |
| --- | --- |
| `orcut/_core.py` | the solver: bitsets, relays, witness relays, twin merging, warm-started Dinic |
| `orcut/_witness.py` | the linear-size witness network |
| `orcut/_layered.py` | the layered closure for costs up to D (balls, multi-step walks) |
| `orcut/certify.py` | certificates (an optimal plan and potential), checked exactly |
| `orcut/_kernels.py` | the solver of 1.0.0, kept unchanged (`orcut.exact`) |
| `orcut/_oracle.py` | the oracle: explicit distances, three-level closure, its own max flow |
| `orcut/api.py`, `graph.py`, `measures.py` | entry points, CSR graphs, measures |
| `tests/` | the kernels against each other, the oracle and linear programming; the theorems, numerically |
| `experiments/` | E1–E6 of the paper ([experiments/README.md](experiments/README.md)) |
| `tools/` | `check_formulas.py`, `check_phases.py` (the theorems on larger samples), `provenance.py` (the node behind each row) |
