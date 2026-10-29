# Changelog

## 1.1.0

- `orcut.curvature(graph, measure)`: the general one-step kernel (`orcut/_core.py`) for lazy,
  non-lazy, LLY, uniform idleness p, and weighted walks (`orcut.Walk`: positive integer weights,
  rational idleness), in exact integer arithmetic (common denominators up to 2^50; edges above it
  are NaN and flagged). Optional exact W1 fractions and per-edge diagnostics (path, nodes, arcs, row
  groups after twin merging, warm-start flow). Every engineering step can be switched off
  (`flags`) for ablations; values do not change (tested).
- The TLW certificate is no longer computed by the new kernel (it stays in the frozen kernel).
- `orcut.certify(graph, edge, measure)`: an optimal transport plan and an optimal potential, read
  off the minimum cut and a maximum flow on the tight pairs, verified exactly; `Certificate.verify`
  re-checks them from scratch.
- `orcut.ball_curvature`, `orcut.walk_curvature`: the layered closure (D = 2r + 1) for measures
  supported on balls of radius r (reference implementation, explicit distances).
- `orcut.affected_edges(graph, pairs)`: the edges whose curvature can change after inserting or
  deleting vertex pairs (an endpoint in the union of N[a] u N[b]), for exact incremental updates
  in rewiring and edge-removal algorithms.
- Certificates of one-step measures take the P x N distances from adjacency and 2-paths (numba)
  instead of a BFS per vertex; `Certificate.verify` explores distances only up to the largest
  potential difference (exactly for the pairs of the plan). An edge whose two measures coincide is
  reported as certified by the layered closure.
- Tests: the new kernel against the frozen one (bit for bit, 16 flag combinations) and against
  linear programming; layered closure against LP; certificates; kappa_p = (1 - p) kappa_LLY for
  p >= 1/(deg_max + 1); locality of `affected_edges` under random insertions and deletions.
- `experiments/`: the experiments of the paper (E1--E7), with `main.py`, SLURM scripts, and
  `artifacts.py`, which writes every number, table and figure of the paper.

## 1.0.0

- The exact solver of `curvature_transfer` v11, extracted without changes (`orcut/_kernels.py`,
  `orcut/_reference.py`), with its tests.
