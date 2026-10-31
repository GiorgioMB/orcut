# Changelog

## Unreleased

- Experiments, paths: `main.py` gives every step absolute paths and runs it from the directory it
  was started from. It used to run the steps from `experiments/` and pass the relative `--out` and
  `--data-dir` of the SLURM scripts unchanged, so the result tables went to `experiments/results/`
  while the logs and `status.json` went to `results/`, the downloads went to
  `experiments/experiments/data/real/` instead of the cache filled on the login node, and
  `artifacts` did not find `environment.json`. Tables left in the old place by such a run stop
  `main.py` until they are moved (it prints the command). The SLURM scripts find the repository
  wherever `sbatch` is called from (`ORCUT_ROOT` overrides), and downloads and the data cache are
  written under process-unique temporary names (steps of a job array may fetch the same graph at
  once).
- Experiments, checkpoints: `main.py` records every step in `<out>/status/<step>.json` and does not
  run again a step that ended without errors (`--force` runs it anyway, `--status` shows every
  step); a step that failed, was killed or was interrupted runs again and resumes. The
  `status.json` of earlier runs is read. Rows that hold no result (an error, or GraphRicciCurvature
  unavailable) no longer count as done, and a step that ends with error rows exits with code 3.
  `artifacts` runs again only when the result tables changed, and E5 (whole) and E6 (BORF) once GRC
  becomes available. A lock per running step keeps two jobs from running it at once (both would
  rewrite its tables); SIGTERM (time limit, `scancel`) is recorded as an interruption. A resumed
  step no longer redoes finished work before skipping it (E1 recomputed the oracle on all edges, E3
  part threads a warm-up solve, E5 the exact values, E6 regenerated every LFR graph), and computes
  what an uninterrupted run would: the same samples (E1, E2 part B), the methods that ran out of
  budget stay off (E2 parts A and C), values checked against the full configuration (E4), times
  relative to the lazy measure (E7; they were NaN after a restart).

## 1.2.0

- Witness relays (`F_WITNESS`, on by default: `F_ALL = 31`). A pair of P x N at distance 2 has a
  common neighbour w, and every pair in (N(w) & P) x (N(w) & N) is within distance 2, so the
  implications of these pairs can go through one zero-weight relay per common neighbour. When the
  pairs left at distance 2 after the relays H1--H4 would need many explicit arcs, the kernel routes
  them through per-witness relays instead, if this is at least four times smaller (an upper bound
  is checked first, so the test costs little when it fails). Values are unchanged (tested bit for
  bit against the previous kernels); diagnostics report the new path `PATH_CUTW = 4`.
- Bounded bitset work (part of `F_WITNESS`; `core_ids(..., dispatch=4)`): when the bitsets of an
  edge would cost more than four times its local input (the sum of the degrees of the rows and
  columns), before they are built or during the coverage while most rows are not yet within
  distance 2 of all of U_j, the edge is solved on the witness network instead (`PATH_WIT = 5`).
  On large sparse neighbourhoods this is 2--13x faster; on the real graphs we tried it never
  triggers and costs under 3%.
- `orcut._witness.witness_ids`: the linear-size witness network on its own (one relay per vertex
  adjacent to both P and N; twin merging; direct arcs for witnesses that reach one row class or one
  column; warm start). It has at most 3 m_ij + 2 nodes and 5 m_ij arcs, m_ij the sum of the degrees
  of P and N, and is built in O(m_ij) from the adjacency lists, with no bitsets and no distances.
  Exact fractions identical to `orcut.curvature` (tested for six measures and every flag).
- Experiments: the graph min-cost-flow formulation of W1 (Beckmann's problem on the edge ij and the
  edges with an endpoint in P u N, OR-Tools) as a baseline in E2 and E4; the solver without witness relays and the
  witness network alone in E2 and E4; network sizes against the local input in E2; E2 part C, the
  lower-bound instances of DasGupta, Grigorescu and Mukherjee (2023) up to n = 2^20.

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
