# Changelog

## 1.3.1

The solver is unchanged from 1.3.0 (the same values); the changes are in the experiments, the
paper's artifacts, and the tools.

- GraphRicciCurvature's memory (`experiments/baselines.py`): its ATD needs about 48 n^2 bytes at
  its peak for the all-pairs table it builds from n^2 Python floats (measured on the whole GRC
  process tree), not 40 n^2, so a 300 GB job could start Epinions (n = 75,877) and be killed. The
  estimate is now the measured one, and an ATD row skipped under the old estimate is rewritten
  with the new one without loading the graph. `--grc-atd-apsp array` (E5) builds the same table as
  NetworKit's array (`grc_runner.py --apsp-array`, about 16 n^2 bytes, the same values bit for bit;
  recorded in the `apsp` column).
- A GRC process killed by a signal used to leave an empty `error: ` (no stderr). The job's cgroup
  OOM counter (`partial.oom_kill_count`) now tells a kill at the memory limit, recorded as
  `out of memory with <job's> GB` and retried only by a job with more memory, from other signals
  and exit codes, which are named in the status.
- `experiments/main.py`: a step that completed with rows skipped for memory (or killed at the memory
  limit) runs again, for these rows, in a job with more memory than the one that completed it; the
  step records the job's memory. E6 is not part of the current paper: its steps (`e6-borf`,
  `e6-community`) run only when named in `--steps`, and `array.sbatch` skips tasks 10 and 11 unless
  asked; the step list and its indices are unchanged.
- `experiments/artifacts.py`: no SyntaxWarning (a raw docstring) and no All-NaN warnings (NaN-aware
  statistics of empty columns give NaN quietly; the numbers are unchanged); figure legends get as
  many columns as fit and the room they need; figures are drawn at the width they are printed at
  (`FIG_W`, and `COMMUNITY_W` for the half-width community figure); the trade-off figure shows the
  regularization by marker and gives every graph its own colour; the table of graphs with metadata
  labels adds the modularity of the returned partition and its number of groups.
- `tools/anonymize.py`: the earlier paper's current title and arXiv identifier, third-person wording
  ("an earlier paper"), no venue labels or pointer to this paper's preprint, no sentence naming the
  package on PyPI; the copy is also checked for paths under a user's folder and for links that
  would point to the placeholder; `--show` prints every changed line.
- Tests: the classification of killed GRC runs, the cgroup counter, the ATD estimate on a graph of
  Epinions' size, the reruns with more memory, and the paper's new results checked numerically
  (`tests/test_theory.py`: the matching identity, the min-max formula, and the bound on Dinic's
  phases on every kind of network the solver builds; the full checks are `tools/check_formulas.py`
  and `tools/check_phases.py`).

## 1.3.0

The version of the arXiv paper. The solver is unchanged from 1.2.0 (the same values); the changes
are in the experiments, which can now run for weeks on a cluster, be stopped at any point, and
resume without losing or repeating work.

- Pipeline (`experiments/main.py`): every step gets absolute paths. main.py up to 1.2.0 ran the
  steps from `experiments/` with the relative `--out` and `--data-dir` of the SLURM scripts, so
  tables went to `experiments/results/` and downloads to `experiments/experiments/data/real/`; what
  such a run left there is moved into place (a table found in both places is merged row by row).
  A step that ends without errors is recorded in `<out>/status/<step>.json` and not run again;
  `--status` lists the steps; `--force` recomputes the selected steps from scratch, after moving
  their tables to `<out>/replaced/`; options after `--` go to the step itself; a step whose rules
  changed runs again for the rows they now compute. A lock per running step keeps two jobs from
  running it at once, and the lock of a job that Slurm reports as ended is taken over at once;
  SIGTERM (time limit, `scancel`) is recorded as an interruption.
- Checkpoints inside the steps: rows that hold no result (`error: ...`, `unavailable`) are computed
  again, and a step that ends with such rows exits with code 3. Long work is saved as it goes (each
  method's solve in E2, each timed edge of E3's POT estimate and of E5's Sinkhorn part, each chunk
  of E3's long solves), and a part killed by its job's memory or time limit, as reported by Slurm,
  is tried again only by a job with more of it (`experiments/partial.py`). A resumed step draws the
  same samples as an uninterrupted one.
- No time budgets: E2 parts A and C time every method at every size (the 30 s rule and the fixed
  caps of the dense solvers are gone; a method is skipped only if its problem cannot fit in the
  job's memory, with the reason in `<method>_status`), except HiGHS, which runs on LPs of up to
  4098^2 variables (`--highs-max-vars`: the sym sweep up to s = 4096 and the whole hub sweep; it
  grows about tenfold per doubling of s, half an hour per solve at s = 4096, days beyond); E3's
  POT estimate and E5's
  Sinkhorn part time every sampled edge, with dense problems up to what fits in the job's memory;
  GraphRicciCurvature runs (E5, E6) have no time limit.
- E3: graphs above 10^7 edges, and synthetic rows predicted to take over an hour, are solved in one
  pass by saved chunks (balanced by the size of their two-hop scans); their time is the sum over
  the chunks. Skitter and LiveJournal also get the non-lazy and LLY measures (Orkut: lazy only).
  Synthetic rows are matched by their requested size (`n_target`: rows recorded only the size of
  the largest component, and were computed again on every run), and rows predicted to take over
  two days per solve run after all the others.
- E5: GraphRicciCurvature's ATD gets all-pairs distances (in its pairwise mode, used above 2 x 10^4
  vertices, it fails with a KeyError). E7 (radius): large supports run with fewer threads instead
  of being skipped.
- SLURM scripts: both find the repository wherever `sbatch` is called from and pass their
  arguments on to `main.py`; `array.sbatch` also finds the `orcut` and `orcut-grc` environments
  itself (none needs to be active) and stops a GRC step at once if GraphRicciCurvature does not
  import.
- Environments: `environment.yml` and `experiments/grc-environment.yml` pin the versions that ran
  the experiments of the paper; `pyproject.toml` pins numpy 2.2.5 and scipy 1.16.3 accordingly.

## 1.2.1

- Experiments: the unpickler of the downloaded Planetoid datasets accepts the Python 2 name of the
  builtins (`__builtin__`) that the files use; Cora, CiteSeer and PubMed did not load.

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
