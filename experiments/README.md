# Experiments

`main.py` runs the experiments of the paper, and `artifacts.py` writes every number, table and
figure of the paper from their result tables (`results/paper/`: `solver_numbers.tex`, read by
`\res{...}` and `\resrows{...}`; `KEYS.md`, the source of every number; `fig-*.pdf`, drawn at a
text width of 5.5 in, or 6.75 in for a two-column layout with `--fig-width 6.75`).

| | | script | paper |
|---|---|---|---|
| E1 | exactness, certificates | `e1_exactness.py` | Sec. 6.1 |
| E2 | cost of one edge, lower-bound instances | `e2_per_edge.py --part A\|B\|C` | Sec. 6.2 |
| E3 | whole graphs, threads, random families | `e3_scale.py --part real\|threads\|synth` | Sec. 6.3 |
| E4 | ablation | `e4_ablation.py` | Sec. 6.3 |
| E5 | GraphRicciCurvature, proxies, Sinkhorn | `e5_approx.py --part whole\|sinkhorn` | Sec. 6.4 |
| E6 | other measures, balls, multi-step walks | `e6_generality.py --part measures\|radius` | App. D |

```bash
python experiments/datasets.py --data-dir experiments/data/real   # download the 20 networks once
python experiments/main.py --quick --out results-quick             # the whole pipeline at toy size
python experiments/main.py --threads 32                            # everything (days)
sbatch experiments/slurm/array.sbatch                              # or one step per Slurm task
python experiments/main.py --status
```

A step that ends without errors is not run again (`--force` recomputes it), and a step that is
stopped resumes where it stopped: rows, parts of rows, and the chunks of long solves are saved as
they are computed. The protocol is that of App. C of the paper. GraphRicciCurvature (E5) needs
`scipy <= 1.13` and runs in its own environment (`grc-environment.yml`), found next to the `orcut`
one or given by `--grc-python`; without it, its rows stay `unavailable`. The step `calib` times two
graphs again on the type of node it runs on (`sbatch --array=12 --constraint=<type>
experiments/slurm/array.sbatch` for each type), and before the artifacts `main.py` traces every row
to its node with `tools/provenance.py` when Slurm's `sacct` is available.
