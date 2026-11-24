# Experiments

`main.py` runs the experiments of the paper, and `artifacts.py` writes every number, table and
figure of the paper from their result tables (`results/paper/`: `solver_numbers.tex`, read by
`\res{...}` and `\resrows{...}`; `KEYS.md`, the source of every number; `fig-*.pdf`).

The paper exists in three versions, each a LaTeX project with its own folder of artifacts, all
written from the same result tables by

```bash
python experiments/paper_dirs.py --results results --out paper   # paper/arXiv, paper/LoG, paper/ICML
```

| folder | version | figures | solver's name |
|---|---|---|---|
| `arXiv/generated` | preprint (`log_2026`, `[preprint]`) | one column of 5.5 in | `orcut` |
| `LoG/generated` | LoG review (`log_2026`, `[review]`) | one column of 5.5 in | `ours` |
| `ICML/ICML_generated` | ICML review (`icml2027`) | 3.25 in (main text), 6.75 in (appendix) | `ours` |

The figures are typeset by LaTeX in Times (`--fonts type1`, the default; it needs `latex` on the
PATH), so that the PDFs contain Type-1 fonts only, as LoG and ICML require; every figure is checked
when it is written, and a Type-3 font stops the run. `--fonts truetype` works without LaTeX.
`artifacts.py` alone takes `--layout log|icml`, `--method-name` and `--strip-zeros keys|all|none`
(numbers whose decimals are all zeros lose them in the keys: 6.0x is written 6x). Each folder also
records how it was written (`ArtifactsLayout`, `ArtifactsMethod`, `ArtifactsFonts`), and each paper
prints a red line on its first page if its folder was written for another version.

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
