"""Hardware, for artifacts.py: calibration across node types, and the processor behind each row.

A run can resume on another type of node, so the rows of a table may come from different
processors. Two optional inputs say how much that matters (without them nothing is produced):
``results/calib/<type>/e3/real.csv``, a few graphs timed again on one node of each type (step
``calib``; ``<type>/cpu.txt`` names its processor), gives the table ``calib`` and the keys
``ECalib...``; ``results/provenance/rows.csv``, from ``tools/provenance.py``, gives the table
``provenance`` and the keys ``EProv...``. The processor of a node comes from
``results/provenance/nodes.txt``, which main.py writes, or else from ``results/provenance/types.txt``
(one line per kind of node: the letters that start its names, then its processor).
"""
from __future__ import annotations

import csv
import glob
import os
import re
from collections import Counter, defaultdict

import numpy as np

# ------------------------------------------------------------------ calibration across node types


def load_calib(results: str):
    """{node type label: DataFrame of its lazy rows of e3/real.csv}."""
    import pandas as pd
    out = {}
    for path in sorted(glob.glob(os.path.join(results, "calib", "*", "e3", "real.csv"))):
        root = os.path.dirname(os.path.dirname(path))
        label = os.path.basename(root)
        cpu = os.path.join(root, "cpu.txt")
        if os.path.exists(cpu):
            label = open(cpu).read().strip() or label
        d = pd.read_csv(path)
        out[label] = d[d["measure"].astype(str) == "lazy"] if "measure" in d else d
    return out


def block_calib(results: str, O, A) -> None:
    """A is the artifacts module (its formatting helpers and Out)."""
    runs = load_calib(results)
    if not runs:
        return
    graphs = sorted({g for d in runs.values() for g in d["graph"]}, key=lambda g: min(
        float(d[d["graph"] == g]["m"].iloc[0]) for d in runs.values() if g in set(d["graph"])))
    lines, spread_t, spread_p, spread_r, solve = [], [], [], [], []
    for g in graphs:
        t, p, r = [], [], []
        for label, d in runs.items():
            row = d[d["graph"] == g]
            if not len(row):
                continue
            row = row.iloc[0]
            ours = float(row["time_s"])
            pot = float(row.get("pot_est_s", np.nan))
            pot_solve = float(row.get("pot_solve_est_1thread_s", np.nan)) / float(row.get("threads", 1))
            t.append(ours); p.append(pot); r.append(pot / ours)
            if np.isfinite(pot_solve):
                solve.append(pot_solve / ours)
            lines.append(" & ".join([A.tex_name(label), A.tex_name(g), A.f_time(ours), A.f_time(pot),
                                     A.f_x(pot / ours), A.f_x(pot_solve / ours)]) + r" \\")
        lines.append(r"\addlinespace")
        if len(t) > 1:
            spread_t.append(max(t) / min(t))
            if all(np.isfinite(p)):
                spread_p.append(max(p) / min(p)); spread_r.append(max(r) / min(r))
    if lines:
        lines.pop()
    O.table("calib", lines)
    O.put("ECalibNodeTypes", A.f_int(len(runs)), "calib: node types timed")
    names = [A.tex_name(g) for g in graphs]
    O.put("ECalibGraphs", " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + ", and " + names[-1],
          "calib: graphs timed on every node type")
    if spread_t:
        O.put("ECalibTimeSpreadMax", A.f_x(max(spread_t)), "calib: max over graphs of slowest/fastest node type, our time")
    if spread_p:
        O.put("ECalibPotSpreadMax", A.f_x(max(spread_p)), "calib: the same for POT's estimate")
        O.put("ECalibRatioSpreadMax", A.f_x(max(spread_r)), "calib: the same for POT's estimate over our time")
    if solve:
        O.put("ECalibSolveRatioMin", A.f_x(min(solve)), "calib: min of POT's solve alone over our time")
        O.put("ECalibSolveRatioMax", A.f_x(max(solve)), "calib: max of POT's solve alone over our time")


# ------------------------------------------------------------------ the processors behind each row

PROV_STEPS = [  # step, its name in the table, key stem if its comparisons span several rows of a graph
    ("e1", "E1", None), ("e2-controlled", "E2, controlled edges", None), ("e2-real", "E2, real graphs", None),
    ("e2-dasgupta", "E2, lower bounds", None), ("e3-real", "E3, real networks", None),
    ("e3-threads", "E3, threads", None), ("e3-synth", "E3, random graphs", None), ("e4", "E4", "Four"),
    ("e5-whole", "E5, whole graphs", "FiveWhole"), ("e5-sinkhorn", "E5, Sinkhorn", "FiveSinkhorn"),
    ("e6-measures", "E6, measures", "SixMeasures"), ("e6-radius", "E6, balls and walks", "SixRadius"),
]


def load_types(path: str) -> dict:
    types = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            prefix, _, cpu = line.partition(" ")
            types[prefix] = cpu.strip()
    return types


def processor(nodes: str, types: dict, hosts: dict = None) -> str:
    """The processor of a row's candidate nodes, or '' if they differ (or are unknown). ``hosts``
    names the processor of single nodes, ``types`` that of the nodes whose names start with a prefix."""
    kinds = set()
    for n in filter(None, nodes.split(";")):
        m = re.match(r"[A-Za-z]+", n)
        kinds.add((hosts or {}).get(n) or types.get(m.group() if m else n, ""))
    return kinds.pop() if len(kinds) == 1 else ""


def block_provenance(results: str, O, A) -> None:
    rows_path = os.path.join(results, "provenance", "rows.csv")
    types_path = os.path.join(results, "provenance", "types.txt")
    if not os.path.exists(rows_path):
        return
    nodes_path = os.path.join(results, "provenance", "nodes.txt")
    types = load_types(types_path) if os.path.exists(types_path) else {}
    hosts = load_types(nodes_path) if os.path.exists(nodes_path) else {}
    if not types and not hosts:
        O.missing.append("provenance/nodes.txt or types.txt (the processor of each node; tools/provenance.py)")
        return
    per_step, graphs = defaultdict(Counter), defaultdict(lambda: defaultdict(set))
    total = undetermined = 0
    for r in csv.DictReader(open(rows_path)):
        cpu = processor(r["nodes"], types, hosts)
        per_step[r["step"]][cpu or "ND"] += 1
        graphs[r["step"]][r["key"].split(" | ")[0]].add(cpu or "?")
        total += 1
        undetermined += not cpu
    lines = []
    for step, name, stem in PROV_STEPS:
        if step not in per_step:
            continue
        c = per_step[step]
        cells = ", ".join(f"{A.tex_name(cpu)} ({n})" for cpu, n in sorted(c.items(), key=lambda x: (x[0] == "ND", -x[1])))
        lines.append(f"{name} & {cells} \\\\")
        if stem:
            g = graphs[step]
            O.put(f"EProv{stem}Graphs", A.f_int(len(g)), f"provenance: graphs of {step}")
            O.put(f"EProv{stem}Mixed", A.f_int(sum(1 for s in g.values() if len(s) > 1)),
                  f"provenance: graphs of {step} whose rows come from more than one processor")
    O.table("provenance", lines)
    O.put("EProvRows", A.f_int(total), "provenance: rows traced to their node")
    O.put("EProvOpen", A.f_int(undetermined), "provenance: rows whose candidate nodes have different processors")
