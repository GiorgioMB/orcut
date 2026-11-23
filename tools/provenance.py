"""Which node produced each row of each experiment, from the step logs and the scheduler's records.

A step can be stopped by its time limit and resumed on another node, so the rows of one table can
come from different processors. The logs record, for each run, when it started and which script it
ran, and, for each row, when it was written; the scheduler records which job ran on which node and
when. A row is attributed to the jobs of its step that were running when its last log line was
written (the line that wrote the row of the table): one job, or several jobs on the same type of
node, settle it; workers of one step on different types of node at that moment leave it open.

Usage, on the cluster:

    sacct -u $USER -S <first day> -X -n -o JobID%25,JobName%25,NodeList%20,Start,End > sacct.txt
    python tools/provenance.py --logs results/logs --sacct sacct.txt --out results/provenance/rows.csv

Then write ``results/provenance/types.txt``, one line per kind of node: the letters that start its
node names, then its processor (``abc Xeon Gold 6148``); ``artifacts.py`` turns the rows into the
table ``provenance``. Jobs are those of ``experiments/slurm`` (``orcut-step`` with its array index,
``orcut-all`` for every step); others are ignored.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import sys

# step: (array index of the step's run by script, regex of a row line giving the row's key). A name
# can fill its column, leaving one space before the next field, so each pattern ends the name at a
# field it knows (a method, a measure, "m=", "est.").
STEPS = {
    "e1": ({"e1_exactness": 0}, r"^E1 (.+?)\s+m="),
    "e2-controlled": ({"e2_per_edge": 1}, r"^E2A (\S+) (\S+)\s+s=\s*(\d+) seed=(\d+)"),
    "e2-real": ({"e2_per_edge": 2}, r"^E2B (.+?)\s+bin (\[[^)]*\))"),
    "e2-dasgupta": ({"e2_per_edge": 3}, r"^E2C (.+?)\s+n=\s*(\d+)"),
    "e3-real": ({"e3_scale": 4}, r"^E3 (.+?)\s+(lazy|nonlazy|lly)\s+m="),
    "e3-threads": ({"e3_scale": 5}, r"^E3T (.+?)\s+T=\s*(\d+)\s+[\d.]+s"),
    "e3-synth": ({"e3_scale": 6}, r"^E3S (\S+)\s+(\S+)\s+n=\s*([\d,]+) m=\s*[\d,]+\s+[\d.]+s\s*$"),
    "e4": ({"e4_ablation": 7}, r"^E4 (.+?)\s+(\S+)\s+est\."),
    "e5-whole": ({"e5_approx": 8}, r"^E5 (.+?)\s+((?:grc|orcut|jl|bf|af3)\S*)\s"),
    "e5-sinkhorn": ({"e5_approx": 9}, r"^E5S (.+?)\s+(\S+)\s+reg=(\S+)"),
    # the logs of runs up to 1.3 name these steps E7 (e7_generality.py, array tasks 12 and 13)
    "e6-measures": ({"e7_generality": 12, "e6_generality": 10}, r"^E[67]M (.+?)\s+(lazy|nonlazy|lly|idle \d+/\d+|weighted walk)\s+[\d.]+s"),
    "e6-radius": ({"e7_generality": 13, "e6_generality": 11}, r"^E[67]R (.+?)\s+((?:walk|ball) r=\d+)"),
}
JOB_NAMES = ("orcut-step", "orcut-all")


def read_sacct(path: str) -> list:
    jobs = []
    for line in open(path):
        f = line.split()
        if len(f) < 5 or f[1] not in JOB_NAMES or not f[3][:1].isdigit():
            continue
        task = f[0].split("_")[1] if "_" in f[0] else None
        jobs.append(dict(id=f[0], name=f[1], node=f[2], start=dt.datetime.fromisoformat(f[3]),
                         end=dt.datetime.fromisoformat(f[4]) if f[4][:1].isdigit() else dt.datetime.max,
                         task=int(task) if task is not None and task.isdigit() else None))
    return jobs


def trace(step: str, log: str, jobs: list) -> list:
    """One dict per row of the step's table: its key, when its last line was written, the candidate jobs."""
    tasks_by_script, rx = STEPS[step]
    rx = re.compile(rx)
    last, task, day, prev = {}, None, None, None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$", line.rstrip("\n"))
        if m:
            start = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            day, prev = start.date(), start.time()
            task = next((t for s, t in tasks_by_script.items() if s in m.group(2)), None)
            continue
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] (.*)$", line.rstrip("\n"))
        if not m or day is None:
            continue
        t = dt.datetime.strptime(m.group(1), "%H:%M:%S").time()
        if t < prev:  # past midnight
            day += dt.timedelta(days=1)
        prev = t
        k = rx.match(m.group(2))
        if k:
            last[tuple(" ".join(g.split()) for g in k.groups())] = (dt.datetime.combine(day, t), task)
    rows = []
    for key, (when, task) in last.items():
        cand = [j for j in jobs if j["start"] <= when <= j["end"] and (j["task"] == task or j["name"] == "orcut-all")]
        if any(j["task"] == task for j in cand):  # an array task of the step, rather than the all-steps job
            cand = [j for j in cand if j["task"] == task]
        rows.append(dict(step=step, key=" | ".join(key), written=when.isoformat(sep=" "),
                         jobs=";".join(j["id"] for j in cand), nodes=";".join(sorted({j["node"] for j in cand}))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="results/logs")
    ap.add_argument("--sacct", required=True)
    ap.add_argument("--out", default="results/provenance/rows.csv")
    a = ap.parse_args()
    jobs = read_sacct(a.sacct)
    rows = []
    for step in STEPS:
        log = os.path.join(a.logs, f"{step}.log")
        if os.path.exists(log):
            rows += trace(step, log, jobs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "key", "written", "jobs", "nodes"])
        w.writeheader()
        w.writerows(rows)
    open_rows = sum(1 for r in rows if not r["nodes"])
    print(f"wrote {a.out}: {len(rows)} rows of {len({r['step'] for r in rows})} steps"
          + (f"; {open_rows} without a job" if open_rows else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
