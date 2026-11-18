"""Check E3's chunked rows against their saved chunk logs, and repair the synthetic ones.

A row of E3 that is solved in chunks has as its time the sum of the times of its chunks, which are
saved in ``<results>/e3/partial/`` as they are computed (``*.json`` while under way, ``*.done.json``
once the row is written). Up to 1.4.0, several workers could leave two logs of one row and the row
could be written from one of them: WS at average degree 2.1 n^0.4 and n = 10^6 was written from 27
of its 43 chunks (42.3 h of the 62.0 h its 43 chunks took).

For every log, this script takes the union of the chunks of its live and completed copies, finds the
row of e3/real.csv or e3/synth.csv that it belongs to (same graph, measure or family, number of
edges and threads) and compares the row's time with the sum over the union. It reports:

  ok          the row sums every chunk of its plan
  one call    the row was timed in one call (no ``chunks``); the log is a second measurement by
              another worker, and the row keeps its own
  mismatch    the row sums fewer chunks than its logs hold, and the logs cover the whole plan
  incomplete  the logs do not cover the whole plan (the row cannot be repaired from them)
  conflict    a chunk appears in two copies with different times
  no row      a log without a row (a row still under way, or one dropped from the plan)

With ``--fix``, a mismatched row of e3/synth.csv gets the time of the union, ``chunks_timed`` and
``chunk_hosts``, and its logs are merged into one ``*.done.json``; the original table and logs go to
``<results>/replaced/check-chunks-<time>/`` first (the layout of ``main.py --force``). Rows of
e3/real.csv are only reported: their diagnostic columns come from the chunks too, so a mismatched
one is recomputed (``main.py --steps e3-real``) rather than patched.

Usage: python tools/check_chunks.py [--results results] [--fix]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import common  # noqa: E402


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def logs(results: str) -> dict:
    """{log name without .json/.done.json: {"live": path or None, "done": path or None}}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(results, "e3", "partial", "*.json"))):
        b = os.path.basename(p)
        done = b.endswith(".done.json")
        base = b[:-len(".done.json")] if done else b[:-len(".json")]
        out.setdefault(base, {"live": None, "done": None})["done" if done else "live"] = p
    return out


def union(paths) -> tuple:
    """(plan of the first readable copy, {chunk: stats} over all copies, [conflicting chunks])."""
    plan, chunks, conflicts = None, {}, []
    for p in paths:
        try:
            s = json.load(open(p))
        except (OSError, ValueError):
            continue
        if not isinstance(s, dict) or "bounds" not in s:
            return None, {}, []  # not a chunk log (a POT sample)
        if plan is None:
            plan = s
        elif s.get("key") != plan.get("key") or s.get("bounds") != plan.get("bounds"):
            conflicts.append("plan")
            continue
        for c, st in s.get("chunks", {}).items():
            if c in chunks and abs(chunks[c]["seconds"] - st["seconds"]) > 1e-6:
                conflicts.append(c)
            chunks.setdefault(c, st)
    return plan, chunks, conflicts


def find_row(rows: list, key: dict, synth: bool):
    for i, r in enumerate(rows):
        if common.num(r, "m") != key.get("m") or int(common.num(r, "threads", -1)) != key.get("threads"):
            continue
        if synth and r.get("family") == str(key.get("graph", "")).split(" ")[0]:
            return i
        if not synth and _norm(r.get("graph")) == _norm(key.get("graph")) and r.get("measure") == key.get("measure"):
            return i
    return None


def check(results: str) -> list:
    """One dict per chunk log: base, table, row index, status, numbers."""
    tables = {t: common.read_csv(os.path.join(results, "e3", f"{t}.csv"))
              for t in ("real", "synth") if os.path.exists(os.path.join(results, "e3", f"{t}.csv"))}
    found = []
    for base, copies in logs(results).items():
        paths = [p for p in (copies["done"], copies["live"]) if p]
        plan, chunks, conflicts = union(paths)
        if plan is None:
            continue
        table = "synth" if base.startswith("synth.") else "real"
        rows = tables.get(table, [])
        i = find_row(rows, plan["key"], table == "synth")
        nch = len(plan["bounds"]) - 1
        total = sum(c["seconds"] for c in chunks.values())
        hosts = len({str(c.get("host")) for c in chunks.values()})
        t = common.num(rows[i], "time_s") if i is not None else float("nan")
        if conflicts:
            status = "conflict"
        elif i is None:
            status = "no row"
        elif not common.num(rows[i], "chunks", 0.0) > 0:
            status = "one call"
        elif len(chunks) < nch:
            status = "incomplete"
        elif abs(t - total) <= max(1e-6, 1e-9 * total):
            status = "ok"
        else:
            status = "mismatch"
        found.append(dict(base=base, table=table, row=i, status=status, paths=paths, chunks=chunks, plan=plan,
                          nch=nch, total=total, hosts=hosts, time=t))
    return found


def fix(results: str, found: list) -> list:
    """Repair the mismatched synthetic rows; returns what was done."""
    todo = [f for f in found if f["status"] == "mismatch" and f["table"] == "synth"]
    if not todo:
        return []
    dest = os.path.join(results, "replaced", f"check-chunks-{time.strftime('%Y%m%d-%H%M%S')}")
    path = os.path.join(results, "e3", "synth.csv")
    rows = common.read_csv(path)
    os.makedirs(os.path.join(dest, "e3", "partial"), exist_ok=True)
    shutil.copy2(path, os.path.join(dest, "e3", "synth.csv"))
    done = []
    for f in todo:
        r = rows[f["row"]]
        old = common.num(r, "time_s")
        r.update(time_s=f["total"], chunks=f["nch"], chunks_timed=len(f["chunks"]), chunk_hosts=f["hosts"])
        merged = dict(f["plan"], chunks=f["chunks"])
        target = os.path.join(results, "e3", "partial", f["base"] + ".done.json")
        for p in f["paths"]:
            shutil.copy2(p, os.path.join(dest, "e3", "partial", os.path.basename(p)))
        with open(target + ".tmp", "w") as fh:
            json.dump(merged, fh)
        os.replace(target + ".tmp", target)
        for p in f["paths"]:
            if os.path.abspath(p) != os.path.abspath(target):
                os.remove(p)
        done.append(f"{r.get('family')} {r.get('regime')} n={r.get('n')}: time {old:,.0f} s -> {f['total']:,.0f} s "
                    f"({len(f['chunks'])} of {f['nch']} chunks); logs merged into {os.path.basename(target)}")
    common.write_csv(rows, path)
    done.append(f"originals in {dest}")
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=common.RESULTS)
    ap.add_argument("--fix", action="store_true", help="repair the mismatched rows of e3/synth.csv")
    a = ap.parse_args()
    found = check(a.results)
    for f in found:
        what = (f"row sums {f['time']:,.0f} s, its {len(f['chunks'])}/{f['nch']} saved chunks {f['total']:,.0f} s"
                if f["row"] is not None else f"{len(f['chunks'])}/{f['nch']} chunks saved")
        print(f"{f['status']:10s} {f['table']:5s} {f['base']:46s} {what}")
    bad = [f for f in found if f["status"] in ("mismatch", "conflict")]
    if a.fix:
        for line in fix(a.results, found):
            print("fixed:", line)
        bad = [f for f in bad if not (f["status"] == "mismatch" and f["table"] == "synth")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
