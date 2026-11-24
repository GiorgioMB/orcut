"""The generated folders of the three versions of the paper, from the same result tables.

    python experiments/paper_dirs.py --results results --out paper

writes, under ``--out`` (the folder that holds the three LaTeX projects), each project's ``generated/``:

| folder                 | version                     | layout                 | solver's name |
|------------------------|-----------------------------|------------------------|---------------|
| arXiv/generated        | preprint (log_2026 preprint) | one column, 5.5 in     | orcut         |
| LoG/generated          | LoG review (log_2026 review) | one column, 5.5 in     | ours          |
| ICML/generated         | ICML review (icml2027)       | 3.25 in / 6.75 in      | ours          |

Each folder gets ``solver_numbers.tex``, ``KEYS.md`` and the figures, written by ``artifacts.py``
with the same numbers; only the size of the figures and the name of the solver in their legends and
in the table labels differ. The figures are set with Type-1 fonts (``--fonts type1``, which needs
``latex``); every PDF is checked again here, and a Type-3 font stops the run. ``--targets`` writes
only some of the folders, ``--fonts truetype`` works without LaTeX (TrueType fonts: fine for arXiv,
not what LoG and ICML ask for).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TARGETS = {
    "arxiv": {"dir": os.path.join("arXiv", "generated"), "layout": "log", "method": "orcut"},
    "log": {"dir": os.path.join("LoG", "generated"), "layout": "log", "method": "ours"},
    "icml": {"dir": os.path.join("ICML", "generated"), "layout": "icml", "method": "ours"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="result tables (default: that of artifacts.py, results/)")
    ap.add_argument("--out", default="paper", help="folder with the LaTeX projects arXiv/, LoG/ and ICML/")
    ap.add_argument("--targets", nargs="+", choices=sorted(TARGETS), default=["arxiv", "log", "icml"])
    ap.add_argument("--fonts", choices=("type1", "truetype"), default="type1")
    ap.add_argument("--provisional", action="store_true", help="mark every number as provisional (quick run)")
    a = ap.parse_args()

    import artifacts as A  # for the font check only

    failed = []
    for name in a.targets:
        t = TARGETS[name]
        out = os.path.join(a.out, t["dir"])
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "artifacts.py"), "--paper-dir", out, "--layout", t["layout"],
               "--method-name", t["method"], "--fonts", a.fonts]
        if a.results:
            cmd += ["--results", a.results]
        if a.provisional:
            cmd += ["--provisional"]
        print(f"[{name}] {' '.join(cmd)}", flush=True)
        if subprocess.call(cmd) != 0:
            failed.append(f"{name}: artifacts.py failed")
            continue
        for pdf in sorted(glob.glob(os.path.join(out, "fig-*.pdf"))):
            fonts = sorted(set(A.pdf_fonts(pdf)))
            bad = [f for f in fonts if f == "Type3" or (a.fonts == "type1" and f not in ("Type1", "Type1C"))]
            print(f"  {os.path.relpath(pdf, a.out)}: {', '.join(fonts) or 'no fonts'}" + ("  <- not accepted" if bad else ""))
            if bad:
                failed.append(f"{pdf}: {bad}")
    if failed:
        print("paper_dirs: " + "; ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
