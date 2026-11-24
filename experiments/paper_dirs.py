"""The generated folders of the versions of the paper, from the same result tables.

    python experiments/paper_dirs.py --results results \
        --preprint ../paper/A/generated --onecolumn ../paper/B/generated --twocolumn ../paper/C/generated

writes each folder that is given (at least one):

| option        | version                         | figures                              | solver's name |
|---------------|---------------------------------|--------------------------------------|---------------|
| --preprint    | the preprint (one column)       | one column of 5.5 in                 | orcut         |
| --onecolumn   | an anonymous one-column version | one column of 5.5 in                 | ours          |
| --twocolumn   | an anonymous two-column version | 3.25 in (main text), 6.75 in (appendix) | ours       |

Each folder gets ``solver_numbers.tex``, ``KEYS.md`` and the figures, written by ``artifacts.py``
with the same numbers; only the size of the figures and the name of the solver in their legends and
in the table labels differ. The figures are set with Type-1 fonts (``--fonts type1``, which needs
``latex``); every PDF is checked again here, and a Type-3 font stops the run. ``--fonts truetype``
works without LaTeX (TrueType fonts: fine for a preprint, not for styles that ask for Type 1).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TARGETS = {  # option: (layout of artifacts.py, solver's name)
    "preprint": ("onecolumn", "orcut"),
    "onecolumn": ("onecolumn", "ours"),
    "twocolumn": ("twocolumn", "ours"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="result tables (default: that of artifacts.py, results/)")
    for name, (layout, method) in TARGETS.items():
        ap.add_argument(f"--{name}", default=None, metavar="DIR",
                        help=f"folder to write: layout {layout}, solver named {method}")
    ap.add_argument("--fonts", choices=("type1", "truetype"), default="type1")
    ap.add_argument("--provisional", action="store_true", help="mark every number as provisional (quick run)")
    a = ap.parse_args()
    jobs = [(name, getattr(a, name)) for name in TARGETS if getattr(a, name)]
    if not jobs:
        ap.error("give at least one of " + ", ".join(f"--{n}" for n in TARGETS))

    import artifacts as A  # for the font check only

    failed = []
    for name, out in jobs:
        layout, method = TARGETS[name]
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "artifacts.py"), "--paper-dir", out, "--layout", layout,
               "--method-name", method, "--fonts", a.fonts]
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
            print(f"  {pdf}: {', '.join(fonts) or 'no fonts'}" + ("  <- not accepted" if bad else ""))
            if bad:
                failed.append(f"{pdf}: {bad}")
    if failed:
        print("paper_dirs: " + "; ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
