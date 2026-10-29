"""Anonymized copy of the repository for double-blind review (supplementary material).

Copies the files tracked by git (``git ls-files``; the downloaded data and the results are not
tracked), replaces author names, URLs and the names of the companion paper and its code by neutral
placeholders, checks that no identifying string is left, and writes a zip archive.

Usage: python tools/anonymize.py [--out orcut-anonymous.zip] [--extra STRING ...]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPLACEMENTS = [
    (r'\(the code used for the results of Micaletto and\s+Nigrelli, "What Balanced Forman Curvature Certifies About'
     r'\s+Ollivier--Ricci\s+Curvature"\)', "(the code of a companion paper)"),
    (r"in Micaletto and Nigrelli,\s+\*What Balanced Forman Curvature Certifies About Ollivier(--|–)Ricci Curvature\*:",
     "in a companion paper:"),
    (r"Copyright \(c\) (\d{4}) Giorgio Micaletto", r"Copyright (c) \1 Anonymous Authors"),
    (r"https?://github\.com/GiorgioMB/[A-Za-z0-9_.\-]+", "https://anonymous.4open.science/r/orcut"),
    (r"git\+https://anonymous\.4open\.science/r/orcut(\.git)?@v[0-9.]+", "<anonymous repository>"),
    (r"\*?What Balanced Forman Curvature Certifies About Ollivier(--|–)Ricci\s+Curvature\*?", "[companion paper]"),
    (r'"What Balanced Forman Curvature Certifies About Ollivier--Ricci\s+Curvature"', "[companion paper]"),
    (r"Micaletto\s+and\s+Nigrelli", "the authors of the companion paper"),
    (r"Giorgio\s+Micaletto", "Anonymous"),
    (r"Micaletto", "Anonymous"),
    (r"Nigrelli", "Anonymous"),
    (r"GiorgioMB", "anonymous"),
    (r"gmicaletto@uchicago\.edu", "anonymous@example.org"),
    (r"curvature[_-]transfer", "companion_code"),
    (r"Curvature-Transfer-Code", "companion-code"),
]
FORBIDDEN = ["micaletto", "nigrelli", "giorgio", "giorgiomb", "uchicago", "bocconi", "triton", "aalto", "curvature_transfer",
             "curvature-transfer"]
TEXT_EXT = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".sh", ".sbatch", ".tex", ".bib", ".json", ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "orcut-anonymous.zip"))
    ap.add_argument("--extra", nargs="*", default=[], help="further strings that must not appear")
    a = ap.parse_args()
    files = subprocess.check_output(["git", "-C", ROOT, "ls-files"], text=True).split()
    forbidden = [f.lower() for f in FORBIDDEN + a.extra]
    leaks = []
    tmp = a.out + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            if rel.startswith("tools/anonymize.py"):
                continue  # this script lists the strings it removes
            path = os.path.join(ROOT, rel)
            ext = os.path.splitext(rel)[1]
            data = open(path, "rb").read()
            if ext in TEXT_EXT:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
                if text is not None:
                    for pat, rep in REPLACEMENTS:
                        text = re.sub(pat, rep, text)
                    low = text.lower()
                    leaks += [(rel, f) for f in forbidden if f in low]
                    data = text.encode("utf-8")
            zf.writestr(os.path.join("orcut", rel), data)
    if leaks:
        os.remove(tmp)
        for rel, f in leaks:
            print(f"identifying string {f!r} left in {rel}", file=sys.stderr)
        sys.exit(1)
    os.replace(tmp, a.out)
    print(f"wrote {a.out} ({len(files)} files)")


if __name__ == "__main__":
    main()
