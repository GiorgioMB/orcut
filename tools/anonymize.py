"""Anonymized copy of the repository for double-blind review (supplementary material).

Copies the files tracked by git and removes every identifying name from their contents and their
paths: people, accounts and email addresses, institutions and the cluster, the earlier paper whose
solver this package started from (its title, arXiv identifier and code) and the repository, and
the package itself. ``orcut`` and its PyPI name ``faster-or-curvature``
become a neutral placeholder everywhere (folder, imports, environment variables, conda environments,
table labels, LaTeX macros), so the copy still installs and runs under that name. Matching ignores
case and the separators between words: ORCUT_DATA, vOrcut, faster_or_curvature and a name broken
across two lines are all caught. The full name and email address of every author and committer in
the git history, and the owner of every remote, are removed too (single names go in NAMES).

Double-blind rules (ICML 2026, LoG 2026: the supplementary material is published on OpenReview) also
ask that the authors' earlier work be cited without revealing that it is theirs, and that the
submission not point to a non-anonymous version of itself. So "the companion paper" becomes "an
earlier paper", links to the authors' arXiv papers are removed, "the arXiv paper" (this paper's
preprint) becomes "the paper", venue labels such as "(LoG 2027 version)" are dropped, and the
sentence that names the package on PyPI goes (its page names the authors).

The copy is then rescanned, text and binary files and their paths, for any trace of these names, for
email addresses, for absolute paths under a user's home or scratch folder, and for links that now
point to the placeholder (dead, or someone else's); if one is left, nothing is written. The archive has fixed timestamps (no dates
or time zone), so the same tree always gives the same archive, byte for byte.

Usage: python tools/anonymize.py [--out anonymous.zip] [--placeholder anonpkg] [--repo-url URL]
                                 [--extra NAME ...] [--dry-run] [--show]
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/")  # not copied: it lists the names
PKG = None  # in NAMES: the placeholder (--placeholder)
OWN_ARXIV = ["2603.13535"]  # arXiv identifiers of the authors' papers: links to them are removed
ARXIV = r"(?:{})(?:v\d+)?".format("|".join(re.escape(x) for x in OWN_ARXIV))

# Rewrites applied first, so that the sentence around a name still reads well ({url}: --repo-url).
CONTEXT = [
    ("install commands", r"pip install (['\"]?)faster[\W_]*or[\W_]*curvature(\[[^\]\s'\"]*\])?\1",
     r"pip install \1.\2\1"),
    ("install commands", r"(['\"]?)[\w.\-]+ ?@ ?git\+https?://github\.com/GiorgioMB/[^\s'\"`]+\1", "."),
    ("repository links", r"https?://github\.com/GiorgioMB\b[^\s'\"`)>\]]*", "{url}"),
    # PyPI's page of the package names its authors
    ("package index", r"`?orcut`? is on PyPI as [^\n]*?, but that package is the solver alone; for the experiments"
     r" and everything else beyond the solver, install from this repository\.", "Install from this repository."),
    # the earlier paper, cited in the third person and without its title or link
    ("earlier paper", r"\(the code used for the results of Micaletto and\s+Nigrelli,\s+\"[^\"]*\"\)",
     "(the code of an earlier paper)"),
    ("earlier paper", r"in Micaletto and\s+Nigrelli,\s+\[?\*[^*]*\*\]?(?:\(\s*https?://[^)\s]*\s*\))?\s*:",
     "in an earlier paper:"),
    ("earlier paper", r"\b(?:the|a|our)\s+companion\s+paper\b", "an earlier paper"),
    ("own arXiv links", r"\[([^\]]*)\]\(\s*https?://arxiv\.org/(?:abs|pdf)/" + ARXIV + r"[^)\s]*\s*\)", r"\1"),
    ("own arXiv links", r"[(<]?https?://arxiv\.org/(?:abs|pdf)/" + ARXIV + r"[^\s)>]*[)>]?", ""),
    ("own arXiv links", r"\barXiv:\s*" + ARXIV, ""),
    # this paper: no pointer to its non-anonymous preprint, nor to the venue a version was written for
    ("own preprint", r"\bthe\s+arXiv\s+(?:paper|preprint|version)\b", "the paper"),
    ("venue labels", r"[ \t]*\((?:LoG|ICML|NeurIPS|ICLR|KDD|AISTATS|TMLR|AAAI|WWW|SDM)[ \t]*\d{4}[ \t]+version\)", ""),
    ("copyright", r"Copyright \(c\) (\d{4}) Giorgio Micaletto", r"Copyright (c) \1 Anonymous Authors"),
]

# Identifying names and what replaces them (applied longest first; the case of each match is kept).
NAMES = {
    "orcut": PKG, "faster-or-curvature": PKG,
    "Giorgio Micaletto": "Anonymous", "Giorgio": "Anonymous", "Micaletto": "Anonymous", "Nigrelli": "Anonymous",
    "Micaletto and Nigrelli": "an earlier paper",
    "GiorgioMB": "anonymous", "gmicaletto": "anonymous", "gmicaletto@uchicago.edu": "anonymous@example.org",
    "micaleg1": "anonymous",  # the cluster account (paths in logs, results and status records)
    "University of Chicago": "[institution]", "uchicago": "[institution]", "Bocconi": "[institution]",
    "Aalto": "[institution]", "Triton": "[cluster]",
    # the earlier paper: its first title and the one it has on arXiv, and its code
    "What Balanced Forman Curvature Certifies About Ollivier-Ricci Curvature": "[earlier paper]",
    "From Proxy to Certificate: Two-Sided Ollivier-Ricci Bounds from Balanced Forman Curvature": "[earlier paper]",
    "Curvature-Transfer-Code": "prior-code", "curvature-transfer": "prior_code",
    **{x: "[withheld]" for x in OWN_ARXIV},
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]{2,}(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ALLOWED = ("@example.org", "@example.com")
# a user's folder: /home/<user>, /Users/<user>, /scratch/<user>, /scratch/work/<user> (not .../, <user>, $USER)
USER_PATH = re.compile(r"(?<![\w.~$])/(?:home|Users|scratch(?:/work)?)/(?!\.\.\.|work/|[<${])([A-Za-z][\w.-]*)")
URL = re.compile(r"https?://[^\s<>\"'`)\]]+")


def pattern(term: str, whole_words: bool) -> re.Pattern:
    """``term`` in any case, with any or no separators (-, _, ., spaces, line breaks) between its words."""
    core = r"[\W_]*".join(re.escape(w) for w in re.findall(r"[^\W_]+", term))
    return re.compile(rf"(?<![^\W_]){core}(?![^\W_])" if whole_words else core, re.IGNORECASE)


def cased(rep: str):
    """A replacement that follows the case of the match: orcut -> anonpkg, ORCUT -> ANONPKG, Orcut -> Anonpkg."""
    def sub(m):
        s = "".join(c for c in m.group(0) if c.isalpha())
        if len(s) > 1 and s.isupper():
            return rep.upper()
        if s.islower():
            return rep.lower()
        if s[:1].isupper() and s[1:].islower():
            return rep[:1].upper() + rep[1:]
        return rep
    return sub


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def from_git() -> dict:
    """Full names and email addresses of every author and committer in the history, GitHub usernames
    in no-reply addresses, and the owner of every remote (matched as whole words)."""
    found = {}
    for line in git("log", "--all", "--format=%an%x00%ae%x00%cn%x00%ce").splitlines():
        f = line.split("\0")
        for name, email in zip(f[0::2], f[1::2]):
            if " " in name.strip() and "[bot]" not in name and not name.lower().startswith("github"):
                found[name.strip()] = "Anonymous"
            local, _, domain = email.lower().partition("@")
            if domain == "users.noreply.github.com":
                found[local.split("+")[-1]] = "anonymous"
            elif domain and domain != "github.com":
                found[email] = "anonymous@example.org"
    for url in git("remote", "-v").split():
        m = re.search(r"[:/]([^/:\s]+)/[^/\s]+/?$", url) if re.match(r"\w+://|[\w.-]+@", url) else None
        if m:
            found[m.group(1)] = "anonymous"
    return found


def write(out: str, top: str, entries: dict) -> None:
    """Zip in sorted order with fixed timestamps and permissions: no dates, time zone or user ids."""
    tmp = out + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(entries):
            data, mode = entries[rel]
            info = zipfile.ZipInfo(f"{top}/{rel}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o755 if mode & 0o111 else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "anonymous.zip"), help="archive to write")
    ap.add_argument("--placeholder", default="anonpkg", help="neutral name of the package in the copy")
    ap.add_argument("--repo-url", default="<anonymous repository>", help="what links to the repository become")
    ap.add_argument("--extra", nargs="*", default=[], help="further identifying strings (matched as whole words)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change and write nothing")
    ap.add_argument("--show", action="store_true", help="print every changed line (old -, new +) to check how it reads")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z]+", a.placeholder):
        sys.exit("--placeholder: lowercase letters only (it names a module and goes into LaTeX macros)")

    known = {t.lower() for t in NAMES}
    terms = [(t, r, False) for t, r in NAMES.items()]            # curated names: also inside words (vOrcut)
    terms += [(t, "Anonymous", True) for t in a.extra]            # the rest: whole words only
    terms += [(t, r, True) for t, r in from_git().items() if t.lower() not in known]
    terms = sorted((x for x in terms if re.search(r"[^\W_]", x[0])), key=lambda x: -len(x[0]))
    subs = [(t, pattern(t, whole), cased(a.placeholder if r is PKG else r)) for t, r, whole in terms]
    context = [(lb, re.compile(rx, re.IGNORECASE), rep.replace("{url}", a.repo_url)) for lb, rx, rep in CONTEXT]
    checks = [(t, pattern(t, False)) for t, _, _ in terms]        # the check is stricter than the replacement
    holder = re.compile(re.escape(a.placeholder), re.IGNORECASE)

    files = [f for f in git("ls-files", "-z").split("\0") if f and f != SELF and os.path.isfile(os.path.join(ROOT, f))]
    if not files:
        sys.exit("no tracked files: run it inside the git repository")
    entries, counts, leaks = {}, {}, []
    for rel in files:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            data = fh.read()
        if holder.search(rel) or holder.search(data.decode("utf-8", "replace")):
            sys.exit(f"{a.placeholder!r} already occurs in {rel}: choose another --placeholder")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None                                           # binary: copied as it is, checked below
        if text is not None:
            before = text
            for label, rx, rep in context + subs:
                text, n = rx.subn(rep, text)
                counts[label] = counts.get(label, 0) + n
            data = text.encode("utf-8")
            if a.show and text != before:
                sys.stdout.writelines(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                                           f"{rel}", f"(anonymized) {rel}", n=0))
        new = rel
        for _, rx, rep in subs:
            new = rx.sub(rep, new)
        if new in entries:
            sys.exit(f"two files would both become {new}")
        seen = data.decode("utf-8", "replace")
        leaks += [(new, t) for t, rx in checks if rx.search(new) or rx.search(seen)]
        leaks += [(new, e) for e in EMAIL.findall(seen) if not e.lower().endswith(ALLOWED)]
        leaks += [(new, f"/.../{m.group(1)} (a user's folder)") for m in USER_PATH.finditer(seen)
                  if m.group(1).lower() not in ("anonymous", a.placeholder)]
        leaks += [(new, f"{u} (a link to the placeholder)") for u in URL.findall(seen)
                  if holder.search(u) and not u.startswith(a.repo_url)]
        entries[new] = (data, os.stat(os.path.join(ROOT, rel)).st_mode)
    leaks += [(a.out, t) for t, rx in checks if rx.search(os.path.basename(a.out))]
    if leaks:
        for rel, t in leaks:
            print(f"identifying string {t!r} left in {rel}", file=sys.stderr)
        sys.exit(1)
    done = ", ".join(f"{k} x{n}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n)
    if a.dry_run:
        print(f"{len(entries)} files, nothing written; would replace: {done}")
        return
    write(a.out, a.placeholder, entries)
    print(f"wrote {a.out}: {len(entries)} files under {a.placeholder}/; replaced: {done}")


if __name__ == "__main__":
    main()
