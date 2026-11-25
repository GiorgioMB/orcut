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
point to the placeholder (dead, or someone else's); if one is left, nothing is written. A name
the copy would show replaced ("Anonymous, G.", "[earlier paper]") stops it too: a visible redaction
says the work is the authors' own, so the sentence is rewritten in the source or by a rule of
CONTEXT instead. So do an arXiv identifier not in CITED_ARXIV (this paper's preprint), first-person
mentions of earlier work, venues named in full, scheduler settings that name the site (account,
partition, QOS, node types, modules), and binary files the check cannot read inside (the members
of zip-based files such as .npz are read; any other binary must be listed in BINARY_OK). Files
that contain NUL bytes (UTF-16 text, such as the file names in a .DS_Store) are also checked with
the NUL bytes removed. Desktop metadata (.DS_Store, ._*, Thumbs.db, desktop.ini) and editor settings
(.vscode, .idea, workspaces) are never copied, the anonymizer is not mentioned in the copy, the
earlier paper's code is not said to be this package's, and no venue is named. The archive has
fixed timestamps (no dates or time zone), so the same tree always gives the same archive, byte for byte.

Usage: python tools/anonymize.py [--out anonymous.zip] [--placeholder anonpkg] [--repo-url URL]
                                 [--extra NAME ...] [--dry-run] [--show]
"""
from __future__ import annotations

import argparse
import difflib
import io
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/")  # not copied: it lists the names
PKG = None  # in NAMES: the placeholder (--placeholder)
OWN_ARXIV = ["2603.13535", "2611.32528"]  # the authors' papers (2611.32528: this one): links to them are removed
ARXIV = r"(?:{})(?:v\d+)?".format("|".join(re.escape(x) for x in OWN_ARXIV))
CITED_ARXIV: set[str] = set()  # arXiv identifiers of papers by others that the copy may name; any other fails
BINARY_OK: set[str] = set()    # tracked binary files checked by hand (paths in the repository); any other fails

# Rewrites applied first, so that the sentence around a name still reads well ({url}: --repo-url).
CONTEXT = [
    ("install commands", r"pip install (['\"]?)faster[\W_]*or[\W_]*curvature(\[[^\]\s'\"]*\])?\1",
     r"pip install \1.\2\1"),
    ("install commands", r"(['\"]?)[\w.\-]+ ?@ ?git\+https?://github\.com/GiorgioMB/[^\s'\"`]+\1", "."),
    ("repository links", r"https?://github\.com/GiorgioMB\b[^\s'\"`)>\]]*", "{url}"),
    # PyPI's page of the package names its authors
    ("package index", r"(?m)^The solver alone is on PyPI as [^\n]*\n\n?", ""),
    # the README's citation of this paper (its preprint names the author): the whole section goes
    ("own preprint", r"(?ms)^## Citation\n.*?(?=^## |\Z)", ""),
    ("earlier paper", r"\(the code used for the results of Micaletto and\s+Nigrelli,\s+\"[^\"]*\"\)",
     "(the code of an earlier paper)"),
    ("earlier paper", r"\b(?:the|a|our)\s+companion\s+paper\b", "an earlier paper"),
    ("own arXiv links", r"\[([^\]]*)\]\(\s*https?://arxiv\.org/(?:abs|pdf)/" + ARXIV + r"[^)\s]*\s*\)", r"\1"),
    ("own arXiv links", r"[(<]?https?://arxiv\.org/(?:abs|pdf)/" + ARXIV + r"[^\s)>]*[)>]?", ""),
    ("own arXiv links", r"\barXiv:\s*" + ARXIV, ""),
    # this paper: no pointer to its non-anonymous preprint, nor to the venue a version was written for
    ("own preprint", r"\bthe\s+arXiv\s+(?:paper|preprint|version)\b", "the paper"),
    ("venue labels", r"[ \t]*\((?:LoG|ICML|NeurIPS|ICLR|KDD|AISTATS|TMLR|AAAI|WWW|SDM)[ \t]*\d{4}[ \t]+version\)", ""),
    # the anonymizer itself is not copied, nor is its archive named
    ("anonymizer", r"(?m)^anonymous\.zip(?:\.tmp)?\n", ""),
    ("copyright", r"Copyright \(c\) (\d{4}) Giorgio Micaletto", r"Copyright (c) \1 Anonymous Authors"),
    # ICML asks that licenses be removed from submitted code: the LICENSE file is not copied (LICENSES)
    # and the package metadata stops pointing to it
    ("license", r"(?m)^license\s*=\s*\{[^}\n]*\}[ \t]*\n", ""),
]

# license files: not copied (ICML: "author names and licenses should be removed"); nor is the citation
# file, which names the author and points to the preprint
LICENSES = re.compile(r"(?:^|/)(?:(?:LICEN[CS]E|COPYING)(?:\.[A-Za-z]+)?|CITATION\.cff)$")

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
USER_PATH = re.compile(r"(?<![\w.~$])/(?:home|Users|scratch(?:/work)?|project\d*|work|gpfs|lustre\w*)/"
                       r"(?!\.\.\.|work/|[<${])([A-Za-z][\w.-]*)")
WIN_USER_PATH = re.compile(r"(?i)\b[A-Z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)([A-Za-z][^\\/\s\"'<>%]*)")
ARXIV_ID = re.compile(r"arxiv(?:\.org/(?:abs|pdf)/|:\s*|\s+)(\d{4}\.\d{4,5})", re.IGNORECASE)
URL = re.compile(r"https?://[^\s<>\"'`)\]]+")
# desktop metadata: never copied (a .DS_Store lists the names of the folder it was in, in UTF-16)
JUNK = re.compile(r"(?:^|/)(?:\.DS_Store|\._[^/]+|Thumbs\.db|desktop\.ini|\.directory|\.env)$"
                  r"|(?:^|/)(?:__MACOSX|\.vscode|\.idea|\.ipynb_checkpoints)/|\.code-workspace$", re.I)
# must not be left in the copy at all (checked, never replaced): the name the earlier paper's code gets
# (a sentence still naming it says how this package relates to that code), the anonymizer and its
# archive, and venues (case-sensitive, so that "log" is not one)
FORBIDDEN = [
    ("the earlier paper's code", re.compile(r"prior[\W_]*code", re.I)),
    ("the anonymizer", re.compile(r"anonymi[sz]e\.py|anonymous\.zip", re.I)),
    ("a venue", re.compile(r"(?<![\w-])(?:LoG|ICML|NeurIPS|ICLR|KDD|AISTATS|TMLR|AAAI|SDM|UAI|COLT|CIKM|WSDM)(?![\w-])")),
    ("a venue", re.compile(r"Learning\s+on\s+Graphs\s+Conference|International\s+Conference\s+on\s+(?:Machine\s+"
                           r"Learning|Learning\s+Representations)|Neural\s+Information\s+Processing\s+Systems"
                           r"|Knowledge\s+Discovery\s+and\s+Data\s+Mining")),
    # "our previous work" says whose it is, however it is named
    ("the authors' earlier work", re.compile(r"\b(?:our|my)\s+(?:own\s+)?(?:earlier|previous|prior|past|companion)\s+"
                                             r"(?:paper|work|article|study|code|package|solver|implementation|results?)\b", re.I)),
    # the site: an account, partition, QOS, node type or module belongs to one cluster
    ("a cluster setting", re.compile(r"(?im)^\s*#SBATCH\s+(?:-[ApMq]\b|--(?:account|partition|qos|reservation|clusters|"
                                     r"mail-user|constraint=(?!<)))|^\s*module\s+(?:load|add|use)\s")),
]


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


def visible(data: bytes) -> str:
    """The text the checks read: the bytes as UTF-8 and, if they hold NUL bytes, without them (UTF-16
    and UTF-32 text, such as the names in a .DS_Store or a NumPy string array, hides otherwise)."""
    seen = data.decode("utf-8", "replace")
    if b"\0" in data:
        seen += "\n" + data.replace(b"\0", b"").decode("latin-1")
    return seen


def members(data: bytes):
    """(name, bytes) of every member of a zip-based file (.zip, .npz, .docx, .xlsx, .whl), or None:
    compressed members hide their text from the checks."""
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return None
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return [(i.filename, zf.read(i)) for i in zf.infolist() if not i.is_dir()]


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
    subs = [(t, pattern(t, whole), cased(a.placeholder if r is PKG else r), r) for t, r, whole in terms]
    context = [(lb, re.compile(rx, re.IGNORECASE), rep.replace("{url}", a.repo_url)) for lb, rx, rep in CONTEXT]
    checks = [(t, pattern(t, False)) for t, _, _ in terms]        # the check is stricter than the replacement
    holder = re.compile(re.escape(a.placeholder), re.IGNORECASE)

    def scan(where: str, seen: str) -> list:
        out = [(where, f"{m.group(0)!r} ({what})") for what, rx in FORBIDDEN for m in rx.finditer(seen)]
        out += [(where, t) for t, rx in checks if rx.search(seen)]
        out += [(where, e) for e in EMAIL.findall(seen) if not e.lower().endswith(ALLOWED)]
        out += [(where, f"/.../{m.group(1)} (a user's folder)") for rx in (USER_PATH, WIN_USER_PATH)
                for m in rx.finditer(seen) if m.group(1).lower() not in ("anonymous", a.placeholder)]
        out += [(where, f"{u} (a link to the placeholder)") for u in URL.findall(seen)
                if holder.search(u) and not u.startswith(a.repo_url)]
        out += [(where, f"arXiv {m.group(1)} (not in CITED_ARXIV: this paper's preprint?)")
                for m in ARXIV_ID.finditer(seen) if m.group(1) not in CITED_ARXIV]
        return out

    files = [f for f in git("ls-files", "-z").split("\0") if f and f != SELF and os.path.isfile(os.path.join(ROOT, f))]
    junk = [f for f in files if JUNK.search(f)]
    files = [f for f in files if f not in junk]
    if junk:
        print(f"not copied (desktop metadata; untrack it with git rm --cached): {', '.join(junk)}", file=sys.stderr)
    licenses = [f for f in files if LICENSES.search(f)]
    files = [f for f in files if f not in licenses]
    if licenses:
        print(f"not copied (license and citation files name the authors): {', '.join(licenses)}", file=sys.stderr)
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
        redacted = []
        if text is not None:
            before = text
            for label, rx, rep in context:
                text, n = rx.subn(rep, text)
                counts[label] = counts.get(label, 0) + n
            for label, rx, rep, r in subs:
                text, n = rx.subn(rep, text)
                counts[label] = counts.get(label, 0) + n
                if n and r is not PKG:  # "Anonymous, G." or "[earlier paper]" in the copy says whose work it is
                    redacted.append(f"{label!r} shown as {r!r} x{n} (cite it in the third person, in the source"
                                    f" or by a rule of CONTEXT)")
            data = text.encode("utf-8")
            if a.show and text != before:
                sys.stdout.writelines(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                                           f"{rel}", f"(anonymized) {rel}", n=0))
        new = rel
        for _, rx, rep, _ in subs:
            new = rx.sub(rep, new)
        if new in entries:
            sys.exit(f"two files would both become {new}")
        leaks += [(new, x) for x in redacted]
        leaks += [(new, t) for t, rx in checks if rx.search(new)]
        leaks += scan(new, visible(data))
        if text is None:
            inner = members(data)
            if inner is not None:
                for name, blob in inner:
                    leaks += scan(f"{new}!{name}", visible(blob))
            elif rel not in BINARY_OK:
                leaks += [(new, "binary contents (compressed streams and metadata are not read: convert it to"
                                " text, or check it by hand and add it to BINARY_OK)")]
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
