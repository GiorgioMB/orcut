"""Numbers, tables and figures of the paper, from the result tables of E1--E7.

Writes into ``--paper-dir`` (default ``results/paper``):

* ``solver_numbers.tex``: one macro per number, read by ``\\res{Key}`` in the paper
  (``\\csname res@Key\\endcsname``); keys are listed in ``KEYS.md`` with their source;
* ``rows-<table>.tex``: table bodies, read by ``\\resrows{<table>}``;
* ``fig-*.pdf``: figures.

Every block is independent: a missing result table only leaves its keys undefined (the paper then
shows a red placeholder), so the script can be run on partial results. ``--provisional`` marks
every number with a dagger (results of a quick run, not of the full protocol).

Usage: python artifacts.py [--results results] [--paper-dir results/paper] [--provisional]
"""
from __future__ import annotations

import argparse
import math
import os
import traceback
from typing import Dict, List

import numpy as np

import common
import xio


class Out:
    def __init__(self, provisional: bool):
        self.keys: Dict[str, str] = {}
        self.src: Dict[str, str] = {}
        self.rows: Dict[str, str] = {}
        self.provisional = provisional
        self.missing: List[str] = []

    def put(self, key: str, value: str, source: str):
        self.keys[key] = value + (r"\textsuperscript{\dag}" if self.provisional else "")
        self.src[key] = source

    def table(self, name: str, lines: List[str]):
        self.rows[name] = "\n".join(lines)


# ------------------------------------------------------------- formatting

def f_int(x) -> str:
    x = int(round(float(x)))
    s = f"{x:,}".replace(",", "{,}")
    return s


def _m(s: str) -> str:
    """Wrap math (\\times, powers) so that the value can be used in text and in tables."""
    return rf"\ensuremath{{{s}}}" if ("\\times" in s or "^" in s) else s


def f_num(x, sig: int = 3) -> str:
    return _m(_f_num(x, sig))


def _f_num(x, sig: int = 3) -> str:
    x = float(x)
    if not math.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    if mag < -3:
        return _f_sci(x, sig)
    if mag >= 5:
        m = x / 10 ** mag
        return rf"{m:.{max(0, sig - 1)}f}\times10^{{{mag}}}"
    dec = max(0, sig - 1 - mag)
    s = f"{x:,.{dec}f}".replace(",", "{,}")
    return s


def f_sci(x, sig: int = 2) -> str:
    return _m(_f_sci(x, sig))


def _f_sci(x, sig: int = 2) -> str:
    x = float(x)
    if not math.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    m = x / 10 ** mag
    return rf"{m:.{sig - 1}f}\times10^{{{mag}}}"


def f_time(sec) -> str:
    """Seconds with a unit: 12.3 ms, 4.56 s, 1.2 h."""
    s = float(sec)
    if not math.isfinite(s):
        return "--"
    if s < 1e-3:
        return rf"{f_num(s * 1e6)}\,\textmu s"
    if s < 1:
        return rf"{f_num(s * 1e3)}\,ms"
    if s < 600:
        return rf"{f_num(s)}\,s"
    if s < 3 * 3600:
        return rf"{f_num(s / 60)}\,min"
    if s < 3 * 86400:
        return rf"{f_num(s / 3600)}\,h"
    return rf"{f_num(s / 86400)}\,d"


def f_x(r) -> str:
    r = float(r)
    if not math.isfinite(r):
        return "--"
    return rf"\ensuremath{{{_f_num(r, 2 if r < 100 else 3)}\times}}"


def f_pct(p, dec: int = 1) -> str:
    p = float(p)
    if not math.isfinite(p):
        return "--"
    return rf"{100 * p:.{dec}f}\%"


def tex_name(s: str) -> str:
    """A graph or method name for LaTeX: K_{20} -> $K_{20}$, mu=0.3 -> $\mu{=}0.3$, specials escaped."""
    import re

    s = str(s).replace("&", r"\&").replace("%", r"\%")
    s = re.sub(r"([A-Za-z])_\{([^}]*)\}", r"$\1_{\2}$", s)
    s = re.sub(r"(?<!\$)_", r"\\_", s) if "$" not in s else s
    return s.replace("mu=", r"$\mu{=}$")


def load(results: str, rel: str):
    import pandas as pd

    p = os.path.join(results, rel)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p)


# ------------------------------------------------------------------ E1

def block_e1(R, O: Out):
    d = load(R, "e1/exactness.csv")
    O.put("EOneGraphs", f_int(d["graph"].nunique()), "e1/exactness.csv: graphs")
    O.put("EOneMeasures", f_int(d["measure"].nunique()), "e1: measures")
    O.put("EOneEdgeMeasurePairs", f_int(d["m"].sum()), "e1: sum of edges over graphs x measures (solved)")
    chk = int(d["ortools_checked"].sum())
    bad = int(d["ortools_mismatch"].sum())
    O.put("EOneOrtoolsChecked", f_int(chk), "e1: edges compared as exact fractions with OR-Tools")
    O.put("EOneOrtoolsMismatches", f_int(bad), "e1: mismatches with OR-Tools")
    mm = 0
    for c in ("frozen_mismatch", "oracle_mismatch", "layered_mismatch"):
        if c in d:
            mm += int(np.nansum(d[c].astype(float)))
    O.put("EOneOracleMismatches", f_int(mm), "e1: mismatches with the frozen kernel, the oracle, the layered closure")
    O.put("EOnePotMaxDiff", f_sci(np.nanmax(d["pot_maxdiff"].astype(float))), "e1: max |orcut - POT|")
    O.put("EOneCertChecked", f_int(d["cert_checked"].sum()), "e1: certificates computed")
    O.put("EOneCertFailed", f_int(d["cert_failed"].sum() + d["verify_failed"].sum()), "e1: certificates failed")
    O.put("EOneVerifyChecked", f_int(d["verify_checked"].sum()), "e1: certificates re-verified from scratch")


# ------------------------------------------------------------------ E2

E2_METHODS = [("ours", "orcut", "-", "o"), ("pot", "POT emd (given $D$)", "--", "s"), ("emd2", "ot.emd2 (given $D$)", ":", "^"),
              ("ortools", "OR-Tools (given $D$)", "-.", "D"), ("highs", "HiGHS LP (given $D$)", ":", "v"),
              ("oracle", "previous solver", "--", "x")]


def block_e2(R, O: Out, figdir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load(R, "e2/controlled.csv")
    g = d.groupby(["sweep", "regime", "s"]).median(numeric_only=True).reset_index()
    # headline: largest common symmetric size where POT was run
    sym = g[g["sweep"] == "sym"]
    smax = int(sym.dropna(subset=["pot_s", "ours_s"])["s"].max())
    at = sym[sym["s"] == smax]
    r_pot = (at["pot_s"] / at["ours_s"]).to_numpy()
    r_build = ((at["pot_s"] + at["build_s"]) / at["ours_s"]).to_numpy()
    O.put("ETwoSymMaxS", f_int(smax), "e2/controlled: largest symmetric size")
    O.put("ETwoSymPotMin", f_x(r_pot.min()), "e2: POT/ours at the largest symmetric size, min over regimes")
    O.put("ETwoSymPotMax", f_x(r_pot.max()), "e2: POT/ours, max over regimes")
    O.put("ETwoSymPotBuildMax", f_x(r_build.max()), "e2: (POT + distance matrix)/ours, max over regimes")
    hub = g[g["sweep"] == "hub"]
    if len(hub):
        hmax = int(hub.dropna(subset=["pot_s", "ours_s"])["s"].max())
        ah = hub[hub["s"] == hmax]
        O.put("ETwoHubMaxS", f_int(hmax), "e2: largest hub size")
        O.put("ETwoHubPotMax", f_x((ah["pot_s"] / ah["ours_s"]).max()), "e2: POT/ours at the largest hub size, max")
        hl = ah[ah["regime"] == "hubleaf"]
        if len(hl):
            O.put("ETwoHubLeafPot", f_x(float((hl["pot_s"] / hl["ours_s"]).iloc[0])), "e2: POT/ours, hub-leaf regime")
    if "ortools_s" in at:
        O.put("ETwoSymOrtoolsMax", f_x((at["ortools_s"] / at["ours_s"]).max()), "e2: OR-Tools/ours, max")
    # memory of the general-purpose input at the largest size
    O.put("ETwoDistBytesMax", f_num(d["pot_input_bytes"].max() / 2**20) + r"\,MB", "e2: POT cost matrix at largest size")
    # figure: two panels per sweep, regime 'mixed' (others in the appendix figure)
    for name, regimes in (("fig-per-edge.pdf", ["mixed"]), ("fig-per-edge-all.pdf", list(g["regime"].unique()))):
        fig, axes = plt.subplots(len(regimes), 2, figsize=(6.75, 2.1 * len(regimes)), squeeze=False)
        for r_i, regime in enumerate(regimes):
            for c_i, sweep in enumerate(["sym", "hub"]):
                ax = axes[r_i, c_i]
                sub = g[(g["sweep"] == sweep) & (g["regime"] == regime)].sort_values("s")
                for key, lab, ls, mk in E2_METHODS:
                    col = f"{key}_s"
                    if col in sub and sub[col].notna().any():
                        ax.loglog(sub["s"], sub[col], ls, marker=mk, ms=3, lw=1.2, label=lab)
                if "build_s" in sub:
                    ax.loglog(sub["s"], sub["build_s"], color="0.6", lw=0.8, label="build $D$ only")
                ax.set_xlabel(r"$|\mathcal{U}_i| = |\mathcal{U}_j|$" if sweep == "sym" else r"$|\mathcal{U}_j|$ ($|\mathcal{U}_i| = 32$)",
                              fontsize=8)
                ax.set_ylabel("seconds per edge", fontsize=8)
                ax.set_title(f"{'symmetric' if sweep == 'sym' else 'hub'} sweep, regime '{regime}'", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, which="major", lw=0.3, alpha=0.5)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=6, frameon=False)
        fig.tight_layout(rect=(0, 0.1 if len(regimes) == 1 else 0.04, 1, 1))
        fig.savefig(os.path.join(figdir, name))
        plt.close(fig)
    # real graphs, per bin
    try:
        b = load(R, "e2/real_bins.csv")
        w = b["share"].to_numpy()
        rows = []
        for gname, sub in b.groupby("graph", sort=False):
            ours = float(np.sum(sub["share"] * sub["ours_s"]))
            pot = float(np.sum(sub["share"] * sub["pot_simplex_s"]))
            build = float(np.sum(sub["share"] * sub["build_s"]))
            rows.append((gname, ours, pot, build))
        rp = np.array([p / o for _, o, p, _ in rows])
        rb = np.array([(p + bb) / o for _, o, p, bb in rows])
        O.put("ETwoRealPotMedian", f_x(np.median(rp)), "e2/real_bins: median over graphs of POT(solve)/ours, per edge")
        O.put("ETwoRealPotBuildMedian", f_x(np.median(rb)), "e2/real_bins: median of (POT + build)/ours")
        O.put("ETwoRealGraphs", f_int(len(rows)), "e2/real_bins: graphs")
    except FileNotFoundError:
        O.missing.append("e2/real_bins.csv")


# ------------------------------------------------------------------ E3

def block_e3(R, O: Out, figdir: str):
    d = load(R, "e3/real.csv")
    lz = d[d["measure"] == "lazy"].copy().sort_values("m")
    lines = []
    for _, r in lz.iterrows():
        pot = float(r.get("pot_est_s", float("nan")))
        sp = pot / float(r["time_s"]) if math.isfinite(pot) else float("nan")
        tw = float(r.get("twin_ratio_hub", float("nan")))
        lines.append(" & ".join([tex_name(r["graph"]), f_num(r["n"]), f_num(r["m"]), f_int(r["deg_max"]),
                                 f_time(r["time_s"]), f_time(pot), f_x(sp),
                                 f_num(float(r["peak_extra_mb"]), 2) if math.isfinite(float(r["peak_extra_mb"])) else "--",
                                 f_num(tw, 2) if math.isfinite(tw) else "--"]) + r" \\")
    O.table("large", lines)
    big = lz.iloc[-1]
    O.put("EThreeLargestName", tex_name(big["graph"]), "e3/real: largest graph")
    O.put("EThreeLargestEdges", f_num(big["m"]), "e3/real: edges of the largest graph")
    O.put("EThreeLargestTime", f_time(big["time_s"]), "e3/real: time on the largest graph")
    O.put("EThreeThreads", f_int(big["threads"]), "e3/real: threads")
    O.put("EThreeGraphs", f_int(len(lz)), "e3/real: graphs")
    ok = lz[np.isfinite(lz.get("pot_est_s", np.nan).astype(float))]
    if len(ok):
        sp = ok["pot_est_s"].astype(float) / ok["time_s"].astype(float)
        O.put("EThreePotSpeedupMedian", f_x(np.median(sp)), "e3/real: median POT-estimate / orcut over graphs")
        O.put("EThreePotSpeedupMax", f_x(np.max(sp)), "e3/real: max POT-estimate / orcut")
        O.put("EThreePotSpeedupMin", f_x(np.min(sp)), "e3/real: min POT-estimate / orcut")
    if "twin_ratio_hub_max" in lz:
        O.put("EThreeTwinMax", f_num(np.nanmax(lz["twin_ratio_hub_max"].astype(float))), "e3/real: max rows/groups on hub edges")
    if "share_fast" in lz:
        O.put("EThreeShareFastMedian", f_pct(np.nanmedian(lz["share_fast"].astype(float))), "e3/real: median share of the fast path")
    other = d[d["measure"] != "lazy"]
    if len(other):
        rel = []
        for _, r in other.iterrows():
            base = lz[lz["graph"] == r["graph"]]
            if len(base):
                rel.append(float(r["time_s"]) / float(base["time_s"].iloc[0]))
        if rel:
            O.put("EThreeOtherMeasuresRelMax", f_x(max(rel)), "e3/real: max time(non-lazy or LLY)/time(lazy)")
    # strong scaling
    try:
        t = load(R, "e3/threads.csv")
        best = []
        for gname, sub in t.groupby("graph"):
            t1 = float(sub[sub["threads"] == sub["threads"].min()]["time_s"].iloc[0])
            tT = float(sub[sub["threads"] == sub["threads"].max()]["time_s"].iloc[0])
            best.append((gname, int(sub["threads"].max()), t1 / tT))
        O.put("EThreeScalingThreads", f_int(best[0][1]), "e3/threads: max threads")
        O.put("EThreeScalingMin", f_x(min(b[2] for b in best)), "e3/threads: min speedup at max threads")
        O.put("EThreeScalingMax", f_x(max(b[2] for b in best)), "e3/threads: max speedup at max threads")
    except FileNotFoundError:
        O.missing.append("e3/threads.csv")
    # synthetic scaling: exponent of time against sum_e (d_i + d_j)^2 and against m
    try:
        import matplotlib.pyplot as plt

        s = load(R, "e3/synth.csv")
        lines = []
        fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3))
        for ax, regime in zip(axes, ["fixed16", "grow0.4"]):
            sub = s[s["regime"] == regime]
            for fam, ss in sub.groupby("family"):
                ss = ss.groupby("n").median(numeric_only=True).reset_index().sort_values("m")
                ax.loglog(ss["m"], ss["time_s"], marker="o", ms=3, lw=1, label=fam)
                if len(ss) >= 3:
                    k = np.polyfit(np.log(ss["m"]), np.log(ss["time_s"]), 1)[0]
                    lines.append(f"{tex_name(fam)} & {regime.replace('fixed16', 'avg.~16').replace('grow0.4', r'$2.1n^{0.4}$')} & "
                                 f"{f_num(ss['m'].max())} & {k:.2f}" + r" \\")
            ax.set_xlabel("edges $m$", fontsize=8)
            ax.set_ylabel("seconds (all edges)", fontsize=8)
            ax.set_title("average degree 16" if regime == "fixed16" else r"average degree $2.1\,n^{0.4}$", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)
        axes[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig-synth.pdf"))
        plt.close(fig)
        O.table("synth", lines)
    except FileNotFoundError:
        O.missing.append("e3/synth.csv")


# ------------------------------------------------------------------ E4

E4_ORDER = [("cut", "two-layer cut, bitsets"), ("+relays", "+ relays"), ("+twins", "+ twin merging"),
            ("+fast", "+ fast path"), ("full", "+ warm start (orcut)"), ("-relays", "full $-$ relays"),
            ("-twins", "full $-$ twins"), ("-fast", "full $-$ fast path"), ("oracle", "previous solver (explicit $D$)"),
            ("pot", "explicit $D$ + POT")]


def block_e4(R, O: Out):
    d = load(R, "e4/ablation.csv")
    graphs = list(dict.fromkeys(d["graph"]))
    lines = []
    for cfg, label in E4_ORDER:
        cells = [label]
        for gname in graphs:
            sub = d[(d["graph"] == gname)]
            full = sub[sub["config"] == "full"]["est_total_s"]
            row = sub[sub["config"] == cfg]
            if len(full) and len(row):
                cells.append(f_num(float(row["est_total_s"].iloc[0]) / float(full.iloc[0]), 2))
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
        if cfg == "full":
            lines.append(r"\midrule")
    O.table("ablation", lines)
    O.table("ablation-head", [" & ".join(["configuration"] + [tex_name(g) for g in graphs]) + r" \\"])
    O.put("EFourGraphs", f_int(len(graphs)), "e4: graphs")
    # headline: full vs cut, median over graphs; twins on the most hub-dominated graph
    rat, tw = [], []
    for gname in graphs:
        sub = d[d["graph"] == gname].set_index("config")["est_total_s"].astype(float)
        if "cut" in sub and "full" in sub:
            rat.append(sub["cut"] / sub["full"])
        if "-twins" in sub and "full" in sub:
            tw.append(sub["-twins"] / sub["full"])
        if "pot" in sub and "full" in sub:
            pass
    if rat:
        O.put("EFourCutOverFullMedian", f_x(np.median(rat)), "e4: median (plain cut)/(full)")
        O.put("EFourCutOverFullMax", f_x(np.max(rat)), "e4: max (plain cut)/(full)")
    if tw:
        O.put("EFourTwinsMax", f_x(np.max(tw)), "e4: max (full without twins)/(full)")
    pot = []
    for gname in graphs:
        sub = d[d["graph"] == gname].set_index("config")["est_total_s"].astype(float)
        if "pot" in sub and "full" in sub:
            pot.append(sub["pot"] / sub["full"])
    if pot:
        O.put("EFourPotOverFullMedian", f_x(np.median(pot)), "e4: median (explicit D + POT)/(full), single thread")
        O.put("EFourPotOverFullMin", f_x(np.min(pot)), "e4: min (explicit D + POT)/(full)")
        O.put("EFourPotOverFullMax", f_x(np.max(pot)), "e4: max (explicit D + POT)/(full)")
    mism = int(np.nansum(d.get("value_mismatch", np.zeros(1)).astype(float)))
    O.put("EFourMismatches", f_int(mism), "e4: value mismatches between configurations")


# ------------------------------------------------------------------ E5

E5_ORDER = [("grc-mix", "GRC default (mixed)"), ("grc-otd", "GRC OTD"), ("grc-otd-full", "GRC OTD, no truncation"),
            ("grc-sinkhorn", r"GRC Sinkhorn ($\varepsilon=0.1$)"), ("grc-atd", "GRC ATD"),
            ("jl", "Jost--Liu mean"), ("bf@lly", "Balanced Forman"), ("af3@lly", "AF$_3$")]


def block_e5(R, O: Out, figdir: str):
    d = load(R, "e5/whole.csv")
    lines = []
    for meth, label in E5_ORDER:
        sub = d[(d["method"] == meth)]
        if "status" in sub:
            sub_ok = sub[(sub["status"].isna()) | (sub["status"] == "ok")]
        else:
            sub_ok = sub
        if not len(sub):
            continue
        n_to = int((sub.get("status", "") == "timeout").sum()) if "status" in sub else 0
        def med(col):
            return float(np.nanmedian(sub_ok[col].astype(float))) if col in sub_ok and len(sub_ok) else float("nan")
        def mx(col):
            return float(np.nanmax(sub_ok[col].astype(float))) if col in sub_ok and len(sub_ok) else float("nan")
        cells = [label, f"{len(sub_ok)}/{len(sub)}",
                 f_sci(med("mae")) if math.isfinite(med("mae")) else "--",
                 f_num(mx("max_err"), 2) if math.isfinite(mx("max_err")) else "--",
                 f_pct(med("sign_disagree")), f_num(med("spearman"), 3), f_pct(med("recall_neg"), 0)]
        lines.append(" & ".join(cells) + r" \\")
    O.table("approx", lines)
    oc = d[d["method"] == "orcut"].set_index("graph")["time_s"].astype(float)
    for meth, key in (("grc-mix", "Mix"), ("grc-otd", "Otd"), ("grc-sinkhorn", "Sinkhorn")):
        sub = d[(d["method"] == meth) & ((d.get("status") == "ok") if "status" in d else True)]
        if len(sub):
            r = [float(t) / oc[g] for g, t in zip(sub["graph"], sub["time_s"]) if g in oc and oc[g] > 0]
            if r:
                O.put(f"EFiveGrc{key}SlowdownMedian", f_x(np.median(r)), f"e5/whole: median time({meth})/time(orcut)")
                O.put(f"EFiveGrc{key}SlowdownMax", f_x(np.max(r)), f"e5/whole: max time({meth})/time(orcut)")
            if "hub_max_err" in sub:
                O.put(f"EFiveGrc{key}HubMaxErr", f_num(np.nanmax(sub["hub_max_err"].astype(float)), 2),
                      f"e5/whole: max error on hub edges ({meth})")
            if "max_err" in sub:
                O.put(f"EFiveGrc{key}MaxErr", f_num(np.nanmax(sub["max_err"].astype(float)), 2), f"e5/whole: max error ({meth})")
    to = d[d.get("status", "") == "timeout"] if "status" in d else d.iloc[0:0]
    O.put("EFiveGrcTimeouts", f_int(len(to)), "e5/whole: GRC runs over the time limit")
    O.put("EFiveGraphs", f_int(d["graph"].nunique()), "e5/whole: graphs")
    if "jl_bounds_hold" in d:
        ok = d[d["method"] == "jl"]["jl_bounds_hold"].astype(str).str.lower().eq("true").all()
        O.put("EFiveJLBoundsHold", "all" if ok else "not all", "e5/whole: JL bounds bracket the non-lazy curvature")
    try:
        import matplotlib.pyplot as plt

        s = load(R, "e5/sinkhorn.csv")
        fig, ax = plt.subplots(figsize=(3.25, 2.3))
        for gname, sub in s.groupby("graph"):
            sk = sub[sub["method"] == "sinkhorn"].sort_values("reg")
            line, = ax.loglog(sk["time_edge_weighted_s"], sk["mae_weighted"].clip(lower=1e-12), marker="o", ms=3, lw=1,
                              label=gname)
            for _, r in sk.iterrows():
                ax.annotate(f"{r['reg']:g}", (r["time_edge_weighted_s"], max(r["mae_weighted"], 1e-12)), fontsize=5,
                            xytext=(2, 2), textcoords="offset points")
            o = sub[sub["method"] == "orcut"]
            if len(o):
                ax.axvline(float(o["time_edge_weighted_s"].iloc[0]), color=line.get_color(), ls=":", lw=0.8)
        ax.plot([], [], color="0.3", ls=":", lw=0.8, label="orcut (exact)")
        ax.set_xlabel("seconds per edge (one thread)", fontsize=8)
        ax.set_ylabel("mean absolute error", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(fontsize=5, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig-tradeoff.pdf"))
        plt.close(fig)
        r01 = s[(s["method"] == "sinkhorn") & (np.isclose(s["reg"].astype(float), 0.1))]
        oc = s[s["method"] == "orcut"].set_index("graph")["time_edge_weighted_s"].astype(float)
        if len(r01):
            ratio = [float(t) / oc[g] for g, t in zip(r01["graph"], r01["time_edge_weighted_s"]) if g in oc]
            O.put("EFiveSinkhornTenthSlowdownMedian", f_x(np.median(ratio)), "e5/sinkhorn: time(eps=0.1)/time(orcut), median")
            O.put("EFiveSinkhornTenthMaeMedian", f_sci(np.median(r01["mae_weighted"].astype(float))), "e5/sinkhorn: MAE at eps=0.1")
        rmin = s[(s["method"] == "sinkhorn")]["reg"].astype(float).min()
        rl = s[(s["method"] == "sinkhorn") & (np.isclose(s["reg"].astype(float), rmin))]
        if len(rl):
            ratio = [float(t) / oc[g] for g, t in zip(rl["graph"], rl["time_edge_weighted_s"]) if g in oc and math.isfinite(float(t))]
            O.put("EFiveSinkhornSmallReg", f"{rmin:g}", "e5/sinkhorn: smallest regularization")
            if ratio:
                O.put("EFiveSinkhornSmallSlowdownMedian", f_x(np.median(ratio)), "e5/sinkhorn: time(smallest eps)/time(orcut)")
    except FileNotFoundError:
        O.missing.append("e5/sinkhorn.csv")


# ------------------------------------------------------------------ E6

def block_e6(R, O: Out, figdir: str):
    try:
        c = load(R, "e6/community.csv")
        lfr = c[c["graph"].str.startswith("LFR")].copy()
        if len(lfr):
            lfr["n_"] = lfr["graph"].str.extract(r"LFR\((\d+)")[0].astype(int)
            lfr["mu"] = lfr["graph"].str.extract(r"mu=([0-9.]+)")[0].astype(float)
            methods = [("exact", "exact"), ("bf", "BF"), ("jl", "JL"), ("af3", "AF$_3$"), ("louvain", "Louvain")]
            lines = []
            for (n_, mu), sub in lfr.groupby(["n_", "mu"]):
                cells = [f_int(n_), f"{mu:.1f}"]
                vals = {}
                for meth, _ in methods:
                    x = sub[sub["method"] == meth]["nmi"].astype(float)
                    vals[meth] = (x.mean(), x.std(ddof=1) if len(x) > 1 else float("nan"), len(x))
                best = max(v[0] for k, v in vals.items() if k != "louvain" and math.isfinite(v[0]))
                for meth, _ in methods:
                    mval, sd, k = vals[meth]
                    s_ = f"{mval:.3f}" if math.isfinite(mval) else "--"
                    if meth != "louvain" and math.isfinite(mval) and abs(mval - best) < 1e-12:
                        s_ = r"\textbf{" + s_ + "}"
                    if math.isfinite(sd):
                        s_ += rf"\,{{\scriptsize$\pm${sd:.3f}}}"
                    cells.append(s_)
                lines.append(" & ".join(cells) + r" \\")
            O.table("community", lines)
            ex = lfr[lfr["method"] == "exact"]
            O.put("ESixLfrRuns", f_int(len(ex)), "e6/community: LFR runs per method")
            hard = lfr[lfr["mu"] >= 0.3]
            if len(hard):
                gap = []
                for (gname), sub in hard.groupby("graph"):
                    e_ = sub[sub["method"] == "exact"]["nmi"]
                    p_ = sub[sub["method"].isin(["bf", "jl", "af3"])]["nmi"]
                    if len(e_) and len(p_):
                        gap.append(float(e_.iloc[0]) - float(p_.max()))
                if gap:
                    O.put("ESixNmiGapHardMedian", f_num(np.median(gap), 2), "e6: median NMI(exact) - best proxy NMI, mu >= 0.3")
                    O.put("ESixExactWinsHard", f"{int(np.sum(np.array(gap) > 0))}/{len(gap)}", "e6: exact beats every proxy, mu >= 0.3")
            if "est_full_recompute_s" in ex:
                r = ex["est_full_recompute_s"].astype(float) / ex["time_s"].astype(float)
                O.put("ESixIncrementalSpeedupMedian", f_x(np.nanmedian(r)), "e6: est. full recomputation / incremental, median")
            if "check_mismatch" in ex:
                O.put("ESixCheckMismatches", f_int(np.nansum(ex["check_mismatch"].astype(float))), "e6: incremental vs full mismatches")
            O.put("ESixLargestLfr", f_int(lfr["n_"].max()), "e6: largest LFR size")
            t_big = ex[lfr.loc[ex.index, "n_"] == lfr["n_"].max()]["time_s"].astype(float)
            if len(t_big):
                O.put("ESixLargestLfrTime", f_time(np.median(t_big)), "e6: median time of the exact run on the largest LFR")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, lfr["n_"].nunique(), figsize=(6.75 if lfr["n_"].nunique() > 1 else 3.25, 2.2),
                                     squeeze=False)
            for ax, (n_, sub) in zip(axes[0], lfr.groupby("n_")):
                for meth, lab in methods:
                    ss = sub[sub["method"] == meth].groupby("mu")["nmi"].agg(["mean", "std"]).reset_index()
                    ax.errorbar(ss["mu"], ss["mean"], yerr=ss["std"], marker="o", ms=3, lw=1, capsize=2,
                                label=lab.replace("$", ""), ls="--" if meth == "louvain" else "-")
                ax.set_xlabel(r"mixing $\mu$", fontsize=8)
                ax.set_ylabel("NMI", fontsize=8)
                ax.set_title(f"LFR, n = {n_}", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, lw=0.3, alpha=0.5)
            axes[0, 0].legend(fontsize=6, frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, "fig-community.pdf"))
            plt.close(fig)
        real = c[~c["graph"].str.startswith("LFR") & ~c["graph"].str.startswith("PP")]
        lines = []
        for gname, sub in real.groupby("graph"):
            cells = [tex_name(gname)]
            for meth in ("exact", "bf", "jl", "af3", "louvain"):
                x = sub[sub["method"] == meth]
                cells.append(f"{float(x['nmi'].iloc[0]):.3f} / {float(x['ari'].iloc[0]):.3f}" if len(x) else "--")
            lines.append(" & ".join(cells) + r" \\")
        O.table("community-real", lines)
    except FileNotFoundError:
        O.missing.append("e6/community.csv")
    try:
        b = load(R, "e6/borf.csv")
        lines = []
        sp_grc, sp_inc = [], []
        for gname, sub in b.groupby("graph", sort=False):
            s_ = sub.set_index("backend")
            def get(bk, col):
                try:
                    return float(s_.loc[bk, col])
                except Exception:
                    return float("nan")
            grc = get("grc", "curv_per_iter_s")
            full = get("orcut-full", "curv_per_iter_s")
            inc = get("orcut-incremental", "curv_per_iter_s")
            plan = get("orcut-incremental", "plan_per_iter_s")
            aff = get("orcut-incremental", "affected_share")
            m = get("orcut-full", "m")
            status = str(s_.loc["grc", "status"]) if "grc" in s_.index and "status" in s_ else "ok"
            lines.append(" & ".join([tex_name(gname), f_num(m), f_time(grc) if status == "ok" else r"\textit{" + status + "}",
                                     f_time(full), f_time(inc), f_pct(aff, 0), f_time(plan)]) + r" \\")
            if math.isfinite(grc) and math.isfinite(full) and full > 0:
                sp_grc.append(grc / full)
            if math.isfinite(inc) and math.isfinite(full) and inc > 0:
                sp_inc.append(full / inc)
        O.table("borf", lines)
        if sp_grc:
            O.put("ESixBorfGrcSpeedupMedian", f_x(np.median(sp_grc)), "e6/borf: GRC / orcut-full per iteration, median")
            O.put("ESixBorfGrcSpeedupMax", f_x(np.max(sp_grc)), "e6/borf: GRC / orcut-full per iteration, max")
        if sp_inc:
            O.put("ESixBorfIncSpeedupMedian", f_x(np.median(sp_inc)), "e6/borf: full / incremental per iteration, median")
        if "check_mismatch" in b:
            O.put("ESixBorfMismatches", f_int(np.nansum(b["check_mismatch"].astype(float))), "e6/borf: mismatches")
        O.put("ESixBorfGraphs", f_int(b["graph"].nunique()), "e6/borf: graphs")
    except FileNotFoundError:
        O.missing.append("e6/borf.csv")


# ------------------------------------------------------------------ E7

def block_e7(R, O: Out):
    try:
        d = load(R, "e7/measures.csv")
        lines = []
        for meas, sub in d.groupby("measure", sort=False):
            r = sub["rel_lazy"].astype(float)
            lines.append(f"{meas} & {f_num(np.median(r), 2)} & {f_num(np.max(r), 2)}" + r" \\")
        O.table("measures", lines)
        nl = d[d["measure"] != "lazy"]["rel_lazy"].astype(float)
        O.put("ESevenRelMax", f_x(np.nanmax(nl)), "e7/measures: max time relative to lazy")
        O.put("ESevenRelMedian", f_x(np.nanmedian(nl)), "e7/measures: median time relative to lazy")
    except FileNotFoundError:
        O.missing.append("e7/measures.csv")
    try:
        d = load(R, "e7/radius.csv")
        lines = []
        for cfg, sub in d.groupby("config", sort=False):
            r = sub["pot_est_total_1thread_s"].astype(float) / sub["est_total_1thread_s"].astype(float)
            kind, rr = cfg.split(" r=")
            label = ("uniform on $B_{%s}$" if kind == "ball" else "%s-step lazy walk") % rr
            lines.append(f"{label} & {f_num(np.nanmedian(r), 2)} & {f_num(np.nanmin(r), 2)}--{f_num(np.nanmax(r), 2)}"
                         f" & {f_sci(np.nanmax(sub['max_diff_vs_pot'].astype(float)))}"
                         f" & {f_pct(np.nanmin(sub['certified_share'].astype(float)), 0)}" + r" \\")
        O.table("radius", lines)
        O.put("ESevenCertifiedMin", f_pct(np.nanmin(d["certified_share"].astype(float)), 0), "e7/radius: min certified share")
        r2 = d[d["config"] == "ball r=2"]
        if len(r2):
            rr = r2["pot_est_total_1thread_s"].astype(float) / r2["est_total_1thread_s"].astype(float)
            O.put("ESevenBallTwoPotMedian", f_x(np.nanmedian(rr)), "e7/radius: POT/orcut, ball r=2, median")
    except FileNotFoundError:
        O.missing.append("e7/radius.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=common.RESULTS)
    ap.add_argument("--paper-dir", default=None)
    ap.add_argument("--provisional", action="store_true", help="mark every number as provisional (quick run)")
    a = ap.parse_args()
    paper = a.paper_dir or os.path.join(a.results, "paper")
    os.makedirs(paper, exist_ok=True)
    O = Out(a.provisional)
    for name, fn in (("E1", lambda: block_e1(a.results, O)), ("E2", lambda: block_e2(a.results, O, paper)),
                     ("E3", lambda: block_e3(a.results, O, paper)), ("E4", lambda: block_e4(a.results, O)),
                     ("E5", lambda: block_e5(a.results, O, paper)), ("E6", lambda: block_e6(a.results, O, paper)),
                     ("E7", lambda: block_e7(a.results, O))):
        try:
            fn()
        except FileNotFoundError as ex:
            O.missing.append(str(ex))
        except Exception:
            O.missing.append(f"{name}: error\n" + traceback.format_exc())
    lines = ["% generated by experiments/artifacts.py -- do not edit",
             r"\makeatletter"]
    for k, v in sorted(O.keys.items()):
        lines.append(rf"\@namedef{{res@{k}}}{{{v}}}")
    for k, v in sorted(O.rows.items()):
        lines.append(rf"\@namedef{{resrows@{k}}}{{%" + "\n" + v + "\n}")
    lines.append(r"\makeatother")
    xio.atomic_text("\n".join(lines) + "\n", os.path.join(paper, "solver_numbers.tex"))
    md = ["# Keys of solver_numbers.tex", "", "| key | value | source |", "|---|---|---|"]
    md += [f"| `{k}` | `{O.keys[k]}` | {O.src[k]} |" for k in sorted(O.keys)]
    md += ["", "## Tables", ""] + [f"- `{k}`: {len(v.splitlines())} rows" for k, v in sorted(O.rows.items())]
    if O.missing:
        md += ["", "## Missing or failed", ""] + [f"- {m}" for m in O.missing]
    xio.atomic_text("\n".join(md) + "\n", os.path.join(paper, "KEYS.md"))
    common.log(f"artifacts: {len(O.keys)} numbers, {len(O.rows)} tables -> {paper}"
               + (f"; missing: {len(O.missing)}" if O.missing else ""))
    for m in O.missing:
        print("  missing:", m.splitlines()[0])


if __name__ == "__main__":
    main()
