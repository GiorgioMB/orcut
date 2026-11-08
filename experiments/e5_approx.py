"""E5. What approximate pipelines get wrong, and what they cost.

Part ``whole``: on every graph, the exact curvature at idleness 1/2 (GraphRicciCurvature's default
alpha = 0.5) and the non-lazy curvature (alpha = 0, as in BORF) are computed with orcut; against
them we score

  grc-mix       GraphRicciCurvature 0.6.1 with its defaults (OTDSinkhornMix, nbr_topk = 3000,
                ot.emd2 with numItermax = 100000), alpha = 0.5;
  grc-otd       method OTD (ot.emd2 everywhere), nbr_topk = 3000;
  grc-otd-full  method OTD with no neighbour truncation (only when some degree exceeds 3000);
  grc-sinkhorn  method Sinkhorn (regularization 0.1);
  grc-atd       method ATD (average transport distance, an upper bound of W1);
  bf, af3, jl   one-pass proxies (Balanced Forman, triangle-augmented Forman, mean of the
                Jost--Liu bounds), scored by rank (Spearman, sign, recall of the 1% most negative /
                positive edges) against the non-lazy and the Lin--Lu--Yau curvature, and jl also by
                its error against the non-lazy curvature, which it estimates. On every edge we
                check that the Jost--Liu bounds bracket the non-lazy curvature and that BF is below
                the Lin--Lu--Yau curvature (Topping et al., 2022).

Errors are reported on all edges and on hub edges (an endpoint of degree >= ``--hub``); every
method gets the same number of threads (``--threads``) and is timed end to end (GRC includes its
shortest paths and its process pool). GRC runs in its own environment (``--grc-python`` or
$ORCUT_GRC_PYTHON). A GRC run has no time limit. On graphs above ``--grc-chunk-above`` edges, and
wherever one call did not finish before, it runs in contiguous chunks of edges of about
``--grc-chunk`` seconds, each saved as soon as it is done (its time is then the sum over the chunks,
in ``chunks`` of them); a single call killed by its job's memory limit is tried again only by a job
with more memory.

Part ``sinkhorn``: time-accuracy trade-off on a stratified sample of edges (bins of d_i + d_j), on
the local problems at idleness 1/2: entropic OT at regularization ``--regs`` against the exact value,
per-edge solve time (given the distance matrix) against the per-edge time of orcut and of POT's
network simplex.

Usage: python e5_approx.py --part whole|sinkhorn [--graphs SPEC ...] [--threads 32] [--grc-python PY]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from fractions import Fraction

import numpy as np

import common
from common import Rows, agreement, log, out_dir
import baselines as B
import registry as R
import xio

GRAPHS = ["bundled:yeast", "bundled:arxiv", "real:Deezer", "real:Twitch-DE", "real:Facebook", "real:Squirrel",
          "real:Crocodile", "real:Amazon-Computers", "real:Enron", "real:AstroPh", "real:GitHub", "real:Epinions",
          "synth:ER:20000:20:0", "synth:CL2.1:10000:16:0", "synth:CL2.1:30000:12:0"]
GRC_METHODS = [("grc-mix", "OTDSinkhornMix", 3000), ("grc-otd", "OTD", 3000), ("grc-otd-full", "OTD", 10**9),
               ("grc-sinkhorn", "Sinkhorn", 3000), ("grc-atd", "ATD", 3000)]
HALF = Fraction(1, 2)


def part_whole(a):
    import orcut

    rows = Rows(f"{out_dir(a.out, 'e5')}/whole.csv", ["graph", "method"])
    cache = out_dir(a.out, "e5", "values")
    import partial as PT

    grc_ok = B.grc_python(a.grc_python) is not None
    grc_runs = PT.Parts(os.path.join(out_dir(a.out, "e5", "partial"), "grc.json"))  # the GRC runs under way
    kept = 0  # GRC rows left as unavailable because GRC is not available in this run
    log(f"E5 GraphRicciCurvature: {B.grc_python(a.grc_python)}" if grc_ok else
        "E5 GraphRicciCurvature unavailable: its rows are recorded as unavailable (see main.py's log)")
    proxy_rows = [meth if target == "nonlazy" else f"{meth}@lly" for meth in ("bf", "af3", "jl")
                  for target in ("nonlazy", "lly")]
    for spec in a.graphs:
        # what is left for this graph; without GRC, its rows that already say "unavailable" stay as they are
        name = R.display(spec)
        need_orcut = not rows.done(graph=name, method="orcut")
        need_px = not all(rows.done(graph=name, method=m) for m in proxy_rows)
        grc_left = []
        for meth, grc_method, _ in GRC_METHODS:
            r = rows.get(graph=name, method=meth)
            # an ATD row skipped because its all-pairs distances did not fit runs when this job has the memory
            fits_now = r is not None and grc_ok and grc_method == "ATD" and \
                str(r.get("status") or "").startswith("skipped: needs") and \
                B.grc_atd_need(int(common.num(r, "n"))) <= (PT.memory_available() or 0)
            # a timeout of the old 2-hour limit, or a method it skipped, runs now (there is no limit any more);
            # so does a run given up by a job whose time or memory this one exceeds
            again = r is not None and grc_ok and (str(r.get("status") or "") in ("timeout", "skipped") or PT.retry_now(r))
            if fits_now or again or \
                    (not rows.done(graph=name, method=meth) and (grc_ok or r is None or common.row_failed(r))):
                grc_left.append(meth)
                if r is not None:
                    log(f"E5 {name:16s} {meth:12s} computed again (was: {str(r.get('status'))[:60]})")
            elif r is not None and not rows.done(graph=name, method=meth):
                kept += 1
        d0 = common.num(rows.get(graph=name, method="orcut") or {}, "deg_max")
        if d0 == d0 and d0 <= 3000:  # grc-otd-full is only run when some degree exceeds 3000
            grc_left = [m for m in grc_left if m != "grc-otd-full"]
        if not (need_orcut or need_px or grc_left):
            continue
        name, g, meta = R.load(spec, a.data_dir)
        if meta["deg_max"] <= 3000:
            grc_left = [m for m in grc_left if m != "grc-otd-full"]
        if not (need_orcut or need_px or grc_left):
            continue
        hub = np.maximum(g.deg[g.edges[:, 0]], g.deg[g.edges[:, 1]]) >= a.hub
        base = dict(graph=name, n=g.num_nodes, m=g.num_edges, deg_max=meta["deg_max"], threads=a.threads)
        k_half = k_zero = None  # the exact values: needed by the orcut and proxy rows, and to score GRC
        if need_orcut or need_px or (grc_ok and grc_left):
            orcut.curvature(g, HALF, n_jobs=a.threads)  # warm
            t0 = time.perf_counter()
            k_half = orcut.curvature(g, HALF, n_jobs=a.threads)
            t_half = time.perf_counter() - t0
            t0 = time.perf_counter()
            k_zero = orcut.curvature(g, "nonlazy", n_jobs=a.threads)
            t_zero = time.perf_counter() - t0
            np.savez_compressed(os.path.join(cache, f"{R.slug(name)}.npz"), half=k_half, zero=k_zero, edges=g.edges)
        if need_orcut:
            rows.add(dict(base, method="orcut", target="idle 1/2", time_s=t_half, time_nonlazy_s=t_zero,
                          hub_edges=int(hub.sum()), share_negative=float(np.mean(k_half < 0))))
        if need_px:
            B.proxies(g)  # compile
            t0 = time.perf_counter()
            px = B.proxies(g)
            t_px = time.perf_counter() - t0
            # the Jost--Liu bounds bracket the non-lazy curvature, and BF bounds the Lin--Lu--Yau
            # curvature from below (= 2 x the curvature at idleness 1/2); checked on every edge
            k_lly = orcut.curvature(g, "lly", n_jobs=a.threads)
            jl_ok = bool(np.all(px["jl_lower"] <= k_zero + 1e-12) and np.all(k_zero <= px["jl_upper"] + 1e-12))
            bf_ok = bool(np.all(px["bf"] <= k_lly + 1e-12))
            for meth, key in (("bf", "bf"), ("af3", "af3"), ("jl", "jl_mean")):
                for target, ref in (("nonlazy", k_zero), ("lly", k_lly)):
                    row = dict(base, method=meth if target == "nonlazy" else f"{meth}@lly", target=target, time_s=t_px,
                               jl_bounds_hold=jl_ok, bf_bound_holds=bf_ok)
                    row.update(agreement(ref, px[key], hub, errors=(meth == "jl" and target == "nonlazy")))
                    rows.add(row)
        for meth, grc_method, topk in GRC_METHODS:
            if meth not in grc_left:
                continue
            extra = {}
            state_dir = os.path.join(out_dir(a.out, "e5", "partial"), f"grc.{R.slug(name)}.{meth}")
            if grc_ok:
                st, gave = grc_runs.resume((name, meth), "grc")
                prev = str((rows.get(graph=name, method=meth) or {}).get("status") or "")
                # in saved chunks: a run that did not finish in one call before (a timeout, a method it
                # skipped, killed by its job's time limit), one under way in chunks, or a large graph
                chunked = os.path.isdir(state_dir) or g.num_edges > a.grc_chunk_above or \
                    prev in ("timeout", "skipped") or prev.startswith("did not finish") or \
                    (st == "gave_up" and gave["status"].startswith("did not finish"))
                if chunked:
                    val, t, status, nch = B.grc_chunked(g, grc_method, 0.5, topk, a.threads, a.grc_python, state_dir,
                                                        chunk_s=a.grc_chunk, log=lambda msg: log(f"E5 {name:16s} {msg}"))
                    extra = dict(chunks=nch)
                elif st == "gave_up":  # killed by its job's memory limit: only a bigger job runs it again
                    val, t, status = None, float("nan"), gave["status"]
                    extra = {k: v for k, v in gave.items() if k != "status"}
                else:
                    grc_runs.begin((name, meth), "grc")
                    val, t, status = B.grc(g, grc_method, 0.5, topk, a.threads, a.grc_python)
            else:
                val, t, status = None, float("nan"), "unavailable"
            row = dict(base, method=meth, target="idle 1/2", status=status, time_s=t, **extra)
            if val is not None:
                row.update(agreement(k_half, val, hub))
                np.save(os.path.join(cache, f"{R.slug(name)}.{meth}.npy"), val)
            rows.add(row)
            grc_runs.drop((name, meth))
            if status == "ok":
                shutil.rmtree(state_dir, ignore_errors=True)
            log(f"E5 {name:16s} {meth:12s} {status:8s} {t:9.1f}s "
                + (f"MAE {row.get('mae', float('nan')):.2e} max {row.get('max_err', float('nan')):.3f} "
                   f"recall- {row.get('recall_neg', float('nan')):.3f}" if val is not None else ""))
        if k_half is not None:
            log(f"E5 {name:16s} orcut idle 1/2 {t_half:.2f}s, non-lazy {t_zero:.2f}s ({a.threads} threads)")
    if kept:
        log(f"E5 {kept} GRC rows left as unavailable: GraphRicciCurvature is not available in this run")


SINK_BYTES = 80.0  # bytes per entry of a dense local problem for POT or Sinkhorn (distances, kernel, plan), rounded up


def sinkhorn_cap():
    """The largest dense local problem (entries) this job can give POT or Sinkhorn: what fits in its
    memory (one edge at a time, single-threaded); at least 5 x 10^7, the fixed limit of before."""
    import partial as PT

    return max(5e7, (PT.memory_available() or 0.0) / SINK_BYTES)


def sinkhorn_done(rows, a, cap=None, **key):
    """The row is in the table, is not one of the earlier versions' budget (fewer edges timed than
    sampled, and no count of the edges skipped for memory), and did not skip edges for memory that
    this job can hold (``cap``; rows that do not record theirs had 5 x 10^7)."""
    if not rows.done(**key):
        return False
    r = rows.get(**key)
    skipped = common.num(r, "skipped_memory")
    if skipped != skipped:
        return common.num(r, "timed") >= common.num(r, "sample")
    if skipped > 0:
        had = common.num(r, "max_pairs")
        had = 5e7 if had != had else had
        return not (cap if cap is not None else sinkhorn_cap()) > 1.01 * had
    return True


def part_sinkhorn(a):
    """Per stratum and method: per-edge time (solve given the distance matrix; for orcut everything)
    and error against the exact value; the graph-level time and mean error are weighted by the bin
    counts (what the method would cost, and how wrong it would be, on average over all edges).
    Every sampled edge is timed, however long it takes; each is saved as soon as it is
    (<out>/e5/partial/), so a run stopped in the middle of a row resumes at the next edges. Edges
    whose dense local problem does not fit in the job's memory (``sinkhorn_cap``, or ``--max-pairs``)
    are counted in skipped_memory; the saved times are kept (<name>.done.json), so that a job with more
    memory computes those edges only."""
    import numba
    import orcut

    numba.set_num_threads(1)
    rows = Rows(f"{out_dir(a.out, 'e5')}/sinkhorn.csv", ["graph", "method", "reg"])
    methods = [("orcut", 0.0), ("pot", 0.0)] + [("sinkhorn", r) for r in a.regs]
    for spec in a.graphs:
        cap = a.max_pairs or sinkhorn_cap()
        if all(sinkhorn_done(rows, a, cap, graph=R.display(spec), method=meth, reg=reg) for meth, reg in methods):
            continue
        name, g, meta = R.load(spec, a.data_dir)
        cap = a.max_pairs or sinkhorn_cap()  # with the graph loaded
        max_pairs = cap
        strata = common.stratified_bins(g, a.per_bin, seed=1)
        exact = [orcut.curvature(g, HALF, edges=s, n_jobs=1) for _, s, _ in strata]
        cnt = np.array([c for _, _, c in strata], float)
        hubs = [np.maximum(g.deg[g.edges[s, 0]], g.deg[g.edges[s, 1]]) >= a.hub for _, s, _ in strata]
        too_big = [(g.deg[g.edges[s, 0]] + 1.0) * (g.deg[g.edges[s, 1]] + 1.0) > max_pairs for _, s, _ in strata]
        for meth, reg in methods:
            if sinkhorn_done(rows, a, cap, graph=name, method=meth, reg=reg):
                continue
            r0 = rows.get(graph=name, method=meth, reg=reg)
            if r0 is not None:
                why = (f"{r0.get('skipped_memory')} edges were too large for the memory of the job that ran it"
                       if common.num(r0, "skipped_memory") > 0 else "without the old time budget")
                log(f"E5S {name:16s} {meth:8s} reg={reg:<6g} computed again: {why} "
                    f"(it timed {r0.get('timed')}/{r0.get('sample')} edges)")
            ck = os.path.join(out_dir(a.out, "e5", "partial"), f"sinkhorn.{R.slug(name)}.{meth}.{reg:g}.json")
            ck_done = ck[:-len(".json")] + ".done.json"  # the times of a finished row
            sample_ids = [int(x) for _, s, _ in strata for x in s]
            saved = xio.read_json(ck, default=None)
            if not (isinstance(saved, dict) and saved.get("sample") == sample_ids):
                saved = xio.read_json(ck_done, default=None)
            if not (isinstance(saved, dict) and saved.get("sample") == sample_ids):
                saved = dict(sample=sample_ids, edges={})
            done = saved["edges"]  # edge id -> [seconds, error] (NaN where skipped for memory)
            fit_now = [str(int(e)) for (_, s, _), big in zip(strata, too_big) for e, b in zip(s, big)
                       if not b and str(int(e)) in done and done[str(int(e))][0] != done[str(int(e))][0]]
            for e in fit_now:  # skipped for memory by a smaller job: this one holds them
                del done[e]
            if fit_now:
                log(f"E5S {name:16s} {meth:8s} reg={reg:<6g} {len(fit_now)} edges skipped for memory before fit now")
            for (bn, s, c), ex, big in zip(strata, exact, too_big):
                if meth == "orcut":
                    if not all(str(int(e)) in done for e in s):
                        t = common.time_per_edge(lambda sub: orcut.curvature(g, HALF, edges=sub, n_jobs=1), s, reps=3)
                        done.update({str(int(e)): [float(x), 0.0] for e, x in zip(s, t)})
                        xio.atomic_json(saved, ck)
                    continue
                todo = [k for k, e in enumerate(s) if str(int(e)) not in done]
                for q in range(0, len(todo), 5):
                    ks = np.array(todo[q:q + 5], np.int64)
                    t = np.full(len(ks), np.nan)
                    v, _, _, _ = B.run_local(g, s[ks], "pot_simplex" if meth == "pot" else "sinkhorn", kind="idle:1/2",
                                             reps=1, reg=reg or None, times=t, max_pairs=max_pairs)
                    done.update({str(int(s[k])): [float(x), float(abs(y - ex[k]))] for k, x, y in zip(ks, t, v)})
                    xio.atomic_json(saved, ck)
            times = [np.array([done[str(int(e))][0] for e in s], float) for _, s, _ in strata]
            errs = [np.array([done[str(int(e))][1] for e in s], float) for _, s, _ in strata]
            ok_bins = np.array([np.isfinite(e).any() for e in errs])
            mean_err_bin = np.array([np.nanmean(e) if np.isfinite(e).any() else np.nan for e in errs])
            hub_err = np.concatenate([e[h] for e, h in zip(errs, hubs)])
            all_err = np.concatenate(errs)
            skipped = 0 if meth == "orcut" else int(sum(int(b.sum()) for b in too_big))
            row = dict(graph=name, method=meth, reg=reg, m=g.num_edges, sample=int(sum(len(s) for _, s, _ in strata)),
                       timed=int(np.isfinite(all_err).sum()), skipped_memory=skipped, max_pairs=max_pairs,
                       bins_complete=int(ok_bins.sum()), bins=len(strata),
                       est_total_1thread_s=common.estimate_total(strata, times),
                       mae_weighted=float(np.sum(cnt[ok_bins] * mean_err_bin[ok_bins]) / np.sum(cnt[ok_bins])),
                       max_err=float(np.nanmax(all_err)) if np.isfinite(all_err).any() else float("nan"),
                       hub_mae=float(np.nanmean(hub_err)) if np.isfinite(hub_err).any() else float("nan"))
            row["time_edge_weighted_s"] = row["est_total_1thread_s"] / g.num_edges
            rows.add(row)
            xio.atomic_json(saved, ck_done)  # kept: a job with more memory then computes the skipped edges only
            try:
                os.remove(ck)
            except OSError:
                pass
            log(f"E5S {name:16s} {meth:8s} reg={reg:<6g} {1e6 * row['time_edge_weighted_s']:10.1f}us/edge "
                f"MAE {row['mae_weighted']:.2e} max {row['max_err']:.3f} ({row['timed']}/{row['sample']} timed"
                + (f", {skipped} too large for memory" if skipped else "") + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["whole", "sinkhorn"], required=True)
    ap.add_argument("--graphs", nargs="+", default=GRAPHS)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--hub", type=int, default=1000, help="hub edges: an endpoint of at least this degree")
    ap.add_argument("--grc-python", default=None, help="python of the GRC environment (default $ORCUT_GRC_PYTHON)")
    ap.add_argument("--grc-chunk", type=float, default=3600.0, help="part whole: seconds per saved chunk of a long GRC run")
    ap.add_argument("--grc-chunk-above", type=int, default=10**6,
                    help="part whole: GRC runs in saved chunks on graphs with more edges (and wherever one call did "
                         "not finish before)")
    ap.add_argument("--regs", nargs="+", type=float, default=[1.0, 0.3, 0.1, 0.03, 0.01])
    ap.add_argument("--per-bin", type=int, default=50)
    ap.add_argument("--budget", type=float, default=None, help="ignored: part sinkhorn has no time budget any more")
    ap.add_argument("--max-pairs", type=float, default=None,
                    help="part sinkhorn: largest dense local problem (entries) for POT and Sinkhorn "
                         "(default: what fits in the job's memory)")
    ap.add_argument("--out", default=common.RESULTS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.graphs = ["bundled:jazz", "synth:CL2.1:5000:8:0"]
        a.hub, a.per_bin, a.regs = 200, 10, [1.0, 0.1, 0.01]
    common.warm_jit()
    (part_whole if a.part == "whole" else part_sinkhorn)(a)


if __name__ == "__main__":
    main()
    sys.exit(common.finish())
