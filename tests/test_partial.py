"""Checkpoints inside a row (experiments/partial.py, e2's methods): a part killed in the middle is
computed again, unless it was killed by the job's memory or time limit, in which case only a job
with more memory or more time tries it again."""
import os
import sys

import numpy as np
import pytest

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, EXP)

import common  # noqa: E402
import partial as PT  # noqa: E402


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """Stand-ins for sacct (how the last job ended) and squeue (this job's time limit)."""
    b = tmp_path / "bin"
    b.mkdir()

    def set_(end="", limit="2-00:00:00", mem_mb=64000):
        (b / "sacct").write_text(f"#!/bin/sh\necho '{end}'\n")
        (b / "squeue").write_text(f"#!/bin/sh\necho '{limit}'\n")
        for f in ("sacct", "squeue"):
            (b / f).chmod(0o755)
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(mem_mb))

    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    return set_


def test_slurm_times():
    assert PT._slurm_seconds("2-00:00:00") == 172800 and PT._slurm_seconds("12:00:00") == 43200
    assert PT._slurm_seconds("30:00") == 1800 and PT._slurm_seconds("UNLIMITED") is None


def test_part_killed_by_something_else_is_tried_again_up_to_three_times(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))  # no Slurm
    p = PT.Parts(str(tmp_path / "t.json"))
    for k in range(PT.ATTEMPTS):
        assert PT.Parts(str(tmp_path / "t.json")).resume(("r",), "pot") == ("todo", None)
        p.begin(("r",), "pot")  # ... and the process dies
    st, fields = PT.Parts(str(tmp_path / "t.json")).resume(("r",), "pot")
    assert st == "gave_up" and "interrupted 3 times" in fields["status"]


def test_part_killed_by_the_memory_limit_waits_for_a_job_with_more(tmp_path, slurm):
    slurm(end="OUT_OF_MEMORY", mem_mb=64000)
    p = PT.Parts(str(tmp_path / "t.json"))
    p.begin(("r",), "pot")
    st, fields = PT.Parts(str(tmp_path / "t.json")).resume(("r",), "pot")
    assert st == "gave_up" and fields["status"].startswith("out of memory with 62 GB")
    row = {"pot_s": "nan", "pot_status": fields["status"], "pot_mem_bytes": fields["mem_bytes"]}
    assert not PT.retry_now(row, "pot")
    slurm(end="OUT_OF_MEMORY", mem_mb=256000)  # resubmitted with --mem=256G
    assert PT.Parts(str(tmp_path / "t.json")).resume(("r",), "pot") == ("todo", None)
    assert PT.retry_now(row, "pot")


def test_part_killed_by_the_time_limit_waits_for_a_longer_job(tmp_path, slurm):
    slurm(end="TIMEOUT", limit="2-00:00:00")
    p = PT.Parts(str(tmp_path / "t.json"))
    p.begin(("r",), "highs")
    st, fields = PT.Parts(str(tmp_path / "t.json")).resume(("r",), "highs")
    assert st == "gave_up" and fields["status"].startswith("did not finish") and fields["time_limit_s"] == 172800
    row = {"highs_s": "nan", "highs_status": fields["status"], "highs_time_limit_s": fields["time_limit_s"]}
    assert not PT.retry_now(row, "highs")
    slurm(end="TIMEOUT", limit="5-00:00:00")
    assert PT.retry_now(row, "highs")


def test_finished_parts_are_reused_and_dropped_with_their_row(tmp_path):
    p = PT.Parts(str(tmp_path / "t.json"))
    p.begin(("r", 1), "ours")
    p.finish(("r", 1), "ours", {"ours_s": 0.5})
    assert PT.Parts(str(tmp_path / "t.json")).resume(("r", 1), "ours") == ("done", {"ours_s": 0.5})
    p.drop(("r", 1))
    assert PT.Parts(str(tmp_path / "t.json")).resume(("r", 1), "ours") == ("todo", None)


def test_e2_method_interrupted_in_the_middle_of_a_row_resumes(tmp_path, monkeypatch):
    import e2_per_edge as E2
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    parts = PT.Parts(str(tmp_path / "c.json"))
    calls = []

    def ok():
        calls.append(1)
        return 0.25

    def dies():
        raise KeyboardInterrupt  # the job is killed while this method runs

    pk = ("blocks", 4096, 0)
    assert E2._method(parts, pk, "ours", ok, 0, 0.25, 30.0)["ours_err"] == 0.0
    with pytest.raises(KeyboardInterrupt):
        E2._method(parts, pk, "pot", dies, 0, 0.25, 30.0)
    parts = PT.Parts(str(tmp_path / "c.json"))  # the next run
    n = len(calls)
    assert E2._method(parts, pk, "ours", ok, 0, 0.25, 30.0)["ours_err"] == 0.0 and len(calls) == n  # saved: not again
    assert "pot_s" in E2._method(parts, pk, "pot", ok, 0, 0.25, 30.0)  # computed now
    # too large for the job's memory: skipped with the reason, and run once a job has enough
    got = E2._method(parts, pk, "highs", ok, 1e18, 0.25, 30.0)
    assert got["highs_status"].startswith("skipped: needs") and not PT.retry_now(got, "highs")


def test_e2_rows_of_earlier_budgets_fill_in_what_they_skipped(tmp_path, monkeypatch):
    import e2_per_edge as E2
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "d.csv"), ["kind", "n", "seed"])
    t.add(dict(kind="blocks", n=16384, seed=0, ours_s=0.1, pot_s=float("nan")))             # an old row: no reason
    t.add(dict(kind="blocks", n=4096, seed=0, ours_s=0.1, pot_s=3.9))                       # complete
    t.add(dict(kind="blocks", n=2**20, seed=0, ours_s=0.1, pot_s=float("nan"), pot_need_bytes=1e18,
               pot_status="skipped: needs ~1e9 GB, the job has 64 GB"))                    # cannot fit
    t = common.Rows(str(tmp_path / "d.csv"), ["kind", "n", "seed"])
    assert E2._row_left(t, ["ours", "pot"], kind="blocks", n=16384, seed=0)[0] == ["pot"]
    assert E2._row_left(t, ["ours", "pot"], kind="blocks", n=4096, seed=0)[0] is None
    assert E2._row_left(t, ["ours", "pot"], kind="blocks", n=2**20, seed=0)[0] is None


def test_synth_rows_are_found_by_their_requested_size(tmp_path, monkeypatch):
    import e3_scale as E3
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "synth.csv"), ["family", "regime", "n", "seed"])
    t.add(dict(family="CL2.1", regime="fixed16", n=288752, seed=0, m=1966735, time_s=2020.75))   # older row
    t.add(dict(family="CL2.1", regime="fixed16", n=958274, seed=0, m=6653529, time_s=30737.7))   # requested 10^6
    t.add(dict(family="ER", regime="fixed16", n=9999998, n_target=10**7, seed=0, m=79999925, time_s=56.2))
    t = common.Rows(str(tmp_path / "synth.csv"), ["family", "regime", "n", "seed"])
    assert E3.synth_done(t, "CL2.1", "fixed16", 3 * 10**5, 0) and E3.synth_done(t, "CL2.1", "fixed16", 10**6, 0)
    assert not E3.synth_done(t, "CL2.1", "fixed16", 3 * 10**6, 0) and not E3.synth_done(t, "CL2.1", "fixed16", 10**5, 0)
    assert E3.synth_done(t, "ER", "fixed16", 10**7, 0) and not E3.synth_done(t, "ER", "fixed16", 3 * 10**6, 0)
    m = E3.synth_m(t, "CL2.1", "fixed16", 10**7)
    assert 6e7 < m < 7.5e7  # from 958,274 vertices / 6.65M edges
    assert E3.predicted_time(t, "CL2.1", "fixed16", 0, m) > 2 * 86400  # waits until the others are done


def test_unstarted_chunk_plan_takes_the_new_chunk_size(tmp_path):
    import json
    import datasets as D
    import e3_scale as E3
    g = D.to_graph(*D.synthetic("BA", 400, 6, 0))
    ck = str(tmp_path / "c.json")
    stop = type("Stop", (Exception,), {})

    def halt(c):
        raise stop

    with pytest.raises(stop):  # plan of 4 chunks, killed during... after the first
        E3.solve_chunked(g, "lazy", 1, ck, g.num_edges // 4, "G", after_chunk=halt)
    k4, k40 = -(-g.num_edges // (g.num_edges // 4)), -(-g.num_edges // (g.num_edges // 40))
    st = json.load(open(ck))
    assert len(st["chunks"]) == 1 and st["k"] == k4
    E3.solve_chunked(g, "lazy", 1, ck, g.num_edges // 40, "G")  # a chunk is done: the plan stays
    assert json.load(open(ck))["k"] == k4
    ck2 = str(tmp_path / "d.json")
    json.dump(dict(key=dict(graph="G", measure="lazy", n=g.num_nodes, m=g.num_edges, threads=1), k=4,
                   bounds=[0, g.num_edges], chunks={}), open(ck2, "w"))
    E3.solve_chunked(g, "lazy", 1, ck2, g.num_edges // 40, "G")  # nothing done: the new size
    assert json.load(open(ck2))["k"] == k40


def test_grc_atd_gets_all_pairs_when_they_fit(tmp_path, monkeypatch):
    import baselines as B
    import datasets as D
    g = D.to_graph(*D.synthetic("ER", 300, 6, 0))
    py = tmp_path / "python"  # a GRC interpreter that records how it was called
    py.write_text("#!/bin/sh\necho \"$@\" > %s\nexit 1\n" % (tmp_path / "argv"))
    py.chmod(0o755)
    monkeypatch.setattr(PT, "memory_available", lambda: 1e15)
    _, _, st = B.grc(g, "ATD", 0.5, 3000, 1, str(py), timeout=60)
    assert "--shortest-path all_pairs" in (tmp_path / "argv").read_text() and st.startswith("error")
    B.grc(g, "OTD", 0.5, 3000, 1, str(py), timeout=60)
    assert "--shortest-path auto" in (tmp_path / "argv").read_text()  # the other methods are unchanged
    monkeypatch.setattr(PT, "memory_available", lambda: 1e3)
    _, _, st = B.grc(g, "ATD", 0.5, 3000, 1, str(py), timeout=60)
    assert st.startswith("skipped: needs") and "ATD" in st


def test_highs_skipped_for_its_lp_size_runs_when_the_limit_is_raised(tmp_path, monkeypatch):
    import e2_per_edge as E2
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "c.csv"), ["sweep", "regime", "s", "seed"])
    t.add(dict(sweep="sym", regime="tree", s=8192, seed=0, ours_s=0.1, highs_s=float("nan"), highs_lp_vars=8194 ** 2,
               highs_status=f"skipped: LP of {8194 ** 2:,} variables, above --highs-max-vars (16,793,604)"))
    t = common.Rows(str(tmp_path / "c.csv"), ["sweep", "regime", "s", "seed"])
    k = dict(sweep="sym", regime="tree", s=8192, seed=0)
    assert E2._row_left(t, ["ours", "highs"], {"highs": 4098.0 ** 2}, **k)[0] is None    # the same limit: done
    assert E2._row_left(t, ["ours", "highs"], {"highs": 8194.0 ** 2}, **k)[0] == ["highs"]  # raised: HiGHS only


def test_grc_in_saved_chunks_resumes_after_a_kill(tmp_path, monkeypatch):
    import baselines as B
    import datasets as D
    g = D.to_graph(*D.synthetic("ER", 2000, 8, 0))
    calls = []

    def fake(g_, method, alpha, topk, proc, python, ids=None, **kw):  # GRC: value = edge id, 2 s per call
        calls.append((int(ids[0]), int(ids[-1]) + 1))
        if len(calls) == 3:
            raise KeyboardInterrupt  # the job is killed during the third chunk
        return ids.astype(float), 2.0, "ok"

    monkeypatch.setattr(B, "grc", fake)
    d = str(tmp_path / "state")
    with pytest.raises(KeyboardInterrupt):
        B.grc_chunked(g, "OTD", 0.5, 3000, 1, "py", d, chunk_s=1e-6 * g.num_edges)
    n = len(calls)
    vals, t, st, nch, ndone = B.grc_chunked(g, "OTD", 0.5, 3000, 1, "py", d, chunk_s=1e-6 * g.num_edges)
    assert st == "ok" and ndone == g.num_edges and np.array_equal(vals, np.arange(g.num_edges, dtype=float))
    assert calls[n][0] == calls[n - 1][0]  # it restarts the killed chunk, not the two saved before it
    assert t == 2.0 * nch and nch == len(calls) - 1


def test_borf_resumes_after_its_last_saved_iteration(tmp_path, monkeypatch):
    import datasets as D
    import e6_downstream as E6
    import orcut
    g = D.to_graph(*D.synthetic("BA", 300, 4, 0))
    ref, _ = E6.borf(g, 4, 5, 3, threads=1)
    ck = str(tmp_path / "b.npz")
    real = orcut.certify
    n = {"calls": 0}

    def dies(*args, **kw):
        n["calls"] += 1
        if n["calls"] > 12:  # killed during the third iteration (5 plans per iteration)
            raise KeyboardInterrupt
        return real(*args, **kw)

    monkeypatch.setattr(orcut, "certify", dies)
    with pytest.raises(KeyboardInterrupt):
        E6.borf(g, 4, 5, 3, threads=1, ckpt=ck)
    monkeypatch.setattr(orcut, "certify", real)
    got, recs = E6.borf(g, 4, 5, 3, threads=1, ckpt=ck)
    assert np.array_equal(got.edges, ref.edges) and [r["iteration"] for r in recs] == [-1, 0, 1, 2, 3]


def test_grc_reference_stops_at_its_budget_and_continues_with_a_larger_one(tmp_path, monkeypatch):
    import baselines as B
    import datasets as D
    g = D.to_graph(*D.synthetic("ER", 3000, 10, 0))
    monkeypatch.setattr(B, "grc", lambda g_, *a, ids=None, **kw: (ids.astype(float), 1.0, "ok"))  # 1 s per call
    d = str(tmp_path / "state")
    vals, t, st, nch, ndone = B.grc_chunked(g, "OTDSinkhornMix", 0.0, 3000, 1, "py", d, chunk_s=1.0, budget_s=3.0)
    assert st == "partial" and t == 3.0 and nch == 3 and 0 < ndone < g.num_edges
    assert np.isnan(vals[ndone:]).all() and np.array_equal(vals[:ndone], np.arange(ndone, dtype=float))
    vals, t, st, nch, ndone = B.grc_chunked(g, "OTDSinkhornMix", 0.0, 3000, 1, "py", d, chunk_s=1e6, budget_s=1e6)
    assert st == "ok" and ndone == g.num_edges and nch == 4  # the three saved chunks, and one for the rest


def test_workers_share_the_chunks_of_one_row(tmp_path):
    import checkpoint as CK
    import datasets as D
    import e3_scale as E3
    g = D.to_graph(*D.synthetic("BA", 400, 6, 0))
    ck, claims = str(tmp_path / "row.json"), str(tmp_path / "claims")
    other = CK.StepLock(claims, "row.c2").acquire()  # another worker is computing chunk 2
    mine = []
    assert E3.solve_chunked(g, "lazy", 1, ck, g.num_edges // 5, "G", after_chunk=mine.append, claims=claims) is None
    assert 2 not in mine and len(mine) >= 4  # all the others: the row is not written by this worker
    other.release()  # ... which finished chunk 2 and then finds the row complete
    theirs = []
    state, tot, finish = E3.solve_chunked(g, "lazy", 1, ck, g.num_edges // 5, "G", after_chunk=theirs.append,
                                          claims=claims)
    assert theirs == [2] and finish is not None and tot["edges"] == g.num_edges
    with pytest.raises(CK.Busy):  # nobody else writes the row meanwhile
        CK.StepLock(claims, "row.row").acquire()
    finish.release()


# ------------------------------------------- GRC killed at the memory limit (only its process)

def _fake_open(files):
    """open() that serves ``files`` (path -> text) and raises OSError for any other path."""
    import io

    def fake(path, *a, **kw):
        if path in files:
            return io.StringIO(files[path])
        raise OSError(path)
    return fake


def test_oom_kill_count_reads_the_cgroup_of_the_job(monkeypatch):
    v2 = "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_7/step_batch/user/task_0"
    monkeypatch.setattr(PT, "open", _fake_open({
        "/proc/self/cgroup": "0::/system.slice/slurmstepd.scope/job_7/step_batch/user/task_0\n",
        f"{v2}/memory.events": "low 0\nhigh 0\nmax 12\noom 2\noom_kill 2\noom_group_kill 0\n"}), raising=False)
    assert PT.oom_kill_count() == 2
    # cgroup v1: only the job's level has the file, the step's does not; the unified line has nothing
    monkeypatch.setattr(PT, "open", _fake_open({
        "/proc/self/cgroup": "12:memory:/slurm/uid_1/job_7/step_batch\n4:cpu,cpuacct:/slurm/uid_1/job_7\n0::/\n",
        "/sys/fs/cgroup/memory/slurm/uid_1/job_7/memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 3\n"}),
        raising=False)
    assert PT.oom_kill_count() == 3
    monkeypatch.setattr(PT, "open", _fake_open({"/proc/self/cgroup": "0::/\n"}), raising=False)
    assert PT.oom_kill_count() is None


def _script(path, body):
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return str(path)


def test_grc_killed_by_a_signal_says_so_instead_of_an_empty_error(tmp_path, monkeypatch):
    """GRC's own process killed at the job's memory limit leaves no stderr and its parent alive; the row
    says "out of memory with <job's> GB", which only a job with more memory runs again."""
    import baselines as B
    import datasets as D
    g = D.to_graph(*D.synthetic("ER", 300, 6, 0))
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(300 * 1024))
    kill9 = _script(tmp_path / "kill9", "kill -9 $$")
    counts = iter([4, 5])  # the job's cgroup counted one OOM kill while GRC ran
    monkeypatch.setattr(PT, "oom_kill_count", lambda: next(counts))
    _, _, st = B.grc(g, "OTD", 0.5, 3000, 1, kill9, timeout=60)
    assert st == "out of memory with 300 GB (more --mem runs it again)"
    row = dict(status=st, **B.grc_row_fields(st, g.num_nodes))
    assert row["mem_bytes"] == 300 * 2**30 and not common.row_failed(row) and not PT.retry_now(row)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(320 * 1024))
    assert PT.retry_now(row)  # resubmitted with --mem=320G
    counts = iter([4, 4])  # SIGKILL from something else: an error, computed again
    _, _, st = B.grc(g, "OTD", 0.5, 3000, 1, kill9, timeout=60)
    assert st == "error: killed by SIGKILL"
    monkeypatch.setattr(PT, "oom_kill_count", lambda: None)  # the kernel does not say: SIGKILL is the memory limit
    assert B.grc(g, "OTD", 0.5, 3000, 1, kill9, timeout=60)[2].startswith("out of memory with 320 GB")
    assert B.grc(g, "OTD", 0.5, 3000, 1, _script(tmp_path / "segv", "kill -11 $$"), timeout=60)[2] == \
        "error: killed by SIGSEGV"
    assert B.grc(g, "OTD", 0.5, 3000, 1, _script(tmp_path / "fails", "echo boom >&2; exit 1"), timeout=60)[2] == \
        "error: exit 1: boom"


def test_grc_atd_on_epinions_needs_more_than_a_300g_job(tmp_path, monkeypatch):
    """GRC's ATD on a graph of Epinions' size (75,877 vertices) needs ~258 GB as GRC builds its
    all-pairs table (48 n^2 bytes), which a 300 GB job does not plan for; 16 n^2 with --apsp-array."""
    import baselines as B
    import datasets as D
    n = 75877
    g = D.to_graph(n, np.stack([np.arange(n - 1), np.arange(1, n)], 1).astype(np.int64))  # a path
    assert round(B.grc_atd_need(n) / 2**30) == 258 and round(B.grc_atd_need(n, "array") / 2**30) == 87
    py = _script(tmp_path / "python", f'echo "$@" > {tmp_path / "argv"}; exit 1')
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(300 * 1024))
    _, _, st = B.grc(g, "ATD", 0.5, 3000, 1, py, timeout=60)
    assert st == "skipped: needs ~258 GB for the all-pairs distances of GRC's ATD, the job has 300 GB"
    assert B.grc_row_fields(st, n)["need_bytes"] == B.grc_atd_need(n)
    assert not (tmp_path / "argv").exists()  # nothing was started
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(320 * 1024))
    assert B.grc(g, "ATD", 0.5, 3000, 1, py, timeout=60)[2] == "error: exit 1"
    assert "--shortest-path all_pairs" in (tmp_path / "argv").read_text()
    assert "--apsp-array" not in (tmp_path / "argv").read_text()
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(128 * 1024))  # the lean table fits in the 128 GB jobs
    B.grc(g, "ATD", 0.5, 3000, 1, py, timeout=60, apsp="array")
    assert "--apsp-array" in (tmp_path / "argv").read_text()
    B.grc(g, "OTD", 0.5, 3000, 1, py, timeout=60, apsp="array")
    assert "--apsp-array" not in (tmp_path / "argv").read_text()  # only ATD's table is built that way


def test_e5_rewrites_an_atd_skip_made_under_the_old_estimate(tmp_path, monkeypatch):
    """The row the 128 GB job left (needs ~214 GB: 40 n^2) is rewritten with the measured need, and no
    graph is loaded or GRC run when it still does not fit."""
    import argparse
    import e5_approx as E5
    import registry as R
    monkeypatch.setattr(common, "_TABLES", [])
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(300 * 1024))
    monkeypatch.setattr(R, "load", lambda *a, **k: pytest.fail("no graph should be loaded"))
    out = tmp_path / "out"
    t = common.Rows(str(out / "e5" / "whole.csv"), ["graph", "method"])
    base = dict(graph="Epinions", n=75877, m=405739, deg_max=3044, threads=32)
    for meth in ["orcut", "bf", "af3", "jl", "bf@lly", "af3@lly", "jl@lly"]:
        t.add(dict(base, method=meth))
    for meth in ["grc-mix", "grc-otd", "grc-otd-full", "grc-sinkhorn"]:
        t.add(dict(base, method=meth, status="ok"))
    old = "skipped: needs ~214 GB for the all-pairs distances of GRC's ATD, the job has 128 GB"
    t.add(dict(base, method="grc-atd", status=old))
    py = _script(tmp_path / "python", "exit 1")
    a = argparse.Namespace(out=str(out), graphs=["real:Epinions"], grc_python=py, grc_atd_apsp="grc", data_dir=None,
                           threads=32, hub=1000, grc_chunk=3600.0, grc_budget=None, grc_chunk_above=10**6)
    E5.part_whole(a)
    r = common.Rows(str(out / "e5" / "whole.csv"), ["graph", "method"]).get(graph="Epinions", method="grc-atd")
    assert r["status"] == "skipped: needs ~258 GB for the all-pairs distances of GRC's ATD, the job has 300 GB"
    assert float(r["need_bytes"]) == E5.B.grc_atd_need(75877) and r["apsp"] == "grc"
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(128 * 1024))  # a smaller job leaves the row as it is
    monkeypatch.setattr(common, "_TABLES", [])
    E5.part_whole(a)
    r = common.Rows(str(out / "e5" / "whole.csv"), ["graph", "method"]).get(graph="Epinions", method="grc-atd")
    assert r["status"].endswith("the job has 300 GB")
