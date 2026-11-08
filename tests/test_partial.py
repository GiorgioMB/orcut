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
    vals, t, st, nch = B.grc_chunked(g, "OTD", 0.5, 3000, 1, "py", d, chunk_s=1e-6 * g.num_edges)
    assert st == "ok" and np.array_equal(vals, np.arange(g.num_edges, dtype=float))
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
