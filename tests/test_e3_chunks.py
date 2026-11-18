"""E3's saved chunks (experiments/e3_scale.py): one pass in chunks gives the row of the one-call path,
and a run stopped between chunks resumes without computing the saved ones again."""
import json
import os
import sys

import numpy as np
import pytest

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, EXP)

import common  # noqa: E402
import datasets as D  # noqa: E402
import e3_scale as E  # noqa: E402
from orcut import _core as C  # noqa: E402


@pytest.fixture(scope="module")
def g():
    nn, e = D.synthetic("CL2.1", 1500, 10, 0)
    return D.to_graph(nn, e)


def test_chunk_bounds_cover_all_edges(g):
    b = E.chunk_bounds(g, 10)
    assert b[0] == 0 and b[-1] == g.num_edges and all(x < y for x, y in zip(b, b[1:])) and len(b) <= 11
    assert E.chunk_bounds(g, 1) == [0, g.num_edges]


def _one_call(g):
    """The diagnostic columns as part_real computes them on all edges at once."""
    d = E.solve_all(g, "lazy", 1, diagnostics=True)
    val, dg = d["curvature"], d["diagnostics"]
    path = dg[:, C.D_PATH]
    hub = np.maximum(g.deg[g.edges[:, 0]], g.deg[g.edges[:, 1]]) >= 100
    ok = hub & (dg[:, C.D_GROUPS] > 0)
    ref = dict(share_zero=np.mean(path == C.PATH_ZERO),
               share_fast=np.mean((path == C.PATH_FAST) | (path == C.PATH_NOADJ)),
               share_cut=np.mean((path == C.PATH_CUT) | (path == C.PATH_CUTW) | (path == C.PATH_WIT)),
               share_witness=np.mean((path == C.PATH_CUTW) | (path == C.PATH_WIT)),
               mean_nodes=dg[:, C.D_NODES].mean(), mean_arcs=dg[:, C.D_ARCS].mean(), max_arcs=dg[:, C.D_ARCS].max(),
               hub_edges=hub.sum(),
               twin_ratio_hub=np.mean(dg[ok, C.D_ROWS] / dg[ok, C.D_GROUPS]) if ok.any() else np.nan,
               twin_ratio_hub_max=np.max(dg[ok, C.D_ROWS] / dg[ok, C.D_GROUPS]) if ok.any() else np.nan)
    return ref, int(np.isnan(val).sum())


def test_chunks_resume_and_match_one_call(g, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_TABLES", [])
    path = str(tmp_path / "x.json")
    first = []

    class Stop(Exception):
        pass

    def stop_after_2(c):
        first.append(c)
        if len(first) == 2:
            raise Stop

    with pytest.raises(Stop):
        E.solve_chunked(g, "lazy", 1, path, g.num_edges // 6, "G", after_chunk=stop_after_2)
    assert sorted(json.load(open(path))["chunks"]) == ["0", "1"]
    second = []
    state, tot, _ = E.solve_chunked(g, "lazy", 1, path, g.num_edges // 6, "G", after_chunk=second.append)
    nch = len(state["bounds"]) - 1
    assert first + second == list(range(nch)) and nch >= 6  # the saved chunks are not computed again
    ref, nan = _one_call(g)
    got = E.diag_fields(tot, g.num_edges)
    for k, v in ref.items():
        assert got[k] == pytest.approx(float(v), rel=1e-12, nan_ok=True), k
    assert tot["edges"] == g.num_edges and tot["nan"] == nan
    # a later run with all chunks saved computes nothing
    third = []
    E.solve_chunked(g, "lazy", 1, path, g.num_edges // 6, "G", after_chunk=third.append)
    assert third == []
    # another graph (or measure, or thread count) does not reuse them
    fourth = []
    E.solve_chunked(g, "lazy", 1, path, g.num_edges // 6, "another graph", after_chunk=fourth.append)
    assert len(fourth) >= 6


def test_predicted_time(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_TABLES", [])
    rows = common.Rows(str(tmp_path / "s.csv"), ["family", "regime", "n", "seed"])
    assert E.predicted_time(rows, "CL2.1", "fixed16", 0, 10**6) == 0.0
    rows.add(dict(family="CL2.1", regime="fixed16", n=10**5, seed=0, m=646069, time_s=222.15))
    assert E.predicted_time(rows, "CL2.1", "fixed16", 0, 10**6) == 0.0  # one size is not enough
    rows.add(dict(family="CL2.1", regime="fixed16", n=3 * 10**5, seed=0, m=1966735, time_s=2496.87))
    rows.add(dict(family="ER", regime="fixed16", n=3 * 10**5, seed=0, m=2400000, time_s=1.0))
    rows = common.Rows(str(tmp_path / "s.csv"), ["family", "regime", "n", "seed"])  # as read back from the CSV
    k = np.log(2496.87 / 222.15) / np.log(1966735 / 646069)
    assert E.predicted_time(rows, "CL2.1", "fixed16", 0, 6.5e6) == pytest.approx(2496.87 * (6.5e6 / 1966735) ** k)
    assert E.predicted_time(rows, "CL2.1", "fixed16", 0, 6.5e6) > 3600  # CL2.1 at n = 10^6: chunked
    assert E.predicted_time(rows, "CL2.1", "grow0.4", 0, 6.5e6) == 0.0


def test_synth_plan_stops_at_the_sizes_that_were_run():
    p = E.synth_plan()
    assert len(p) == 89 and len(E.synth_plan(all_sizes=True)) == 96
    have = {(f, r, n) for f, r, n, _ in p}
    assert ("CL2.1", "fixed16", 3 * 10**6) in have and ("CL2.1", "fixed16", 10**7) not in have
    assert ("RGG", "grow0.4", 10**6) in have and ("ER", "grow0.4", 10**6) not in have
    assert ("CL2.5", "grow0.4", 10**5) in have and ("CL2.5", "grow0.4", 3 * 10**5) not in have


def _solve(g, path, claims, **kw):
    return E.solve_chunked(g, "lazy", 1, str(path), max(1, g.num_edges // 4), "CL2.1 n=1500", claims=str(claims), **kw)


def test_a_completed_log_gives_its_row_again_without_solving(g, tmp_path, monkeypatch):
    """A row whose log is complete but which is missing from the table is written from the log."""
    monkeypatch.setattr(common, "_TABLES", [])
    path, claims = tmp_path / "x.json", tmp_path / "claims"
    state, tot, finish = _solve(g, path, claims)
    finish.release()
    E._finish_chunked(str(path))
    import orcut
    monkeypatch.setattr(orcut, "curvature", lambda *a, **k: pytest.fail("solved again"))
    state2, tot2, finish2 = _solve(g, path, claims)
    finish2.release()
    assert tot2["seconds"] == tot["seconds"] and len(state2["chunks"]) == len(state["bounds"]) - 1


def test_a_chunk_finished_after_the_row_was_written_is_dropped(g, tmp_path, monkeypatch):
    """Up to 1.4.0, such a chunk started a second log from an old copy of the first."""
    monkeypatch.setattr(common, "_TABLES", [])
    full, claims, path = tmp_path / "full.json", tmp_path / "claims", tmp_path / "x.json"
    state, tot, finish = _solve(g, full, claims)  # the complete log of another worker
    finish.release()

    def written_meanwhile(c):
        if c == 0:  # the other worker writes the row: its log becomes x.done.json
            os.replace(str(full), str(tmp_path / "x.done.json"))
            os.remove(str(path))

    assert _solve(g, path, claims, after_chunk=written_meanwhile) is None
    assert not path.exists()


def test_an_incomplete_completed_log_is_completed(g, tmp_path, monkeypatch):
    """The case of WS at 2.1 n^0.4, n = 10^6: a completed log with some of its chunks."""
    monkeypatch.setattr(common, "_TABLES", [])
    full, claims = tmp_path / "full.json", tmp_path / "claims"
    state, tot, finish = _solve(g, full, claims)
    finish.release()
    part = dict(state, chunks=dict(list(state["chunks"].items())[:2]))
    (tmp_path / "x.done.json").write_text(json.dumps(part))
    state2, tot2, finish2 = _solve(g, tmp_path / "x.json", claims)
    finish2.release()
    assert len(state2["chunks"]) == len(state["bounds"]) - 1 and set(part["chunks"]) <= set(state2["chunks"])
