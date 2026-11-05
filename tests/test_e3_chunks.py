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
    state, tot = E.solve_chunked(g, "lazy", 1, path, g.num_edges // 6, "G", after_chunk=second.append)
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
