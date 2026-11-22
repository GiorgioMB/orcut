"""experiments/hardware.py and tools/provenance.py: the calibration across node types, and the
processors behind each row; without their inputs, artifacts.py produces nothing from them."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import artifacts as A  # noqa: E402
import common  # noqa: E402
import hardware  # noqa: E402
import provenance as P  # noqa: E402


def _calib(root, node, rows, cpu=None):
    d = root / "calib" / node / "e3"
    d.mkdir(parents=True)
    common.write_csv(rows, str(d / "real.csv"))
    if cpu:
        (root / "calib" / node / "cpu.txt").write_text(cpu + "\n")


def test_calibration(tmp_path):
    row = lambda g, m, t, pot, solve: dict(graph=g, measure="lazy", m=m, threads=32, time_s=t, pot_est_s=pot,
                                           pot_solve_est_1thread_s=solve * 32)
    _calib(tmp_path, "a", [row("G1", 10, 1.0, 4.0, 2.0), row("G2", 20, 10.0, 30.0, 12.0)], cpu="Processor A")
    _calib(tmp_path, "b", [row("G1", 10, 2.0, 3.0, 1.0), row("G2", 20, 9.0, 33.0, 11.0)])
    O = A.Out(False)
    hardware.block_calib(str(tmp_path), O, A)
    assert O.keys["ECalibNodeTypes"] == "2" and O.keys["ECalibGraphs"] == "G1 and G2"
    assert "2.0" in O.keys["ECalibTimeSpreadMax"] and "2.7" in O.keys["ECalibRatioSpreadMax"]
    assert "0.50" in O.keys["ECalibSolveRatioMin"] and "2.0" in O.keys["ECalibSolveRatioMax"]
    assert "Processor A & G1" in O.rows["calib"] and "b & G2" in O.rows["calib"]


def test_nothing_without_inputs(tmp_path):
    O = A.Out(False)
    hardware.block_calib(str(tmp_path), O, A)
    hardware.block_provenance(str(tmp_path), O, A)
    assert not O.keys and not O.rows and not O.missing


def test_provenance_from_logs_and_sacct(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "e4.log").write_text(
        "===== 2026-11-01 10:00:00 python e4_ablation.py\n"
        "[10:05:00] E4 ER(20k, 20)      full      est.   1.0s single-threaded\n"
        "[23:59:00] E4 ER(20k, 20)      cut       est.   2.0s single-threaded\n"
        "[00:10:00] E4 GitHub           full      est.   3.0s single-threaded\n"   # next day, two workers
        "===== 2026-11-03 09:00:00 python e4_ablation.py\n"
        "[09:30:00] E4 ER(20k, 20)      full      est.   1.1s single-threaded\n")  # overwrites the first
    sacct = tmp_path / "sacct.txt"
    sacct.write_text(
        "1_7 orcut-step abc1 2026-11-01T09:59:00 2026-11-02T12:00:00\n"
        "2_7 orcut-step xyz2 2026-11-02T00:00:00 2026-11-02T01:00:00\n"
        "3_7 orcut-step abc3 2026-11-03T08:59:00 2026-11-03T10:00:00\n"
        "4_6 orcut-step xyz4 2026-11-03T08:00:00 2026-11-03T10:00:00\n"   # another step: not a candidate
        "5 bochner-polytope xyz5 2026-11-01T00:00:00 2026-11-04T00:00:00\n")  # another project
    rows = {r["key"]: r for r in P.trace("e4", str(logs / "e4.log"), P.read_sacct(str(sacct)))}
    assert rows["ER(20k, 20) | full"]["nodes"] == "abc3"
    assert rows["ER(20k, 20) | cut"]["nodes"] == "abc1"
    assert rows["GitHub | full"]["nodes"] == "abc1;xyz2"
    prov = tmp_path / "results" / "provenance"
    prov.mkdir(parents=True)
    common.write_csv(list(rows.values()), str(prov / "rows.csv"))
    (prov / "types.txt").write_text("abc Processor A\nxyz Processor B\n")
    O = A.Out(False)
    hardware.block_provenance(str(tmp_path / "results"), O, A)
    assert "E4 & Processor A (2), not determined (1)" in O.rows["provenance"]
    assert O.keys["EProvRows"] == "3" and O.keys["EProvOpen"] == "1"
    assert O.keys["EProvFourGraphs"] == "2" and O.keys["EProvFourMixed"] == "0"  # abc1 and abc3: one processor
    rows["ER(20k, 20) | full"]["nodes"] = "xyz9"  # the same graph's rows on two processors
    common.write_csv(list(rows.values()), str(prov / "rows.csv"))
    O = A.Out(False)
    hardware.block_provenance(str(tmp_path / "results"), O, A)
    assert O.keys["EProvFourMixed"] == "1"


# ------------------------------------------------------------------ main.py: the calibration and provenance hooks
import subprocess  # noqa: E402
import types as _types  # noqa: E402

import checkpoint as CK  # noqa: E402
import main as M  # noqa: E402


def _args(tmp_path, quick=False):
    return _types.SimpleNamespace(out=str(tmp_path), data_dir=None, threads=4, grc_python=None, quick=quick, step_args=[])


def test_calib_step_runs_on_this_node_type(tmp_path):
    a = _args(tmp_path)
    cmd = M._command("calib", a, M.STEPS)
    out = cmd[cmd.index("--out") + 1]
    name = M.cpu_name(M.cpu_model())
    assert out == os.path.join(str(tmp_path), "calib", M.cpu_slug(name))
    assert open(os.path.join(out, "cpu.txt")).read().strip() == name
    assert cmd[cmd.index("--graphs") + 1:cmd.index("--graphs") + 3] == M.CALIB_GRAPHS and "--threads" in cmd
    q = M._command("calib", _args(tmp_path, quick=True), M.STEPS)
    assert "--quick" not in q and q[q.index("--graphs") + 1:q.index("--graphs") + 3] == M.CALIB_QUICK_GRAPHS
    assert M._config("calib", a)["cpu"] == M.cpu_slug(name)  # another type of node: another calibration
    assert M.cpu_name("Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz") == "Xeon Gold 6248"
    assert M.cpu_name("AMD EPYC 7713 64-Core Processor") == "EPYC 7713"


def test_record_node_once(tmp_path):
    M.record_node(str(tmp_path))
    M.record_node(str(tmp_path))
    lines = (tmp_path / "provenance" / "nodes.txt").read_text().splitlines()
    assert len(lines) == 1 and lines[0].endswith(M.cpu_name(M.cpu_model()))


def test_provenance_hook(tmp_path, monkeypatch):
    a = _args(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "e4.log").write_text("===== 2026-11-01 10:00:00 python e4_ablation.py  [host abc1, X]\n"
                                              "[10:05:00] E4 ER(20k, 20) full      est.   1.0s single-threaded\n")
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    M._provenance(a)  # without Slurm: nothing
    assert not (tmp_path / "provenance").exists()
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(M.subprocess, "run", lambda *x, **k: subprocess.CompletedProcess(
        x, 0, stdout="1_7 orcut-step abc1 2026-11-01T09:00:00 2026-11-01T11:00:00\n", stderr=""))
    M._provenance(a)
    rows = tmp_path / "provenance" / "rows.csv"
    assert "ER(20k, 20) | full" in rows.read_text() and "abc1" in rows.read_text()
    fp, mtime = CK.inputs_fingerprint(str(tmp_path)), rows.stat().st_mtime_ns
    M._provenance(a)  # unchanged: not written again, and the artifacts need not run again
    assert rows.stat().st_mtime_ns == mtime and CK.inputs_fingerprint(str(tmp_path)) == fp
    (tmp_path / "provenance" / "nodes.txt").write_text("abc1 Processor A\n")
    assert CK.inputs_fingerprint(str(tmp_path)) != fp
    assert hardware.processor("abc1", {}, hardware.load_types(str(tmp_path / "provenance" / "nodes.txt"))) == "Processor A"


def test_fingerprint_sees_the_calibration(tmp_path):
    fp = CK.inputs_fingerprint(str(tmp_path))
    _calib(tmp_path, "a", [dict(graph="G1", measure="lazy", m=10, threads=32, time_s=1.0)])
    assert CK.inputs_fingerprint(str(tmp_path)) != fp
