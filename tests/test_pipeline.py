"""The checkpoints of the experiment pipeline (experiments/main.py, checkpoint.py, common.Rows), with
fake steps: a step that ended without errors is not run again, anything else is."""
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

import numpy as np

import pytest

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, EXP)

import checkpoint as CK  # noqa: E402
import common  # noqa: E402
import main as M  # noqa: E402

FAKE = textwrap.dedent("""\
    import json, os, sys, time
    name, out = sys.argv[1], sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calls.jsonl"), "a") as fh:
        fh.write(json.dumps({"step": name, "argv": sys.argv[1:], "cwd": os.getcwd(),
                             "grc_env": os.environ.get("ORCUT_GRC_PYTHON")}) + "\\n")
    rc = os.path.join(out, "rc-" + name)
    code = int(open(rc).read()) if os.path.exists(rc) else 0
    if code == 99:  # run until killed
        time.sleep(600)
    sys.exit(code)
""")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A table of fake steps s1, s2 and e5-whole (a GRC step); returns (run, calls, out)."""
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    steps = {s: ([str(script), s], "fake") for s in ("s1", "s2", "e5-whole", "e7-radius")}
    out = tmp_path / "out"
    monkeypatch.setattr(common, "environment", lambda: {"git": "abc"})
    monkeypatch.delenv("ORCUT_GRC_PYTHON", raising=False)
    monkeypatch.setattr(M, "GRC_ENV", "no-such-environment")  # do not find a real orcut-grc next to this python
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "no-conda"))  # nor through conda

    def run(*args):
        return M.main(["--out", str(out), *args], steps=steps)

    def calls():
        p = out / "calls.jsonl"
        return [json.loads(x)["step"] for x in p.read_text().splitlines()] if p.exists() else []

    return run, calls, out


def _rc(out, step, code):
    out.mkdir(exist_ok=True)
    (out / f"rc-{step}").write_text(str(code))


# ------------------------------------------------------------------ steps

def test_completed_steps_are_not_run_again(fake):
    run, calls, out = fake
    assert run("--steps", "s1", "s2") == 0
    assert calls() == ["s1", "s2"]
    assert run("--steps", "s1", "s2") == 0
    assert calls() == ["s1", "s2"]
    rec = json.loads((out / "status" / "s1.json").read_text())
    assert rec["returncode"] == 0 and rec["state"] == "ok" and rec["git"] == "abc" and rec["config"] == {"quick": False}
    assert set(json.loads((out / "status.json").read_text())) == {"s1", "s2"}


def test_failed_and_incomplete_steps_run_again(fake):
    run, calls, out = fake
    for code, state in ((1, "failed"), (CK.EXIT_INCOMPLETE, "incomplete")):
        _rc(out, "s1", code)
        n = len(calls())
        assert run("--steps", "s1") == 1
        assert calls()[n:] == ["s1"]
        assert CK.load(str(out), "s1")["state"] == state
    _rc(out, "s1", 0)
    n = len(calls())
    assert run("--steps", "s1") == 0
    assert run("--steps", "s1") == 0
    assert calls()[n:] == ["s1"]


def test_killed_step_is_recorded_and_runs_again(fake, monkeypatch):
    run, calls, out = fake
    _rc(out, "s1", 0)
    real_run = M._run

    def killed(cmd, log_path, lock, env=None):
        real_run(cmd, log_path, lock, env)
        return -9  # what subprocess reports for a step killed by SIGKILL (the out-of-memory killer)

    monkeypatch.setattr(M, "_run", killed)
    assert run("--steps", "s1") == 1
    rec = CK.load(str(out), "s1")
    assert rec["state"] == "killed" and rec["returncode"] == -9
    monkeypatch.setattr(M, "_run", real_run)
    assert run("--steps", "s1") == 0
    assert calls() == ["s1", "s1"]


def test_force_recomputes_everything_and_keeps_the_old_results(fake, tmp_path):
    run, calls, out = fake
    check = tmp_path / "check.py"  # a step that reports which of its tables exist when it starts
    check.write_text(textwrap.dedent("""\
        import json, os, sys
        out = sys.argv[sys.argv.index("--out") + 1]
        seen = {p: os.path.exists(os.path.join(out, p)) for p in
                ("e5/sinkhorn.csv", "e5/whole.csv", "e3/real.csv", "e3/partial/orkut.lazy.T32.json",
                 "e3/partial/synth.cl2.1.fixed16.n1000000.s0.T32.json")}
        open(os.path.join(out, "seen-" + sys.argv[1] + ".json"), "w").write(json.dumps(seen))
    """))
    steps = {s_: ([str(check), s_], "x") for s_ in ("e5-sinkhorn", "e3-real")}
    for rel, text in (("e5/sinkhorn.csv", "graph,method,reg\nG,orcut,0.0\nG,pot,0.0\n"), ("e5/whole.csv", "graph\nG\n"),
                      ("e3/real.csv", "graph,measure\nG,lazy\n"), ("e3/partial/orkut.lazy.T32.json", "{}"),
                      ("e3/partial/synth.cl2.1.fixed16.n1000000.s0.T32.json", "{}")):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
    assert M.main(["--out", str(out), "--steps", "e5-sinkhorn", "e3-real", "--force"], steps=steps) == 0
    s5 = json.loads((out / "seen-e5-sinkhorn.json").read_text())
    s3 = json.loads((out / "seen-e3-real.json").read_text())
    assert not s5["e5/sinkhorn.csv"] and s5["e5/whole.csv"]               # its own table only
    assert not s3["e3/real.csv"] and not s3["e3/partial/orkut.lazy.T32.json"]
    assert s3["e3/partial/synth.cl2.1.fixed16.n1000000.s0.T32.json"]      # e3-synth's chunks stay
    kept = sorted(os.path.relpath(f, out) for f in glob.glob(str(out / "replaced" / "**" / "*"), recursive=True)
                  if os.path.isfile(f))
    assert [k.split(os.sep, 2)[2] for k in kept] == ["e3/partial/orkut.lazy.T32.json", "e3/real.csv", "e5/sinkhorn.csv"]
    assert "G,pot" in open(glob.glob(str(out / "replaced" / "e5-sinkhorn-*" / "e5" / "sinkhorn.csv"))[0]).read()


def test_every_table_belongs_to_one_step():
    owners = {}
    for step, pats in M.STEP_OUTPUTS.items():
        assert step in M.STEPS
        for p_ in pats:
            if p_.endswith(".csv"):
                owners.setdefault(p_, []).append(step)
    assert sorted(owners) == sorted(M.TABLE_KEYS) and all(len(v) == 1 for v in owners.values())
    assert set(M.STEP_OUTPUTS) == set(M.STEPS)


def test_force_on_a_completed_step_runs_it(fake):
    run, calls, out = fake
    assert run("--steps", "s1") == 0
    assert run("--steps", "s1", "--force") == 0
    assert calls() == ["s1", "s1"]

def test_records_of_old_main_py(fake):
    """status.json written by main.py up to v1.2.0: exit 0 counts as done, -9 does not."""
    run, calls, out = fake
    out.mkdir()
    old = {"s1": {"returncode": 0, "seconds": 1.0, "finished": "x", "quick": False},
           "s2": {"returncode": -9, "seconds": 2.0, "finished": "y", "quick": False}}
    (out / "status.json").write_text(json.dumps(old))
    assert run("--steps", "s1", "s2") == 0
    assert calls() == ["s2"]
    merged = json.loads((out / "status.json").read_text())
    assert merged["s1"] == old["s1"] and merged["s2"]["returncode"] == 0 and "record" in merged["s2"]
    # a deleted record is not replaced by its copy in status.json
    os.remove(out / "status" / "s2.json")
    assert CK.load(str(out), "s2") is None


def _fake_grc(path, imports=True):
    """A stand-in for the python of the GRC environment, answering main.py's import check."""
    path.write_text("#!/bin/sh\n" + ("echo 0.6.1\n" if imports else
                                       "echo \"ModuleNotFoundError: No module named 'GraphRicciCurvature'\" >&2; exit 1\n"))
    path.chmod(0o755)
    return path


def test_grc_step_runs_again_once_grc_is_available(fake, tmp_path):
    run, calls, out = fake
    assert run("--steps", "e5-whole") == 0
    py = _fake_grc(tmp_path / "grcpython")
    assert run("--steps", "e5-whole", "--grc-python", str(py)) == 0
    assert run("--steps", "e5-whole", "--grc-python", str(py)) == 0
    assert calls() == ["e5-whole", "e5-whole"]
    assert json.loads((out / "calls.jsonl").read_text().splitlines()[1])["argv"][-2:] == ["--grc-python", str(py)]


def test_quick_and_full_runs_do_not_share_a_directory(fake):
    run, calls, out = fake
    assert run("--steps", "s1", "--quick") == 0
    assert run("--steps", "s2") == 2
    assert calls() == ["s1"]


def test_status_lists_every_step(fake, capsys):
    run, calls, out = fake
    _rc(out, "s2", 1)
    run("--steps", "s1", "s2")
    capsys.readouterr()
    assert run("--status") == 0
    lines = capsys.readouterr().out.splitlines()
    state = {ln.split()[0]: ln.split()[1] for ln in lines[1:]}
    assert state == {"s1": "done", "s2": "failed", "e5-whole": "not", "e7-radius": "not", "artifacts": "not"}


# ---------------------------------------------------------------- locks

def test_a_step_running_elsewhere_is_skipped(fake):
    run, calls, out = fake
    lock = out / "status" / "s1.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"token": "x", "host": "another-node", "pid": 1, "started": "now"}))
    assert run("--steps", "s1", "s2") == 0
    assert calls() == ["s2"]
    old = time.time() - 2 * CK.STALE_S  # its job was killed: the lock is stale
    os.utime(lock, (old, old))
    assert run("--steps", "s1") == 0
    assert calls() == ["s2", "s1"] and not lock.exists()


def test_lock_of_a_dead_process_is_stale(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    lk = CK.StepLock(str(tmp_path), "s")
    os.makedirs(os.path.dirname(lk.path))
    with open(lk.path, "w") as fh:
        json.dump({"token": "x", "host": socket.gethostname(), "pid": p.pid}, fh)
    with lk:
        assert json.load(open(lk.path))["token"] == lk.token
        with pytest.raises(CK.Busy):
            CK.StepLock(str(tmp_path), "s").acquire()
    assert not os.path.exists(lk.path)


# ---------------------------------------------------------------- paths

def test_relative_paths_reach_the_steps_absolute(fake, tmp_path, monkeypatch):
    run, calls, out = fake
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    steps = {"s1": ([str(tmp_path / "fake.py"), "s1"], "fake")}
    assert M.main(["--out", "res", "--data-dir", "dd", "--steps", "s1"], steps=steps) == 0
    call = json.loads((work / "res" / "calls.jsonl").read_text())
    assert call["cwd"] == str(work)
    assert call["argv"][call["argv"].index("--out") + 1] == str(work / "res")
    assert call["argv"][call["argv"].index("--data-dir") + 1] == str(work / "dd")


def test_tables_in_the_old_place_are_moved_and_merged(fake, tmp_path, monkeypatch):
    """What main.py up to v1.2.0 left in experiments/results and experiments/experiments/data/real
    is moved into place, and a table in both places is merged row by row; then the steps run."""
    run, calls, out = fake
    root = tmp_path / "repo"
    here = root / "experiments"
    monkeypatch.setattr(M, "HERE", str(here))
    old, new = here / "results", root / "results"
    for d in ("e1", "e5", "e5/values"):
        (old / d).mkdir(parents=True, exist_ok=True)
    (new / "e5").mkdir(parents=True)
    (old / "e1" / "exactness.csv").write_text("graph,measure,x\nG,lazy,1\n")       # only in the old place
    (old / "e5" / "values" / "g.npz").write_bytes(b"npz")
    (new / "e5" / "whole.csv").write_text("graph,method,status,t\nG,orcut,,1\nG,grc-mix,ok,5\nH,orcut,,2\n")
    os.utime(new / "e5" / "whole.csv", (1, 1))  # the older copy
    (old / "e5" / "whole.csv").write_text("graph,method,status,t\nG,orcut,,9\nG,grc-mix,unavailable,\n"
                                          "G,grc-otd,error: x,\n")
    (here / "experiments" / "data" / "real").mkdir(parents=True)
    (here / "experiments" / "data" / "real" / "com_x.npz").write_bytes(b"data")
    steps = {"s1": ([str(tmp_path / "fake.py"), "s1"], "fake")}
    assert M.main(["--out", str(new), "--steps", "s1"], steps=steps) == 0
    assert not glob.glob(str(old / "e*" / "**" / "*"), recursive=True)
    assert (new / "e1" / "exactness.csv").read_text() == "graph,measure,x\nG,lazy,1\n"
    assert (new / "e5" / "values" / "g.npz").read_bytes() == b"npz"
    got = {(r["graph"], r["method"]): r for r in common.read_csv(str(new / "e5" / "whole.csv"))}
    assert got[("G", "orcut")]["t"] == "9"             # the newer copy's row
    assert got[("G", "grc-mix")]["status"] == "ok"     # a row with a result beats the newer one without
    assert got[("G", "grc-otd")]["status"] == "error: x" and got[("H", "orcut")]["t"] == "2"
    assert (here / "data" / "real" / "com_x.npz").read_bytes() == b"data"
    assert calls() == []  # the fake step writes its calls into --out: here new/calls.jsonl
    assert len((new / "calls.jsonl").read_text().splitlines()) == 1
    # nothing left to move: the next run does not touch anything
    assert M._adopt_old_layout(str(new), None) == []


def test_table_keys_match_the_experiments():
    import re
    found = {}
    for f in glob.glob(os.path.join(EXP, "e[1-7]_*.py")):
        for d, name, key in re.findall(r"Rows\(f\"\{out_dir\(a\.out, '(e[1-7])'\)\}/([\w.]+)\", (\[[^\]]*\])\)",
                                       open(f).read()):
            found[f"{d}/{name}"] = json.loads(key)
    assert found == M.TABLE_KEYS

def test_artifacts_run_again_when_the_tables_change(fake, tmp_path, monkeypatch):
    run, calls, out = fake
    here = tmp_path / "exp"
    here.mkdir()
    (here / "artifacts.py").write_text(FAKE.replace('name, out = sys.argv[1], sys.argv[sys.argv.index("--out") + 1]',
                                                    'name, out = "artifacts", sys.argv[sys.argv.index("--results") + 1]'))
    monkeypatch.setattr(M, "HERE", str(here))
    (out / "e1").mkdir(parents=True)
    (out / "e1" / "t.csv").write_text("a\n1\n")
    assert run("--steps", "artifacts") == 0
    assert run("--steps", "artifacts") == 0
    assert calls() == ["artifacts"]
    (out / "e1" / "t.csv").write_text("a\n1\n2\n")
    assert run("--steps", "artifacts") == 0
    assert calls() == ["artifacts", "artifacts"]


def test_inputs_fingerprint_ignores_host_and_job(tmp_path):
    (tmp_path / "e3").mkdir()
    (tmp_path / "e3" / "real.csv").write_text("x\n")
    env = {"cpu_model": "c", "host": "a", "slurm_job": "1"}
    (tmp_path / "environment.json").write_text(json.dumps(env))
    f0 = CK.inputs_fingerprint(str(tmp_path))
    (tmp_path / "environment.json").write_text(json.dumps(dict(env, host="b", slurm_job="2")))
    assert CK.inputs_fingerprint(str(tmp_path)) == f0
    (tmp_path / "environment.json").write_text(json.dumps(dict(env, cpu_model="d")))
    assert CK.inputs_fingerprint(str(tmp_path)) != f0


# -------------------------------------------------------------- signals

def test_sigterm_records_the_step_as_interrupted(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    out = tmp_path / "out"
    out.mkdir()
    (out / "rc-s1").write_text("99")
    driver = tmp_path / "drive.py"
    driver.write_text(textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {EXP!r})
        import common, main
        common.environment = lambda: {{}}
        sys.exit(main.main(["--out", {str(out)!r}, "--steps", "s1"], steps={{"s1": ([{str(script)!r}, "s1"], "x")}}))
    """))
    p = subprocess.Popen([sys.executable, str(driver)])
    lock = out / "status" / "s1.lock"
    for _ in range(200):
        if lock.exists() and (out / "calls.jsonl").exists():
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 128 + signal.SIGTERM
    rec = CK.load(str(out), "s1")
    assert rec["state"] == "interrupted" and rec["returncode"] == -signal.SIGTERM
    assert not lock.exists()


# ------------------------------------------------------------------ rows

def test_rows_without_a_result_are_computed_again(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "t.csv"), ["graph", "method"])
    t.add(dict(graph="G", method="orcut", time_s=1.0))
    t.add(dict(graph="G", method="grc-mix", status="timeout"))
    t.add(dict(graph="G", method="grc-otd", status="unavailable"))
    assert common.finish() == 0
    t.add(dict(graph="G", method="grc-atd", status="error: MemoryError"))
    t = common.Rows(str(tmp_path / "t.csv"), ["graph", "method"])  # as read back by the next run
    assert t.done(graph="G", method="orcut") and t.done(graph="G", method="grc-mix")
    assert not t.done(graph="G", method="grc-otd") and not t.done(graph="G", method="grc-atd")
    assert t.get(graph="G", method="grc-atd")["status"].startswith("error")
    assert common.finish() == CK.EXIT_INCOMPLETE
    t.add(dict(graph="G", method="grc-atd", status="ok"))
    assert t.done(graph="G", method="grc-atd") and len(t.rows) == 4
    monkeypatch.setattr(common, "_TABLES", [t])
    assert common.finish() == 0


def test_radius_rows_skipped_for_size_run_again_when_they_fit(tmp_path, monkeypatch):
    import argparse
    import e7_generality as E7
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "radius.csv"), ["graph", "config"])
    t.add(dict(graph="BA", config="ball r=3", max_support_pairs=22810160,
               status="skipped: supports up to 22810160 pairs (> --max-pairs)"))
    t.add(dict(graph="BA", config="ball r=2", max_support_pairs=4000000, certified_share=1.0))
    t = common.Rows(str(tmp_path / "radius.csv"), ["graph", "config"])
    old, new = argparse.Namespace(max_pairs=5e6), argparse.Namespace(max_pairs=1.6e8)
    assert E7.radius_done(t, "BA", "ball r=3", old) and not E7.radius_done(t, "BA", "ball r=3", new)
    assert E7.radius_done(t, "BA", "ball r=2", new) and not E7.radius_done(t, "BA", "walk r=2", new)


def test_grc_that_does_not_import_counts_as_unavailable(fake, tmp_path, monkeypatch, capsys):
    run, calls, out = fake
    py = _fake_grc(tmp_path / "grcpython", imports=False)
    monkeypatch.setenv("ORCUT_GRC_PYTHON", str(py))
    assert run("--steps", "e5-whole") == 0
    assert "graphriccicurvature unavailable" in capsys.readouterr().out.lower()
    call = json.loads((out / "calls.jsonl").read_text().splitlines()[-1])
    assert "--grc-python" not in call["argv"] and call["grc_env"] is None  # the step is not given it either
    assert CK.load(str(out), "e5-whole")["config"]["grc"] is None
    _fake_grc(py, imports=True)  # installed: the step runs again, with it
    assert run("--steps", "e5-whole") == 0
    call = json.loads((out / "calls.jsonl").read_text().splitlines()[-1])
    assert call["argv"][-2:] == ["--grc-python", str(py)] and calls() == ["e5-whole", "e5-whole"]


def test_grc_environment_next_to_this_one_is_found(tmp_path, monkeypatch):
    envs = tmp_path / "conda_envs"
    (envs / "orcut" / "bin").mkdir(parents=True)
    py = _fake_grc((envs / "orcut-grc" / "bin").mkdir(parents=True) or envs / "orcut-grc" / "bin" / "python")
    monkeypatch.delenv("ORCUT_GRC_PYTHON", raising=False)
    monkeypatch.setattr(M.sys, "prefix", str(envs / "orcut"))
    p, src = M._grc_python(None)
    assert p == str(py) and "next to" in src
    monkeypatch.setenv("ORCUT_GRC_PYTHON", "/elsewhere/python")  # an explicit choice wins
    assert M._grc_python(None)[0] == "/elsewhere/python"


def test_steps_completed_under_older_rules_run_again(fake):
    """e7-radius recorded by main.py up to v1.2.0 (it skipped large supports) runs again, once."""
    run, calls, out = fake
    out.mkdir()
    (out / "status.json").write_text(json.dumps({"e7-radius": {"returncode": 0, "seconds": 1.0, "finished": "x",
                                                              "quick": False}}))
    assert run("--steps", "e7-radius") == 0
    assert run("--steps", "e7-radius") == 0
    assert calls() == ["e7-radius"] and CK.load(str(out), "e7-radius")["config"]["rules"] == M.RULES["e7-radius"]


def _squeue(tmp_path, monkeypatch, out="", rc=0, err=""):
    """A stand-in for Slurm's squeue on PATH, answering every job with ``out``."""
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (b / "squeue").write_text(f"#!/bin/sh\nprintf '%s' '{out}'\nprintf '%s' '{err}' >&2\nexit {rc}\n")
    (b / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")


def _lock_of(tmp_path, job, age=0.0, host="another-node"):
    lk = CK.StepLock(str(tmp_path), "s")
    os.makedirs(os.path.dirname(lk.path), exist_ok=True)
    with open(lk.path, "w") as fh:
        json.dump({"token": "x", "host": host, "pid": 1, "job": job, "started": "then"}, fh)
    os.utime(lk.path, (time.time() - age,) * 2)
    return lk


@pytest.mark.parametrize("answer,rc,err,taken", [
    ("", 0, "", True),                                   # the job is no longer in the queue
    ("", 1, "slurm_load_jobs error: Invalid job id specified", True),
    ("CANCELLED", 0, "", True), ("OUT_OF_MEMORY", 0, "", True), ("TIMEOUT", 0, "", True),
    ("RUNNING", 0, "", False),
])
def test_lock_of_a_job_that_ended_is_taken_over_at_once(tmp_path, monkeypatch, answer, rc, err, taken):
    _squeue(tmp_path, monkeypatch, answer, rc, err)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    _lock_of(tmp_path, "21348862_6", age=5)  # touched 5 s ago: a heartbeat alone would call it alive
    lk = CK.StepLock(str(tmp_path), "s")
    if taken:
        lk.acquire()
        assert lk.held and "21348862_6" in lk.took_over
        lk.release()
    else:
        with pytest.raises(CK.Busy, match="RUNNING"):
            lk.acquire()


def test_lock_of_a_completing_job_waits_then_is_taken_over(tmp_path, monkeypatch):
    _squeue(tmp_path, monkeypatch, "COMPLETING")
    monkeypatch.setattr(CK, "ENDING_WAIT_S", 0.0)
    _lock_of(tmp_path, "123", age=5)
    lk = CK.StepLock(str(tmp_path), "s").acquire()
    assert lk.held and "completing" in lk.took_over
    lk.release()


def test_lock_without_slurm_falls_back_to_the_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))  # no squeue
    _lock_of(tmp_path, "123", age=5)
    with pytest.raises(CK.Busy):
        CK.StepLock(str(tmp_path), "s").acquire()
    _lock_of(tmp_path, "123", age=CK.STALE_S + 1)
    assert CK.StepLock(str(tmp_path), "s").acquire().took_over.startswith("not touched")


def test_lock_of_this_very_job_is_taken_over(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "21348862")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "6")
    _lock_of(tmp_path, "21348862_6", age=5)
    assert "this job" in CK.StepLock(str(tmp_path), "s").acquire().took_over


def test_step_killed_with_sigkill_does_not_block_its_resubmission(tmp_path, monkeypatch):
    """The incident: a run of a step is killed hard (main.py too, so its lock stays); the run that is
    submitted next asks Slurm, finds the old job gone, takes the lock over and runs the step."""
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    out = tmp_path / "out"
    out.mkdir()
    (out / "rc-e3-synth").write_text("99")  # the step runs until killed
    driver = tmp_path / "drive.py"
    driver.write_text(textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {EXP!r})
        import common, main
        common.environment = lambda: {{}}
        sys.exit(main.main(["--out", {str(out)!r}, "--steps", "e3-synth"],
                           steps={{"e3-synth": ([{str(script)!r}, "e3-synth"], "x")}}))
    """))
    env = dict(os.environ, SLURM_ARRAY_JOB_ID="21348862", SLURM_ARRAY_TASK_ID="6", SLURM_JOB_ID="21348869")
    p = subprocess.Popen([sys.executable, str(driver)], env=env, start_new_session=True)
    lock = out / "status" / "e3-synth.lock"
    for _ in range(200):
        if lock.exists() and (out / "calls.jsonl").exists():
            break
        time.sleep(0.05)
    os.killpg(p.pid, signal.SIGKILL)  # main.py and the step, no clean-up
    p.wait()
    assert json.load(open(lock))["job"] == "21348862_6"  # left behind
    (out / "rc-e3-synth").write_text("0")
    _squeue(tmp_path, monkeypatch, "")  # Slurm: that job is gone
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "21348870")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "6")
    monkeypatch.setattr(common, "environment", lambda: {})
    assert M.main(["--out", str(out), "--steps", "e3-synth", "--force"],
                  steps={"e3-synth": ([str(script), "e3-synth"], "x")}) == 0
    assert len((out / "calls.jsonl").read_text().splitlines()) == 2 and not lock.exists()
    assert CK.load(str(out), "e3-synth")["returncode"] == 0


def test_options_after_double_dash_reach_the_step(fake):
    run, calls, out = fake
    assert run("--steps", "s1", "--", "--budget", "3600") == 0
    argv = json.loads((out / "calls.jsonl").read_text().splitlines()[-1])["argv"]
    assert argv[-2:] == ["--budget", "3600"] and CK.load(str(out), "s1")["config"]["step_args"] == ["--budget", "3600"]
    assert run("--steps", "s1", "--", "--budget", "3600") == 0      # the same options: done
    assert run("--steps", "s1", "--", "--budget", "7200") == 0      # other options: runs again
    assert run("--steps", "s1") == 0                                 # no options: the last run counts
    assert calls() == ["s1", "s1"]


def test_sinkhorn_rows_of_the_old_budget_run_again(tmp_path, monkeypatch):
    import argparse
    import e5_approx as E5
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "sinkhorn.csv"), ["graph", "method", "reg"])
    t.add(dict(graph="Twitch-DE", method="sinkhorn", reg=0.03, sample=558, timed=369))        # cut by the old budget
    t.add(dict(graph="Twitch-DE", method="sinkhorn", reg=0.1, sample=558, timed=558))         # complete
    t.add(dict(graph="Squirrel", method="sinkhorn", reg=0.03, sample=500, timed=470, skipped_memory=30,
               max_pairs=5e7, time_edge_weighted_s=1e-3, untimed_share=0.0))                   # new: done
    t = common.Rows(str(tmp_path / "sinkhorn.csv"), ["graph", "method", "reg"])
    a = argparse.Namespace()
    assert not E5.sinkhorn_done(t, a, graph="Twitch-DE", method="sinkhorn", reg=0.03)
    assert E5.sinkhorn_done(t, a, graph="Twitch-DE", method="sinkhorn", reg=0.1)
    assert E5.sinkhorn_done(t, a, graph="Squirrel", method="sinkhorn", reg=0.03)


def test_sinkhorn_rows_that_skipped_edges_for_memory_run_again_with_more(tmp_path, monkeypatch):
    import argparse
    import e5_approx as E5
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "sinkhorn.csv"), ["graph", "method", "reg"])
    t.add(dict(graph="GitHub", method="pot", reg=0.0, sample=651, timed=650, skipped_memory=1))   # old cap: 5e7
    t.add(dict(graph="Squirrel", method="pot", reg=0.0, sample=500, timed=500, skipped_memory=0))
    t.add(dict(graph="Big", method="pot", reg=0.0, sample=10, timed=9, skipped_memory=1, max_pairs=7e8,
               time_edge_weighted_s=1e-3, untimed_share=0.0))
    t = common.Rows(str(tmp_path / "sinkhorn.csv"), ["graph", "method", "reg"])
    a = argparse.Namespace()
    assert not E5.sinkhorn_done(t, a, 7e8, graph="GitHub", method="pot", reg=0.0)   # fits in this job now
    assert E5.sinkhorn_done(t, a, 7e8, graph="Squirrel", method="pot", reg=0.0)
    assert E5.sinkhorn_done(t, a, 7e8, graph="Big", method="pot", reg=0.0)          # no more memory than before
    assert not E5.sinkhorn_done(t, a, 2e9, graph="Big", method="pot", reg=0.0)      # a job with more


def test_sinkhorn_estimate_covers_the_timed_bins_and_rows_redo_once(tmp_path, monkeypatch):
    import argparse
    import e5_approx as E5
    strata = [(0, np.arange(3), 600), (1, np.arange(2), 399), (2, np.arange(1), 1)]   # the top bin: one hub edge
    times = [np.array([1.0, 1.0, 1.0]), np.array([2.0, np.nan]), np.array([np.nan])]   # never timed
    tot, untimed = E5._estimate_covered(strata, times, 1000)
    assert tot == pytest.approx((600 * 1.0 + 399 * 2.0) / 999 * 1000) and untimed == pytest.approx(0.001)
    monkeypatch.setattr(common, "_TABLES", [])
    t = common.Rows(str(tmp_path / "s.csv"), ["graph", "method", "reg"])
    t.add(dict(graph="GitHub", method="sinkhorn", reg=0.01, sample=651, timed=650, skipped_memory=1,
               time_edge_weighted_s=float("nan")))                                     # the old NaN row
    t.add(dict(graph="GitHub", method="sinkhorn", reg=0.03, sample=651, timed=650, skipped_memory=1,
               max_pairs=5e7, time_edge_weighted_s=2e-3, untimed_share=1e-6))          # redone: finite
    t = common.Rows(str(tmp_path / "s.csv"), ["graph", "method", "reg"])
    a = argparse.Namespace(sinkhorn_max_pairs=5e7)
    assert not E5.sinkhorn_done(t, a, 7e8, graph="GitHub", method="sinkhorn", reg=0.01)   # redone once
    assert E5.sinkhorn_done(t, a, 7e8, graph="GitHub", method="sinkhorn", reg=0.03)       # then done
    assert E5.method_limit("pot", 7e8, a) == 7e8 and E5.method_limit("sinkhorn", 7e8, a) == 5e7


def test_two_writers_of_one_table_keep_each_others_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_TABLES", [])
    a = common.Rows(str(tmp_path / "t.csv"), ["graph", "method"])
    b = common.Rows(str(tmp_path / "t.csv"), ["graph", "method"])  # another worker, opened at the same time
    a.add(dict(graph="G", method="grc-mix", status="ok"))
    b.add(dict(graph="G", method="grc-otd", status="ok"))
    a.add(dict(graph="H", method="grc-mix", status="ok"))
    got = {(r["graph"], r["method"]) for r in common.read_csv(str(tmp_path / "t.csv"))}
    assert got == {("G", "grc-mix"), ("G", "grc-otd"), ("H", "grc-mix")}


def test_workers_only_for_steps_that_share_work(fake):
    run, calls, out = fake
    with pytest.raises(SystemExit):
        run("--steps", "s1", "--worker", "2")
    with pytest.raises(SystemExit):
        run("--steps", "e5-whole", "--worker", "2", "--force")
