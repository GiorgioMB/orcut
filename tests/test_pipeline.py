"""The checkpoints of the experiment pipeline (experiments/main.py, checkpoint.py, common.Rows), with
fake steps: a step that ended without errors is not run again, anything else is."""
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

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
        fh.write(json.dumps({"step": name, "argv": sys.argv[1:], "cwd": os.getcwd()}) + "\\n")
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
    steps = {s: ([str(script), s], "fake") for s in ("s1", "s2", "e5-whole")}
    out = tmp_path / "out"
    monkeypatch.setattr(common, "environment", lambda: {"git": "abc"})
    monkeypatch.delenv("ORCUT_GRC_PYTHON", raising=False)

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

    def killed(cmd, log_path, lock):
        real_run(cmd, log_path, lock)
        return -9  # what subprocess reports for a step killed by SIGKILL (the out-of-memory killer)

    monkeypatch.setattr(M, "_run", killed)
    assert run("--steps", "s1") == 1
    rec = CK.load(str(out), "s1")
    assert rec["state"] == "killed" and rec["returncode"] == -9
    monkeypatch.setattr(M, "_run", real_run)
    assert run("--steps", "s1") == 0
    assert calls() == ["s1", "s1"]


def test_force_runs_a_completed_step(fake):
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


def test_grc_step_runs_again_once_grc_is_available(fake, tmp_path):
    run, calls, out = fake
    assert run("--steps", "e5-whole") == 0
    py = tmp_path / "grcpython"
    py.write_text("")
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
    assert state == {"s1": "done", "s2": "failed", "e5-whole": "not", "artifacts": "not"}


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


def test_tables_in_the_old_place_stop_the_run(fake, tmp_path, monkeypatch):
    run, calls, out = fake
    root = tmp_path / "repo"
    monkeypatch.setattr(M, "HERE", str(root / "experiments"))
    (root / "experiments" / "results" / "e1").mkdir(parents=True)
    (root / "experiments" / "results" / "e1" / "exactness.csv").write_text("graph,measure\n")
    steps = {"s1": ([str(tmp_path / "fake.py"), "s1"], "fake")}
    assert M.main(["--out", str(root / "results"), "--steps", "s1"], steps=steps) == 2
    assert M.main(["--out", str(root / "results"), "--steps", "s1", "--ignore-old-layout"], steps=steps) == 0


# ------------------------------------------------------------ artifacts

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
