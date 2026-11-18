"""tools/check_chunks.py: a chunked row whose time does not sum all the chunks of its logs is found
and repaired from them; a row timed in one call keeps its own time."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import check_chunks as CC  # noqa: E402
import common  # noqa: E402


def _log(chunks, nch=4, m=100):
    return dict(key=dict(graph="WS n=1000", measure="lazy", n=1000, m=m, threads=32), k=nch,
                bounds=list(range(0, m + 1, m // nch)),
                chunks={str(c): dict(seconds=float(s), host=h) for c, s, h in chunks})


def test_a_row_written_from_part_of_its_chunks_is_repaired(tmp_path):
    e3 = tmp_path / "e3"
    (e3 / "partial").mkdir(parents=True)
    common.write_csv([dict(family="WS", regime="grow0.4", n=1000, seed=0, m=100, threads=32, time_s=3.0, chunks=4,
                           chunk_hosts=1),
                      dict(family="WS", regime="grow0.4", n=300, seed=0, m=50, threads=32, time_s=2.0)],
                     str(e3 / "synth.csv"))
    p = e3 / "partial"
    (p / "synth.ws.grow0.4.n1000.s0.T32.done.json").write_text(json.dumps(_log([(0, 1, "a"), (1, 2, "a")])))
    (p / "synth.ws.grow0.4.n1000.s0.T32.json").write_text(json.dumps(_log([(1, 2, "a"), (2, 3, "b"), (3, 4, "c")])))
    (p / "synth.ws.grow0.4.n300.s0.T32.json").write_text(json.dumps(_log([(0, 1, "a"), (1, 2, "a")], nch=2, m=50)))
    found = {f["base"]: f["status"] for f in CC.check(str(tmp_path))}
    assert found == {"synth.ws.grow0.4.n1000.s0.T32": "mismatch", "synth.ws.grow0.4.n300.s0.T32": "one call"}
    CC.fix(str(tmp_path), CC.check(str(tmp_path)))
    rows = common.read_csv(str(e3 / "synth.csv"))
    assert float(rows[0]["time_s"]) == 10.0 and float(rows[0]["chunks_timed"]) == 4 and float(rows[0]["chunk_hosts"]) == 3
    assert float(rows[1]["time_s"]) == 2.0  # timed in one call: its own time
    assert sorted(os.listdir(p)) == ["synth.ws.grow0.4.n1000.s0.T32.done.json", "synth.ws.grow0.4.n300.s0.T32.json"]
    assert set(json.loads((p / "synth.ws.grow0.4.n1000.s0.T32.done.json").read_text())["chunks"]) == {"0", "1", "2", "3"}
    assert (tmp_path / "replaced").is_dir()
    assert {f["status"] for f in CC.check(str(tmp_path))} == {"ok", "one call"}
