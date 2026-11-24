"""artifacts.py, E4: a configuration that did not run on some graphs (POT where the dense local
problems exceed the memory guard: est_total_s empty) gives its ratios over the graphs where it ran,
not an empty key."""
import os
import sys

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, EXP)
import artifacts as A  # noqa: E402
import common  # noqa: E402


def test_pot_ratios_are_over_the_graphs_where_it_ran(tmp_path):
    rows = []
    for g, pot in (("ER", 8.0), ("BA", 4.0), ("CL2.1", None)):
        for cfg, t in (("full", 1.0), ("cut", 2.0), ("-twins", 1.5), ("pot", pot), ("graphflow", 20.0)):
            rows.append(dict(graph=g, config=cfg, m=100, deg_max=10, value_mismatch=0,
                             est_total_s="" if t is None else t))
    (tmp_path / "e4").mkdir()
    common.write_csv(rows, str(tmp_path / "e4" / "ablation.csv"))
    O = A.Out(False)
    A.block_e4(str(tmp_path), O)
    assert O.keys["EFourPotGraphs"] == "2"
    # 8.0x and 4.0x: decimals that are all zeros are dropped in the keys (artifacts.strip_zeros)
    assert O.keys["EFourPotOverFullMax"] == r"\ensuremath{8\times}" and O.keys["EFourPotOverFullMin"] == r"\ensuremath{4\times}"
    assert "--" not in O.keys["EFourPotOverFullMedian"]


def test_method_name_reaches_the_labels(tmp_path, monkeypatch):
    """--method-name (e.g. 'ours' for an anonymous version) replaces the solver's name in the labels."""
    rows = [dict(graph="ER", config=cfg, m=100, deg_max=10, value_mismatch=0, est_total_s=t)
            for cfg, t in (("full", 1.0), ("cut", 2.0))]
    (tmp_path / "e4").mkdir()
    common.write_csv(rows, str(tmp_path / "e4" / "ablation.csv"))
    monkeypatch.setattr(A, "METHOD", "ours")
    O = A.Out(False)
    A.block_e4(str(tmp_path), O)
    table = O.rows["ablation"]
    assert "+ witness relays (ours)" in table and "orcut" not in table
