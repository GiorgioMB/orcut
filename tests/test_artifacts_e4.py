"""artifacts.py, E4: a configuration that did not run on some graphs (POT where the dense local
problems exceed the memory guard: est_total_s empty) gives its ratios over the graphs where it ran,
not an empty key (up to 1.4.0: '--')."""
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
    assert "8.0" in O.keys["EFourPotOverFullMax"] and "4.0" in O.keys["EFourPotOverFullMin"]
    assert "--" not in O.keys["EFourPotOverFullMedian"]
