"""artifacts.py: numbers without decimals that are all zeros, the memory column, the keys behind
the numbers of the text, one colour per method in every figure, the sizes of the two layouts, and
figures with Type-1 fonts only."""
import os
import shutil
import sys

import pytest

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, EXP)
import artifacts as A  # noqa: E402
import common  # noqa: E402


@pytest.mark.parametrize("raw, out", [
    (r"\ensuremath{6.0\times}", r"\ensuremath{6\times}"),
    (r"0.0\%", r"0\%"),
    (r"\ensuremath{1.0\times10^{-14}}", r"\ensuremath{1\times10^{-14}}"),
    ("3.00", "3"), ("10.0", "10"),
    ("0.40", "0.40"), (r"10.4\%", r"10.4\%"), (r"2{,}048\,MB", r"2{,}048\,MB"),
    ("9.15.9999", "9.15.9999"), ("3.10.0", "3.10.0"),  # versions are not numbers
])
def test_strip_zeros(raw, out):
    assert A.strip_zeros(raw) == out


def test_keys_lose_zero_decimals_but_names_and_versions_do_not(monkeypatch):
    monkeypatch.setattr(A, "STRIP_ZEROS", "keys")
    O = A.Out(False)
    O.put("K", A.f_x(6.04), "test")
    O.put("EnvPython", "3.10", "test", strip=False)
    O.table("t", ["a & 1.0 & 0.99 \\\\"])
    assert O.keys == {"K": r"\ensuremath{6\times}", "EnvPython": "3.10"}
    assert O.rows["t"] == "a & 1.0 & 0.99 \\\\"  # the tables keep their digits unless --strip-zeros all
    monkeypatch.setattr(A, "STRIP_ZEROS", "none")
    O.put("K", A.f_x(6.04), "test")
    assert O.keys["K"] == r"\ensuremath{6.0\times}"


def test_memory_of_one_page_is_zero():
    assert A.f_mem(4096 / 2**20) == "0" and A.f_mem(0.0) == "0"
    assert A.f_mem(1.3) == "1.3" and A.f_mem(float("nan")) == "--"


def test_every_method_has_its_own_colour():
    colours = [m[4] for m in A.E2_METHODS]
    assert len(set(colours)) == len(colours)
    assert all(len(m) == 5 for m in A.E2_METHODS)


def test_layout_sizes(monkeypatch):
    monkeypatch.setattr(A, "LAYOUT", "twocolumn")
    monkeypatch.setattr(A, "FIG_W", A.LAYOUTS["twocolumn"]["text"])
    assert A.fig_size("fig-per-edge")[0] == 3.25  # one column
    assert A.fig_size("fig-tradeoff")[0] == 6.75  # the one-column appendix
    monkeypatch.setattr(A, "LAYOUT", "onecolumn")
    monkeypatch.setattr(A, "FIG_W", 5.5)
    assert A.fig_size("fig-per-edge")[0] == 5.5 and A.fig_size("fig-per-edge-all", 7.5) == (5.5, 7.5)


def test_keys_of_the_text(tmp_path):
    """The numbers the text quoted by hand: speedups per graph at the most threads, the share of the
    most negatively curved edges each proxy recovers, the graphs with other measures."""
    (tmp_path / "e3").mkdir()
    common.write_csv([dict(graph=g, threads=T, time_s=t) for g, sp in (("DBLP", 22.0), ("Twitch-DE", 12.0))
                      for T, t in ((1, sp), (32, 1.0))], str(tmp_path / "e3" / "threads.csv"))
    common.write_csv([dict(graph=g, measure=m, n=10, m=20, deg_max=3, deg_mean=2.0, threads=32, time_s=1.0,
                           peak_extra_mb=4096 / 2**20, pot_est_s=4.0)
                      for g in ("A", "B") for m in ("lazy", "nonlazy")], str(tmp_path / "e3" / "real.csv"))
    O = A.Out(False)
    A.block_e3(str(tmp_path), O, str(tmp_path))
    assert O.keys["EThreeScalingDBLP"] == r"\ensuremath{22\times}"
    assert O.keys["EThreeScalingTwitchDE"] == r"\ensuremath{12\times}"
    assert O.keys["EThreeOtherMeasuresGraphs"] == "2"
    assert " & 0 & " in O.rows["large"]  # one page of growth is no growth
    (tmp_path / "e5").mkdir()
    common.write_csv([dict(graph="A", method=m, time_s=1.0, mae="", max_err="", sign_disagree=0.1, spearman=0.5,
                           recall_neg=r) for m, r in (("bf@lly", 0.15), ("af3@lly", 0.04), ("jl", 0.12), ("orcut", 1.0))],
                     str(tmp_path / "e5" / "whole.csv"))
    A.block_e5(str(tmp_path), O, str(tmp_path))
    assert (O.keys["EFiveRecallBF"], O.keys["EFiveRecallAFThree"], O.keys["EFiveRecallJL"]) == (r"15\%", r"4\%", r"12\%")


@pytest.mark.parametrize("mode", ["type1", "truetype"])
def test_figures_have_no_type3_fonts(tmp_path, mode):
    if mode == "type1" and not shutil.which("latex"):
        pytest.skip("latex is not installed")
    import matplotlib

    old = dict(matplotlib.rcParams)
    try:
        A.setup_fonts(mode)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.25, 2))
        ax.loglog([1, 10], [1, 100], label=A.lab("CL2.1(10k, 16) & 50%"))
        ax.set_xlabel(r"$|\mathcal{U}_i| = |\mathcal{U}_j|$")
        ax.set_title(f"hub sweep, regime {A.quoted('mixed')}")
        ax.legend()
        monkey = A.FONTS
        A.FONTS = mode
        A.save_figure(fig, str(tmp_path / "f.pdf"))
        A.FONTS = monkey
        fonts = set(A.pdf_fonts(str(tmp_path / "f.pdf")))
        assert "Type3" not in fonts and fonts
        if mode == "type1":
            assert fonts <= {"Type1", "Type1C"}
        plt.close(fig)
    finally:
        matplotlib.rcParams.update(old)


def test_type3_fonts_are_refused(tmp_path, monkeypatch):
    import matplotlib

    old = dict(matplotlib.rcParams)
    try:
        matplotlib.use("Agg")
        matplotlib.rcParams.update({"text.usetex": False, "pdf.fonttype": 3})
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_title("x")
        monkeypatch.setattr(A, "FONTS", "truetype")
        with pytest.raises(A.FontError):
            A.save_figure(fig, str(tmp_path / "f.pdf"))
        plt.close(fig)
    finally:
        matplotlib.rcParams.update(old)
