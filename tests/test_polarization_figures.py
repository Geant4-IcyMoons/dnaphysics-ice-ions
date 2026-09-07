"""Polarization PDF figures share the nonlinear benchmark's paper style."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pytest

from physics.inelastic_dielectric.polarization.benchmarking import compare_screened_barkas_point_projectiles as comparison
from physics.inelastic_dielectric.polarization.benchmarking import compare_close_collisions as close_comparison


@pytest.mark.parametrize("projectile", ["proton", "alpha"])
def test_pdf_geometry_and_style(tmp_path, monkeypatch, projectile):
    rows = [dict(phase=phase, T_MeV=t, S_numerical_MeV_cm2_g=1.0001/t,
                 S_salvat_MeV_cm2_g=1/t, relative_difference=1e-4)
            for phase in comparison.PHASES for t in (.1, 1, 10, 100)]
    saved = []
    def inspect(fig, path, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert fig.get_figwidth() == comparison.AASTEX_FULL_WIDTH_IN
        colors = comparison.plt.get_cmap("plasma")(np.linspace(0, 1, 3))[:-1]
        for axis in fig.axes[:2]:
            for line, color in zip(axis.lines, colors):
                np.testing.assert_array_equal(line.get_color(), color)
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text():
                assert item.get_fontsize() == comparison.PAPER_FONTSIZE
                assert item.get_fontfamily()[0] in (comparison.FONT_COURIER, "Courier New", "Courier")
        texts = fig.texts + [a.xaxis.label for a in fig.axes] + [a.yaxis.label for a in fig.axes]
        texts += [a._left_title for a in fig.axes] + list(fig.legends[0].get_texts())
        for item in texts:
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            assert 0 <= box.x0 <= box.x1 <= fig.bbox.width
            assert 0 <= box.y0 <= box.y1 <= fig.bbox.height
        lower = fig.axes[2].xaxis.label.get_window_extent(renderer)
        legend = fig.legends[0].get_window_extent(renderer)
        assert legend.y1 < lower.y0
        assert Path(path).suffix == ".pdf"
        saved.append(path)
    monkeypatch.setattr(Figure, "savefig", inspect)
    comparison._plot(rows, tmp_path, projectile)
    assert len(saved) == 1


def test_plot_only_keeps_report_and_never_integrates(tmp_path, monkeypatch):
    report = tmp_path/"comparison.json"
    original = json.dumps(dict(projectile="alpha", rows=[]))
    report.write_text(original)
    monkeypatch.setattr(comparison.sys, "argv", ["benchmark", "--projectile", "alpha", "--plot-only", "--out-dir", str(tmp_path)])
    monkeypatch.setattr(comparison, "_plot", lambda *args: tmp_path/"alpha.pdf")
    def forbidden(*args, **kwargs):
        raise AssertionError("Plot-only mode must not start workers")
    monkeypatch.setattr(comparison, "ProcessPoolExecutor", forbidden)
    assert comparison.main() == 0
    assert report.read_text() == original


def test_close_comparison_plot_style_and_spacing(tmp_path, monkeypatch):
    report = dict(states=["H:1","He:0","He:1","C:3","O:4","S:8"],
                  energies_MeV=[10,30,100])
    report["rows"] = [dict(state=state,phase=phase,T_MeV=t,
        S_cutoff_MeV_cm2_g=10.**i/t,S_matched_MeV_cm2_g=1.1*10.**i/t,
        relative_difference=.1)
        for i,state in enumerate(report["states"]) for phase in close_comparison.PHASES
        for t in report["energies_MeV"]]

    def inspect(fig, path, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert fig.get_figwidth() == close_comparison.AASTEX_FULL_WIDTH_IN
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text():
                assert item.get_fontsize() == close_comparison.PAPER_FONTSIZE
                assert item.get_fontfamily()[0] in (close_comparison.FONT_COURIER,"Courier New","Courier")
        colors = close_comparison.plt.get_cmap("plasma")(np.linspace(0,1,7))[:-1]
        for axis in fig.axes[:2]:
            for i,color in enumerate(colors):
                for line in axis.lines[2*i:2*i+2]:
                    np.testing.assert_array_equal(line.get_color(),color)
        legend = fig.legends[0].get_window_extent(renderer)
        assert legend.x0 >= 0 and legend.x1 <= fig.bbox.width
        assert legend.y0 >= 0
        for axis in fig.axes[2:]:
            assert legend.y1 < axis.xaxis.label.get_window_extent(renderer).y0
        for upper,lower in zip(fig.axes[:2],fig.axes[2:]):
            title = lower._left_title.get_window_extent(renderer)
            assert lower.bbox.y1 <= title.y0 < title.y1 <= upper.bbox.y0
        assert Path(path).suffix == ".pdf"

    monkeypatch.setattr(Figure,"savefig",inspect)
    close_comparison._plot(report,tmp_path)


def test_close_comparison_plot_only_preserves_provenance(tmp_path, monkeypatch):
    report = tmp_path/"comparison.json"
    original = json.dumps(dict(source_sha256={"original-code":"unchanged"},rows=[]))
    report.write_text(original)
    monkeypatch.setattr(sys,"argv",["comparison","--plot-only","--output-dir",str(tmp_path)])
    monkeypatch.setattr(close_comparison,"_plot",lambda *args: tmp_path/"comparison.pdf")
    def forbidden(*args, **kwargs):
        raise AssertionError("Plot-only mode must not start workers")
    monkeypatch.setattr(close_comparison,"ProcessPoolExecutor",forbidden)
    assert close_comparison.main() == 0
    assert report.read_text() == original
