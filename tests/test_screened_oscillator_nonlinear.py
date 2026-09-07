"""Regression checks for the independent all-state oscillator benchmark."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[1]/"physics/inelastic_dielectric/polarization/benchmarking/check_screened_oscillator_nonlinear.py")
SPEC = importlib.util.spec_from_file_location("nonlinear_oscillator_check", SCRIPT)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@pytest.mark.parametrize("element,charge", check.STATES)
def test_independent_field_matches_frozen_density(element, charge):
    r = np.r_[0., np.geomspace(1e-6, 100., 101)]
    field = check.independent_field(element, charge)
    actual = field(r)
    expected = check.load_density(element, charge).radial_charge_direct(r)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)
    assert actual[0] == check.ELEMENTS[element]
    assert actual[-1] == pytest.approx(charge, abs=1e-15)
    assert field.electron_integral == pytest.approx(check.ELEMENTS[element]-charge, abs=1e-8)


def test_all_states_and_charge_limits_are_covered():
    assert len(check.STATES) == 38
    for element, z in check.ELEMENTS.items():
        assert [q for e, q in check.STATES if e == element] == list(range(z+1))


@pytest.mark.parametrize("radius", [-1., np.inf, np.nan])
def test_invalid_radius_fails(radius):
    with pytest.raises(ValueError):
        check.independent_field("He", 1)(radius)


def test_cubic_extraction_removes_even_terms_and_fifth_order_bias():
    eta = check.SCALED_STRENGTHS/2
    # E(eta)=eta^2*(3+7*eta+11*eta^2+13*eta^3).
    def scaled_energy(e):
        return 3+7*e+11*e**2+13*e**3
    value, error, _ = check.extrapolate_cubic(scaled_energy(eta), scaled_energy(-eta), eta)
    assert value == pytest.approx(7., abs=1e-12)
    assert error < 1e-12


@pytest.mark.parametrize("strengths", [[.02, .01], [.02, .01, .004], [-.02, -.01, -.005]])
def test_invalid_extrapolation_strengths_fail(strengths):
    with pytest.raises(ValueError):
        check.extrapolate_cubic([1., 2., 3.], [1., 2., 3.], strengths)


@pytest.mark.parametrize("x,b", [(0., .5), (.1, -1.)])
def test_invalid_oscillator_parameters_fail(x, b):
    with pytest.raises(ValueError):
        check.nonlinear_energy(x, b, .01, check.independent_field("He", 1))


def test_zero_coupling_ode_matches_independent_linear_fourier_integral():
    field = check.independent_field("He", 1)
    actual = check.nonlinear_energy(.3, .5, 0., field, tail=128., rtol=2e-12)
    expected = check.linear_energy_quadrature(.3, .5, field)
    assert actual == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("element,charge", [("H", 0), ("He", 1), ("C", 2), ("S", 0)])
def test_screened_cubic_term_matches_independent_nonlinear_solver(element, charge):
    row = check.check_case((element, charge, 1., .3))
    assert row["status"] == "PASS"
    assert row["relative_checks"]["nonlinear_vs_kernel"] < 1e-4
    assert row["extended"]["value"] > 0


def test_unresolved_case_is_recorded_as_failure(monkeypatch):
    def unresolved(case):
        raise RuntimeError("Unresolved coefficient")
    monkeypatch.setattr(check, "check_case", unresolved)
    row = check.checked_case(("S", 0, 2., 1.))
    assert row["status"] == "FAIL"
    assert row["charge"] == 0
    assert "Unresolved coefficient" in row["error"]


def test_plot_paper_style_and_panel_height_colorbar(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    rows = [dict(element=e, charge=q, b_over_radius=b, status="PASS",
                 relative_checks={"nonlinear_vs_kernel": 1e-7*(q+1)*(b+1),
                                  "ode_tolerance": 1e-6*(q+1)*(b+1)})
            for e, q in check.STATES for b in check.IMPACT_RADIUS_RATIOS]
    outputs = []
    def inspect_figure(fig, path, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes = fig.axes[:5]
        assert [a.get_title(loc="left") for a in axes] == [f"({s})" for s in "abcde"]
        np.testing.assert_allclose(fig.get_size_inches(),
                                   [check.AASTEX_FULL_WIDTH_IN, 2*check.THREE_PANEL_ROW_HEIGHT_IN])
        color_ax = axes[2].child_axes[0]
        assert color_ax.get_position().height == pytest.approx(axes[0].get_position().height)
        gap = color_ax.get_position().x0-axes[2].get_position().x1
        assert 0 < gap < .02*axes[2].get_position().width
        assert "plasma" in color_ax.collections[-1].cmap.name
        assert axes[0].get_xlabel() == axes[1].get_xlabel() == ""
        assert all(axes[i].get_xlabel() for i in (2, 3, 4))
        assert all(not a.get_ylabel() for a in axes)
        assert sum(t.get_text() == "Maximum difference (%)" for t in fig.texts) == 1
        legend_box = fig.axes[5].get_legend().get_window_extent(renderer)
        assert axes[2].xaxis.label.get_window_extent(renderer).y0-legend_box.y1 > 5
        assert any("$a_0$ for bare ions" in t.get_text() for t in fig.axes[5].texts)
        from matplotlib import pyplot as plt
        expected_colors = plt.get_cmap("plasma")(np.linspace(0., 1., 4))[:-1]
        np.testing.assert_allclose([line.get_color() for line in axes[0].lines[::2]], expected_colors)
        assert not any(line.get_linestyle() == ":" for axis in axes for line in axis.lines)
        assert all("tolerance" not in t.get_text() for t in fig.axes[5].get_legend().get_texts())
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                assert text.get_fontsize() == check.PAPER_FONTSIZE
                assert text.get_fontfamily()[0] in (check.FONT_COURIER, "Courier New", "Courier")
        # Check non-tick text against the canvas, and labels between rows.
        labels = [a.xaxis.label for a in axes]+[a.yaxis.label for a in axes]
        labels += [a._left_title for a in axes]+[color_ax.yaxis.label]+fig.texts
        labels += fig.axes[5].texts+fig.axes[5].get_legend().get_texts()
        for text in labels:
            if not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= fig.bbox.width
            assert box.y0 >= 0 and box.y1 <= fig.bbox.height
        for top, bottom in zip(axes[:2], axes[3:]):
            tick_boxes = [t.get_window_extent(renderer) for t in top.get_xticklabels() if t.get_visible()]
            assert min(b.y0 for b in tick_boxes) > bottom._left_title.get_window_extent(renderer).y1
        outputs.append(Path(path).suffix)
    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    check.plot_results(rows, tmp_path)
    assert outputs == [".pdf"]


def test_plot_only_preserves_numerical_report(tmp_path, monkeypatch):
    report_path = tmp_path/"comparison.json"
    raw = json.dumps(dict(rows=[dict(element="He", charge=1)], relative_tolerance=.003,
                          code_sha256={"original_solver": "preserved"})).encode()
    report_path.write_bytes(raw)
    calls = []
    monkeypatch.setattr(check, "plot_results", lambda rows, out, **kw: calls.append((rows, out, kw)))
    def no_solver(*args, **kwargs):
        raise AssertionError("Plot-only must not start benchmark workers")
    monkeypatch.setattr(check, "ProcessPoolExecutor", no_solver)
    monkeypatch.setattr(check.sys, "argv", [str(SCRIPT), "--plot-only", "--out-dir", str(tmp_path)])
    assert check.main() == 0
    assert report_path.read_bytes() == raw
    assert calls[0][2] == {}
