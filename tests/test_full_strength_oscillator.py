"""Physical-strength encounter diagnostics do not export or replace ice DCS."""
import json

import numpy as np
import pytest

from physics.inelastic_dielectric.polarization.benchmarking import compare_full_strength_oscillator as check


@pytest.mark.parametrize("v,w,b", [(0,15,1), (20,15,1), (5,0,1), (5,15,.5), (np.nan,15,1)])
def test_invalid_kinematics_fail(v, w, b):
    with pytest.raises(ValueError):
        check.encounter_parameters(v,w,b)


def test_physical_strength_and_salvat_cutoff():
    b, x, eta = check.encounter_parameters(5., 15., 2.)
    assert b == pytest.approx(2*.5616/5)
    assert x == pytest.approx(15/check.HARTREE_EV*b/5)
    assert eta == pytest.approx(1/(b*25))
    assert check.HARTREE_EV == pytest.approx(27.21138625, rel=1e-8)


def test_energy_units_parity_and_exact_additivity():
    v, eta = 5., .02
    # H(eta)=3+7*eta+11*eta^2+13*eta^3.
    h = lambda e: 3+7*e+11*e**2+13*e**3
    data = check.energy_components(v, eta, 3., 7., h(eta), h(-eta))
    scale = check.HARTREE_EV*v*v
    assert data["leading_eV"] == pytest.approx(scale*3*eta**2)
    assert data["cubic_eV"] == pytest.approx(scale*7*eta**3)
    assert data["higher_even_eV"] == pytest.approx(scale*11*eta**4)
    assert data["higher_odd_eV"] == pytest.approx(scale*13*eta**5)
    assert data["full_eV"] == pytest.approx(data["full_even_eV"]+data["full_odd_eV"])
    assert data["full_eV"] == pytest.approx(data["leading_plus_cubic_eV"]+data["higher_even_eV"]+data["higher_odd_eV"])


def test_negative_cubic_not_clipped():
    result = check.energy_components(5., .1, 1., -20., .3, .4)
    assert result["leading_plus_cubic_eV"] < 0
    assert result["full_odd_eV"] < 0
    assert result["full_eV"] > 0


def test_summary_serializes_numpy_results():
    row=dict(numerical_status="PASS", full_odd_resolved=True,
             cubic_over_leading=np.float64(2), truncation_error_over_full=np.float64(.2))
    result=json.loads(json.dumps(check.summarize([row]),allow_nan=False))
    assert result["cubic_at_least_leading_count"] == 1
    assert result["truncation_error_above_10_percent_count"] == 1


def test_nuclear_singularity_failure_is_retained(monkeypatch):
    def fail(task):
        raise RuntimeError("Nuclear singularity")
    monkeypatch.setattr(check, "evaluate_case", fail)
    row = check.checked_case(("He:0",5.,15.,1.))
    assert row["numerical_status"] == "FAIL"
    assert "Nuclear singularity" in row["error"]
    assert "full_eV" not in row
    assert check.summarize([row])["failed_count"] == 1


def test_window_failure_does_not_zero_uncertainty(monkeypatch):
    def fake_energy(x, b, eta, field, *, tail, rtol):
        return 2.+(tail/1000 if eta else 0.)
    monkeypatch.setattr(check.reference, "nonlinear_energy", fake_energy)
    monkeypatch.setattr(check.reference, "linear_energy_quadrature", lambda *args: 2.)
    monkeypatch.setattr(check, "cubic_reference", lambda *args, **kw: (2.,0.))
    row = check.evaluate_case(("H:1",5.,15.,1.))
    assert row["numerical_status"] == "FAIL"
    assert row["tail_history"][-1]["tail"] == 512.
    assert row["odd_absolute_error_estimate_eV"] > 0
    assert not row["full_odd_resolved"]


def test_actual_full_strength_encounter_and_refinements():
    row = check.evaluate_case(("H:0",5.,15.,4.))
    assert row["numerical_status"] == "PASS"
    assert row["full_eV"] > 0
    assert row["eta"] == pytest.approx(1/(row["b_bohr"]*25))
    assert row["T_total_MeV_NR"] > 0
    assert row["full_eV"] == pytest.approx(row["full_even_eV"]+row["full_odd_eV"])


@pytest.mark.parametrize("element,charge", [("H",0),("He",0)])
def test_weak_coupling_remainder_has_fourth_order_energy_scaling(element, charge):
    field=check.reference.frozen_field(element,charge)
    b,x=.5,.3
    leading=check.reference.linear_energy_quadrature(x,b,field)
    cubic=check.cubic_reference(x,b,field)[0]
    remainders=[]
    for eta in (.008,.004):
        full=check.reference.nonlinear_energy(x,b,eta,field,tail=128,rtol=2e-12)
        remainders.append(eta**2*(full-leading-eta*cubic))
    assert remainders[0]/remainders[1] == pytest.approx(16,rel=.03)


def test_pdf_style_and_missing_cases(tmp_path, monkeypatch):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.text import Text
    from matplotlib import pyplot as plt
    report = dict(states=list(check.DEFAULT_STATES), velocities_au=[5.],
                  oscillator_energies_eV=[15.], impact_ratios=[1.,4.,16.], rows=[])
    for state in report["states"]:
        for ratio in report["impact_ratios"]:
            report["rows"].append(dict(state=state, velocity_au=5., oscillator_eV=15.,
                impact_ratio=ratio, numerical_status="PASS" if ratio != 4 else "FAIL",
                leading_eV=1., leading_plus_cubic_eV=2., full_eV=1.5))
    calls = []
    def inspect(pdf, fig, **kwargs):
        fig.canvas.draw()
        assert len(fig.axes) == 4
        np.testing.assert_allclose(fig.get_size_inches(), [check.constants.AASTEX_FULL_WIDTH_IN,4.8])
        colors = plt.get_cmap("plasma")(np.linspace(0,1,4))[:-1]
        for ax in fig.axes:
            np.testing.assert_allclose([line.get_color() for line in ax.lines],colors)
            assert all(np.isnan(line.get_ydata()[1]) for line in ax.lines)
            assert not any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                assert text.get_fontsize() == check.constants.PAPER_FONTSIZE
                assert text.get_fontfamily()[0] in (check.constants.FONT_COURIER,"Courier New","Courier")
        calls.append(fig)
    monkeypatch.setattr(PdfPages,"savefig",inspect)
    assert check.plot_results(report,tmp_path).suffix == ".pdf"
    assert len(calls) == 1


def test_plot_only_leaves_provenance_untouched(tmp_path, monkeypatch):
    path=tmp_path/"comparison.json"
    raw=json.dumps(dict(rows=[],source_sha256={"original":"unchanged"})).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(check,"plot_results",lambda report,out:out/"figure.pdf")
    def fail(*args,**kw):
        raise AssertionError("Plot-only must not run workers")
    monkeypatch.setattr(check,"ProcessPoolExecutor",fail)
    assert check.main(["--plot-only","--output-dir",str(tmp_path)]) == 0
    assert path.read_bytes() == raw
