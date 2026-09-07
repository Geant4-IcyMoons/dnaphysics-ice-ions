"""Screened oscillator math, charge limits, and production table integration."""
import json

import numpy as np
import pytest
from scipy.integrate import quad

from test_projectile_relativistic_dcs import MODULE as gen
from physics.inelastic_dielectric.polarization import barkas_dcs as bd
from physics.inelastic_dielectric.polarization import screened_barkas as sb
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import ELEMENTS, load_density

STATES = [(element, q) for element, z in ELEMENTS.items() for q in range(z+1)]


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(gen, "ICE_TYPE", "amorphous")
    gen.set_projectile("proton")
    gen._set_projectile_relativistic_dcs(False)
    yield
    gen.set_projectile("proton")
    gen._set_projectile_relativistic_dcs(False)


@pytest.mark.parametrize("element,charge", STATES)
def test_radial_field_gauss_law_and_derivative(element, charge):
    d = load_density(element, charge)
    for r in (0., .03, .2, 2., 20.):
        g, rgprime = d.radial_charge_direct(r)
        enclosed = quad(lambda x: 4*np.pi*x*x*d.density(x), 0, r,
                        epsabs=1e-10, epsrel=1e-10)[0]
        assert g == pytest.approx(d.z-enclosed, abs=3e-8)
        assert rgprime == pytest.approx(-4*np.pi*r**3*d.density(r), abs=1e-12)
    r = np.geomspace(1e-5, 100, 501)
    for interp, direct in zip(d.radial_charge(r), d.radial_charge_direct(r)):
        np.testing.assert_allclose(interp, direct, rtol=2e-5, atol=2e-7)
    r, step = .2, 1e-6
    derivative = (d.radial_charge_direct(r+step)[0]-d.radial_charge_direct(r-step)[0])/(2*step)
    assert derivative*r == pytest.approx(d.radial_charge_direct(r)[1], abs=1e-7)


@pytest.mark.parametrize("beta", [.01, .4])
def test_independent_point_field_recovers_salvat_integral(beta):
    gamma = 1/np.sqrt(1-beta*beta)
    xi = np.array([1e-5, .001, .1, 1.])
    w = xi*gamma*beta*beta*bd.MEC2_EV/.5616
    actual = sb.oscillator_kernel(w, beta, gamma, load_density("H", 1),
                                 n_impact=144, n_time=8193, tail_cycles=128.)
    expected = bd.arbi1(xi)+bd.arbi2(xi)/gamma**2
    # SBETHE itself is a piecewise fit with a quoted 0.1% tolerance.
    np.testing.assert_allclose(actual, expected, rtol=.001)


def test_force_gradient_is_cubic_not_effective_charge_squared():
    density = load_density("C", 2)
    class ScaledField:
        def __init__(self, scale):
            self.scale = scale
        def radial_charge(self, r):
            return tuple(self.scale*v for v in density.radial_charge(r))
    args = (np.array([.001, .1, 1.]), np.array([.02, .2, 2.]))
    baseline = sb._impulse_products(*args, density, 2049, 32.)
    for scale in (-1., 2.):
        actual = sb._impulse_products(*args, ScaledField(scale), 2049, 32.)
        np.testing.assert_allclose(actual, np.asarray(baseline)*scale**3, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("element,charge", STATES)
@pytest.mark.parametrize("relativistic", [False, True])
def test_all_states_kernel_and_explicit_physical_rejections(element, charge, relativistic):
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    gen._set_projectile_relativistic_dcs(relativistic)
    d = load_density(element, charge)
    s, c = gen.model.epsilon_optical("amorphous"), gen.model.default_dispersion_coefficients()
    w = np.array([15., 50., 1000.])
    t = np.full_like(w, gen.PROJECTILE_MASS_NUMBER*1e7)
    exc = np.array([[gen._selected_dsigma_excitation(v, t[0], i, s, c, 256)
                     for i in range(len(s.excitations))] for v in w])
    ion = np.array([[gen._selected_dsigma_ionization(v, t[0], i, s, c, 256)
                     for i in range(len(s.ionizations))]
                    + [gen._selected_dsigma_kshell(v, t[0], s, c, 256)] for v in w])
    data = {"T_line": t, "E_line": w, "exc_vals": exc, "ion_vals": ion}
    # These are recorded failures of the proposed spectral matching, not
    # successful all-state production validation. Do not hide them by scaling.
    rejected = {("He", 0), ("C", 0), ("C", 1), ("O", 0), ("O", 1),
                *(("S", q) for q in range(6))}
    if (element, charge) in rejected:
        with pytest.raises(RuntimeError, match="uncontrolled"):
            bd.apply_barkas_correction_to_dcs_data(
                data, s, "amorphous", gen.PROJECTILE_MASS_AU, d.z,
                include_barkas_dcs=True, born_reference_charge="bare_Z",
                projectile_density=d, workers=1)
        return
    corrected, diag = bd.apply_barkas_correction_to_dcs_data(
        data, s, "amorphous", gen.PROJECTILE_MASS_AU, d.z,
        include_barkas_dcs=True, born_reference_charge="bare_Z",
        projectile_density=d, workers=1)
    born = exc.sum(axis=1)+ion.sum(axis=1)
    np.testing.assert_array_equal(diag.DCS_Born_m2_per_eV, born)
    np.testing.assert_allclose(diag.DCS_total_m2_per_eV, born+diag.DCS_Barkas_m2_per_eV, rtol=1e-14)
    assert np.all(np.isfinite(diag.DCS_total_m2_per_eV))
    assert np.all(diag.DCS_total_m2_per_eV >= 0)
    assert diag.TCS_total_m2[0] == pytest.approx(np.trapezoid(diag.DCS_total_m2_per_eV, w), rel=1e-14)
    assert diag.S_Barkas_check_eV_m2[0] == pytest.approx(np.trapezoid(w*diag.DCS_Barkas_m2_per_eV, w), rel=1e-14)
    if d.electrons:
        assert diag.barkas_model_metadata["barkas_model"] == sb.MODEL
        assert np.any(diag.DCS_Barkas_m2_per_eV > 0)
    else:
        oos = bd.oos_density(w, s, "amorphous")
        expected = bd.barkas_dcs_m2_per_eV(t, w, d.z, gen.PROJECTILE_MASS_AU, oos.df_dW_total)
        np.testing.assert_array_equal(diag.DCS_Barkas_m2_per_eV, expected)


def test_neutral_is_not_zero_and_cutoff_and_units_are_explicit():
    d = load_density("H", 0)
    t, mass = 1e6, 1837.15
    w = np.array([15., 50., 2*bd.wmax_eV(t, mass)])
    result, error = sb.dcs_m2_per_eV(t, w, mass, np.ones(3), d, workers=1)
    beta, gamma = bd.projectile_beta_gamma(t, mass)
    kernel, _ = sb.converged_kernel(w[:2], beta, gamma, d)
    pref_cm2 = 4*np.pi*bd.RE_CLASSICAL_CM**2*bd.ALPHA_FINE/(gamma**2*beta**5)
    np.testing.assert_allclose(result[:2], 1e-4*pref_cm2*kernel, rtol=1e-14)
    assert result[2] == 0 and error[2] == 0
    assert np.all(result[:2] > 0)
    with pytest.raises(ValueError, match="Bohr velocity"):
        sb.dcs_m2_per_eV(1e4, [10.], mass, [1.], d, workers=1)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_oos_normalization_remains_eight_plus_two(phase):
    s = gen.model.epsilon_optical(phase)
    w = np.geomspace(1e-6, 1e8, 50000)
    oos = bd.oos_density(w, s, phase)
    assert np.trapezoid(oos.df_dW_valence, w) == pytest.approx(8., abs=1e-10)
    assert np.trapezoid(oos.df_dW_OK, w) == pytest.approx(2., abs=1e-10)


@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("element,charge", [(e, z-1) for e, z in ELEMENTS.items()])
def test_generator_exports_corrected_totals_and_metadata(tmp_path, monkeypatch, relativistic, element, charge):
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    gen._set_projectile_relativistic_dcs(relativistic)
    monkeypatch.setattr(gen, "_export_to_custom_geant4", lambda paths: None)
    s, c = gen.model.epsilon_optical("amorphous"), gen.model.default_dispersion_coefficients()
    t = gen.PROJECTILE_MASS_NUMBER*np.array([1e7, 1e8])
    data = gen.write_emfietzoglou_dcs_tables(
        s, c, out_dir=tmp_path, T_list=t, NE=12, Nq=128, return_data=True,
        parallel_channels=False, max_workers=1, include_kshell=True,
        include_barkas_dcs=True, charge_mode="bare", merge_energy_patches=False)
    for channel in ("excitation", "ionisation"):
        path = next(tmp_path.glob(f"sigmadiff_{channel}_*.dat"))
        meta = json.loads(path.read_text().splitlines()[0].split(": ", 1)[1])
        assert meta["barkas_model"] == sb.MODEL
        assert meta["barkas_charge_state"] == charge
        assert meta["projectile_element"] == element
        assert meta["projectile_kernel"] == (gen.RPWBA_MODEL_NAME if relativistic else "pwba")
        dcs = np.loadtxt(path)
        total = np.loadtxt(next(tmp_path.glob(f"sigma_{channel}_*.dat")))
        for i, energy in enumerate(t):
            rows = dcs[dcs[:, 0] == energy]
            np.testing.assert_allclose(np.trapezoid(rows[:, 2:], rows[:, 1], axis=0),
                                       total[i, 1:], rtol=2e-14)
    diagnostic_rows = [dict(excitation_sigma_pwba=[0.]*len(s.excitations),
                            ionization_sigma_pwba=[0.]*len(s.ionizations)) for _ in t]
    sigma_list = gen._replace_sigma_list_with_dcs_totals(t, diagnostic_rows, data)
    cache = tmp_path/"cache.npz"
    gen.save_cross_section_corrections_npz(t, sigma_list, out_path=cache, NE=12,
        Nq=128, dcs_data=data, ice_label="amorphous", include_barkas_dcs=True)
    with np.load(cache) as saved:
        assert gen._npz_matches_params(saved, NE=12, Nq=128, T_list=t, include_barkas_dcs=True)
        assert "DCS_Barkas_m2_per_eV" in saved
        assert "barkas_quadrature_error_m2_per_eV" in saved
        assert gen._dcs_data_from_npz(saved)["barkas_model_metadata"] == sb.metadata(load_density(element, charge))
        wrong = dict(saved)
    wrong["barkas_model"] = "stale"
    assert not gen._npz_matches_params(wrong, NE=12, Nq=128, T_list=t, include_barkas_dcs=True)


def test_unconverged_kernel_and_uncontrolled_correction_fail(monkeypatch):
    count = [0]
    def inconsistent(w, *args, **kwargs):
        count[0] += 1
        return np.full_like(w, float(count[0]))
    monkeypatch.setattr(sb, "oscillator_kernel", inconsistent)
    with pytest.raises(RuntimeError, match="did not converge"):
        sb.converged_kernel(np.array([10.]), .1, 1.01, load_density("H", 0))
    monkeypatch.setattr(sb, "dcs_m2_per_eV", lambda *a, **k: (np.ones(2), np.zeros(2)))
    data = dict(T_line=np.array([1e6]*2), E_line=np.array([10., 20.]),
                exc_vals=np.full((2, 1), 1e-20), ion_vals=np.full((2, 1), 1e-20))
    with pytest.raises(RuntimeError, match="uncontrolled"):
        bd.apply_barkas_correction_to_dcs_data(data, gen.model.epsilon_optical("amorphous"),
            "amorphous", 1837., 1, include_barkas_dcs=True,
            born_reference_charge="bare_Z", projectile_density=load_density("H", 0), workers=1)


@pytest.mark.parametrize("element,charge", STATES)
@pytest.mark.parametrize("relativistic", [False, True])
def test_corrected_cache_export_preserves_totals_without_second_correction(
        tmp_path, monkeypatch, element, charge, relativistic):
    """Synthetic cache/interface regression, not a physical DCS benchmark."""
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    gen._set_projectile_relativistic_dcs(relativistic)
    monkeypatch.setattr(gen, "_export_to_custom_geant4", lambda paths: None)
    def forbidden_reapplication(*args, **kwargs):
        raise AssertionError("Corrected cached DCS must not receive the correction twice")
    monkeypatch.setattr(bd, "apply_barkas_correction_to_dcs_data", forbidden_reapplication)
    s, c = gen.model.epsilon_optical("amorphous"), gen.model.default_dispersion_coefficients()
    state_meta = gen._projectile_state_metadata()
    model_meta = gen._barkas_generation_metadata(True)
    # Values are arbitrary, positive and nonconstant, with a known linear-W
    # integral. No synthetic result is reported as generated physical data.
    t, w = np.array([1e7]), np.array([10., 20., 30.])
    exc = np.arange(1., 3*len(s.excitations)+1).reshape(3, -1)*1e-22
    ion = np.arange(1., 3*(len(s.ionizations)+1)+1).reshape(3, -1)*1e-22
    cache = dict(dcs_T_line=np.repeat(t, 3), dcs_E_line=w, dcs_exc_vals=exc,
                 dcs_ion_vals=ion, charge_mode="bare", include_barkas_dcs=True,
                 born_reference_charge="bare_Z", **state_meta, **model_meta,
                 projectile_state_metadata_json=json.dumps(state_meta))
    loaded = gen._dcs_data_from_npz(cache)
    assert loaded is not None
    assert loaded["barkas_charge_applied"]
    gen.write_emfietzoglou_dcs_tables(s, c, out_dir=tmp_path, T_list=t,
        NE=12, Nq=128, dcs_data=loaded, include_kshell=True,
        include_barkas_dcs=True, charge_mode="bare", merge_energy_patches=False)
    for channel, expected in (("excitation", exc), ("ionisation", ion)):
        table = np.loadtxt(next(tmp_path.glob(f"sigmadiff_{channel}_*.dat")))
        np.testing.assert_array_equal(table[:, 2:], expected)
        # Same piecewise-linear area used by the C++ table loader's CDF.
        areas = .5*(table[1:, 2:]+table[:-1, 2:])*np.diff(table[:, 1])[:, None]
        cumulative = np.vstack([np.zeros(expected.shape[1]), np.cumsum(areas, axis=0)])
        assert np.all(np.isfinite(cumulative)) and np.all(np.diff(cumulative, axis=0) >= 0)
        total = np.loadtxt(next(tmp_path.glob(f"sigma_{channel}_*.dat")))
        np.testing.assert_allclose(cumulative[-1], total[1:], rtol=2e-14)
        np.testing.assert_allclose((cumulative/cumulative[-1])[-1], 1.)
