"""Fixed-state screening: atomic transforms, kernel placement, and table provenance."""
import json

import numpy as np
import pytest
from scipy.integrate import quad

from test_projectile_relativistic_dcs import MODULE as gen

ff = gen.projectile_ff
STATES = [(element, q) for element, z in ff.ELEMENTS.items() for q in range(z+1)]


def _optical_and_dispersion():
    return gen.model.epsilon_optical("amorphous"), gen.model.default_dispersion_coefficients()


@pytest.mark.parametrize("element,charge", STATES)
@pytest.mark.parametrize("inline", [False, True])
def test_charge_state_cli_is_not_scalar_charge(monkeypatch, element, charge, inline):
    option = [f"--charge-state={charge}"] if inline else ["--charge-state", str(charge)]
    monkeypatch.setattr(gen.sys, "argv", ["generator", *option])
    monkeypatch.delenv("ICE_EXPLICIT_CHARGE", raising=False)
    assert gen._explicit_charge_from_argv() is None
    gen.set_projectile(element)
    gen._set_projectile_charge_state(gen._argv_value(("--charge-state",)))
    assert gen.PROJECTILE_CHARGE_STATE == charge


@pytest.mark.parametrize("option", ["--explicit-charge", "--q-charge"])
def test_scalar_charge_cli_remains_positive_only(monkeypatch, option):
    monkeypatch.delenv("ICE_EXPLICIT_CHARGE", raising=False)
    monkeypatch.setattr(gen.sys, "argv", ["generator", option, "0"])
    with pytest.raises(ValueError, match="finite and positive"):
        gen._explicit_charge_from_argv()
    monkeypatch.setattr(gen.sys, "argv", ["generator", option, "1"])
    assert gen._explicit_charge_from_argv() == 1


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(gen, "ICE_TYPE", "amorphous")
    gen.set_projectile("proton")
    gen._set_projectile_relativistic_dcs(False)
    gen._set_kshell_model("hydrogenic-gos")
    yield
    gen.set_projectile("proton")
    gen._set_projectile_relativistic_dcs(False)


@pytest.mark.parametrize("element,charge", STATES)
def test_all_states_normalized_with_correct_limits(element, charge):
    density = ff.load_density(element, charge)
    assert density.moment(0) == pytest.approx(density.electrons, abs=1e-10)
    assert density.form_factor(0) == density.electrons
    assert density.charge_amplitude(0) == charge
    assert density.charge_amplitude(1e7) == pytest.approx(density.z, abs=1e-9)
    values = density.squared_charge(np.r_[0, np.geomspace(1e-12, 1e7, 200)])
    assert np.all(np.isfinite(values)) and np.all(values >= 0)
    if charge == 0:
        k = 1e-7
        assert density.squared_charge(k) == pytest.approx(k**4*(density.moment(2)/6)**2, rel=1e-10)
        assert density.squared_charge(1.0) > 0


@pytest.mark.parametrize("element,charge", STATES)
def test_radial_integral_and_independent_fourier_transform(element, charge):
    density = ff.load_density(element, charge)
    if not density.electrons:
        return
    # Oscillatory quadrature is independent of the analytic Gaussian transform.
    count = quad(lambda r: 4*np.pi*r*r*density.density(r), 0, 60, epsabs=2e-10, epsrel=1e-11)[0]
    assert count == pytest.approx(density.electrons, abs=3e-8)
    for k in (0.3, 3.0, 30.0):
        integral = quad(lambda r: 4*np.pi*r*density.density(r)/k,
                        0, 60, weight="sin", wvar=k, epsabs=1e-9, limit=200)[0]
        assert density.form_factor(k, direct=True) == pytest.approx(integral, abs=3e-8)


@pytest.mark.parametrize("element,z", list(ff.ELEMENTS.items()))
def test_hydrogenic_reference(element, z):
    density = ff.load_density(element, z-1)
    k = np.geomspace(1e-5, 1000, 300)
    expected = (1+(k/(2*z))**2)**-2
    np.testing.assert_allclose(density.form_factor(k), expected, atol=3e-16)


@pytest.mark.parametrize("element,charge", STATES)
def test_interpolated_amplitude_and_basis_convergence(element, charge):
    density = ff.load_density(element, charge)
    k = np.geomspace(1.123e-6, 9.876e5, 611)
    expected = charge + density.electron_deficit(k, direct=True)
    np.testing.assert_allclose(density.charge_amplitude(k), expected, rtol=2e-7)
    if density.electrons > 1:
        assert density.record["basis_comparison"]["max_squared_charge_relative_difference"] < 0.005
        assert density.record["ao_radial_export_max_scaled_error"] < 1e-10


def test_invalid_states_and_timelike_transfer_fail():
    for q in (-1, 7, 2.5, np.nan):
        with pytest.raises(ValueError):
            ff.load_density("C", q)
    with pytest.raises(ValueError, match="Unsupported"):
        ff.load_density("Fe", 1)
    with pytest.raises(ValueError, match="spacelike"):
        ff.screening_momentum(0.01, 1000, relativistic=True)
    with pytest.raises(ValueError):
        ff.load_density("H", 0).form_factor(-1)


def test_breit_momentum_matches_lorentz_rigid_cloud_limit():
    beta = 0.6
    parallel = np.array([0.1, 1, 10])
    perpendicular = 0.7
    k = np.hypot(parallel, perpendicular)
    loss = ff.C_AU*ff.EH*beta*parallel
    actual = ff.screening_momentum(k, loss, relativistic=True)
    np.testing.assert_allclose(actual, np.sqrt(perpendicular**2+(1-beta**2)*parallel**2), rtol=1e-14)
    np.testing.assert_array_equal(ff.screening_momentum(k, loss), k)


@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("element,charge", STATES)
def test_real_target_kernels_finite_all_charge_states(element, charge, relativistic):
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    gen._set_projectile_relativistic_dcs(relativistic)
    s, C = _optical_and_dispersion()
    T = gen.PROJECTILE_MASS_NUMBER*1e7
    values = [gen._selected_dsigma_excitation(15, T, 0, s, C, 160),
              gen._selected_dsigma_ionization(50, T, 0, s, C, 160),
              gen._selected_dsigma_kshell(1000, T, s, C, 160)]
    assert np.all(np.isfinite(values)) and np.all(np.asarray(values) > 0)


@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("element,z", list(ff.ELEMENTS.items()))
def test_explicit_bare_state_matches_old_default_exactly(element, z, relativistic):
    gen.set_projectile(element)
    gen._set_projectile_relativistic_dcs(relativistic)
    s, C = _optical_and_dispersion()
    T = gen.PROJECTILE_MASS_NUMBER*1e7
    old_mass = gen.PROJECTILE_MASS_AU
    old = gen._selected_dsigma_ionization(50, T, 0, s, C, 160)
    gen._set_projectile_charge_state(z)
    assert gen.PROJECTILE_MASS_AU == old_mass
    assert gen._selected_dsigma_ionization(50, T, 0, s, C, 160) == old


def test_screening_is_inside_integral_and_forbids_double_scaling(monkeypatch):
    gen.set_projectile("C")
    gen._set_projectile_charge_state(3)
    s, C = _optical_and_dispersion()
    seen = []
    original = gen._projectile_screening_ratio
    def record(k, w, **kwargs):
        values = original(k, w, **kwargs)
        seen.append(values)
        return values
    monkeypatch.setattr(gen, "_projectile_screening_ratio", record)
    for relativistic in (False, True):
        gen._set_projectile_relativistic_dcs(relativistic)
        gen._selected_dsigma_ionization(50, 1e8, 0, s, C, 160)
        gen._selected_dsigma_kshell(1000, 1e8, s, C, 160)
    assert len(seen) == 4
    assert all(np.ptp(ratio) > 0.01 for ratio in seen)
    for mode in ("zeff", "explicit"):
        with pytest.raises(ValueError, match="scalar"):
            gen._validate_projectile_state_options(mode, False)
    gen._validate_projectile_state_options("bare", True)
    assert gen._barkas_generation_metadata(True)["barkas_charge_state"] == 3


@pytest.mark.parametrize("relativistic", [False, True])
def test_real_export_tcs_and_state_provenance(tmp_path, monkeypatch, relativistic):
    gen.set_projectile("H")
    gen._set_projectile_charge_state(0)
    gen._set_projectile_relativistic_dcs(relativistic)
    monkeypatch.setattr(gen, "_export_to_custom_geant4", lambda paths: None)
    s, C = _optical_and_dispersion()
    T = np.array([1e6, 1e7])
    data = gen.write_emfietzoglou_dcs_tables(
        s, C, out_dir=tmp_path, T_list=T, NE=32, Nq=80, return_data=True,
        parallel_channels=False, include_kshell=True, include_barkas_dcs=False,
        charge_mode="bare", merge_energy_patches=False)
    metadata = gen._projectile_state_metadata()
    assert data["projectile_state_metadata"] == metadata
    for channel in ("excitation", "ionisation"):
        path = next(tmp_path.glob(f"sigmadiff_{channel}_*.dat"))
        assert "q0_frozen_hf" in path.name
        saved = json.loads(path.read_text().splitlines()[0].split(": ", 1)[1])
        assert all(saved[k] == v for k, v in metadata.items())
        dcs = np.loadtxt(path)
        total = np.loadtxt(next(tmp_path.glob(f"sigma_{channel}_*.dat")))
        for i, t in enumerate(T):
            rows = dcs[dcs[:, 0] == t]
            np.testing.assert_allclose(np.trapezoid(rows[:, 2:], rows[:, 1], axis=0), total[i, 1:], rtol=2e-14)
        assert np.all(np.isfinite(dcs)) and np.all(dcs[:, 2:] >= 0)
    # Metadata must survive loading under a different current selection.
    archive = {"dcs_T_line": data["T_line"], "dcs_E_line": data["E_line"],
               "dcs_exc_vals": data["exc_vals"], "dcs_ion_vals": data["ion_vals"],
               "projectile_state_metadata_json": json.dumps(metadata), **metadata}
    diagnostic_rows = [dict(excitation_sigma_pwba=[0.0]*len(s.excitations),
                            ionization_sigma_pwba=[0.0]*len(s.ionizations)) for _ in T]
    sigma_list = gen._replace_sigma_list_with_dcs_totals(T, diagnostic_rows, data)
    cache_path = tmp_path/"cache.npz"
    gen.save_cross_section_corrections_npz(T, sigma_list, out_path=cache_path,
                                         NE=32, Nq=80, dcs_data=data, ice_label="amorphous")
    with np.load(cache_path) as saved_cache:
        assert gen._npz_matches_params(saved_cache, NE=32, Nq=80, T_list=T)
        assert gen._dcs_data_from_npz(saved_cache)["projectile_state_metadata"] == metadata
    gen.set_projectile("proton")
    assert gen._dcs_data_from_npz(archive)["projectile_state_metadata"] == metadata
    with pytest.raises(ValueError, match="provenance"):
        gen.write_emfietzoglou_dcs_tables(s, C, out_dir=tmp_path, dcs_data=data, charge_mode="bare", include_barkas_dcs=False)


def test_cache_rejects_different_charge_or_data_hash():
    gen.set_projectile("C")
    gen._set_projectile_charge_state(3)
    meta = gen._projectile_state_metadata()
    wrong = dict(meta, projectile_charge_state=2)
    assert not gen._npz_matches_params(wrong, NE=100, Nq=100, T_list=[])
    wrong = dict(meta, projectile_atomic_data_sha256="wrong")
    assert not gen._npz_matches_params(wrong, NE=100, Nq=100, T_list=[])
    gen.set_projectile("proton")
    assert not gen._npz_matches_params(meta, NE=100, Nq=100, T_list=[])


def test_missing_or_corrupt_atomic_data_fails(monkeypatch, tmp_path):
    ff.load_density.cache_clear()
    ff._atomic_data.cache_clear()
    try:
        monkeypatch.setattr(ff, "DATA_PATH", tmp_path/"missing.json")
        with pytest.raises(FileNotFoundError, match="Missing"):
            ff.load_density("C", 3)
        monkeypatch.setattr(ff, "_atomic_data", lambda: ({"states": {}}, "missing"))
        with pytest.raises(ValueError, match="no effective-charge fallback"):
            ff.load_density("C", 3)
    finally:
        ff.load_density.cache_clear()


def test_incomplete_cache_provenance_fails():
    with pytest.raises(ValueError, match="Incomplete"):
        gen._projectile_state_from_npz({"projectile_form_factor_model": ff.VERSION})
    meta = ff.density_metadata(ff.load_density("C", 3))
    archive = dict(meta, projectile_state_metadata_json=json.dumps(meta), projectile_charge_state=4)
    with pytest.raises(ValueError, match="Inconsistent"):
        gen._projectile_state_from_npz(archive)


def test_spawn_workers_preserve_multielectron_state(tmp_path, monkeypatch):
    import importlib
    # Use an importable module name for multiprocessing's spawn protocol.
    worker_gen = importlib.import_module("physics.inelastic_dielectric.generate_cross_sections")
    worker_gen.set_projectile("C")
    worker_gen._set_projectile_charge_state(3)
    worker_gen._set_projectile_relativistic_dcs(True)
    monkeypatch.setattr(worker_gen, "_export_to_custom_geant4", lambda paths: None)
    s, C = _optical_and_dispersion()
    try:
        kwargs = dict(T_list=[1e7], NE=20, Nq=40, return_data=True,
                      a_vec=C.a_fj, b_vec=C.b_fj, c_vec=C.c_fj,
                      ice_type="amorphous", include_kshell=True,
                      charge_mode="bare", include_barkas_dcs=False,
                      merge_energy_patches=False)
        serial = worker_gen.write_emfietzoglou_dcs_tables(s, C, out_dir=tmp_path/"serial", parallel_channels=False, **kwargs)
        parallel = worker_gen.write_emfietzoglou_dcs_tables(s, C, out_dir=tmp_path/"parallel", parallel_channels=True, max_workers=2, **kwargs)
        for key in ("T_line", "E_line", "exc_vals", "ion_vals"):
            np.testing.assert_array_equal(parallel[key], serial[key])
        assert parallel["projectile_state_metadata"] == serial["projectile_state_metadata"]
    finally:
        worker_gen.set_projectile("proton")
        worker_gen._set_projectile_relativistic_dcs(False)
