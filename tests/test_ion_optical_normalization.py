"""Optical sum-rule and table provenance checks; no stopping-curve fitting."""

from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys

import numpy as np
import pytest
from scipy.constants import elementary_charge, electron_mass, epsilon_0, hbar

from physics.inelastic_dielectric import generate_cross_sections as gen


@pytest.fixture(autouse=True)
def reset_modes(monkeypatch):
    gen.set_projectile("proton")
    gen._set_kshell_model("hydrogenic-gos")
    gen._set_projectile_relativistic_dcs(False, False, use_density_effect=False)
    monkeypatch.setattr(gen, "ICE_TYPE", "amorphous")
    monkeypatch.setattr(gen, "_export_to_custom_geant4", lambda paths: None)
    yield
    gen.set_projectile("proton")
    gen._set_kshell_model("hydrogenic-gos")
    gen._set_projectile_relativistic_dcs(False, False, use_density_effect=False)


def dispersion():
    return gen.model.DispersionCoeffs(
        [3.82, 2.47, 2.47, 3.01, 2.44],
        [0.0272, 0.0295, 0.0311, 0.0111, 0.0633],
        [0.098, 0.075, 0.074, 0.765, 0.425],
    )


@pytest.mark.parametrize("phase,rho", [
    ("amorphous", 0.9343471678603292), ("hexagonal", 0.9335),
])
def test_single_material_density_and_plasma_energy(phase, rho):
    data = gen._ion_normalization_metadata(phase)
    number = rho * 1e6 * gen.AVOGADRO / gen.H2O_MOLAR_MASS_G_MOL
    plasma = hbar / elementary_charge * np.sqrt(
        10 * number * elementary_charge**2 / (electron_mass * epsilon_0)
    )
    assert data["material_density_g_cm3"] == rho
    assert data["material_molecular_density_m3"] == pytest.approx(number, rel=1e-14)
    assert data["physical_plasma_energy_eV"] == pytest.approx(plasma, rel=1e-14)
    assert data["density_scale_factor"] == 1.0
    assert data["material_density_applied_to_cross_sections"] is False
    assert not hasattr(gen, "N")
    assert not hasattr(gen, "REL_MC_TARGET_DENSITY_G_CM3")


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_independent_full_optical_sum_and_convergence(phase):
    s = gen.model.epsilon_optical(phase)
    sums = []
    for upper, points in [(1e8, 120001), (1e9, 240001)]:
        W = np.geomspace(s.Bmin, upper, points)
        q_zero = np.array([0.0])
        e1 = gen._ion_epsilon1_valence(W, q_zero, s, dispersion())
        e2 = gen._ion_epsilon2_valence(W, q_zero, s, dispersion())
        denominator = np.maximum(
            e1["total"][0] ** 2 + e2["total"][0] ** 2,
            np.finfo(float).tiny,
        )
        valence = e2["total"][0] / denominator
        # Integrate the actual hydrogenic curve independently of its analytic
        # normalization target, on a grid starting immediately above its edge.
        K_W = np.geomspace(gen.KSHELL_B_EV + 1e-6, upper, points)
        core = gen.model.oxygen_K_ion_hydrogenic_gos_elf(K_W, 0.0, Ep_eV=s.Ep)
        moment = np.trapezoid(W * valence, W) + np.trapezoid(K_W * core, K_W)
        sums.append(moment * gen._ion_elf_per_molecule_factor(s) / gen.OPTICAL_SUM_UNIT_EV2_M3)
        rolled_moment = np.trapezoid(W * valence * gen._elf_rolloff_factor(W), W)
        rolled_moment += np.trapezoid(K_W * core, K_W)
        assert rolled_moment * gen._ion_elf_per_molecule_factor(s) / gen.OPTICAL_SUM_UNIT_EV2_M3 == pytest.approx(10.0, rel=1e-5)
    assert sums == pytest.approx([10.0, 10.0], rel=2e-6)
    assert sums[1] == pytest.approx(sums[0], rel=2e-6)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_no_kshell_does_not_renormalize_valence_or_change_barkas(phase):
    s = gen.model.epsilon_optical(phase)
    factor = gen._ion_elf_per_molecule_factor(s)
    gen._set_kshell_model("none")
    assert gen._ion_elf_per_molecule_factor(s) == factor
    gen._set_kshell_model("old-optical")
    assert gen._ion_elf_per_molecule_factor(s) == factor
    oos = gen.barkas_dcs.oos_density(np.geomspace(7.0, 1e6, 1000), s, phase)
    assert oos.valence_integral_raw * oos.valence_norm == pytest.approx(8.0)
    assert oos.ok_integral_raw * oos.ok_norm == pytest.approx(2.0)
    assert gen.KSHELL_B_EV == 543.4
    assert gen.KSHELL_ZEFF == 7.7
    assert gen.HYDROGENIC_KSHELL_ROLLOFF_APPLIED is False


def test_density_enters_macroscopic_rate_once(monkeypatch):
    s = gen.model.epsilon_optical("amorphous")
    factor = gen._ion_elf_per_molecule_factor(s)
    before = gen._ion_normalization_metadata("amorphous")
    monkeypatch.setattr(gen, "ICE_AMORPHOUS_DENSITY_G_CM3", 2 * gen.ICE_AMORPHOUS_DENSITY_G_CM3)
    after = gen._ion_normalization_metadata("amorphous")
    assert gen._ion_elf_per_molecule_factor(s) == pytest.approx(factor, rel=1e-14)
    assert after["material_molecular_density_m3"] / before["material_molecular_density_m3"] == 2.0


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("projectile", tuple(gen.PROJECTILE_LIBRARY))
def test_all_born_kernels_get_same_normalization(phase, relativistic, projectile, monkeypatch):
    gen.set_projectile(projectile)
    gen._set_projectile_relativistic_dcs(relativistic)
    s = gen.model.epsilon_optical(phase)
    C = dispersion()
    T = gen.PROJECTILE_MASS_NUMBER * 1e7
    def values():
        return np.array([
            gen._selected_dsigma_excitation(20.0, T, 0, s, C, 24),
            gen._selected_dsigma_ionization(100.0, T, 0, s, C, 24),
            gen._selected_dsigma_kshell(1000.0, T, s, C, 24),
        ])
    current = values()
    expected_ratio = 3.34e28 * gen._ion_elf_per_molecule_factor(s)
    monkeypatch.setattr(gen, "_ion_elf_per_molecule_factor", lambda model: 1.0 / 3.34e28)
    legacy = values()
    assert np.all(np.isfinite(current)) and np.all(current > 0.0)
    np.testing.assert_allclose(current, legacy * expected_ratio, rtol=1e-13, atol=0.0)


def test_wrong_or_missing_phase_fails():
    s = gen.model.epsilon_optical("amorphous")
    with pytest.raises(ValueError, match="phase differ"):
        gen._ion_normalization_metadata("hexagonal", s)
    with pytest.raises(ValueError, match="Unsupported ice phase"):
        gen._ion_elf_per_molecule_factor(replace(s, material=None))


def test_legacy_cache_rejected_and_current_cache_roundtrips(tmp_path):
    path = tmp_path / "cache.npz"
    gen.save_cross_section_corrections_npz([1e6], [{}], path, NE=10, Nq=12)
    cached = gen.load_cross_section_corrections_npz(path, NE=10, Nq=12)
    assert cached is not None
    with np.load(path) as data:
        assert gen._npz_matches_params(data, 10, 12)
        legacy = dict(data)
    legacy.pop("ion_normalization_version")
    assert not gen._npz_matches_params(legacy, 10, 12)
    bad = dict(legacy, ion_normalization_version=gen.ION_NORMALIZATION_VERSION)
    bad["density_scale_factor"] = 1.01
    assert not gen._npz_matches_params(bad, 10, 12)
    bad["density_scale_factor"] = 1.0
    bad["material_density_g_cm3"] = 1.0
    assert not gen._npz_matches_params(bad, 10, 12)


def small_dcs(s):
    T = np.repeat([1.1e5, 1e6, 1e8], 7)
    W = np.tile([7.1, 20.0, 80.0, 200.0, 550.0, 1000.0, 2000.0], 3)
    C = dispersion()
    args = (s, C, 20, T, W, [s.Bmin] * 5, [o.Bth for o in s.ionizations], gen.KSHELL_B_EV)
    excitation = [gen._compute_dcs_channel_values("excitation", j, *args) for j in range(5)]
    ionization = [gen._compute_dcs_channel_values("ionization", j, *args) for j in range(4)]
    ionization.append(gen._compute_dcs_channel_values("kshell", 0, *args))
    return dict(T_line=T, E_line=W, exc_vals=np.array(excitation).T, ion_vals=np.array(ionization).T)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_barkas_addition_is_not_density_scaled(phase):
    s = gen.model.epsilon_optical(phase)
    current = small_dcs(s)
    ratio = 3.34e28 * gen._ion_elf_per_molecule_factor(s)
    legacy = dict(current, exc_vals=current["exc_vals"] / ratio, ion_vals=current["ion_vals"] / ratio)
    diagnostics = []
    for data in (current, legacy):
        _, diag = gen.barkas_dcs.apply_barkas_correction_to_dcs_data(
            data, s, material=phase, projectile_mass_me=gen.PROJECTILE_MASS_AU,
            nuclear_charge=1.0, charge_mode="bare", include_barkas_dcs=True,
            include_kshell=True, dcs_scale_m2=gen.EMFI_DCS_SCALE_M2,
            born_reference_charge="bare_Z",
        )
        diagnostics.append(diag)
    np.testing.assert_allclose(
        diagnostics[0].DCS_Barkas_m2_per_eV,
        diagnostics[1].DCS_Barkas_m2_per_eV, rtol=1e-14, atol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics[0].DCS_Born_m2_per_eV,
        diagnostics[1].DCS_Born_m2_per_eV * ratio, rtol=1e-14, atol=0.0,
    )


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_ten_worker_generation_matches_serial(phase):
    s = gen.model.epsilon_optical(phase)
    serial = small_dcs(s)
    C = dispersion()
    tasks = [("excitation", j) for j in range(5)]
    tasks += [("ionization", j) for j in range(4)] + [("kshell", 0)]
    tasks = [(channel, index, 0, len(serial["T_line"])) for channel, index in tasks]
    with ProcessPoolExecutor(
        max_workers=10, initializer=gen._init_dcs_worker,
        initargs=(C.a_fj, C.b_fj, C.c_fj, serial["T_line"], serial["E_line"],
                  20, False, False, False, False, phase, "proton",
                  "hydrogenic-gos", False, False, False),
    ) as pool:
        results = list(pool.map(gen._compute_dcs_channel_worker, tasks))
    for channel, index, values in results:
        expected = serial["exc_vals"][:, index] if channel == "excitation" else serial["ion_vals"][:, -1 if channel == "kshell" else index]
        np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("barkas", [False, True])
def test_generated_dat_totals_and_safe_patch_merge(tmp_path, phase, relativistic, barkas):
    gen._set_projectile_relativistic_dcs(relativistic)
    s = gen.model.epsilon_optical(phase)
    dcs = small_dcs(s)
    gen.write_emfietzoglou_dcs_tables(
        s, dispersion(), out_dir=tmp_path, dcs_data=dcs, ice_label=phase,
        ice_type=phase, charge_mode="bare", include_barkas_dcs=barkas,
        T_list=[1.1e5, 1e6, 1e8], merge_energy_patches=False,
    )
    for channel in ("excitation", "ionisation"):
        path = next(tmp_path.glob(f"sigmadiff_{channel}_*.dat"))
        total_path = next(tmp_path.glob(f"sigma_{channel}_*.dat"))
        metadata = json.loads(path.read_text().splitlines()[0].split(": ", 1)[1])
        gen._require_current_normalization(metadata, phase)
        assert metadata["include_barkas_dcs"] == barkas
        data = np.loadtxt(path)
        total = np.loadtxt(total_path)
        assert np.all(np.isfinite(data)) and np.all(data[:, 2:] >= 0.0)
        T, values = gen._integrate_dcs_to_totals(data[:, 0], data[:, 1], data[:, 2:])
        np.testing.assert_allclose(T, total[:, 0], rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(values, total[:, 1:], rtol=2e-9, atol=0.0)
    exc = next(tmp_path.glob("sigmadiff_excitation_*.dat"))
    ion = next(tmp_path.glob("sigmadiff_ionisation_*.dat"))
    existing = gen._load_dcs_pair(exc, ion)
    patch = gen._slice_dcs_data(existing, t_min=1e6, t_max=1e6)
    merged = gen._prepare_dcs_data_for_output(patch, exc, ion, table_metadata=metadata)
    np.testing.assert_array_equal(merged["exc_vals"], existing["exc_vals"])
    with pytest.raises(ValueError, match="Incompatible"):
        gen._prepare_dcs_data_for_output(patch, exc, ion, table_metadata=dict(metadata, include_barkas_dcs=not barkas))
    exc.write_text("1e6 20 1 1 1 1 1\n")
    with pytest.raises(ValueError, match="Legacy/unversioned"):
        gen._prepare_dcs_data_for_output(patch, exc, ion, table_metadata=metadata)
