from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "generate_ice_cross_sections_ion.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_ice_cross_sections_ion_rel_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PROJECTILES = tuple(MODULE.PROJECTILE_LIBRARY)


@pytest.fixture(autouse=True)
def reset_projectile_kernel() -> None:
    MODULE.set_projectile("proton")
    MODULE._set_kshell_model("hydrogenic-gos")
    MODULE._set_projectile_relativistic_dcs(False, False, use_density_effect=False)
    yield
    MODULE.set_projectile("proton")
    MODULE._set_projectile_relativistic_dcs(False, False, use_density_effect=False)


def _optical_and_dispersion():
    optical = MODULE.model.epsilon_optical("amorphous")
    dispersion = MODULE.model.DispersionCoeffs(
        a_fj=np.array([3.82, 2.47, 2.47, 3.01, 2.44]),
        b_fj=np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633]),
        c_fj=np.array([0.098, 0.075, 0.074, 0.765, 0.425]),
    )
    return optical, dispersion


def test_rpwba_mode_is_optional_complete_and_separately_tagged() -> None:
    assert MODULE._projectile_kernel_tag() == ""
    MODULE._set_projectile_relativistic_dcs(True)
    assert MODULE.INCLUDE_TRANSVERSE_DCS
    assert MODULE.RPWBA_DENSITY_EFFECT
    assert MODULE._projectile_kernel_tag() == "_rpwba_dm2022"
    MODULE._set_projectile_relativistic_dcs(True, True, use_density_effect=False)
    assert MODULE._projectile_kernel_tag() == "_rpwba_dm2022_no_density"
    with pytest.raises(ValueError, match="requires"):
        MODULE._set_projectile_relativistic_dcs(False, True)
    with pytest.raises(ValueError, match="requires both"):
        MODULE._set_projectile_relativistic_dcs(True, False)


def test_rpwba_reference_is_dominguez_munoz_2022() -> None:
    assert MODULE.RPWBA_MODEL_NAME == "dominguez-munoz-2022-finite-Q"
    assert MODULE.RPWBA_REFERENCE_DOI == "10.1016/j.radphyschem.2022.110363"


@pytest.mark.parametrize("projectile", PROJECTILES)
def test_per_u_energy_conversion_uses_each_mass_number(projectile: str) -> None:
    MODULE.set_projectile(projectile)
    emin, emax = MODULE._energy_range_to_total_eV(1.0e5, 1.0e6, "per_u")
    assert emin == pytest.approx(1.0e5 * MODULE.PROJECTILE_MASS_NUMBER)
    assert emax == pytest.approx(1.0e6 * MODULE.PROJECTILE_MASS_NUMBER)


@pytest.mark.parametrize("projectile", PROJECTILES)
def test_exact_q_bounds_recover_each_ions_nonrelativistic_limit(projectile: str) -> None:
    MODULE.set_projectile(projectile)
    incident_eV = MODULE.PROJECTILE_MASS_NUMBER * 1.0e6
    loss_eV = 100.0
    nonrel = MODULE._q_bounds_scalar(loss_eV, incident_eV)
    relativistic = MODULE._q_bounds_scalar_rel(loss_eV, incident_eV)
    assert relativistic[0] == pytest.approx(nonrel[0], rel=3.0e-3)
    assert relativistic[1] == pytest.approx(nonrel[1], rel=3.0e-3)


def test_longitudinal_prefactor_uses_selected_ions_z_squared_at_fixed_beta() -> None:
    gamma = 1.02
    optical, _ = _optical_and_dispersion()
    reduced = []
    for projectile in PROJECTILES:
        MODULE.set_projectile(projectile)
        kinetic_eV = (gamma - 1.0) * MODULE.projectile_rest_energy_eV()
        prefactor = MODULE._projectile_relativistic_longitudinal_prefactor(kinetic_eV, optical)
        reduced.append(prefactor / MODULE.PROJECTILE_CHARGE**2)
    assert np.asarray(reduced) == pytest.approx(reduced[0], rel=2.0e-14)


def test_finite_q_transverse_ratio_matches_dominguez_munoz_eq3() -> None:
    W_eV = 100.0
    beta2 = 0.2
    q_au = np.array([0.2, 0.5, 1.0, 2.0])
    recoil_product = (MODULE.C_AU * MODULE.EH * q_au) ** 2
    longitudinal_factor = 2.0 * MODULE.MC2_eV / (W_eV * recoil_product)
    transverse_factor = (
        2.0
        * MODULE.MC2_eV
        * W_eV
        / (recoil_product - W_eV**2) ** 2
        * (beta2 - W_eV**2 / recoil_product)
    )
    expected = transverse_factor / longitudinal_factor
    actual = MODULE._rpwba_transverse_ratio(W_eV, q_au, beta2)
    assert actual == pytest.approx(expected, rel=3.0e-15)


def test_density_corrected_ratio_matches_thesis_eq_2_287() -> None:
    W_eV = 100.0
    beta2 = 0.2
    q_au = np.array([0.2, 0.5, 1.0, 2.0])
    epsilon1 = np.array([1.8, 1.4, 1.1, 1.02])
    epsilon2 = np.array([0.8, 0.4, 0.15, 0.03])
    recoil_product = (MODULE.C_AU * MODULE.EH * q_au) ** 2
    epsilon = epsilon1 + 1j * epsilon2
    medium_transverse_factor = (
        2.0 * MODULE.MC2_eV * W_eV * np.abs(epsilon)**2
        / np.abs(recoil_product - W_eV**2 * epsilon)**2
        * (beta2 - W_eV**2 / recoil_product)
    )
    longitudinal_factor = 2.0 * MODULE.MC2_eV / (W_eV * recoil_product)
    expected = medium_transverse_factor / longitudinal_factor
    actual = MODULE._rpwba_transverse_ratio(
        W_eV,
        q_au,
        beta2,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        use_density_effect=True,
    )
    assert actual == pytest.approx(expected, rel=3.0e-14)


@pytest.mark.parametrize("W", [10.0, 100.0, 1000.0])
@pytest.mark.parametrize("beta2", [0.001, 0.2, 0.9])
def test_density_ratio_recovers_dilute_limit(W, beta2):
    qmin = W / (MODULE.C_AU * MODULE.EH * np.sqrt(beta2))
    q = qmin * np.geomspace(1.0, 1000.0, 100)
    vacuum = MODULE._rpwba_transverse_ratio(W, q, beta2)
    dilute = MODULE._rpwba_transverse_ratio(
        W, q, beta2, epsilon1=1.0, epsilon2=0.0, use_density_effect=True,
    )
    np.testing.assert_allclose(dilute, vacuum, rtol=2e-15, atol=0.0)


def test_relativistic_lower_q_bound_is_stable_for_heaviest_supported_ion() -> None:
    MODULE.set_projectile("sulfur")
    incident_eV = MODULE.PROJECTILE_MASS_NUMBER * 1.0e8
    qlo, qhi = MODULE._q_bounds_scalar_rel(10.0, incident_eV)
    assert np.isfinite(qlo) and qlo > 0.0
    assert np.isfinite(qhi) and qhi > qlo


@pytest.mark.parametrize("projectile", PROJECTILES)
def test_relativistic_kernels_are_finite_for_every_supported_ion(projectile: str) -> None:
    MODULE.set_projectile(projectile)
    MODULE._set_projectile_relativistic_dcs(True, True, use_density_effect=True)
    optical, dispersion = _optical_and_dispersion()
    incident_eV = MODULE.PROJECTILE_MASS_NUMBER * 1.0e8
    longitudinal, transverse = MODULE._integrate_channel_single_E_rpwba_components(
        100.0,
        incident_eV,
        0,
        "ionization",
        optical,
        dispersion,
        Nq=40,
        use_density_effect=True,
    )
    k_longitudinal, k_transverse = MODULE._integrate_kshell_single_E_rpwba_components(
        600.0,
        incident_eV,
        optical,
        dispersion,
        Nq=40,
        include_kshell=True,
        use_density_effect=True,
    )
    values = np.asarray([longitudinal, transverse, k_longitudinal, k_transverse])
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert longitudinal > 0.0


def test_default_kernel_remains_the_existing_nonrelativistic_pwba() -> None:
    optical, dispersion = _optical_and_dispersion()
    selected = MODULE._selected_dsigma_ionization(
        100.0, 1.0e8, 0, optical, dispersion, 60
    )
    baseline = MODULE._integrate_channel_single_E(
        100.0,
        1.0e8,
        0,
        "ionization",
        optical,
        dispersion,
        Nq=60,
        use_rel_bounds=False,
    )
    assert selected == pytest.approx(baseline, rel=1.0e-14)


def test_selected_rpwba_kernel_is_longitudinal_plus_finite_q_transverse() -> None:
    optical, dispersion = _optical_and_dispersion()
    MODULE._set_projectile_relativistic_dcs(True, True, use_density_effect=False)
    longitudinal, transverse = MODULE._integrate_channel_single_E_rpwba_components(
        100.0,
        1.0e8,
        0,
        "ionization",
        optical,
        dispersion,
        Nq=60,
        use_density_effect=False,
    )
    combined = MODULE._selected_dsigma_ionization(
        100.0, 1.0e8, 0, optical, dispersion, 60
    )
    assert transverse > 0.0
    assert combined == pytest.approx(longitudinal + transverse, rel=1.0e-14)


def test_density_effect_uses_finite_q_dielectric_and_changes_transverse_term() -> None:
    optical, dispersion = _optical_and_dispersion()
    T_eV = 300.0e6
    vacuum = MODULE._integrate_channel_single_E_trans(
        30.0,
        T_eV,
        0,
        "ionization",
        optical,
        dispersion,
        Nq=80,
        use_density_effect=False,
    )
    dense = MODULE._integrate_channel_single_E_trans(
        30.0,
        T_eV,
        0,
        "ionization",
        optical,
        dispersion,
        Nq=80,
        use_density_effect=True,
    )
    assert np.isfinite(vacuum) and vacuum >= 0.0
    assert np.isfinite(dense) and dense >= 0.0
    assert not np.isclose(dense, vacuum, rtol=1.0e-6, atol=0.0)
