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
    MODULE._set_projectile_relativistic_dcs(False, False)
    yield
    MODULE.set_projectile("proton")
    MODULE._set_projectile_relativistic_dcs(False, False)


def _optical_and_dispersion():
    optical = MODULE.model.epsilon_optical("amorphous")
    dispersion = MODULE.model.DispersionCoeffs(
        a_fj=np.array([3.82, 2.47, 2.47, 3.01, 2.44]),
        b_fj=np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633]),
        c_fj=np.array([0.098, 0.075, 0.074, 0.765, 0.425]),
    )
    return optical, dispersion


def test_relativistic_mode_is_optional_and_separately_tagged() -> None:
    assert MODULE._projectile_kernel_tag() == ""
    MODULE._set_projectile_relativistic_dcs(True, False)
    assert MODULE._projectile_kernel_tag() == "_relproj"
    MODULE._set_projectile_relativistic_dcs(True, True)
    assert MODULE._projectile_kernel_tag() == "_relproj_transverse"
    with pytest.raises(ValueError, match="requires"):
        MODULE._set_projectile_relativistic_dcs(False, True)


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
    reduced = []
    for projectile in PROJECTILES:
        MODULE.set_projectile(projectile)
        kinetic_eV = (gamma - 1.0) * MODULE.projectile_rest_energy_eV()
        prefactor = MODULE._projectile_relativistic_longitudinal_prefactor(kinetic_eV)
        reduced.append(prefactor / MODULE.PROJECTILE_CHARGE**2)
    assert np.asarray(reduced) == pytest.approx(reduced[0], rel=2.0e-14)


def test_relativistic_lower_q_bound_is_stable_for_heaviest_supported_ion() -> None:
    MODULE.set_projectile("sulfur")
    incident_eV = MODULE.PROJECTILE_MASS_NUMBER * 1.0e8
    qlo, qhi = MODULE._q_bounds_scalar_rel(10.0, incident_eV)
    assert np.isfinite(qlo) and qlo > 0.0
    assert np.isfinite(qhi) and qhi > qlo


@pytest.mark.parametrize("projectile", PROJECTILES)
def test_relativistic_kernels_are_finite_for_every_supported_ion(projectile: str) -> None:
    MODULE.set_projectile(projectile)
    MODULE._set_projectile_relativistic_dcs(True, True)
    optical, dispersion = _optical_and_dispersion()
    incident_eV = MODULE.PROJECTILE_MASS_NUMBER * 1.0e8
    longitudinal = MODULE._integrate_channel_single_E_rel(
        100.0, incident_eV, 0, "ionization", optical, dispersion, Nq=40
    )
    transverse = MODULE._integrate_channel_single_E_trans(
        100.0, incident_eV, 0, "ionization", optical, dispersion
    )
    k_shell = MODULE._integrate_kshell_single_E_rel(
        600.0, incident_eV, optical, Nq=40, include_kshell=True
    )
    values = np.asarray([longitudinal, transverse, k_shell])
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


def test_selected_relativistic_kernel_adds_transverse_only_when_enabled() -> None:
    optical, dispersion = _optical_and_dispersion()
    MODULE._set_projectile_relativistic_dcs(True, False)
    longitudinal = MODULE._selected_dsigma_ionization(
        100.0, 1.0e8, 0, optical, dispersion, 60
    )
    MODULE._set_projectile_relativistic_dcs(True, True)
    combined = MODULE._selected_dsigma_ionization(
        100.0, 1.0e8, 0, optical, dispersion, 60
    )
    transverse = MODULE._integrate_channel_single_E_trans(
        100.0, 1.0e8, 0, "ionization", optical, dispersion
    )
    assert transverse > 0.0
    assert combined == pytest.approx(longitudinal + transverse, rel=1.0e-14)
