from __future__ import annotations

import math

import pytest

from python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl.kernel import (
    PROJECTILES,
    TARGETS,
    cos_theta_cm,
    maximum_recoil_energy_ev,
    pair_cross_section_angstrom2,
    screening,
    screening_length_angstrom,
)


def test_universal_screening_constants() -> None:
    assert screening(0.0) == pytest.approx(1.00007, rel=0.0, abs=1.0e-12)
    assert screening(1.0) == pytest.approx(0.4164406620, rel=1.0e-9)
    assert screening(10.0) > 0.0
    assert screening(10.0) < screening(1.0)


def test_screening_length_uses_nuclear_atomic_number() -> None:
    carbon_oxygen = screening_length_angstrom(6, 8)
    assert carbon_oxygen == pytest.approx(0.1500033140, rel=1.0e-8)
    assert screening_length_angstrom(6, 8) == carbon_oxygen


def test_exact_proton_oxygen_maximum_recoil() -> None:
    proton = PROJECTILES["H"]
    oxygen = TARGETS["O"]
    recoil = maximum_recoil_energy_ev(
        proton.mass_ev, oxygen.mass_ev, 1.0e6
    )
    nonrelativistic = (
        4.0
        * proton.mass_ev
        * oxygen.mass_ev
        / (proton.mass_ev + oxygen.mass_ev) ** 2
        * 1.0e6
    )
    assert recoil > 0.0
    assert recoil == pytest.approx(nonrelativistic, rel=5.0e-4)


@pytest.mark.parametrize("projectile", tuple(PROJECTILES.values()))
@pytest.mark.parametrize("target", tuple(TARGETS.values()))
def test_all_supported_pairs_are_finite(projectile, target) -> None:
    cross_section = pair_cross_section_angstrom2(
        projectile, target, 1.0e6, 10.0
    )
    assert math.isfinite(cross_section)
    assert cross_section > 0.0


def test_transfer_cut_controls_cross_section() -> None:
    projectile = PROJECTILES["C"]
    target = TARGETS["O"]
    sigma_1 = pair_cross_section_angstrom2(projectile, target, 1.0e6, 1.0)
    sigma_10 = pair_cross_section_angstrom2(projectile, target, 1.0e6, 10.0)
    sigma_30 = pair_cross_section_angstrom2(projectile, target, 1.0e6, 30.0)
    assert sigma_1 > sigma_10 > sigma_30 > 0.0


def test_deflection_decreases_with_impact_parameter() -> None:
    projectile = PROJECTILES["O"]
    target = TARGETS["O"]
    length = screening_length_angstrom(8, 8)
    cosines = [
        cos_theta_cm(projectile, target, 1.0e6, factor * length)
        for factor in (0.0, 0.5, 1.0, 2.0, 4.0)
    ]
    assert all(math.isfinite(value) for value in cosines)
    assert all(a < b for a, b in zip(cosines, cosines[1:]))


def test_energy_argument_is_total_projectile_energy() -> None:
    carbon = PROJECTILES["C"]
    oxygen = TARGETS["O"]
    sigma_total_1_mev = pair_cross_section_angstrom2(
        carbon, oxygen, 1.0e6, 10.0
    )
    sigma_total_12_mev = pair_cross_section_angstrom2(
        carbon, oxygen, 12.0e6, 10.0
    )
    assert sigma_total_1_mev != pytest.approx(sigma_total_12_mev)
