from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from bca import (  # noqa: E402
    NLHCollisionKernel,
    hard_cross_section_angstrom2,
    maximum_impact_parameter_angstrom,
    solve_nlh_collision,
    two_body_outcome_from_cm_angle,
    turning_threshold_radius_angstrom,
)


@pytest.mark.parametrize(
    ("projectile", "target"),
    (("C", "H"), ("C", "O"), ("O", "H"), ("S", "O")),
)
def test_retained_boundary_has_requested_turning_potential(projectile, target):
    maximum_impact = maximum_impact_parameter_angstrom(
        projectile, target, 1.0e5, minimum_turning_potential_ev=30.0
    )
    result = solve_nlh_collision(
        projectile,
        target,
        1.0e5,
        maximum_impact,
        minimum_turning_potential_ev=30.0,
    )
    assert result.turning_potential_ev == pytest.approx(30.0, rel=2.0e-12)
    assert hard_cross_section_angstrom2(projectile, target, 1.0e5) == pytest.approx(
        math.pi * maximum_impact**2
    )


def test_scattering_and_recoil_decrease_with_collision_area_quantile():
    kernel = NLHCollisionKernel("C", "O", 1.0e5)
    quantiles = np.linspace(0.0, 1.0, 17)
    results = [
        kernel.solve(kernel.maximum_impact_parameter_angstrom * math.sqrt(quantile))
        for quantile in quantiles
    ]
    theta = np.asarray([result.theta_cm_rad for result in results])
    recoil = np.asarray([result.recoil_energy_ev for result in results])
    assert theta[0] == pytest.approx(math.pi)
    assert np.all(np.diff(theta) < 0.0)
    assert np.all(np.diff(recoil) < 0.0)
    assert all(abs(result.energy_conservation_error_ev) < 1.0e-10 for result in results)


def test_quadrature_is_converged_at_default_scale():
    kernel_64 = NLHCollisionKernel("S", "O", 1.0e6, quadrature_order=64)
    kernel_128 = NLHCollisionKernel("S", "O", 1.0e6, quadrature_order=128)
    impact = 0.5 * kernel_64.maximum_impact_parameter_angstrom
    theta_64 = kernel_64.solve(impact).theta_cm_rad
    theta_128 = kernel_128.solve(impact).theta_cm_rad
    assert theta_64 == pytest.approx(theta_128, rel=1.0e-8, abs=1.0e-11)


def test_below_published_nlh_domain_is_rejected():
    with pytest.raises(ValueError, match="published NLH domain"):
        NLHCollisionKernel(
            "C", "O", 1.0e5, minimum_turning_potential_ev=9.999
        )


def test_exact_hard_cross_section_formula_matches_kernel():
    kernel = NLHCollisionKernel("S", "H", 1.0e3)
    radius = turning_threshold_radius_angstrom("S", "H")
    expected = math.pi * radius**2 * (
        1.0
        - kernel.minimum_turning_potential_ev
        / kernel.kinematics.relative_kinetic_energy_ev
    )
    assert kernel.hard_cross_section_angstrom2 == pytest.approx(expected)


def test_public_cm_transform_reproduces_collision_outcome():
    kernel = NLHCollisionKernel("O", "H", 1.0e6)
    collision = kernel.solve(0.37 * kernel.maximum_impact_parameter_angstrom)
    transformed = two_body_outcome_from_cm_angle(
        kernel.kinematics, collision.theta_cm_rad
    )
    assert transformed.theta_projectile_lab_rad == pytest.approx(
        collision.theta_projectile_lab_rad
    )
    assert transformed.recoil_energy_ev == pytest.approx(collision.recoil_energy_ev)
    assert transformed.projectile_out_energy_ev == pytest.approx(
        collision.projectile_out_energy_ev
    )
