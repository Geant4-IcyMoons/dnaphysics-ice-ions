from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


NLH_PARENT = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NLH_PARENT) not in sys.path:
    sys.path.insert(0, str(NLH_PARENT))

from nlh import (
    NLHDomainError,
    get_coefficients,
    potential_derivative_ev_per_angstrom,
    potential_ev,
    radial_force_ev_per_angstrom,
    screening_derivative_per_angstrom,
    screening_function,
    supported_projectile_target_pairs,
)
from nlh import potential as potential_module


def test_all_requested_projectile_target_pairs_are_available():
    pairs = supported_projectile_target_pairs()
    assert pairs == (
        ("H", "H"),
        ("H", "O"),
        ("He", "H"),
        ("He", "O"),
        ("C", "H"),
        ("C", "O"),
        ("O", "H"),
        ("O", "O"),
        ("S", "H"),
        ("S", "O"),
    )
    assert len({get_coefficients(*pair) for pair in pairs}) == 9


@pytest.mark.parametrize(
    ("projectile", "target", "reference"),
    (
        (
            "H",
            "H",
            (93.494187895, -1434.4069649, 0.649281179168, -3.46859293888),
        ),
        (
            "He",
            "O",
            (1377.36735992, -20824.8991932, 0.597830413132, -3.06050312547),
        ),
        (
            "S",
            "O",
            (9709.92311867, -145405.769253, 0.526810014383, -2.62086194735),
        ),
    ),
)
def test_matches_author_reference_c_evaluator_at_point_one_angstrom(
    projectile, target, reference
):
    expected_v, expected_dv, expected_phi, expected_dphi = reference
    assert potential_ev(0.1, projectile, target) == pytest.approx(expected_v, rel=2e-12)
    assert potential_derivative_ev_per_angstrom(
        0.1, projectile, target
    ) == pytest.approx(expected_dv, rel=2e-12)
    assert screening_function(0.1, projectile, target) == pytest.approx(
        expected_phi, rel=2e-12
    )
    assert screening_derivative_per_angstrom(
        0.1, projectile, target
    ) == pytest.approx(expected_dphi, rel=2e-12)


def test_vector_evaluation_and_force_sign():
    distances = np.array([0.05, 0.1, 0.2])
    potential = potential_ev(distances, "C", "O")
    derivative = potential_derivative_ev_per_angstrom(distances, "C", "O")
    force = radial_force_ev_per_angstrom(distances, "C", "O")
    assert potential.shape == distances.shape
    assert np.all(np.diff(potential) < 0.0)
    assert np.all(derivative < 0.0)
    assert np.allclose(force, -derivative)


@pytest.mark.parametrize("projectile,target", supported_projectile_target_pairs())
def test_scalar_and_vector_potential_paths_agree_to_roundoff(projectile, target):
    distances = np.geomspace(0.02, 0.3, 17)
    vector = potential_ev(
        distances, projectile, target, enforce_fit_domain=False
    )
    scalar = np.asarray(
        [
            potential_ev(
                float(distance),
                projectile,
                target,
                enforce_fit_domain=False,
            )
            for distance in distances
        ]
    )
    np.testing.assert_allclose(
        scalar,
        vector,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )


def test_scalar_fast_path_does_not_construct_numpy_component_arrays(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("scalar potential used the array evaluator")

    monkeypatch.setattr(potential_module, "_raw_components", fail_if_called)
    assert potential_ev(0.1, "C", "O") > 0.0


def test_fit_domain_is_enforced_unless_diagnostics_are_explicit():
    with pytest.raises(NLHDomainError, match="below its published repulsive-fit domain"):
        potential_ev(2.0, "H", "O")
    assert potential_ev(2.0, "H", "O", enforce_fit_domain=False) < 10.0


def test_only_water_target_atoms_are_accepted():
    with pytest.raises(ValueError, match="Ice target atom must be H or O"):
        get_coefficients("C", "C")
