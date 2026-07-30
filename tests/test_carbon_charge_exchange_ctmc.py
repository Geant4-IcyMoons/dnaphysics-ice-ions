from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


PHYSICS_ICE_DIR = (
    Path(__file__).resolve().parents[1] / "python_scripts" / "physics_ice"
)
if str(PHYSICS_ICE_DIR) not in sys.path:
    sys.path.insert(0, str(PHYSICS_ICE_DIR))

import ctmc_numba_backend as numba_backend
import generate_carbon_charge_exchange_ctmc as ctmc


def _reference_state():
    target = ctmc.CorePotential.from_zn(
        ctmc.WATER_PSEUDO_NUCLEAR_CHARGE, 9
    )
    projectile = ctmc.CorePotential.from_zn(
        ctmc.CARBON_NUCLEAR_CHARGE, 3
    )
    state = ctmc._build_initial_state(
        energy_keV_u=100.0,
        impact_parameter_au=3.0,
        separation_au=20.0,
        bound_to="target",
        binding_eV=ctmc.WATER_ORBITALS[0].binding_eV,
        target_core=target,
        projectile_core=projectile,
        radial_grid_points=256,
        rng=np.random.default_rng(12345),
    )
    return target, projectile, state


def test_numba_relative_rhs_matches_full_coordinate_reference():
    target, projectile, state = _reference_state()
    relative = ctmc._state_to_relative_coordinates(state)
    full_rhs = ctmc._three_body_rhs(
        0.0, state, target, projectile, 1.0e-10
    )
    expected = np.concatenate(
        (
            full_rhs[3:6] - full_rhs[0:3],
            full_rhs[6:9] - full_rhs[0:3],
            full_rhs[12:15] - full_rhs[9:12],
            full_rhs[15:18] - full_rhs[9:12],
        )
    )
    actual = np.empty(12)
    numba_backend._relative_rhs(
        relative,
        actual,
        ctmc._core_parameters(target),
        ctmc._core_parameters(projectile),
        1.0e-10,
        ctmc.WATER_MASS_AU,
        ctmc.CARBON_MASS_AU,
    )
    assert np.allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_numba_dop853_matches_scipy_dop853():
    target, projectile, state = _reference_state()
    relative = ctmc._state_to_relative_coordinates(state)
    end_time = 40.0 / ctmc.projectile_velocity_au(100.0)

    reference = solve_ivp(
        ctmc._three_body_rhs,
        (0.0, end_time),
        state,
        method="DOP853",
        t_eval=(end_time,),
        args=(target, projectile, 1.0e-10),
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert reference.success
    expected = ctmc._state_to_relative_coordinates(reference.y[:, -1])

    success, actual, accepted, rejected = (
        numba_backend.integrate_relative_dop853(
            relative,
            end_time,
            ctmc._core_parameters(target),
            ctmc._core_parameters(projectile),
            1.0e-9,
            1.0e-11,
            math.inf,
            1.0e-10,
            ctmc.WATER_MASS_AU,
            ctmc.CARBON_MASS_AU,
            10_000_000,
        )
    )
    assert success
    assert accepted > 0
    assert rejected >= 0
    assert np.allclose(actual, expected, rtol=3.0e-6, atol=3.0e-7)


def test_scheduler_is_fast_energy_first_and_lazy():
    energies = np.array([1.0, 10.0, 100.0])
    charges = np.array([0])
    impact = np.array([0.0, 1.0])
    done = np.zeros((3, 1, 2), dtype=bool)
    done[2, 0, 0] = True

    tasks = list(
        ctmc.iter_pending_probability_tasks(
            energies, charges, impact, done
        )
    )
    assert [task[0] for task in tasks] == [2, 1, 1, 0, 0]
    assert tasks[0][2] == 100.0


def test_optimized_backend_is_the_default():
    args = ctmc.build_parser().parse_args([])
    assert args.backend == "numba"
    assert args.energy_points == 41
    assert args.workers == 12
