"""Numba-compiled relative-coordinate DOP853 backend for carbon CTMC.

The numerical method and coefficients are the same DOP853 implementation used
by :func:`scipy.integrate.solve_ivp`.  Only the Python callback/allocation
overhead and the unneeded centre-of-mass coordinates are removed.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.integrate._ivp import dop853_coefficients


# Copy the public numerical tables into contiguous arrays so Numba can embed
# them in the compiled kernel. Dense-output stages are not needed.
_N_STAGES = int(dop853_coefficients.N_STAGES)
_A = np.ascontiguousarray(
    dop853_coefficients.A[:_N_STAGES, :_N_STAGES], dtype=np.float64
)
_B = np.ascontiguousarray(dop853_coefficients.B, dtype=np.float64)
_C = np.ascontiguousarray(
    dop853_coefficients.C[:_N_STAGES], dtype=np.float64
)
_E3 = np.ascontiguousarray(dop853_coefficients.E3, dtype=np.float64)
_E5 = np.ascontiguousarray(dop853_coefficients.E5, dtype=np.float64)

_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_ERROR_EXPONENT = -1.0 / 8.0
_MACHINE_EPSILON = np.finfo(np.float64).eps


@njit(cache=True, nogil=True)
def _effective_charge(
    nuclear_charge: float,
    spectators: float,
    eta: float,
    zeta: float,
    radius_au: float,
) -> tuple[float, float]:
    """Return Z_eff and dZ_eff/dr for the Garvey screened core."""
    radius = max(radius_au, np.finfo(np.float64).tiny)
    if spectators == 0.0:
        return nuclear_charge, 0.0

    exponent = zeta * radius
    if exponent > 700.0:
        return nuclear_charge - spectators, 0.0

    exp_term = math.exp(exponent)
    ratio = eta / zeta
    denominator = 1.0 + ratio * (exp_term - 1.0)
    omega = 1.0 / denominator
    domega_dr = -eta * exp_term / (denominator * denominator)
    return (
        nuclear_charge - spectators + spectators * omega,
        spectators * domega_dr,
    )


@njit(cache=True, nogil=True)
def _force_on_first(
    dx: float,
    dy: float,
    dz: float,
    charge_first: float,
    derivative_first: float,
    charge_second: float,
    derivative_second: float,
    minimum_radius_au: float,
) -> tuple[float, float, float]:
    """Force for U(r)=Q1(r)Q2(r)/r, matching the reference implementation."""
    radius = max(
        math.sqrt(dx * dx + dy * dy + dz * dz),
        minimum_radius_au,
    )
    product = charge_first * charge_second
    dpotential_dr = (
        derivative_first * charge_second
        + charge_first * derivative_second
    ) / radius - product / (radius * radius)
    coefficient = -dpotential_dr / radius
    return coefficient * dx, coefficient * dy, coefficient * dz


@njit(cache=True, nogil=True)
def _relative_rhs(
    state: np.ndarray,
    derivative: np.ndarray,
    target_parameters: np.ndarray,
    projectile_parameters: np.ndarray,
    minimum_radius_au: float,
    target_mass_au: float,
    projectile_mass_au: float,
) -> None:
    """Evaluate the paper's 12 relative-coordinate equations of motion."""
    # R1 = r_projectile-r_target; R2 = r_electron-r_target.
    r1x, r1y, r1z = state[0], state[1], state[2]
    r2x, r2y, r2z = state[3], state[4], state[5]

    zt_n, zt_s, zt_eta, zt_zeta = (
        target_parameters[0],
        target_parameters[1],
        target_parameters[2],
        target_parameters[3],
    )
    zp_n, zp_s, zp_eta, zp_zeta = (
        projectile_parameters[0],
        projectile_parameters[1],
        projectile_parameters[2],
        projectile_parameters[3],
    )

    # Force on target from projectile. The target-to-projectile displacement
    # used by the full-coordinate reference is r_target-r_projectile = -R1.
    r_tp = max(
        math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z),
        minimum_radius_au,
    )
    zt, dzt = _effective_charge(zt_n, zt_s, zt_eta, zt_zeta, r_tp)
    zp, dzp = _effective_charge(zp_n, zp_s, zp_eta, zp_zeta, r_tp)
    ftp_x, ftp_y, ftp_z = _force_on_first(
        -r1x,
        -r1y,
        -r1z,
        zt,
        dzt,
        zp,
        dzp,
        minimum_radius_au,
    )

    # Force on target from electron: r_target-r_electron = -R2.
    r_te = max(
        math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z),
        minimum_radius_au,
    )
    zt, dzt = _effective_charge(zt_n, zt_s, zt_eta, zt_zeta, r_te)
    fte_x, fte_y, fte_z = _force_on_first(
        -r2x,
        -r2y,
        -r2z,
        zt,
        dzt,
        -1.0,
        0.0,
        minimum_radius_au,
    )

    # Force on projectile from electron: r_projectile-r_electron = R1-R2.
    dpe_x, dpe_y, dpe_z = r1x - r2x, r1y - r2y, r1z - r2z
    r_pe = max(
        math.sqrt(dpe_x * dpe_x + dpe_y * dpe_y + dpe_z * dpe_z),
        minimum_radius_au,
    )
    zp, dzp = _effective_charge(zp_n, zp_s, zp_eta, zp_zeta, r_pe)
    fpe_x, fpe_y, fpe_z = _force_on_first(
        dpe_x,
        dpe_y,
        dpe_z,
        zp,
        dzp,
        -1.0,
        0.0,
        minimum_radius_au,
    )

    # Absolute accelerations followed by R1''=a_p-a_t and R2''=a_e-a_t.
    at_x = (ftp_x + fte_x) / target_mass_au
    at_y = (ftp_y + fte_y) / target_mass_au
    at_z = (ftp_z + fte_z) / target_mass_au

    ap_x = (-ftp_x + fpe_x) / projectile_mass_au
    ap_y = (-ftp_y + fpe_y) / projectile_mass_au
    ap_z = (-ftp_z + fpe_z) / projectile_mass_au

    ae_x = -fte_x - fpe_x
    ae_y = -fte_y - fpe_y
    ae_z = -fte_z - fpe_z

    derivative[0] = state[6]
    derivative[1] = state[7]
    derivative[2] = state[8]
    derivative[3] = state[9]
    derivative[4] = state[10]
    derivative[5] = state[11]
    derivative[6] = ap_x - at_x
    derivative[7] = ap_y - at_y
    derivative[8] = ap_z - at_z
    derivative[9] = ae_x - at_x
    derivative[10] = ae_y - at_y
    derivative[11] = ae_z - at_z


@njit(cache=True, nogil=True)
def _rms_scaled_norm(values: np.ndarray, scale: np.ndarray) -> float:
    total = 0.0
    for index in range(values.size):
        ratio = values[index] / scale[index]
        total += ratio * ratio
    return math.sqrt(total / values.size)


@njit(cache=True, nogil=True)
def _select_initial_step(
    state: np.ndarray,
    derivative: np.ndarray,
    end_time: float,
    max_step_au: float,
    rtol: float,
    atol: float,
    target_parameters: np.ndarray,
    projectile_parameters: np.ndarray,
    minimum_radius_au: float,
    target_mass_au: float,
    projectile_mass_au: float,
) -> float:
    """SciPy/Hairer empirical initial-step selection for an order-7 error."""
    scale = np.empty(state.size, dtype=np.float64)
    scaled_state = np.empty(state.size, dtype=np.float64)
    scaled_derivative = np.empty(state.size, dtype=np.float64)
    for index in range(state.size):
        scale[index] = atol + abs(state[index]) * rtol
        scaled_state[index] = state[index]
        scaled_derivative[index] = derivative[index]

    d0 = _rms_scaled_norm(scaled_state, scale)
    d1 = _rms_scaled_norm(scaled_derivative, scale)
    if d0 < 1.0e-5 or d1 < 1.0e-5:
        h0 = 1.0e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, end_time)

    trial_state = np.empty(state.size, dtype=np.float64)
    trial_derivative = np.empty(state.size, dtype=np.float64)
    for index in range(state.size):
        trial_state[index] = state[index] + h0 * derivative[index]
    _relative_rhs(
        trial_state,
        trial_derivative,
        target_parameters,
        projectile_parameters,
        minimum_radius_au,
        target_mass_au,
        projectile_mass_au,
    )

    delta_derivative = np.empty(state.size, dtype=np.float64)
    for index in range(state.size):
        delta_derivative[index] = (
            trial_derivative[index] - derivative[index]
        ) / h0
    d2 = _rms_scaled_norm(delta_derivative, scale)

    if d1 <= 1.0e-15 and d2 <= 1.0e-15:
        h1 = max(1.0e-6, h0 * 1.0e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / 8.0)

    return min(100.0 * h0, h1, end_time, max_step_au)


@njit(cache=True, nogil=True)
def integrate_relative_dop853(
    initial_state: np.ndarray,
    end_time: float,
    target_parameters: np.ndarray,
    projectile_parameters: np.ndarray,
    rtol: float,
    atol: float,
    max_step_au: float,
    minimum_radius_au: float,
    target_mass_au: float,
    projectile_mass_au: float,
    maximum_steps: int,
) -> tuple[bool, np.ndarray, int, int]:
    """Integrate one 12-component trajectory with adaptive DOP853.

    Returns ``(success, final_state, accepted_steps, rejected_steps)``.
    """
    state = initial_state.copy()
    if end_time <= 0.0:
        return True, state, 0, 0

    derivative = np.empty(12, dtype=np.float64)
    _relative_rhs(
        state,
        derivative,
        target_parameters,
        projectile_parameters,
        minimum_radius_au,
        target_mass_au,
        projectile_mass_au,
    )
    if not np.all(np.isfinite(derivative)):
        return False, state, 0, 0

    finite_max_step = (
        max_step_au if math.isfinite(max_step_au) else math.inf
    )
    step_size = _select_initial_step(
        state,
        derivative,
        end_time,
        finite_max_step,
        rtol,
        atol,
        target_parameters,
        projectile_parameters,
        minimum_radius_au,
        target_mass_au,
        projectile_mass_au,
    )

    stages = np.empty((_N_STAGES + 1, 12), dtype=np.float64)
    trial_state = np.empty(12, dtype=np.float64)
    new_state = np.empty(12, dtype=np.float64)
    scale = np.empty(12, dtype=np.float64)

    time = 0.0
    accepted_steps = 0
    rejected_steps = 0
    attempts = 0
    previous_step_rejected = False

    while time < end_time and attempts < maximum_steps:
        minimum_step = (
            10.0 * _MACHINE_EPSILON * max(abs(time), 1.0)
        )
        step_size = min(step_size, finite_max_step, end_time - time)
        if step_size < minimum_step:
            return False, state, accepted_steps, rejected_steps

        for component in range(12):
            stages[0, component] = derivative[component]

        for stage in range(1, _N_STAGES):
            for component in range(12):
                increment = 0.0
                for previous_stage in range(stage):
                    increment += (
                        _A[stage, previous_stage]
                        * stages[previous_stage, component]
                    )
                trial_state[component] = (
                    state[component] + step_size * increment
                )
            _relative_rhs(
                trial_state,
                stages[stage],
                target_parameters,
                projectile_parameters,
                minimum_radius_au,
                target_mass_au,
                projectile_mass_au,
            )

        for component in range(12):
            increment = 0.0
            for stage in range(_N_STAGES):
                increment += _B[stage] * stages[stage, component]
            new_state[component] = state[component] + step_size * increment

        _relative_rhs(
            new_state,
            stages[_N_STAGES],
            target_parameters,
            projectile_parameters,
            minimum_radius_au,
            target_mass_au,
            projectile_mass_au,
        )

        err5_norm_squared = 0.0
        err3_norm_squared = 0.0
        for component in range(12):
            scale[component] = (
                atol
                + max(abs(state[component]), abs(new_state[component])) * rtol
            )
            err5 = 0.0
            err3 = 0.0
            for stage in range(_N_STAGES + 1):
                err5 += _E5[stage] * stages[stage, component]
                err3 += _E3[stage] * stages[stage, component]
            err5 /= scale[component]
            err3 /= scale[component]
            err5_norm_squared += err5 * err5
            err3_norm_squared += err3 * err3

        if err5_norm_squared == 0.0 and err3_norm_squared == 0.0:
            error_norm = 0.0
        else:
            denominator = (
                err5_norm_squared + 0.01 * err3_norm_squared
            )
            error_norm = (
                abs(step_size)
                * err5_norm_squared
                / math.sqrt(denominator * 12.0)
            )

        attempts += 1
        if math.isfinite(error_norm) and error_norm < 1.0:
            time += step_size
            for component in range(12):
                state[component] = new_state[component]
                derivative[component] = stages[_N_STAGES, component]
            accepted_steps += 1

            if error_norm == 0.0:
                factor = _MAX_FACTOR
            else:
                factor = min(
                    _MAX_FACTOR,
                    _SAFETY * error_norm ** _ERROR_EXPONENT,
                )
            if previous_step_rejected:
                factor = min(1.0, factor)
            step_size *= factor
            previous_step_rejected = False
        else:
            factor = _MIN_FACTOR
            if math.isfinite(error_norm) and error_norm > 0.0:
                factor = max(
                    _MIN_FACTOR,
                    _SAFETY * error_norm ** _ERROR_EXPONENT,
                )
            step_size *= factor
            rejected_steps += 1
            previous_step_rejected = True

        if not np.all(np.isfinite(state)):
            return False, state, accepted_steps, rejected_steps

    success = time >= end_time and np.all(np.isfinite(state))
    return success, state, accepted_steps, rejected_steps


def warm_up_numba_backend() -> None:
    """Compile/cache the kernel before spawning process workers."""
    state = np.array(
        [1.0, 0.0, -10.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    unscreened = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrate_relative_dop853(
        state,
        1.0e-6,
        unscreened,
        unscreened,
        1.0e-8,
        1.0e-10,
        math.inf,
        1.0e-10,
        1836.0,
        21868.0,
        100,
    )
