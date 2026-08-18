"""Numba-compiled relative-coordinate DOP853 backend for charge-exchange CTMC.

The numerical method and coefficients are the same DOP853 implementation used
by :func:`scipy.integrate.solve_ivp`.  Only the Python callback/allocation
overhead and the unneeded centre-of-mass coordinates are removed. A fallback
uses a Sundman time transformation for exceptional close encounters.
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
def _regularized_relative_rhs(
    state: np.ndarray,
    derivative: np.ndarray,
    target_parameters: np.ndarray,
    projectile_parameters: np.ndarray,
    minimum_radius_au: float,
    target_mass_au: float,
    projectile_mass_au: float,
    sundman_power: int,
) -> float:
    """Evaluate the same equations in a close-encounter Sundman time.

    ``dt/ds`` is the shortest pair distance raised to ``sundman_power``,
    capped at one.  Powers one and two are exact positive time
    reparameterizations away from the excluded zero-radius singularity; the
    squared-distance form more strongly regularizes exceptionally eccentric
    Coulomb encounters.
    Multiplying every physical-time derivative by this strictly positive
    factor changes only the trajectory parameterization.  It removes the
    vanishing physical-time steps produced by a near-Coulomb encounter while
    retaining the unmodified screened potentials and Newtonian equations.
    """
    _relative_rhs(
        state,
        derivative,
        target_parameters,
        projectile_parameters,
        minimum_radius_au,
        target_mass_au,
        projectile_mass_au,
    )
    core_distance_squared = (
        state[0] * state[0]
        + state[1] * state[1]
        + state[2] * state[2]
    )
    electron_reference_distance_squared = (
        state[3] * state[3]
        + state[4] * state[4]
        + state[5] * state[5]
    )
    electron_other_x = state[3] - state[0]
    electron_other_y = state[4] - state[1]
    electron_other_z = state[5] - state[2]
    electron_other_distance_squared = (
        electron_other_x * electron_other_x
        + electron_other_y * electron_other_y
        + electron_other_z * electron_other_z
    )
    shortest_distance_squared = min(
        1.0,
        core_distance_squared,
        electron_reference_distance_squared,
        electron_other_distance_squared,
    )
    if sundman_power == 1:
        time_scale = math.sqrt(
            max(shortest_distance_squared, minimum_radius_au**2)
        )
    else:
        time_scale = max(shortest_distance_squared, minimum_radius_au**2)
    for component in range(12):
        derivative[component] *= time_scale
    return time_scale


@njit(cache=True, nogil=True)
def _relative_energy(
    state: np.ndarray,
    reference_parameters: np.ndarray,
    other_parameters: np.ndarray,
    minimum_radius_au: float,
    reference_mass_au: float,
    other_mass_au: float,
) -> float:
    """Return the conserved three-body energy in the relative frame."""
    total_mass = reference_mass_au + other_mass_au + 1.0
    com_x = (other_mass_au * state[6] + state[9]) / total_mass
    com_y = (other_mass_au * state[7] + state[10]) / total_mass
    com_z = (other_mass_au * state[8] + state[11]) / total_mass
    other_vx = state[6] - com_x
    other_vy = state[7] - com_y
    other_vz = state[8] - com_z
    electron_vx = state[9] - com_x
    electron_vy = state[10] - com_y
    electron_vz = state[11] - com_z
    kinetic = 0.5 * (
        reference_mass_au * (com_x * com_x + com_y * com_y + com_z * com_z)
        + other_mass_au
        * (
            other_vx * other_vx
            + other_vy * other_vy
            + other_vz * other_vz
        )
        + electron_vx * electron_vx
        + electron_vy * electron_vy
        + electron_vz * electron_vz
    )

    core_radius = max(
        math.sqrt(
            state[0] * state[0]
            + state[1] * state[1]
            + state[2] * state[2]
        ),
        minimum_radius_au,
    )
    electron_reference_radius = max(
        math.sqrt(
            state[3] * state[3]
            + state[4] * state[4]
            + state[5] * state[5]
        ),
        minimum_radius_au,
    )
    electron_other_x = state[3] - state[0]
    electron_other_y = state[4] - state[1]
    electron_other_z = state[5] - state[2]
    electron_other_radius = max(
        math.sqrt(
            electron_other_x * electron_other_x
            + electron_other_y * electron_other_y
            + electron_other_z * electron_other_z
        ),
        minimum_radius_au,
    )
    reference_core_charge, _ = _effective_charge(
        reference_parameters[0],
        reference_parameters[1],
        reference_parameters[2],
        reference_parameters[3],
        core_radius,
    )
    other_core_charge, _ = _effective_charge(
        other_parameters[0],
        other_parameters[1],
        other_parameters[2],
        other_parameters[3],
        core_radius,
    )
    reference_electron_charge, _ = _effective_charge(
        reference_parameters[0],
        reference_parameters[1],
        reference_parameters[2],
        reference_parameters[3],
        electron_reference_radius,
    )
    other_electron_charge, _ = _effective_charge(
        other_parameters[0],
        other_parameters[1],
        other_parameters[2],
        other_parameters[3],
        electron_other_radius,
    )
    potential = (
        reference_core_charge * other_core_charge / core_radius
        - reference_electron_charge / electron_reference_radius
        - other_electron_charge / electron_other_radius
    )
    return kinetic + potential


@njit(cache=True, nogil=True)
def _project_velocity_to_energy(
    state: np.ndarray,
    conserved_energy: float,
    reference_parameters: np.ndarray,
    other_parameters: np.ndarray,
    minimum_radius_au: float,
    reference_mass_au: float,
    other_mass_au: float,
) -> bool:
    """Project a numerical endpoint back onto its energy invariant."""
    zero_velocity_state = state.copy()
    for component in range(6, 12):
        zero_velocity_state[component] = 0.0
    potential = _relative_energy(
        zero_velocity_state,
        reference_parameters,
        other_parameters,
        minimum_radius_au,
        reference_mass_au,
        other_mass_au,
    )
    kinetic = (
        _relative_energy(
            state,
            reference_parameters,
            other_parameters,
            minimum_radius_au,
            reference_mass_au,
            other_mass_au,
        )
        - potential
    )
    required_kinetic = conserved_energy - potential
    if (
        kinetic <= 0.0
        or required_kinetic < 0.0
        or not math.isfinite(kinetic)
        or not math.isfinite(required_kinetic)
    ):
        return False
    velocity_factor = math.sqrt(required_kinetic / kinetic)
    if not math.isfinite(velocity_factor):
        return False
    for component in range(6, 12):
        state[component] *= velocity_factor
    return True


@njit(cache=True, nogil=True)
def _rms_scaled_norm(values: np.ndarray, scale: np.ndarray) -> float:
    total = 0.0
    for index in range(values.size):
        ratio = values[index] / scale[index]
        total += ratio * ratio
    return math.sqrt(total / values.size)


@njit(cache=True, nogil=True)
def _switch_reference_core(state: np.ndarray) -> None:
    """Apply the paper's p<->t relative-coordinate interchange in place."""
    for component in range(3):
        core_displacement = state[component]
        electron_displacement = state[3 + component]
        core_velocity = state[6 + component]
        electron_velocity = state[9 + component]
        state[component] = -core_displacement
        state[3 + component] = (
            electron_displacement - core_displacement
        )
        state[6 + component] = -core_velocity
        state[9 + component] = electron_velocity - core_velocity


@njit(cache=True, nogil=True)
def _electron_is_closer_to_other_core(state: np.ndarray) -> bool:
    reference_distance_squared = 0.0
    other_distance_squared = 0.0
    for component in range(3):
        electron_reference = state[3 + component]
        electron_other = electron_reference - state[component]
        reference_distance_squared += electron_reference * electron_reference
        other_distance_squared += electron_other * electron_other
    return other_distance_squared < reference_distance_squared


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
def _select_initial_regularized_step(
    state: np.ndarray,
    derivative: np.ndarray,
    max_step_au: float,
    rtol: float,
    atol: float,
    target_parameters: np.ndarray,
    projectile_parameters: np.ndarray,
    minimum_radius_au: float,
    target_mass_au: float,
    projectile_mass_au: float,
    sundman_power: int,
) -> float:
    """SciPy/Hairer initial-step selection in Sundman time."""
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
    h0 = min(h0, max_step_au)

    trial_state = np.empty(state.size, dtype=np.float64)
    trial_derivative = np.empty(state.size, dtype=np.float64)
    for index in range(state.size):
        trial_state[index] = state[index] + h0 * derivative[index]
    _regularized_relative_rhs(
        trial_state,
        trial_derivative,
        target_parameters,
        projectile_parameters,
        minimum_radius_au,
        target_mass_au,
        projectile_mass_au,
        sundman_power,
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

    return min(100.0 * h0, h1, max_step_au)


@njit(cache=True, nogil=True)
def integrate_relative_dop853(
    initial_state: np.ndarray,
    minimum_integration_time_au: float,
    initial_reference_parameters: np.ndarray,
    initial_other_parameters: np.ndarray,
    rtol: float,
    atol: float,
    max_step_au: float,
    minimum_radius_au: float,
    initial_reference_mass_au: float,
    initial_other_mass_au: float,
    maximum_steps: int,
    stop_at_minimum_time: bool = False,
    terminal_core_distance_au: float = -1.0,
) -> tuple[bool, np.ndarray, int, int]:
    """Integrate one 12-component trajectory with adaptive DOP853.

    By default the trajectory ends after the minimum integration time once
    the two screened nuclei have again reached at least their initial
    separation and are moving apart.  ``stop_at_minimum_time`` instead ends
    at the exact disclosed fixed interval used by a formal reference run.
    A positive ``terminal_core_distance_au`` replaces the default initial-
    separation boundary after the minimum interval.

    The exact Appendix-A ``p``/``t`` interchange keeps the electron
    displacement relative to the nearer core during integration.

    Returns ``(success, final_state, accepted_steps, rejected_steps)``.
    ``maximum_steps == 0`` disables the implementation safety ceiling because
    the paper does not define a maximum number of Runge--Kutta steps.
    """
    state = initial_state.copy()
    reference_parameters = initial_reference_parameters
    other_parameters = initial_other_parameters
    reference_mass_au = initial_reference_mass_au
    other_mass_au = initial_other_mass_au
    reference_was_switched = False

    if _electron_is_closer_to_other_core(state):
        _switch_reference_core(state)
        parameters_temporary = reference_parameters
        reference_parameters = other_parameters
        other_parameters = parameters_temporary
        mass_temporary = reference_mass_au
        reference_mass_au = other_mass_au
        other_mass_au = mass_temporary
        reference_was_switched = True

    initial_core_distance = math.sqrt(
        state[0] * state[0]
        + state[1] * state[1]
        + state[2] * state[2]
    )
    terminal_core_distance = (
        terminal_core_distance_au
        if terminal_core_distance_au > 0.0
        else initial_core_distance
    )
    derivative = np.empty(12, dtype=np.float64)
    _relative_rhs(
        state,
        derivative,
        reference_parameters,
        other_parameters,
        minimum_radius_au,
        reference_mass_au,
        other_mass_au,
    )
    if not np.all(np.isfinite(derivative)):
        if reference_was_switched:
            _switch_reference_core(state)
        return False, state, 0, 0

    finite_max_step = (
        max_step_au if math.isfinite(max_step_au) else math.inf
    )
    step_size = _select_initial_step(
        state,
        derivative,
        max(minimum_integration_time_au, 1.0),
        finite_max_step,
        rtol,
        atol,
        reference_parameters,
        other_parameters,
        minimum_radius_au,
        reference_mass_au,
        other_mass_au,
    )

    stages = np.empty((_N_STAGES + 1, 12), dtype=np.float64)
    trial_state = np.empty(12, dtype=np.float64)
    new_state = np.empty(12, dtype=np.float64)
    scale = np.empty(12, dtype=np.float64)

    time = 0.0
    time_compensation = 0.0
    accepted_steps = 0
    rejected_steps = 0
    attempts = 0
    previous_step_rejected = False

    while maximum_steps == 0 or attempts < maximum_steps:
        core_distance_squared = (
            state[0] * state[0]
            + state[1] * state[1]
            + state[2] * state[2]
        )
        radial_motion = (
            state[0] * state[6]
            + state[1] * state[7]
            + state[2] * state[8]
        )
        if time >= minimum_integration_time_au and (
            stop_at_minimum_time
            or (
                core_distance_squared
                >= terminal_core_distance * terminal_core_distance
                and radial_motion > 0.0
            )
        ):
            if reference_was_switched:
                _switch_reference_core(state)
            return True, state, accepted_steps, rejected_steps

        # These equations are autonomous: the RHS has no explicit time
        # dependence.  An absolute-time ULP floor (as used by a generic ODE
        # driver) incorrectly rejects resolvable close-encounter steps after
        # a long low-energy flight.  Treat every step relative to a local time
        # origin and retain the physical elapsed time with compensated
        # summation instead.
        step_size = min(step_size, finite_max_step)
        if time < minimum_integration_time_au:
            step_size = min(
                step_size,
                minimum_integration_time_au - time,
            )
        # The equations are autonomous and every Runge--Kutta update is
        # evaluated relative to the current state, so an absolute epsilon
        # floor on the local step is incorrect.  A close Coulomb passage can
        # require h < eps while h*velocity is still readily resolvable in the
        # state.  Fail only on genuine floating-point stagnation; compensated
        # summation retains the correspondingly small elapsed-time increment.
        if step_size <= 0.0 or not math.isfinite(step_size):
            if reference_was_switched:
                _switch_reference_core(state)
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
                reference_parameters,
                other_parameters,
                minimum_radius_au,
                reference_mass_au,
                other_mass_au,
            )

        for component in range(12):
            increment = 0.0
            for stage in range(_N_STAGES):
                increment += _B[stage] * stages[stage, component]
            new_state[component] = state[component] + step_size * increment

        _relative_rhs(
            new_state,
            stages[_N_STAGES],
            reference_parameters,
            other_parameters,
            minimum_radius_au,
            reference_mass_au,
            other_mass_au,
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
            compensated_step = step_size - time_compensation
            updated_time = time + compensated_step
            time_compensation = (
                updated_time - time
            ) - compensated_step
            time = updated_time
            for component in range(12):
                state[component] = new_state[component]
                derivative[component] = stages[_N_STAGES, component]
            accepted_steps += 1

            if _electron_is_closer_to_other_core(state):
                _switch_reference_core(state)
                parameters_temporary = reference_parameters
                reference_parameters = other_parameters
                other_parameters = parameters_temporary
                mass_temporary = reference_mass_au
                reference_mass_au = other_mass_au
                other_mass_au = mass_temporary
                reference_was_switched = not reference_was_switched
                _relative_rhs(
                    state,
                    derivative,
                    reference_parameters,
                    other_parameters,
                    minimum_radius_au,
                    reference_mass_au,
                    other_mass_au,
                )

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
            if reference_was_switched:
                _switch_reference_core(state)
            return False, state, accepted_steps, rejected_steps

    if reference_was_switched:
        _switch_reference_core(state)
    return False, state, accepted_steps, rejected_steps


@njit(cache=True, nogil=True)
def integrate_relative_dop853_regularized(
    initial_state: np.ndarray,
    minimum_integration_time_au: float,
    initial_reference_parameters: np.ndarray,
    initial_other_parameters: np.ndarray,
    rtol: float,
    atol: float,
    max_step_au: float,
    minimum_radius_au: float,
    initial_reference_mass_au: float,
    initial_other_mass_au: float,
    sundman_power: int,
    maximum_relative_energy_drift: float,
    maximum_steps: int,
    stop_at_minimum_time: bool = False,
    terminal_core_distance_au: float = -1.0,
) -> tuple[bool, np.ndarray, int, int]:
    """DOP853 fallback with a Sundman parameter for close encounters.

    The physical equations, potentials, stopping condition, and tolerances
    are identical to :func:`integrate_relative_dop853`. The independent
    variable changes through ``dt/ds > 0``; ``sundman_power`` must be one or
    two. If accumulated roundoff reaches
    one quarter of the caller's final energy-drift budget, the common
    relative-velocity scale is projected onto the conserved Hamiltonian
    surface; the final endpoint must still pass the caller's independent
    energy check.
    """
    if sundman_power != 1 and sundman_power != 2:
        return False, initial_state.copy(), 0, 0
    state = initial_state.copy()
    reference_parameters = initial_reference_parameters
    other_parameters = initial_other_parameters
    reference_mass_au = initial_reference_mass_au
    other_mass_au = initial_other_mass_au
    reference_was_switched = False

    if _electron_is_closer_to_other_core(state):
        _switch_reference_core(state)
        parameters_temporary = reference_parameters
        reference_parameters = other_parameters
        other_parameters = parameters_temporary
        mass_temporary = reference_mass_au
        reference_mass_au = other_mass_au
        other_mass_au = mass_temporary
        reference_was_switched = True

    initial_core_distance = math.sqrt(
        state[0] * state[0]
        + state[1] * state[1]
        + state[2] * state[2]
    )
    terminal_core_distance = (
        terminal_core_distance_au
        if terminal_core_distance_au > 0.0
        else initial_core_distance
    )
    conserved_energy = _relative_energy(
        state,
        reference_parameters,
        other_parameters,
        minimum_radius_au,
        reference_mass_au,
        other_mass_au,
    )
    derivative = np.empty(12, dtype=np.float64)
    initial_time_scale = _regularized_relative_rhs(
        state,
        derivative,
        reference_parameters,
        other_parameters,
        minimum_radius_au,
        reference_mass_au,
        other_mass_au,
        sundman_power,
    )
    if (
        not np.all(np.isfinite(derivative))
        or not math.isfinite(initial_time_scale)
    ):
        if reference_was_switched:
            _switch_reference_core(state)
        return False, state, 0, 0

    finite_max_step = (
        max_step_au if math.isfinite(max_step_au) else math.inf
    )
    step_size = _select_initial_regularized_step(
        state,
        derivative,
        finite_max_step,
        rtol,
        atol,
        reference_parameters,
        other_parameters,
        minimum_radius_au,
        reference_mass_au,
        other_mass_au,
        sundman_power,
    )

    stages = np.empty((_N_STAGES + 1, 12), dtype=np.float64)
    time_scales = np.empty(_N_STAGES + 1, dtype=np.float64)
    trial_state = np.empty(12, dtype=np.float64)
    new_state = np.empty(12, dtype=np.float64)
    scale = np.empty(12, dtype=np.float64)

    physical_time = 0.0
    time_compensation = 0.0
    accepted_steps = 0
    rejected_steps = 0
    attempts = 0
    previous_step_rejected = False

    while maximum_steps == 0 or attempts < maximum_steps:
        core_distance_squared = (
            state[0] * state[0]
            + state[1] * state[1]
            + state[2] * state[2]
        )
        radial_motion = (
            state[0] * state[6]
            + state[1] * state[7]
            + state[2] * state[8]
        )
        minimum_time_gap = minimum_integration_time_au - physical_time
        if (
            minimum_time_gap > 0.0
            and minimum_time_gap
            <= 100.0
            * _MACHINE_EPSILON
            * max(minimum_integration_time_au, 1.0)
        ):
            physical_time = minimum_integration_time_au
        if physical_time >= minimum_integration_time_au and (
            stop_at_minimum_time
            or (
                core_distance_squared
                >= terminal_core_distance * terminal_core_distance
                and radial_motion > 0.0
            )
        ):
            if reference_was_switched:
                _switch_reference_core(state)
            return True, state, accepted_steps, rejected_steps

        step_size = min(step_size, finite_max_step)
        if physical_time < minimum_integration_time_au:
            step_size = min(
                step_size,
                (minimum_integration_time_au - physical_time)
                / initial_time_scale,
            )
        if step_size <= 0.0 or not math.isfinite(step_size):
            if reference_was_switched:
                _switch_reference_core(state)
            return False, state, accepted_steps, rejected_steps

        for component in range(12):
            stages[0, component] = derivative[component]
        time_scales[0] = initial_time_scale

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
            time_scales[stage] = _regularized_relative_rhs(
                trial_state,
                stages[stage],
                reference_parameters,
                other_parameters,
                minimum_radius_au,
                reference_mass_au,
                other_mass_au,
                sundman_power,
            )

        for component in range(12):
            increment = 0.0
            for stage in range(_N_STAGES):
                increment += _B[stage] * stages[stage, component]
            new_state[component] = state[component] + step_size * increment

        time_scales[_N_STAGES] = _regularized_relative_rhs(
            new_state,
            stages[_N_STAGES],
            reference_parameters,
            other_parameters,
            minimum_radius_au,
            reference_mass_au,
            other_mass_au,
            sundman_power,
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
            denominator = err5_norm_squared + 0.01 * err3_norm_squared
            error_norm = (
                abs(step_size)
                * err5_norm_squared
                / math.sqrt(denominator * 12.0)
            )

        attempts += 1
        if math.isfinite(error_norm) and error_norm < 1.0:
            physical_time_increment = 0.0
            for stage in range(_N_STAGES):
                physical_time_increment += _B[stage] * time_scales[stage]
            physical_time_increment *= step_size
            remaining_minimum_time = (
                minimum_integration_time_au - physical_time
            )
            if (
                remaining_minimum_time > 0.0
                and physical_time_increment
                > remaining_minimum_time
                * (1.0 + 10.0 * _MACHINE_EPSILON)
            ):
                step_size *= max(
                    0.1,
                    0.9
                    * remaining_minimum_time
                    / physical_time_increment,
                )
                rejected_steps += 1
                previous_step_rejected = True
                continue
            compensated_increment = (
                physical_time_increment - time_compensation
            )
            updated_time = physical_time + compensated_increment
            time_compensation = (
                updated_time - physical_time
            ) - compensated_increment
            physical_time = updated_time

            for component in range(12):
                state[component] = new_state[component]
            accepted_steps += 1
            refresh_derivative = False

            if _electron_is_closer_to_other_core(state):
                _switch_reference_core(state)
                parameters_temporary = reference_parameters
                reference_parameters = other_parameters
                other_parameters = parameters_temporary
                mass_temporary = reference_mass_au
                reference_mass_au = other_mass_au
                other_mass_au = mass_temporary
                reference_was_switched = not reference_was_switched
                refresh_derivative = True

            current_energy = _relative_energy(
                state,
                reference_parameters,
                other_parameters,
                minimum_radius_au,
                reference_mass_au,
                other_mass_au,
            )
            relative_energy_drift = abs(
                current_energy - conserved_energy
            ) / max(abs(conserved_energy), 1.0)
            if (
                relative_energy_drift
                > 0.25 * maximum_relative_energy_drift
            ):
                if not _project_velocity_to_energy(
                    state,
                    conserved_energy,
                    reference_parameters,
                    other_parameters,
                    minimum_radius_au,
                    reference_mass_au,
                    other_mass_au,
                ):
                    if reference_was_switched:
                        _switch_reference_core(state)
                    return False, state, accepted_steps, rejected_steps
                refresh_derivative = True

            if refresh_derivative:
                initial_time_scale = _regularized_relative_rhs(
                    state,
                    derivative,
                    reference_parameters,
                    other_parameters,
                    minimum_radius_au,
                    reference_mass_au,
                    other_mass_au,
                    sundman_power,
                )
            else:
                for component in range(12):
                    derivative[component] = stages[_N_STAGES, component]
                initial_time_scale = time_scales[_N_STAGES]

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
            if reference_was_switched:
                _switch_reference_core(state)
            return False, state, accepted_steps, rejected_steps

    if reference_was_switched:
        _switch_reference_core(state)
    return False, state, accepted_steps, rejected_steps


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
    integrate_relative_dop853_regularized(
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
        1,
        1.0e-3,
        100,
    )
