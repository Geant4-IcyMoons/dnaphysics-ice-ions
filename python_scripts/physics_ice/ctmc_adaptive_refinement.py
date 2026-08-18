"""Restart-safe nested-grid refinement for charge-exchange CTMC tables.

The Liamsuwan--Nikjoo equations define the cross-section integral but do not
publish a universally converged impact-parameter or projectile-energy grid.
This module therefore changes no physical model parameter.  It bisects the
implemented impact intervals, validates logarithmic energy interpolation at
calculated midpoints, propagates multinomial/binomial sampling covariance
through the published many-electron estimator, and accepts a curve only when
every physically active reported cross section satisfies the disclosed
discretization and statistical limits.

Refinement has two dependency-ordered stages.  The first calculates each
``(base energy, charge)`` impact curve exactly once.  The second assigns one
charge state and one original energy interval to each independent family,
reuses the two shared endpoint curves, and calculates only new logarithmic
midpoints.  This avoids recomputing interior base-grid endpoints while retaining
deterministic multi-node sharding and restart-safe family checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import signal
from statistics import NormalDist
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tqdm import tqdm

import charge_exchange_ctmc as ctmc


ADAPTIVE_MODEL_VERSION = 2
_ADAPTIVE_SEED_NAMESPACE = 0xAD4B1E5


@dataclass(frozen=True)
class AdaptiveRefinementConfig:
    """Disclosed numerical convergence controls; none are physical inputs."""

    axis_relative_tolerance: float = ctmc.CTMC_ADAPTIVE_AXIS_RELATIVE_TOLERANCE
    combined_relative_tolerance: float = (
        ctmc.CTMC_ADAPTIVE_COMBINED_RELATIVE_TOLERANCE
    )
    statistical_relative_tolerance: float = (
        ctmc.CTMC_ADAPTIVE_STATISTICAL_RELATIVE_TOLERANCE
    )
    statistical_confidence: float = ctmc.CTMC_ADAPTIVE_STATISTICAL_CONFIDENCE
    max_impact_levels: int = ctmc.CTMC_ADAPTIVE_MAX_IMPACT_LEVELS
    max_energy_levels: int = ctmc.CTMC_ADAPTIVE_MAX_ENERGY_LEVELS
    max_sampling_levels: int = ctmc.CTMC_ADAPTIVE_MAX_SAMPLING_LEVELS

    def __post_init__(self) -> None:
        for name, value in (
            ("axis_relative_tolerance", self.axis_relative_tolerance),
            ("combined_relative_tolerance", self.combined_relative_tolerance),
            (
                "statistical_relative_tolerance",
                self.statistical_relative_tolerance,
            ),
            ("statistical_confidence", self.statistical_confidence),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if 2.0 * self.axis_relative_tolerance > self.combined_relative_tolerance:
            raise ValueError(
                "Twice axis_relative_tolerance must not exceed the combined "
                "discretization tolerance"
            )
        if self.max_impact_levels < 1 or self.max_energy_levels < 1:
            raise ValueError("Impact and energy refinement levels must be positive")
        if self.max_sampling_levels < 0:
            raise ValueError("Sampling refinement levels must be nonnegative")


class AdaptiveConvergenceError(RuntimeError):
    """A nested grid reached its safety limit without meeting tolerance."""


def adaptive_config_from_args(args: object) -> AdaptiveRefinementConfig:
    return AdaptiveRefinementConfig(
        axis_relative_tolerance=float(args.adaptive_axis_relative_tolerance),
        combined_relative_tolerance=float(
            args.adaptive_combined_relative_tolerance
        ),
        statistical_relative_tolerance=float(
            args.adaptive_statistical_relative_tolerance
        ),
        statistical_confidence=float(args.adaptive_statistical_confidence),
        max_impact_levels=int(args.adaptive_max_impact_levels),
        max_energy_levels=int(args.adaptive_max_energy_levels),
        max_sampling_levels=int(args.adaptive_max_sampling_levels),
    )


def adaptive_family_count(energies: np.ndarray, charges: np.ndarray) -> int:
    if energies.size < 2:
        return int(charges.size)
    return int((energies.size - 1) * charges.size)


def adaptive_family_indices(
    energies: np.ndarray,
    charges: np.ndarray,
) -> list[tuple[int, int]]:
    """Return ``(charge_index, interval_index)`` with charge varying fastest."""
    interval_count = max(int(energies.size) - 1, 1)
    return [
        (charge_index, interval_index)
        for interval_index in range(interval_count)
        for charge_index in range(int(charges.size))
    ]


def adaptive_base_curve_indices(
    energies: np.ndarray,
    charges: np.ndarray,
) -> list[tuple[int, int]]:
    """Return unique ``(charge_index, energy_index)`` base-curve tasks."""
    return [
        (charge_index, energy_index)
        for energy_index in range(int(energies.size))
        for charge_index in range(int(charges.size))
    ]


def _adaptive_directory(output_dir: Path) -> Path:
    return output_dir / "adaptive_refinement"


def _base_curve_stem(curve_index: int, curve_count: int) -> str:
    return (
        f"{ctmc.PROJECTILE.key}_adaptive_base_curve-"
        f"{curve_index:05d}-of-{curve_count:05d}"
    )


def _base_curve_checkpoint_path(
    output_dir: Path,
    curve_index: int,
    curve_count: int,
) -> Path:
    return _adaptive_directory(output_dir) / (
        _base_curve_stem(curve_index, curve_count) + ".checkpoint.npz"
    )


def _base_curve_result_path(
    output_dir: Path,
    curve_index: int,
    curve_count: int,
) -> Path:
    return _adaptive_directory(output_dir) / (
        _base_curve_stem(curve_index, curve_count) + ".result.npz"
    )


def _interval_stem(interval_index: int, interval_count: int) -> str:
    return (
        f"{ctmc.PROJECTILE.key}_adaptive_interval-"
        f"{interval_index:05d}-of-{interval_count:05d}"
    )


def _interval_checkpoint_path(
    output_dir: Path,
    interval_index: int,
    interval_count: int,
) -> Path:
    return _adaptive_directory(output_dir) / (
        _interval_stem(interval_index, interval_count) + ".checkpoint.npz"
    )


def _interval_result_path(
    output_dir: Path,
    interval_index: int,
    interval_count: int,
) -> Path:
    return _adaptive_directory(output_dir) / (
        _interval_stem(interval_index, interval_count) + ".result.npz"
    )


def _manifest_path(output_dir: Path) -> Path:
    return _adaptive_directory(output_dir) / (
        f"{ctmc.PROJECTILE.key}_adaptive_manifest.json"
    )


def nested_impact_grids(
    base_impact_au: np.ndarray,
    mandatory_impact_au: Sequence[float],
    levels: int,
) -> tuple[np.ndarray, ...]:
    """Build exactly nested grids by bisecting every current interval."""
    initial = np.unique(
        np.concatenate(
            (
                np.asarray(base_impact_au, dtype=float),
                np.asarray(mandatory_impact_au, dtype=float),
            )
        )
    )
    if initial.size < 2 or initial[0] < 0.0:
        raise ValueError("Impact grid must contain at least two nonnegative points")
    grids = [initial]
    for _ in range(levels):
        current = grids[-1]
        midpoints = 0.5 * (current[:-1] + current[1:])
        grids.append(np.sort(np.concatenate((current, midpoints))))
    return tuple(grids)


def nested_interval_energies(
    lower_energy: float,
    upper_energy: float,
    levels: int,
) -> np.ndarray:
    """Return the finest log-energy grid for one original energy interval."""
    if lower_energy <= 0.0 or upper_energy <= lower_energy:
        raise ValueError("Adaptive energy interval must be positive and ordered")
    energies = np.geomspace(lower_energy, upper_energy, 2**levels + 1)
    # Preserve the exact base-grid bit patterns so neighboring families share
    # identical coordinate-keyed trajectory streams at their common endpoint.
    energies[0] = lower_energy
    energies[-1] = upper_energy
    return energies


def _indices_in_finest_grid(
    level_grid: np.ndarray,
    finest_grid: np.ndarray,
) -> np.ndarray:
    lookup = {float(value).hex(): index for index, value in enumerate(finest_grid)}
    try:
        return np.asarray(
            [lookup[float(value).hex()] for value in level_grid],
            dtype=np.int64,
        )
    except KeyError as exc:  # pragma: no cover - guards construction invariant.
        raise RuntimeError("Nested grid lost a parent point") from exc


def _active_cross_section_indices(charge_state: int) -> tuple[int, ...]:
    # E and q occupy columns 0 and 1.  The six independent reported channels
    # follow; decrease/increase are exact sums and add no convergence condition.
    indices = [2, 3, 8, 9]  # SC, TI, SI, DI
    if charge_state < ctmc.PROJECTILE_NUCLEAR_CHARGE:
        indices.extend((4, 5))  # SL, LI
    if charge_state == 0:
        indices = [index for index in indices if index not in (2, 3)]
    return tuple(sorted(indices))


_CROSS_SECTION_NAMES = (
    "E_keV_u",
    "q",
    "sigma_SC_cm2",
    "sigma_TI_cm2",
    "sigma_SL_cm2",
    "sigma_LI_cm2",
    "sigma_decrease_cm2",
    "sigma_increase_cm2",
    "sigma_SI_cm2",
    "sigma_DI_cm2",
)


def _dual_constant(value: float, variable_count: int) -> tuple[float, np.ndarray]:
    return float(value), np.zeros(variable_count, dtype=float)


def _dual_variable(
    value: float, index: int, variable_count: int
) -> tuple[float, np.ndarray]:
    derivative = np.zeros(variable_count, dtype=float)
    derivative[index] = 1.0
    return float(value), derivative


def _dual_add(
    left: tuple[float, np.ndarray], right: tuple[float, np.ndarray]
) -> tuple[float, np.ndarray]:
    return left[0] + right[0], left[1] + right[1]


def _dual_scale(
    value: tuple[float, np.ndarray], factor: float
) -> tuple[float, np.ndarray]:
    return factor * value[0], factor * value[1]


def _dual_multiply(
    left: tuple[float, np.ndarray], right: tuple[float, np.ndarray]
) -> tuple[float, np.ndarray]:
    return (
        left[0] * right[0],
        left[1] * right[0] + right[1] * left[0],
    )


def _dual_power(
    value: tuple[float, np.ndarray], exponent: int
) -> tuple[float, np.ndarray]:
    if exponent < 0:
        raise ValueError("Dual-number exponent must be nonnegative")
    if exponent == 0:
        return _dual_constant(1.0, value[1].size)
    if exponent == 1:
        return value[0], value[1].copy()
    return (
        value[0] ** exponent,
        exponent * value[0] ** (exponent - 1) * value[1],
    )


def _many_electron_probability_jacobian(
    pi: np.ndarray,
    pc: np.ndarray,
    pl: float,
    charge_state: int,
) -> dict[str, tuple[float, np.ndarray]]:
    """Evaluate the published IEVM/IPM weights and their exact Jacobian."""
    orbital_count = len(ctmc.WATER_ORBITALS)
    if pi.shape != (orbital_count,) or pc.shape != (orbital_count,):
        raise ValueError("Unexpected CTMC orbital-probability shape")
    if (
        np.any(pi < 0.0)
        or np.any(pc < 0.0)
        or np.any(pi + pc > 1.0 + 1.0e-12)
        or not 0.0 <= pl <= 1.0
    ):
        raise ValueError("Primitive CTMC probabilities are outside their domain")

    variable_count = 2 * orbital_count + 1
    pi_dual = [
        _dual_variable(float(value), index, variable_count)
        for index, value in enumerate(pi)
    ]
    pc_dual = [
        _dual_variable(float(value), orbital_count + index, variable_count)
        for index, value in enumerate(pc)
    ]
    pl_dual = _dual_variable(float(pl), 2 * orbital_count, variable_count)
    one = _dual_constant(1.0, variable_count)
    zero = _dual_constant(0.0, variable_count)

    si = zero
    di = zero
    for shell_index, orbital in enumerate(ctmc.WATER_ORBITALS):
        electron_count = int(orbital.active_electrons)
        pn = _dual_add(
            one,
            _dual_scale(
                _dual_add(pi_dual[shell_index], pc_dual[shell_index]),
                -1.0,
            ),
        )
        si = _dual_add(
            si,
            _dual_scale(
                _dual_multiply(
                    pi_dual[shell_index],
                    _dual_power(pn, electron_count - 1),
                ),
                float(electron_count),
            ),
        )
        if electron_count >= 2:
            di = _dual_add(
                di,
                _dual_scale(
                    _dual_multiply(
                        _dual_power(pi_dual[shell_index], 2),
                        _dual_power(pn, electron_count - 2),
                    ),
                    0.5 * electron_count * (electron_count - 1),
                ),
            )

    distribution = [[zero, zero], [zero, zero]]
    distribution[0][0] = one
    for shell_index, orbital in enumerate(ctmc.WATER_ORBITALS):
        pn = _dual_add(
            one,
            _dual_scale(
                _dual_add(pi_dual[shell_index], pc_dual[shell_index]),
                -1.0,
            ),
        )
        for _ in range(int(orbital.active_electrons)):
            updated = [[zero, zero], [zero, zero]]
            updated[0][0] = _dual_multiply(distribution[0][0], pn)
            updated[1][0] = _dual_add(
                _dual_multiply(distribution[1][0], pn),
                _dual_multiply(distribution[0][0], pc_dual[shell_index]),
            )
            updated[0][1] = _dual_add(
                _dual_multiply(distribution[0][1], pn),
                _dual_multiply(distribution[0][0], pi_dual[shell_index]),
            )
            updated[1][1] = _dual_add(
                _dual_add(
                    _dual_multiply(distribution[1][1], pn),
                    _dual_multiply(distribution[1][0], pi_dual[shell_index]),
                ),
                _dual_multiply(distribution[0][1], pc_dual[shell_index]),
            )
            distribution = updated
    sc, ti = distribution[1][0], distribution[1][1]
    if charge_state == 0:
        sc = ti = zero

    projectile_orbital = ctmc.PROJECTILE_OUTER_ORBITAL.get(int(charge_state))
    if projectile_orbital is None:
        sl = zero
    else:
        electron_count = int(projectile_orbital.active_electrons)
        survival = _dual_add(one, _dual_scale(pl_dual, -1.0))
        sl = _dual_scale(
            _dual_multiply(pl_dual, _dual_power(survival, electron_count - 1)),
            float(electron_count),
        )
    li = _dual_multiply(si, sl)
    if charge_state == ctmc.PROJECTILE_NUCLEAR_CHARGE:
        sl = li = zero

    return {
        "SC": sc,
        "TI": ti,
        "SL": sl,
        "LI": li,
        "SI": si,
        "DI": di,
    }


def _trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("At least two one-dimensional coordinates are required")
    widths = np.diff(coordinates)
    if np.any(widths <= 0.0):
        raise ValueError("Coordinates must be strictly increasing")
    weights = np.empty_like(coordinates, dtype=float)
    weights[0] = 0.5 * widths[0]
    weights[-1] = 0.5 * widths[-1]
    if coordinates.size > 2:
        weights[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    return weights


def cross_section_relative_confidence_half_widths(
    impact_au: np.ndarray,
    pi: np.ndarray,
    pc: np.ndarray,
    pl: np.ndarray,
    channel_successes: np.ndarray,
    charge_state: int,
    confidence: float,
) -> dict[str, float]:
    """Delta-method Monte Carlo confidence widths for reported cross sections.

    Target ionization and capture counts are treated as multinomial outcomes,
    including their negative covariance.  Different orbitals, projectile-loss
    samples, and impact parameters use independent trajectory streams.
    """
    impact_au = np.asarray(impact_au, dtype=float)
    pi = np.asarray(pi, dtype=float)
    pc = np.asarray(pc, dtype=float)
    pl = np.asarray(pl, dtype=float)
    channel_successes = np.asarray(channel_successes, dtype=np.int64)
    orbital_count = len(ctmc.WATER_ORBITALS)
    expected = (impact_au.size, orbital_count)
    if pi.shape != expected or pc.shape != expected:
        raise ValueError("pi and pc do not match the impact grid")
    if pl.shape != (impact_au.size,):
        raise ValueError("pl does not match the impact grid")
    if channel_successes.shape != (impact_au.size, ctmc.CHANNEL_COUNT):
        raise ValueError("channel_successes does not match the impact grid")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    active_names = tuple(
        _CROSS_SECTION_NAMES[index].removeprefix("sigma_").removesuffix("_cm2")
        for index in _active_cross_section_indices(charge_state)
    )
    probability_values = {name: np.empty(impact_au.size) for name in active_names}
    probability_variances = {
        name: np.empty(impact_au.size) for name in active_names
    }
    variable_count = 2 * orbital_count + 1
    for impact_index in range(impact_au.size):
        weights = _many_electron_probability_jacobian(
            pi[impact_index], pc[impact_index], float(pl[impact_index]), charge_state
        )
        covariance = np.zeros((variable_count, variable_count), dtype=float)
        for orbital_index in range(orbital_count):
            sample_count = int(channel_successes[impact_index, orbital_index])
            if sample_count <= 0:
                continue
            ionized = float(pi[impact_index, orbital_index])
            captured = float(pc[impact_index, orbital_index])
            ionized_index = orbital_index
            captured_index = orbital_count + orbital_index
            covariance[ionized_index, ionized_index] = (
                ionized * (1.0 - ionized) / sample_count
            )
            covariance[captured_index, captured_index] = (
                captured * (1.0 - captured) / sample_count
            )
            cross_covariance = -ionized * captured / sample_count
            covariance[ionized_index, captured_index] = cross_covariance
            covariance[captured_index, ionized_index] = cross_covariance
        loss_samples = int(
            channel_successes[impact_index, ctmc.LOSS_CHANNEL_INDEX]
        )
        if loss_samples > 0:
            loss_index = 2 * orbital_count
            covariance[loss_index, loss_index] = (
                pl[impact_index] * (1.0 - pl[impact_index]) / loss_samples
            )
        for name in active_names:
            value, gradient = weights[name]
            variance = float(gradient @ covariance @ gradient)
            probability_values[name][impact_index] = value
            probability_variances[name][impact_index] = max(variance, 0.0)

    integration_coefficients = (
        2.0
        * math.pi
        * ctmc.BOHR2_CM2
        * impact_au
        * _trapezoid_weights(impact_au)
    )
    normal_quantile = NormalDist().inv_cdf(0.5 * (1.0 + confidence))
    relative_half_widths: dict[str, float] = {}
    for name in active_names:
        cross_section = float(
            np.sum(integration_coefficients * probability_values[name])
        )
        variance = float(
            np.sum(
                integration_coefficients**2 * probability_variances[name]
            )
        )
        if cross_section <= 0.0:
            relative_half_widths[name] = math.inf
        else:
            relative_half_widths[name] = (
                normal_quantile * math.sqrt(variance) / cross_section
            )
    return relative_half_widths


def maximum_relative_change(
    coarse_row: np.ndarray,
    fine_row: np.ndarray,
    charge_state: int,
) -> tuple[float, dict[str, float]]:
    """Return strict relative changes for independent physical channels."""
    changes: dict[str, float] = {}
    for index in _active_cross_section_indices(charge_state):
        coarse = float(coarse_row[index])
        fine = float(fine_row[index])
        if coarse == 0.0 and fine == 0.0:
            change = 0.0
        elif fine == 0.0:
            change = math.inf
        else:
            change = abs(fine - coarse) / abs(fine)
        changes[_CROSS_SECTION_NAMES[index]] = float(change)
    return max(changes.values(), default=0.0), changes


def maximum_log_midpoint_error(
    lower_row: np.ndarray,
    midpoint_row: np.ndarray,
    upper_row: np.ndarray,
    charge_state: int,
) -> tuple[float, dict[str, float]]:
    """Compare a calculated midpoint with log-log endpoint interpolation."""
    errors: dict[str, float] = {}
    for index in _active_cross_section_indices(charge_state):
        lower = float(lower_row[index])
        actual = float(midpoint_row[index])
        upper = float(upper_row[index])
        if lower == 0.0 and actual == 0.0 and upper == 0.0:
            error = 0.0
        elif lower <= 0.0 or actual <= 0.0 or upper <= 0.0:
            # A relative log-log interpolation error is undefined at a sampled
            # zero.  Do not hide that unresolved Monte Carlo limit with a floor.
            error = math.inf
        else:
            predicted = math.sqrt(lower * upper)
            error = abs(actual - predicted) / actual
        errors[_CROSS_SECTION_NAMES[index]] = float(error)
    return max(errors.values(), default=0.0), errors


def _family_signature(
    base_signature: str,
    family_index: int,
    family_count: int,
    energies: np.ndarray,
    impact_au: np.ndarray,
    config: ctmc.CTMCConfig,
    adaptive: AdaptiveRefinementConfig,
    task_kind: str,
) -> str:
    config_payload = asdict(config)
    config_payload.pop("trajectory_chunk_size", None)
    # Adaptive checkpoints created by the first production pass contain the
    # audited 25-million-attempt ceiling.  The 100-million default was selected
    # after an exact failed adaptive seed required 53,384,632 attempts to reach
    # the unchanged physical exit boundary.  Both ceilings produce identical
    # accepted endpoints; canonicalizing only this audited transition preserves
    # the completed trajectory ledger.  Other ceilings remain signature-bound.
    if config_payload["maximum_integration_steps"] in (
        ctmc.CARBON_CTMC_LEGACY_MAXIMUM_INTEGRATION_STEPS,
        ctmc.CARBON_CTMC_MAXIMUM_INTEGRATION_STEPS,
    ):
        config_payload["maximum_integration_steps"] = (
            ctmc.CARBON_CTMC_LEGACY_MAXIMUM_INTEGRATION_STEPS
        )
    if not np.isfinite(config_payload["max_step_au"]):
        config_payload["max_step_au"] = None
    payload = {
        "adaptive_model_version": ADAPTIVE_MODEL_VERSION,
        "base_signature": base_signature,
        "family_index": int(family_index),
        "family_count": int(family_count),
        "energies": energies.tolist(),
        "impact_au": impact_au.tolist(),
        "ctmc_config": config_payload,
        "adaptive_config": asdict(adaptive),
    }
    payload["task_kind"] = task_kind
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float_seed_words(value: float) -> tuple[int, int]:
    bits = int(np.asarray(float(value), dtype=np.float64).view(np.uint64))
    return bits & 0xFFFFFFFF, bits >> 32


def _coordinate_rng(
    master_seed: int,
    energy_keV_u: float,
    charge_state: int,
    impact_au: float,
    channel_index: int,
    trajectory_start: int,
    initial_ensemble: str = ctmc.INITIAL_ENSEMBLE_LIAMSUWAN_OLSON_SALOP,
) -> np.random.Generator:
    """Stable stream keyed by physical coordinates, independent of refinement."""
    energy_low, energy_high = _float_seed_words(energy_keV_u)
    impact_low, impact_high = _float_seed_words(impact_au)
    seed = np.random.SeedSequence(
        [
            int(master_seed),
            _ADAPTIVE_SEED_NAMESPACE,
            energy_low,
            energy_high,
            int(charge_state),
            impact_low,
            impact_high,
            int(channel_index),
        ]
    )
    bit_generator = np.random.PCG64(seed)
    bit_generator.advance(
        int(trajectory_start)
        * ctmc.random_draws_per_trajectory(initial_ensemble)
    )
    return np.random.Generator(bit_generator)


def _compute_adaptive_trajectory_chunk(
    task: tuple[int, int, float, int, int, float, int, int, int, float, float],
) -> tuple[int, int, int, int, int, int, int, int, int, int, float]:
    (
        energy_index,
        charge_index,
        energy,
        charge,
        impact_index,
        impact,
        channel_index,
        trajectory_start,
        trajectory_count,
        start_separation_au,
        minimum_integration_time_au,
    ) = task
    config = ctmc._WORKER_CONFIG
    if config is None:
        raise RuntimeError("Adaptive CTMC worker was not initialized")
    rng = _coordinate_rng(
        config.seed,
        energy,
        charge,
        impact,
        channel_index,
        trajectory_start,
        config.initial_ensemble,
    )

    if channel_index < ctmc.LOSS_CHANNEL_INDEX:
        orbital = ctmc.WATER_ORBITALS[channel_index]
        bound_to = "target"
    elif channel_index == ctmc.LOSS_CHANNEL_INDEX:
        orbital = ctmc.PROJECTILE_OUTER_ORBITAL.get(int(charge))
        if orbital is None:
            raise RuntimeError(
                f"Invalid adaptive projectile-loss task for q={charge}"
            )
        bound_to = "projectile"
    else:  # pragma: no cover - protected by task construction.
        raise RuntimeError(f"Invalid CTMC channel index {channel_index}")

    primary_events = 0
    secondary_events = 0
    failures = 0
    successes = 0
    maximum_energy_drift = 0.0
    for trajectory_offset in range(trajectory_count):
        outcome, success, drift = ctmc.simulate_one_trajectory(
            energy_keV_u=energy,
            charge_state=charge,
            impact_parameter_au=impact,
            bound_to=bound_to,
            binding_eV=orbital.binding_eV,
            start_separation_au=start_separation_au,
            minimum_integration_time_au=minimum_integration_time_au,
            config=config,
            rng=rng,
        )
        if not success:
            if config.max_failure_fraction == 0.0:
                raise ctmc.TrajectoryIntegrationFailure(
                    {
                        "adaptive": True,
                        "energy_index": int(energy_index),
                        "energy_keV_u": float(energy),
                        "charge_state": int(charge),
                        "impact_index": int(impact_index),
                        "impact_parameter_au": float(impact),
                        "channel_index": int(channel_index),
                        "trajectory_index": int(
                            trajectory_start + trajectory_offset
                        ),
                        "failure_outcome": outcome,
                        "relative_total_energy_drift": float(drift),
                    }
                )
            failures += 1
            continue
        successes += 1
        maximum_energy_drift = max(maximum_energy_drift, drift)
        primary_events += int(outcome == "ionized")
        if channel_index < ctmc.LOSS_CHANNEL_INDEX:
            secondary_events += int(outcome == "captured_projectile")

    return (
        energy_index,
        charge_index,
        impact_index,
        channel_index,
        trajectory_start,
        trajectory_count,
        primary_events,
        secondary_events,
        failures,
        successes,
        maximum_energy_drift,
    )


def _bounded_results(
    executor: ProcessPoolExecutor,
    tasks: Iterable[
        tuple[int, int, float, int, int, float, int, int, int, float, float]
    ],
    maximum_pending: int,
    heartbeat_seconds: float = 1.0,
) -> Iterable[ctmc.TrajectoryChunkResult | None]:
    """Return adaptive chunks and periodic checkpoint heartbeats."""
    task_iterator = iter(tasks)
    pending = set()
    for _ in range(maximum_pending):
        try:
            task = next(task_iterator)
        except StopIteration:
            break
        pending.add(executor.submit(_compute_adaptive_trajectory_chunk, task))
    while pending:
        completed, pending = wait(
            pending,
            timeout=heartbeat_seconds,
            return_when=FIRST_COMPLETED,
        )
        if not completed:
            yield None
            continue
        for future in completed:
            yield future.result()
            try:
                task = next(task_iterator)
            except StopIteration:
                continue
            pending.add(executor.submit(_compute_adaptive_trajectory_chunk, task))


def _curve_row(
    energy_index: int,
    impact_indices: np.ndarray,
    energies: np.ndarray,
    charge_state: int,
    impact_au: np.ndarray,
    pi: np.ndarray,
    pc: np.ndarray,
    pl: np.ndarray,
) -> np.ndarray:
    rows, _ = ctmc.build_cross_section_rows(
        np.asarray([energies[energy_index]]),
        np.asarray([charge_state]),
        impact_au[impact_indices],
        pi[energy_index : energy_index + 1, :, impact_indices],
        pc[energy_index : energy_index + 1, :, impact_indices],
        pl[energy_index : energy_index + 1, :, impact_indices],
    )
    return rows[0]


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_savetxt(path: Path, array: np.ndarray, **options: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.savetxt(temporary, array, **options)
    os.replace(temporary, path)


def write_adaptive_manifest(
    output_dir: Path,
    *,
    base_signature: str,
    energies: np.ndarray,
    charges: np.ndarray,
    adaptive: AdaptiveRefinementConfig,
) -> Path:
    family_count = adaptive_family_count(energies, charges)
    base_curve_count = int(energies.size * charges.size)
    path = _manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "adaptive_refinement_required",
        "projectile": ctmc.PROJECTILE.key,
        "base_configuration_signature": base_signature,
        "base_energy_count": int(energies.size),
        "charge_count": int(charges.size),
        "adaptive_family_count": family_count,
        "adaptive_base_curve_count": base_curve_count,
        "execution_stages": [
            "unique_base_energy_charge_curves",
            "energy_intervals_reusing_shared_endpoints",
            "merge",
        ],
        "endpoint_recalculation": False,
        "adaptive_config": asdict(adaptive),
        "error_definition": (
            "separate nested-impact-grid and logarithmic-energy interpolation "
            "limits with their conservative sum bounded by the configured "
            "combined discretization tolerance"
        ),
        "statistical_note": (
            "Every active cross section must separately satisfy the configured "
            "two-sided asymptotic Monte Carlo confidence half-width. Primitive "
            "multinomial/"
            "binomial count covariance is propagated through the published "
            "IEVM/IPM estimator by the delta method."
        ),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _restore_family_checkpoint(
    checkpoint: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    accumulators: ctmc.TrajectoryAccumulators,
) -> None:
    destinations = {
        "pi": arrays["pi"],
        "pc": arrays["pc"],
        "pl": arrays["pl"],
        "done": arrays["done"],
        "failures": arrays["failures"],
        "successes": arrays["successes"],
        "maximum_energy_drift": arrays["maximum_energy_drift"],
        "channel_primary_events": accumulators.primary_events,
        "channel_secondary_events": accumulators.secondary_events,
        "channel_failures": accumulators.failures,
        "channel_successes": accumulators.successes,
        "channel_maximum_energy_drift": accumulators.maximum_energy_drift,
        "channel_completed": accumulators.completed,
    }
    missing = [name for name in destinations if name not in checkpoint]
    if missing:
        raise RuntimeError(
            "Adaptive checkpoint lacks restart arrays: " + ", ".join(missing)
        )
    for name, destination in destinations.items():
        destination[...] = checkpoint[name]


def _copy_base_endpoint(
    *,
    local_energy_index: int,
    base_energy_index: int,
    base_impact_au: np.ndarray,
    finest_impact_au: np.ndarray,
    base_checkpoint: dict[str, np.ndarray],
    charge_index: int,
    arrays: dict[str, np.ndarray],
    accumulators: ctmc.TrajectoryAccumulators,
) -> None:
    lookup = {
        float(value).hex(): index
        for index, value in enumerate(finest_impact_au)
    }
    for base_impact_index, impact in enumerate(base_impact_au):
        fine_impact_index = lookup[float(impact).hex()]
        local = (local_energy_index, 0, fine_impact_index)
        base = (base_energy_index, charge_index, base_impact_index)
        for name in (
            "pi",
            "pc",
            "pl",
            "done",
            "failures",
            "successes",
            "maximum_energy_drift",
        ):
            arrays[name][local] = base_checkpoint[name][base]
        for source_name, destination in (
            ("channel_primary_events", accumulators.primary_events),
            ("channel_secondary_events", accumulators.secondary_events),
            ("channel_failures", accumulators.failures),
            ("channel_successes", accumulators.successes),
            (
                "channel_maximum_energy_drift",
                accumulators.maximum_energy_drift,
            ),
            ("channel_completed", accumulators.completed),
        ):
            destination[local] = base_checkpoint[source_name][base]


def _run_family(
    *,
    output_dir: Path,
    family_index: int,
    family_count: int,
    charge_index: int,
    interval_index: int,
    base_signature: str,
    base_checkpoint: dict[str, np.ndarray],
    base_config: ctmc.CTMCConfig,
    workers: int,
    start_method: str,
    pending_factor: int,
    checkpoint_every: int,
    checkpoint_seconds: float,
    adaptive: AdaptiveRefinementConfig,
    base_energy_index: int | None,
    shared_base_curves: bool,
) -> Path:
    base_energies = np.asarray(base_checkpoint["energies_keV_u"], dtype=float)
    charges = np.asarray(base_checkpoint["charges"], dtype=int)
    base_impact = np.asarray(base_checkpoint["impact_au"], dtype=float)
    charge_state = int(charges[charge_index])

    if base_energy_index is not None:
        if base_energy_index < 0 or base_energy_index >= base_energies.size:
            raise ValueError("base_energy_index is out of range")
        lower_index = upper_index = int(base_energy_index)
        energies = np.asarray([base_energies[base_energy_index]], dtype=float)
        task_kind = "unique_base_curve"
    elif shared_base_curves:
        if base_energies.size == 1:
            lower_index = upper_index = 0
            energies = np.asarray([base_energies[0]], dtype=float)
        else:
            lower_index = interval_index
            upper_index = interval_index + 1
            energies = nested_interval_energies(
                float(base_energies[lower_index]),
                float(base_energies[upper_index]),
                adaptive.max_energy_levels,
            )
        task_kind = "interval_with_shared_endpoints"
    else:
        raise ValueError(
            "An adaptive task must be a unique base curve or a shared-endpoint "
            "energy interval"
        )

    mandatory_b = tuple(base_config.target_bmax_au) + tuple(
        base_config.loss_bmax_au
    )
    impact_levels = nested_impact_grids(
        base_impact,
        mandatory_b,
        adaptive.max_impact_levels,
    )
    impact_au = impact_levels[-1]
    impact_level_indices = tuple(
        _indices_in_finest_grid(level, impact_au) for level in impact_levels
    )

    if energies.size == 1:
        start_separations = (base_config.start_separation_au[lower_index],)
        minimum_times = (base_config.minimum_integration_time_au[lower_index],)
    else:
        endpoint_separations = (
            base_config.start_separation_au[lower_index],
            base_config.start_separation_au[upper_index],
        )
        endpoint_times = (
            base_config.minimum_integration_time_au[lower_index],
            base_config.minimum_integration_time_au[upper_index],
        )
        conservative_separation = max(endpoint_separations)
        conservative_time = max(endpoint_times)
        start_separations = tuple(
            endpoint_separations[0]
            if index == 0
            else endpoint_separations[1]
            if index == energies.size - 1
            else conservative_separation
            for index in range(energies.size)
        )
        minimum_times = tuple(
            endpoint_times[0]
            if index == 0
            else endpoint_times[1]
            if index == energies.size - 1
            else conservative_time
            for index in range(energies.size)
        )
    config = replace(
        base_config,
        start_separation_au=start_separations,
        minimum_integration_time_au=minimum_times,
    )

    shape = (energies.size, 1, impact_au.size)
    arrays: dict[str, np.ndarray] = {
        "pi": np.full(shape + (len(ctmc.WATER_ORBITALS),), np.nan),
        "pc": np.full(shape + (len(ctmc.WATER_ORBITALS),), np.nan),
        "pl": np.full(shape, np.nan),
        "done": np.zeros(shape, dtype=bool),
        "failures": np.zeros(shape, dtype=np.int64),
        "successes": np.zeros(shape, dtype=np.int64),
        "maximum_energy_drift": np.zeros(shape),
    }
    accumulators = ctmc.create_trajectory_accumulators(shape)
    family_signature = _family_signature(
        base_signature,
        family_index,
        family_count,
        energies,
        impact_au,
        config,
        adaptive,
        task_kind,
    )
    if base_energy_index is not None:
        checkpoint_path = _base_curve_checkpoint_path(
            output_dir, family_index, family_count
        )
        result_path = _base_curve_result_path(
            output_dir, family_index, family_count
        )
    else:
        checkpoint_path = _interval_checkpoint_path(
            output_dir, family_index, family_count
        )
        result_path = _interval_result_path(
            output_dir, family_index, family_count
        )
    if result_path.exists():
        with np.load(result_path, allow_pickle=False) as result:
            if str(np.asarray(result["family_signature"]).item()) != family_signature:
                raise RuntimeError(
                    f"Adaptive result {result_path} has an incompatible signature"
                )
        return result_path

    checkpoint = ctmc.load_checkpoint(checkpoint_path, family_signature)
    buffered_results: ctmc.BufferedChunkResults = {}
    if checkpoint is None:
        _copy_base_endpoint(
            local_energy_index=0,
            base_energy_index=lower_index,
            base_impact_au=base_impact,
            finest_impact_au=impact_au,
            base_checkpoint=base_checkpoint,
            charge_index=charge_index,
            arrays=arrays,
            accumulators=accumulators,
        )
        if energies.size > 1:
            _copy_base_endpoint(
                local_energy_index=energies.size - 1,
                base_energy_index=upper_index,
                base_impact_au=base_impact,
                finest_impact_au=impact_au,
                base_checkpoint=base_checkpoint,
                charge_index=charge_index,
                arrays=arrays,
                accumulators=accumulators,
            )
    else:
        _restore_family_checkpoint(checkpoint, arrays, accumulators)
        buffered_results = ctmc.checkpoint_buffered_chunk_results(checkpoint)
        maximum_trajectory_target = (
            config.trajectories * 2**adaptive.max_sampling_levels
        )
        for key, channel_buffer in buffered_results.items():
            energy_index, charge_local_index, impact_index, channel_index = key
            point = (energy_index, charge_local_index, impact_index)
            if (
                energy_index < 0
                or energy_index >= energies.size
                or charge_local_index != 0
                or impact_index < 0
                or impact_index >= impact_au.size
            ):
                raise RuntimeError(
                    f"Adaptive checkpoint contains out-of-range buffered key {key}"
                )
            if arrays["done"][point]:
                raise RuntimeError(
                    f"Completed adaptive point {point} contains buffered chunks"
                )
            if channel_index not in ctmc._active_channels(
                charge_state,
                float(impact_au[impact_index]),
                config,
            ):
                raise RuntimeError(
                    f"Adaptive checkpoint contains inactive buffered channel {key}"
                )
            expected = int(accumulators.completed[key])
            for start, result in channel_buffer.items():
                count = int(result[5])
                if (
                    start < expected
                    or count <= 0
                    or start + count > maximum_trajectory_target
                    or count != int(result[8]) + int(result[9])
                ):
                    raise RuntimeError(
                        "Adaptive checkpoint contains invalid buffered chunk "
                        f"{key}@{start}"
                    )

    def save_checkpoint(announce: bool = False) -> None:
        ctmc.save_checkpoint(
            checkpoint_path,
            signature=family_signature,
            energies=energies,
            charges=np.asarray([charge_state]),
            impact_au=impact_au,
            pi=arrays["pi"],
            pc=arrays["pc"],
            pl=arrays["pl"],
            done=arrays["done"],
            failures=arrays["failures"],
            successes=arrays["successes"],
            maximum_energy_drift=arrays["maximum_energy_drift"],
            accumulators=accumulators,
            execution_workers=workers,
            buffered_results=buffered_results,
        )
        if announce:
            tqdm.write(
                f"Adaptive family {family_index + 1}/{family_count}: "
                f"checkpointed {int(np.count_nonzero(arrays['done'])):,} "
                f"points -> {checkpoint_path}"
            )

    process_executor: ProcessPoolExecutor | None = None
    if workers > 1:
        process_context = multiprocessing.get_context(start_method)
        process_executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=ctmc._init_worker,
            initargs=(config, True),
            mp_context=process_context,
        )
    else:
        ctmc._init_worker(config)

    def run_required_points(
        energy_indices: Sequence[int],
        impact_indices: np.ndarray,
        label: str,
        trajectory_target: int,
    ) -> int:
        nonlocal process_executor
        owner = np.zeros(shape, dtype=bool)
        owner[
            np.ix_(np.asarray(energy_indices), np.asarray([0]), impact_indices)
        ] = True
        effective_target = int(trajectory_target)
        for energy_index, _, impact_index in np.argwhere(owner):
            active_channels = ctmc._active_channels(
                charge_state, float(impact_au[impact_index]), config
            )
            if active_channels:
                effective_target = max(
                    effective_target,
                    max(
                        int(
                            accumulators.completed[
                                energy_index, 0, impact_index, channel
                            ]
                        )
                        for channel in active_channels
                    ),
                )
                for channel in active_channels:
                    key = (energy_index, 0, impact_index, channel)
                    if key in buffered_results:
                        effective_target = max(
                            effective_target,
                            max(
                                start + int(result[5])
                                for start, result in buffered_results[key].items()
                            ),
                        )
        sampling_config = replace(config, trajectories=effective_target)

        # A checkpoint can contain chunks that completed after a missing
        # prefix.  Commit any prefix that became contiguous before rebuilding
        # the pending task stream, then exclude every remaining buffered range.
        for key in list(buffered_results):
            point = key[:3]
            if not owner[point]:
                continue
            channel_buffer = buffered_results[key]
            expected = int(accumulators.completed[key])
            while expected in channel_buffer:
                contiguous = channel_buffer.pop(expected)
                ctmc._commit_trajectory_chunk(
                    contiguous,
                    energies=energies,
                    charges=np.asarray([charge_state]),
                    impact_au=impact_au,
                    config=sampling_config,
                    accumulators=accumulators,
                    pi=arrays["pi"],
                    pc=arrays["pc"],
                    pl=arrays["pl"],
                    done=arrays["done"],
                    failures=arrays["failures"],
                    successes=arrays["successes"],
                    maximum_energy_drift=arrays["maximum_energy_drift"],
                )
                expected += int(contiguous[5])
            if not channel_buffer:
                buffered_results.pop(key)
        for energy_index, _, impact_index in np.argwhere(owner):
            active_channels = ctmc._active_channels(
                charge_state, float(impact_au[impact_index]), sampling_config
            )
            arrays["done"][energy_index, 0, impact_index] = all(
                accumulators.completed[energy_index, 0, impact_index, channel]
                >= effective_target
                for channel in active_channels
            )
        remaining, _ = ctmc.estimate_remaining_trajectory_count(
            np.asarray([charge_state]),
            impact_au,
            arrays["done"],
            accumulators,
            sampling_config,
            owner,
        )
        buffered_trajectories = sum(
            int(result[5])
            for key, channel_buffer in buffered_results.items()
            if owner[key[:3]]
            for result in channel_buffer.values()
        )
        remaining -= buffered_trajectories
        if remaining < 0:
            raise RuntimeError(
                "Buffered adaptive work exceeds the remaining trajectory count"
            )
        if remaining == 0:
            return effective_target
        tasks = ctmc.iter_tasks_excluding_buffered_chunks(
            ctmc.iter_pending_trajectory_tasks(
                energies,
                np.asarray([charge_state]),
                impact_au,
                arrays["done"],
                accumulators,
                sampling_config,
                owner,
            ),
            buffered_results,
        )
        progress = tqdm(
            total=remaining,
            desc=f"Adaptive {family_index + 1}/{family_count} {label}",
            unit="traj",
            unit_scale=True,
            dynamic_ncols=True,
        )
        completed_since_checkpoint = 0
        last_checkpoint_time = time.monotonic()

        def accept(result: tuple) -> None:
            nonlocal completed_since_checkpoint, last_checkpoint_time
            progress.update(result[5])
            key = result[:4]
            start = result[4]
            expected = int(accumulators.completed[key])
            if start < expected:
                raise RuntimeError(
                    f"Duplicate adaptive CTMC result for {key}@{start}"
                )
            channel_buffer = buffered_results.setdefault(key, {})
            if start in channel_buffer:
                raise RuntimeError(
                    f"Duplicate buffered adaptive result for {key}@{start}"
                )
            channel_buffer[start] = result
            while expected in channel_buffer:
                contiguous = channel_buffer.pop(expected)
                finalized = ctmc._commit_trajectory_chunk(
                    contiguous,
                    energies=energies,
                    charges=np.asarray([charge_state]),
                    impact_au=impact_au,
                    config=sampling_config,
                    accumulators=accumulators,
                    pi=arrays["pi"],
                    pc=arrays["pc"],
                    pl=arrays["pl"],
                    done=arrays["done"],
                    failures=arrays["failures"],
                    successes=arrays["successes"],
                    maximum_energy_drift=arrays["maximum_energy_drift"],
                )
                completed_since_checkpoint += int(finalized)
                expected += contiguous[5]
            if not channel_buffer:
                buffered_results.pop(key, None)
            now = time.monotonic()
            if (
                completed_since_checkpoint >= checkpoint_every
                or (
                    checkpoint_seconds > 0.0
                    and now - last_checkpoint_time >= checkpoint_seconds
                )
            ):
                save_checkpoint(announce=True)
                completed_since_checkpoint = 0
                last_checkpoint_time = now

        try:
            if process_executor is None:
                for task in tasks:
                    accept(_compute_adaptive_trajectory_chunk(task))
            else:
                for result in _bounded_results(
                    process_executor,
                    tasks,
                    max(pending_factor * workers, 1),
                ):
                    if result is None:
                        now = time.monotonic()
                        if (
                            checkpoint_seconds > 0.0
                            and now - last_checkpoint_time
                            >= checkpoint_seconds
                        ):
                            save_checkpoint(announce=True)
                            completed_since_checkpoint = 0
                            last_checkpoint_time = now
                        continue
                    accept(result)
        except BaseException:
            if process_executor is not None:
                ctmc.terminate_process_pool(process_executor)
                process_executor = None
            raise
        finally:
            progress.close()
            save_checkpoint()
        return effective_target

    curve_cache: dict[
        int,
        tuple[int, float, np.ndarray, np.ndarray, float, dict[str, float], int],
    ] = {}

    def load_shared_base_curve(
        local_energy_index: int,
        source_base_energy_index: int,
    ) -> None:
        """Load one dependency-complete endpoint without recalculating it."""
        curve_count = int(base_energies.size * charges.size)
        curve_index = int(source_base_energy_index * charges.size + charge_index)
        path = _base_curve_result_path(output_dir, curve_index, curve_count)
        if not path.is_file():
            raise RuntimeError(
                "Missing shared adaptive base curve; complete the base-curve "
                f"stage before interval refinement: {path}"
            )
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["base_signature"]).item()) != base_signature:
                raise RuntimeError(f"Shared adaptive curve {path} uses another base run")
            if int(np.asarray(data["charge_state"]).item()) != charge_state:
                raise RuntimeError(f"Shared adaptive curve {path} has another charge")
            source_energy = np.asarray(data["energy_keV_u"], dtype=float)
            if source_energy.shape != (1,) or float(source_energy[0]).hex() != float(
                base_energies[source_base_energy_index]
            ).hex():
                raise RuntimeError(f"Shared adaptive curve {path} has another energy")
            offsets = np.asarray(data["curve_offsets"], dtype=np.int64)
            if not np.array_equal(offsets, np.asarray([0, offsets[-1]])):
                raise RuntimeError(f"Shared adaptive curve {path} is malformed")
            source_impact = np.asarray(data["impact_au"], dtype=float)
            local_indices = _indices_in_finest_grid(source_impact, impact_au)
            arrays["pi"][local_energy_index, 0, local_indices] = np.asarray(
                data["pi"], dtype=float
            )
            arrays["pc"][local_energy_index, 0, local_indices] = np.asarray(
                data["pc"], dtype=float
            )
            arrays["pl"][local_energy_index, 0, local_indices] = np.asarray(
                data["pl"], dtype=float
            )
            arrays["failures"][local_energy_index, 0, local_indices] = np.asarray(
                data["failures"], dtype=np.int64
            )
            arrays["successes"][local_energy_index, 0, local_indices] = np.asarray(
                data["successes"], dtype=np.int64
            )
            arrays["maximum_energy_drift"][
                local_energy_index, 0, local_indices
            ] = np.asarray(data["maximum_energy_drift"], dtype=float)
            accumulators.successes[local_energy_index, 0, local_indices] = np.asarray(
                data["channel_successes"], dtype=np.int64
            )
            arrays["done"][local_energy_index, 0, local_indices] = True
            statistical_names = tuple(
                str(value) for value in np.asarray(data["statistical_channel_names"])
            )
            statistical_values = np.asarray(
                data["statistical_channel_relative_half_width"], dtype=float
            )[0]
            curve_cache[local_energy_index] = (
                int(np.asarray(data["impact_level"], dtype=np.int64)[0]),
                float(np.asarray(data["impact_relative_error"], dtype=float)[0]),
                np.asarray(data["rows"], dtype=float)[0].copy(),
                local_indices,
                float(
                    np.asarray(
                        data["statistical_relative_half_width"], dtype=float
                    )[0]
                ),
                dict(zip(statistical_names, statistical_values, strict=True)),
                int(np.asarray(data["statistical_trajectory_target"])[0]),
            )

    def converge_impact_curve(
        energy_index: int,
    ) -> tuple[
        int, float, np.ndarray, np.ndarray, float, dict[str, float], int
    ]:
        cached = curve_cache.get(energy_index)
        if cached is not None:
            return cached
        last_error = math.inf
        last_statistical_error = math.inf
        last_statistical_errors: dict[str, float] = {}
        achieved_target = config.trajectories
        for sampling_level in range(adaptive.max_sampling_levels + 1):
            requested_target = config.trajectories * 2**sampling_level
            achieved_target = run_required_points(
                [energy_index],
                impact_level_indices[0],
                f"E={energies[energy_index]:.6g} b-level=0 N={requested_target}",
                requested_target,
            )
            coarse_indices = impact_level_indices[0]
            coarse_row = _curve_row(
                energy_index,
                coarse_indices,
                energies,
                charge_state,
                impact_au,
                arrays["pi"],
                arrays["pc"],
                arrays["pl"],
            )
            converged_curve: tuple[int, np.ndarray, np.ndarray] | None = None
            for level in range(1, adaptive.max_impact_levels + 1):
                fine_indices = impact_level_indices[level]
                achieved_target = run_required_points(
                    [energy_index],
                    fine_indices,
                    f"E={energies[energy_index]:.6g} b-level={level} "
                    f"N={requested_target}",
                    requested_target,
                )
                fine_row = _curve_row(
                    energy_index,
                    fine_indices,
                    energies,
                    charge_state,
                    impact_au,
                    arrays["pi"],
                    arrays["pc"],
                    arrays["pl"],
                )
                last_error, _ = maximum_relative_change(
                    coarse_row, fine_row, charge_state
                )
                if last_error <= adaptive.axis_relative_tolerance:
                    converged_curve = (level, fine_row, fine_indices)
                    break
                coarse_indices = fine_indices
                coarse_row = fine_row

            if converged_curve is not None:
                level, fine_row, fine_indices = converged_curve
                last_statistical_errors = (
                    cross_section_relative_confidence_half_widths(
                        impact_au[fine_indices],
                        arrays["pi"][energy_index, 0, fine_indices],
                        arrays["pc"][energy_index, 0, fine_indices],
                        arrays["pl"][energy_index, 0, fine_indices],
                        accumulators.successes[energy_index, 0, fine_indices],
                        charge_state,
                        adaptive.statistical_confidence,
                    )
                )
                last_statistical_error = max(
                    last_statistical_errors.values(), default=0.0
                )
                if (
                    last_statistical_error
                    <= adaptive.statistical_relative_tolerance
                ):
                    result = (
                        level,
                        last_error,
                        fine_row,
                        fine_indices,
                        last_statistical_error,
                        last_statistical_errors,
                        achieved_target,
                    )
                    curve_cache[energy_index] = result
                    return result

            if sampling_level < adaptive.max_sampling_levels:
                tqdm.write(
                    f"q={charge_state}, E={energies[energy_index]:.9g} keV/u: "
                    f"refining Monte Carlo samples after impact error "
                    f"{last_error:.3%} and statistical half-width "
                    f"{last_statistical_error:.3%}"
                )
                continue
            if converged_curve is None:
                raise AdaptiveConvergenceError(
                    f"q={charge_state}, E={energies[energy_index]:.9g} keV/u: "
                    f"impact-grid error {last_error:.6%} exceeds "
                    f"{adaptive.axis_relative_tolerance:.6%} after "
                    f"{adaptive.max_impact_levels} bisections and "
                    f"{adaptive.max_sampling_levels} sampling refinements"
                )
            raise AdaptiveConvergenceError(
                f"q={charge_state}, E={energies[energy_index]:.9g} keV/u: "
                f"maximum {adaptive.statistical_confidence:.1%} Monte Carlo "
                f"relative half-width {last_statistical_error:.6%} exceeds "
                f"{adaptive.statistical_relative_tolerance:.6%} after "
                f"{achieved_target:,} trajectories per active primitive channel; "
                f"channel widths={last_statistical_errors}"
            )
        raise AdaptiveConvergenceError(
            "Unreachable adaptive sampling state"
        )

    selected_energy_indices: set[int] = set()
    accepted_intervals: list[tuple[int, int, float]] = []
    try:
        left = 0
        right = energies.size - 1
        if shared_base_curves:
            load_shared_base_curve(left, lower_index)
            load_shared_base_curve(right, upper_index)
        else:
            converge_impact_curve(left)
        selected_energy_indices.add(left)
        if right != left:
            if not shared_base_curves:
                converge_impact_curve(right)
            selected_energy_indices.add(right)
            pending_intervals = [(left, right, 1)]
            while pending_intervals:
                lower, upper, level = pending_intervals.pop()
                midpoint = (lower + upper) // 2
                midpoint_result = converge_impact_curve(midpoint)
                lower_result = converge_impact_curve(lower)
                upper_result = converge_impact_curve(upper)
                selected_energy_indices.add(midpoint)
                error, _ = maximum_log_midpoint_error(
                    lower_result[2],
                    midpoint_result[2],
                    upper_result[2],
                    charge_state,
                )
                if error <= adaptive.axis_relative_tolerance:
                    accepted_intervals.append((lower, upper, error))
                elif level >= adaptive.max_energy_levels:
                    raise AdaptiveConvergenceError(
                        f"q={charge_state}, E={energies[lower]:.9g}--"
                        f"{energies[upper]:.9g} keV/u: energy-midpoint error "
                        f"{error:.6%} exceeds "
                        f"{adaptive.axis_relative_tolerance:.6%} after "
                        f"{adaptive.max_energy_levels} bisections"
                    )
                else:
                    pending_intervals.append((midpoint, upper, level + 1))
                    pending_intervals.append((lower, midpoint, level + 1))

        selected = np.asarray(sorted(selected_energy_indices), dtype=np.int64)
        rows = np.stack([curve_cache[index][2] for index in selected])
        impact_level = np.asarray(
            [curve_cache[index][0] for index in selected], dtype=np.int64
        )
        impact_error = np.asarray(
            [curve_cache[index][1] for index in selected], dtype=float
        )
        statistical_error = np.asarray(
            [curve_cache[index][4] for index in selected], dtype=float
        )
        statistical_channels = ("SC", "TI", "SL", "LI", "SI", "DI")
        statistical_channel_errors = np.asarray(
            [
                [
                    curve_cache[index][5].get(name, np.nan)
                    for name in statistical_channels
                ]
                for index in selected
            ],
            dtype=float,
        )
        statistical_trajectory_target = np.asarray(
            [curve_cache[index][6] for index in selected], dtype=np.int64
        )
        maximum_energy_error = max(
            (error for _, _, error in accepted_intervals), default=0.0
        )
        combined_error_bound = float(np.max(impact_error)) + maximum_energy_error
        if combined_error_bound > adaptive.combined_relative_tolerance:
            raise AdaptiveConvergenceError(
                f"q={charge_state}: combined discretization bound "
                f"{combined_error_bound:.6%} exceeds "
                f"{adaptive.combined_relative_tolerance:.6%}"
            )
        offsets = [0]
        flattened_impact: list[np.ndarray] = []
        flattened_pi: list[np.ndarray] = []
        flattened_pc: list[np.ndarray] = []
        flattened_pl: list[np.ndarray] = []
        flattened_failures: list[np.ndarray] = []
        flattened_successes: list[np.ndarray] = []
        flattened_drift: list[np.ndarray] = []
        flattened_channel_successes: list[np.ndarray] = []
        for energy_index in selected:
            indices = curve_cache[int(energy_index)][3]
            flattened_impact.append(impact_au[indices])
            flattened_pi.append(arrays["pi"][energy_index, 0, indices])
            flattened_pc.append(arrays["pc"][energy_index, 0, indices])
            flattened_pl.append(arrays["pl"][energy_index, 0, indices])
            flattened_failures.append(
                arrays["failures"][energy_index, 0, indices]
            )
            flattened_successes.append(
                arrays["successes"][energy_index, 0, indices]
            )
            flattened_drift.append(
                arrays["maximum_energy_drift"][energy_index, 0, indices]
            )
            flattened_channel_successes.append(
                accumulators.successes[energy_index, 0, indices]
            )
            offsets.append(offsets[-1] + indices.size)
        interval_array = np.asarray(accepted_intervals, dtype=float).reshape(-1, 3)
        _atomic_savez(
            result_path,
            family_signature=np.asarray(family_signature),
            base_signature=np.asarray(base_signature),
            family_index=np.asarray(family_index),
            family_count=np.asarray(family_count),
            charge_state=np.asarray(charge_state),
            energy_keV_u=energies[selected],
            rows=rows,
            impact_level=impact_level,
            impact_relative_error=impact_error,
            combined_discretization_error_bound=np.asarray(combined_error_bound),
            statistical_confidence=np.asarray(adaptive.statistical_confidence),
            statistical_channel_names=np.asarray(statistical_channels),
            statistical_relative_half_width=statistical_error,
            statistical_channel_relative_half_width=statistical_channel_errors,
            statistical_trajectory_target=statistical_trajectory_target,
            accepted_intervals=interval_array,
            curve_offsets=np.asarray(offsets, dtype=np.int64),
            impact_au=np.concatenate(flattened_impact),
            pi=np.concatenate(flattened_pi),
            pc=np.concatenate(flattened_pc),
            pl=np.concatenate(flattened_pl),
            failures=np.concatenate(flattened_failures),
            successes=np.concatenate(flattened_successes),
            channel_successes=np.concatenate(flattened_channel_successes),
            maximum_energy_drift=np.concatenate(flattened_drift),
        )
        checkpoint_path.with_suffix(".failure.json").unlink(missing_ok=True)
        return result_path
    except ctmc.TrajectoryIntegrationFailure as error:
        diagnostic_path = checkpoint_path.with_suffix(".failure.json")
        ctmc.save_failure_diagnostic(
            diagnostic_path,
            signature=family_signature,
            checkpoint_path=checkpoint_path,
            detail=error.args[0] if error.args else str(error),
        )
        raise
    finally:
        save_checkpoint()
        if process_executor is not None:
            process_executor.shutdown(wait=True)


def run_adaptive_shard(
    *,
    output_dir: Path,
    base_signature: str,
    base_checkpoint: dict[str, np.ndarray],
    base_config: ctmc.CTMCConfig,
    workers: int,
    start_method: str,
    pending_factor: int,
    checkpoint_every: int,
    checkpoint_seconds: float,
    shard_count: int,
    shard_index: int,
    adaptive: AdaptiveRefinementConfig,
    phase: str = "intervals",
    dry_run: bool = False,
) -> int:
    energies = np.asarray(base_checkpoint["energies_keV_u"], dtype=float)
    charges = np.asarray(base_checkpoint["charges"], dtype=int)
    if phase == "base_curves":
        families = adaptive_base_curve_indices(energies, charges)
        unit = "curve"
        description = "Adaptive base curves"
    elif phase == "intervals":
        families = adaptive_family_indices(energies, charges)
        unit = "interval"
        description = "Adaptive intervals"
        curve_count = int(energies.size * charges.size)
        missing = [
            _base_curve_result_path(output_dir, index, curve_count)
            for index in range(curve_count)
            if not _base_curve_result_path(output_dir, index, curve_count).is_file()
        ]
        if missing and not dry_run:
            raise RuntimeError(
                f"Adaptive interval stage requires all {curve_count} shared "
                f"base curves; {len(missing)} are missing (first: {missing[0]})"
            )
    else:
        raise ValueError("phase must be 'base_curves' or 'intervals'")
    owned = [
        (index, family)
        for index, family in enumerate(families)
        if index % shard_count == shard_index
    ]
    print(
        f"Adaptive {phase}: {len(families)} independent tasks; "
        f"shard {shard_index}/{shard_count - 1} owns {len(owned)}"
    )
    print(
        f"Axis/combined targets: {adaptive.axis_relative_tolerance:.3%}/"
        f"{adaptive.combined_relative_tolerance:.3%}; "
        f"{adaptive.statistical_confidence:.1%} statistical half-width target: "
        f"{adaptive.statistical_relative_tolerance:.3%}; "
        f"maximum impact levels: {adaptive.max_impact_levels}; "
        f"maximum energy levels: {adaptive.max_energy_levels}; "
        f"maximum sampling doublings: {adaptive.max_sampling_levels}"
    )
    if dry_run:
        return 0
    previous_signal_handlers: dict[int, signal.Handlers] = {}

    def handle_scheduler_termination(
        signal_number: int,
        _: object,
    ) -> None:
        signal_name = signal.Signals(signal_number).name
        for installed_signal in previous_signal_handlers:
            signal.signal(installed_signal, signal.SIG_IGN)
        tqdm.write(
            f"Received {signal_name}; stopping adaptive workers and saving "
            "the active family checkpoint..."
        )
        raise KeyboardInterrupt

    for signal_name in ("SIGTERM", "SIGUSR1"):
        termination_signal = getattr(signal, signal_name, None)
        if termination_signal is not None:
            previous_signal_handlers[termination_signal] = signal.getsignal(
                termination_signal
            )
            signal.signal(
                termination_signal,
                handle_scheduler_termination,
            )
    try:
        for family_index, (charge_index, energy_or_interval_index) in tqdm(
            owned,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
        ):
            _run_family(
                output_dir=output_dir,
                family_index=family_index,
                family_count=len(families),
                charge_index=charge_index,
                interval_index=energy_or_interval_index,
                base_signature=base_signature,
                base_checkpoint=base_checkpoint,
                base_config=base_config,
                workers=workers,
                start_method=start_method,
                pending_factor=pending_factor,
                checkpoint_every=checkpoint_every,
                checkpoint_seconds=checkpoint_seconds,
                adaptive=adaptive,
                base_energy_index=(
                    energy_or_interval_index if phase == "base_curves" else None
                ),
                shared_base_curves=phase == "intervals",
            )
    finally:
        for termination_signal, previous_handler in (
            previous_signal_handlers.items()
        ):
            signal.signal(termination_signal, previous_handler)
    return 0


def merge_adaptive_families(
    *,
    output_dir: Path,
    base_signature: str,
    base_checkpoint: dict[str, np.ndarray],
    base_config: ctmc.CTMCConfig,
    adaptive: AdaptiveRefinementConfig,
) -> tuple[Path, Path, Path, Path]:
    base_energies = np.asarray(base_checkpoint["energies_keV_u"], dtype=float)
    charges = np.asarray(base_checkpoint["charges"], dtype=int)
    families = adaptive_family_indices(base_energies, charges)
    curves: dict[tuple[str, int], dict[str, np.ndarray | float | int]] = {}
    maximum_impact_error = 0.0
    maximum_energy_error = 0.0
    maximum_statistical_error = 0.0
    statistical_channel_names: np.ndarray | None = None
    for family_index in tqdm(
        range(len(families)),
        desc="Merging adaptive families",
        unit="family",
        dynamic_ncols=True,
    ):
        path = _interval_result_path(output_dir, family_index, len(families))
        if not path.exists():
            raise RuntimeError(f"Missing adaptive family result {path}")
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["base_signature"]).item()) != base_signature:
                raise RuntimeError(f"Adaptive family {path} uses another base run")
            charge = int(np.asarray(data["charge_state"]).item())
            offsets = np.asarray(data["curve_offsets"], dtype=np.int64)
            family_energies = np.asarray(data["energy_keV_u"], dtype=float)
            rows = np.asarray(data["rows"], dtype=float)
            impact_errors = np.asarray(data["impact_relative_error"], dtype=float)
            statistical_errors = np.asarray(
                data["statistical_relative_half_width"], dtype=float
            )
            family_statistical_names = np.asarray(data["statistical_channel_names"])
            if statistical_channel_names is None:
                statistical_channel_names = family_statistical_names.copy()
            else:
                np.testing.assert_array_equal(
                    statistical_channel_names, family_statistical_names
                )
            maximum_impact_error = max(
                maximum_impact_error, float(np.max(impact_errors))
            )
            maximum_statistical_error = max(
                maximum_statistical_error, float(np.max(statistical_errors))
            )
            accepted = np.asarray(data["accepted_intervals"], dtype=float)
            if accepted.size:
                maximum_energy_error = max(
                    maximum_energy_error, float(np.max(accepted[:, 2]))
                )
            for index, energy in enumerate(family_energies):
                key = (float(energy).hex(), charge)
                start, stop = int(offsets[index]), int(offsets[index + 1])
                candidate = {
                    "energy": float(energy),
                    "charge": charge,
                    "row": rows[index].copy(),
                    "impact": np.asarray(data["impact_au"][start:stop]).copy(),
                    "pi": np.asarray(data["pi"][start:stop]).copy(),
                    "pc": np.asarray(data["pc"][start:stop]).copy(),
                    "pl": np.asarray(data["pl"][start:stop]).copy(),
                    "failures": np.asarray(data["failures"][start:stop]).copy(),
                    "successes": np.asarray(data["successes"][start:stop]).copy(),
                    "channel_successes": np.asarray(
                        data["channel_successes"][start:stop]
                    ).copy(),
                    "drift": np.asarray(
                        data["maximum_energy_drift"][start:stop]
                    ).copy(),
                    "impact_error": float(impact_errors[index]),
                    "statistical_error": float(statistical_errors[index]),
                    "statistical_channel_errors": np.asarray(
                        data["statistical_channel_relative_half_width"][index]
                    ).copy(),
                    "statistical_trajectory_target": int(
                        data["statistical_trajectory_target"][index]
                    ),
                }
                existing = curves.get(key)
                if existing is None:
                    curves[key] = candidate
                else:
                    np.testing.assert_array_equal(
                        existing["row"], candidate["row"]
                    )
                    np.testing.assert_array_equal(
                        existing["impact"], candidate["impact"]
                    )
                    np.testing.assert_array_equal(
                        existing["channel_successes"],
                        candidate["channel_successes"],
                    )

    combined_discretization_error_bound = (
        maximum_impact_error + maximum_energy_error
    )
    if combined_discretization_error_bound > adaptive.combined_relative_tolerance:
        raise AdaptiveConvergenceError(
            "Merged combined discretization bound "
            f"{combined_discretization_error_bound:.6%} exceeds "
            f"{adaptive.combined_relative_tolerance:.6%}"
        )
    if maximum_statistical_error > adaptive.statistical_relative_tolerance:
        raise AdaptiveConvergenceError(
            f"Merged maximum {adaptive.statistical_confidence:.1%} Monte Carlo "
            f"relative half-width {maximum_statistical_error:.6%} exceeds "
            f"{adaptive.statistical_relative_tolerance:.6%}"
        )
    if statistical_channel_names is None:
        raise RuntimeError("Adaptive merge found no statistical channel labels")

    ordered = sorted(
        curves.values(),
        key=lambda curve: (float(curve["energy"]), int(curve["charge"])),
    )
    rows = np.stack([np.asarray(curve["row"]) for curve in ordered])
    columns = [
        "E_keV_u",
        "q",
        "sigma_SC_cm2",
        "sigma_TI_cm2",
        "sigma_SL_cm2",
        "sigma_LI_cm2",
        "sigma_decrease_cm2",
        "sigma_increase_cm2",
        "sigma_SI_cm2",
        "sigma_DI_cm2",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = ctmc.PROJECTILE.key
    dat_path = output_dir / f"{prefix}_charge_exchange_h2o.dat"
    csv_path = output_dir / f"{prefix}_charge_exchange_h2o.csv"
    probabilities_path = output_dir / f"{prefix}_charge_exchange_probabilities.npz"
    metadata_path = output_dir / f"{prefix}_charge_exchange_metadata.json"
    header = (
        " ".join(columns)
        + "\nMicroscopic cross sections in cm2 per H2O molecule; "
        "adaptively converged in impact parameter and projectile energy."
    )
    _atomic_savetxt(
        dat_path,
        rows,
        fmt=["%.9E", "%.0f"] + ["%.9E"] * (rows.shape[1] - 2),
        header=header,
        comments="# ",
    )
    _atomic_savetxt(
        csv_path,
        rows,
        delimiter=",",
        fmt=["%.9E", "%.0f"] + ["%.9E"] * (rows.shape[1] - 2),
        header=",".join(columns),
        comments="",
    )
    offsets = [0]
    for curve in ordered:
        offsets.append(offsets[-1] + len(np.asarray(curve["impact"])))
    flattened_failures = np.concatenate(
        [np.asarray(curve["failures"]) for curve in ordered]
    )
    flattened_successes = np.concatenate(
        [np.asarray(curve["successes"]) for curve in ordered]
    )
    flattened_drift = np.concatenate(
        [np.asarray(curve["drift"]) for curve in ordered]
    )
    flattened_channel_successes = np.concatenate(
        [np.asarray(curve["channel_successes"]) for curve in ordered]
    )
    _atomic_savez(
        probabilities_path,
        adaptive_format=np.asarray(True),
        curve_energy_keV_u=np.asarray([curve["energy"] for curve in ordered]),
        curve_charge=np.asarray([curve["charge"] for curve in ordered], dtype=int),
        curve_offsets=np.asarray(offsets, dtype=np.int64),
        impact_au=np.concatenate([np.asarray(curve["impact"]) for curve in ordered]),
        orbital_labels=np.asarray(
            [orbital.label for orbital in ctmc.WATER_ORBITALS]
        ),
        pi=np.concatenate([np.asarray(curve["pi"]) for curve in ordered]),
        pc=np.concatenate([np.asarray(curve["pc"]) for curve in ordered]),
        pl=np.concatenate([np.asarray(curve["pl"]) for curve in ordered]),
        failures=flattened_failures,
        successes=flattened_successes,
        channel_successes=flattened_channel_successes,
        maximum_energy_drift=flattened_drift,
        impact_relative_error=np.asarray(
            [curve["impact_error"] for curve in ordered]
        ),
        combined_discretization_error_bound=np.asarray(
            combined_discretization_error_bound
        ),
        statistical_confidence=np.asarray(adaptive.statistical_confidence),
        statistical_channel_names=statistical_channel_names,
        statistical_relative_half_width=np.asarray(
            [curve["statistical_error"] for curve in ordered]
        ),
        statistical_channel_relative_half_width=np.stack(
            [np.asarray(curve["statistical_channel_errors"]) for curve in ordered]
        ),
        statistical_trajectory_target=np.asarray(
            [curve["statistical_trajectory_target"] for curve in ordered],
            dtype=np.int64,
        ),
    )
    metadata = {
        "model": ctmc.PROJECTILE.model_name,
        "doi": ctmc.PROJECTILE.primary_doi,
        "target": "isolated spherical H2O pseudotarget",
        "model_references": list(ctmc.PROJECTILE.model_references),
        "projectile": {
            "element": ctmc.PROJECTILE.name,
            "symbol": ctmc.PROJECTILE.symbol,
            "nuclear_charge": ctmc.PROJECTILE.nuclear_charge,
            "bare_nuclear_mass_au": ctmc.PROJECTILE.mass_au,
            "atomic_data_provenance": list(
                ctmc.PROJECTILE.atomic_data_provenance
            ),
        },
        "cross_section_unit": "cm2 per H2O molecule",
        "energy_unit": "keV/u",
        "density_applied": False,
        "phase_density_scaling": ctmc.phase_density_scaling_metadata(),
        "initial_ensemble": ctmc.initial_ensemble_metadata(
            base_config.initial_ensemble
        ),
        "base_configuration_signature": base_signature,
        "base_ctmc_config": asdict(base_config),
        "water_orbitals": [
            asdict(orbital) for orbital in ctmc.WATER_ORBITALS
        ],
        f"{ctmc.PROJECTILE.key}_outer_orbitals": {
            str(charge): asdict(orbital)
            for charge, orbital in ctmc.PROJECTILE_OUTER_ORBITAL.items()
        },
        "trajectory_statistics": {
            "attempted": int(
                np.sum(flattened_failures) + np.sum(flattened_successes)
            ),
            "successful": int(np.sum(flattened_successes)),
            "failed": int(np.sum(flattened_failures)),
            "maximum_relative_total_energy_drift": float(
                np.max(flattened_drift)
            ),
        },
        "adaptive_refinement": {
            "enabled": True,
            **asdict(adaptive),
            "model_version": ADAPTIVE_MODEL_VERSION,
            "family_count": len(families),
            "output_curve_count": len(ordered),
            "maximum_impact_relative_error": maximum_impact_error,
            "maximum_energy_midpoint_relative_error": maximum_energy_error,
            "combined_discretization_error_bound": (
                combined_discretization_error_bound
            ),
            "maximum_statistical_relative_confidence_half_width": (
                maximum_statistical_error
            ),
            "impact_rule": (
                "nested interval bisection; successive 2*pi*integral(b*P db) "
                "trapezoidal estimates"
            ),
            "energy_rule": (
                "calculated geometric midpoints compared with log-log "
                "endpoint interpolation"
            ),
            "acceptance": (
                "every physically active independent channel must satisfy each "
                "per-axis discretization limit, the conservative combined "
                "discretization bound, and the separate Monte Carlo confidence-"
                "width limit; reaching a maximum level alone is never acceptance"
            ),
            "statistical_scope": (
                "The two-sided asymptotic normal confidence half-width uses the delta "
                "method with exact Jacobians of the published IEVM/IPM weights, "
                "multinomial ionization/capture covariance, binomial projectile-"
                "loss variance, and independent impact-parameter streams. It is "
                "enforced separately from the deterministic grid-error budget."
            ),
            "intermediate_boundary_rule": (
                "New midpoint energies use the larger adjacent published/"
                "converged start separation and integration-time lower bound; "
                "this is conservative domain control, not interpolation of a "
                "physical coefficient."
            ),
        },
        "model_notes": list(ctmc.PROJECTILE.model_notes),
    }
    config_metadata = metadata["base_ctmc_config"]
    if not np.isfinite(config_metadata["max_step_au"]):
        config_metadata["max_step_au"] = None
    metadata_temporary = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary, metadata_path)
    manifest = _manifest_path(output_dir)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["status"] = "complete"
        payload["final_metadata"] = str(metadata_path)
        temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest)
    return dat_path, csv_path, probabilities_path, metadata_path
