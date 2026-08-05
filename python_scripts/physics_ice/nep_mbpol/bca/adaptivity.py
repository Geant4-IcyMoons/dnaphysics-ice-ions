"""Error-controlled energy and impact meshes for NLH collision kernels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .scattering import (
    CollisionResult,
    NLHCollisionKernel,
    PairKinematics,
    pair_kinematics,
    two_body_observables_from_cm_angles,
)


_PROBE_FRACTIONS = np.asarray((0.25, 0.5, 0.75), dtype=np.float64)
_BULK_ERROR_FRACTION = 0.8


@dataclass(frozen=True)
class ImpactMesh:
    """One adaptively validated collision-area mesh."""

    area_quantiles: NDArray[np.float64]
    collisions: tuple[CollisionResult, ...]
    recoil_l1_relative_error: float
    transport_l1_relative_error: float
    theta_cm_l1_relative_error: float
    theta_cm_max_error_over_pi: float
    validation_evaluations: int

    @property
    def point_count(self) -> int:
        return int(self.area_quantiles.size)


@dataclass(frozen=True)
class EnergyMesh:
    """One pair-specific logarithmic energy mesh and its validation report."""

    energies_ev: NDArray[np.float64]
    maximum_theta_cm_relative_error: float
    maximum_recoil_relative_error: float
    validation_evaluations: int

    @property
    def point_count(self) -> int:
        return int(self.energies_ev.size)


def _observables(result: CollisionResult) -> NDArray[np.float64]:
    return np.asarray(
        (
            result.recoil_energy_ev,
            1.0 - math.cos(result.theta_projectile_lab_rad),
            result.theta_cm_rad,
        ),
        dtype=np.float64,
    )


def _observables_from_theta(
    kinematics: PairKinematics, theta_cm_rad: NDArray[np.float64]
) -> NDArray[np.float64]:
    theta_lab, recoil = two_body_observables_from_cm_angles(
        kinematics, theta_cm_rad
    )
    return np.stack((recoil, 1.0 - np.cos(theta_lab), theta_cm_rad), axis=-1)


def _bulk_mark(values: NDArray[np.float64], fraction: float) -> set[int]:
    total = float(np.sum(values))
    if total <= 0.0:
        return set()
    marked: set[int] = set()
    accumulated = 0.0
    for index in np.argsort(values)[::-1]:
        marked.add(int(index))
        accumulated += float(values[index])
        if accumulated >= fraction * total:
            break
    return marked


def adaptive_impact_mesh(
    kernel: NLHCollisionKernel,
    *,
    relative_tolerance: float,
    max_points: int,
) -> ImpactMesh:
    """Refine collision-area quantiles until linear interpolation is converged.

    Recoil, lab-frame transport, and CM-angle interpolation are checked at the
    quarter, midpoint, and three-quarter points of every interval.  Integrated
    absolute interpolation errors must satisfy ``relative_tolerance``; the
    maximum CM-angle error must also be below that fraction of pi.
    """

    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie between zero and one.")
    if max_points < 3:
        raise ValueError("max_points must be at least three.")
    if kernel.maximum_impact_parameter_angstrom <= 0.0:
        raise ValueError("The requested energy has no retained NLH collision domain.")

    cache: dict[float, CollisionResult] = {}

    def collision(area_quantile: float) -> CollisionResult:
        key = float(area_quantile)
        if key not in cache:
            impact = kernel.maximum_impact_parameter_angstrom * math.sqrt(key)
            cache[key] = kernel.solve(impact)
        return cache[key]

    quantiles = np.asarray((0.0, 1.0), dtype=np.float64)
    collision(0.0)
    collision(1.0)
    while True:
        widths = np.diff(quantiles)
        probes = quantiles[:-1, None] + widths[:, None] * _PROBE_FRACTIONS
        endpoint_values = np.asarray(
            [_observables(collision(value)) for value in quantiles]
        )
        probe_values = np.asarray(
            [
                [_observables(collision(value)) for value in interval]
                for interval in probes
            ]
        )
        predicted_theta = (
            endpoint_values[:-1, None, 2]
            * (1.0 - _PROBE_FRACTIONS[None, :])
            + endpoint_values[1:, None, 2] * _PROBE_FRACTIONS[None, :]
        )
        predicted = _observables_from_theta(
            kernel.kinematics, predicted_theta
        )
        absolute_error = np.abs(probe_values - predicted)

        # Four-subinterval trapezoidal estimates provide a reference integral
        # for the same validation points.  The interpolation-error integral
        # has zero endpoint error and the three measured interior errors.
        fine_values = np.concatenate(
            (
                endpoint_values[:-1, None, :],
                probe_values,
                endpoint_values[1:, None, :],
            ),
            axis=1,
        )
        fine_integrals = widths[:, None] * np.trapezoid(
            fine_values, dx=0.25, axis=1
        )
        error_values = np.concatenate(
            (
                np.zeros((len(widths), 1, 3)),
                absolute_error,
                np.zeros((len(widths), 1, 3)),
            ),
            axis=1,
        )
        error_integrals = widths[:, None] * np.trapezoid(
            error_values, dx=0.25, axis=1
        )
        totals = np.sum(fine_integrals, axis=0)
        relative_errors = np.sum(error_integrals, axis=0) / np.maximum(
            totals, np.finfo(float).tiny
        )
        maximum_theta_error = float(np.max(absolute_error[:, :, 2])) / math.pi

        converged = bool(
            np.all(relative_errors <= relative_tolerance)
            and maximum_theta_error <= relative_tolerance
        )
        if converged:
            collisions = tuple(collision(value) for value in quantiles)
            return ImpactMesh(
                area_quantiles=quantiles,
                collisions=collisions,
                recoil_l1_relative_error=float(relative_errors[0]),
                transport_l1_relative_error=float(relative_errors[1]),
                theta_cm_l1_relative_error=float(relative_errors[2]),
                theta_cm_max_error_over_pi=maximum_theta_error,
                validation_evaluations=len(cache),
            )

        normalized_contributions = error_integrals / np.maximum(
            totals[None, :], np.finfo(float).tiny
        )
        marked: set[int] = set()
        for observable_index, relative_error in enumerate(relative_errors):
            if relative_error > relative_tolerance:
                marked.update(
                    _bulk_mark(
                        normalized_contributions[:, observable_index],
                        _BULK_ERROR_FRACTION,
                    )
                )
        interval_angle_error = np.max(absolute_error[:, :, 2], axis=1) / math.pi
        if maximum_theta_error > relative_tolerance:
            marked.update(
                int(index)
                for index in np.flatnonzero(
                    interval_angle_error > 0.5 * relative_tolerance
                )
            )
        if not marked:
            score = np.max(normalized_contributions, axis=1)
            score = np.maximum(score, interval_angle_error)
            marked.add(int(np.argmax(score)))

        if len(quantiles) + len(marked) > max_points:
            raise RuntimeError(
                "Adaptive impact mesh exceeded max_points before reaching "
                f"tolerance {relative_tolerance:g}; current points={len(quantiles)}, "
                f"estimated errors={relative_errors.tolist()}, "
                f"theta_max/pi={maximum_theta_error:g}."
            )
        midpoints = 0.5 * (quantiles[:-1] + quantiles[1:])
        quantiles = np.unique(
            np.concatenate((quantiles, midpoints[sorted(marked)]))
        )


def _energy_validation_quantiles() -> NDArray[np.float64]:
    return np.unique(
        np.concatenate(
            (
                np.asarray((0.0,)),
                np.geomspace(1.0e-20, 1.0e-2, 32),
                np.linspace(1.0e-2, 1.0, 33),
            )
        )
    )


def adaptive_energy_mesh(
    projectile: str,
    target: str,
    *,
    energy_min_ev: float,
    energy_max_ev: float,
    base_points: int,
    minimum_turning_potential_ev: float,
    quadrature_order: int,
    relative_tolerance: float,
    max_points: int,
) -> EnergyMesh:
    """Refine a base logarithmic energy grid using direct midpoint checks."""

    if base_points < 2:
        raise ValueError("base_points must be at least two.")
    if max_points < base_points:
        raise ValueError("max_points cannot be smaller than base_points.")
    validation_quantiles = _energy_validation_quantiles()
    cache: dict[float, NDArray[np.float64]] = {}

    def values(energy_ev: float) -> NDArray[np.float64]:
        key = float(energy_ev)
        if key not in cache:
            kernel = NLHCollisionKernel(
                projectile,
                target,
                key,
                minimum_turning_potential_ev=minimum_turning_potential_ev,
                quadrature_order=quadrature_order,
            )
            if kernel.maximum_impact_parameter_angstrom <= 0.0:
                raise ValueError(
                    f"{projectile}-{target} at {key:g} eV does not reach the "
                    "requested NLH turning-potential domain."
                )
            rows = []
            for quantile in validation_quantiles:
                result = kernel.solve(
                    kernel.maximum_impact_parameter_angstrom
                    * math.sqrt(float(quantile))
                )
                rows.append((result.theta_cm_rad, result.recoil_energy_ev))
            cache[key] = np.asarray(rows, dtype=np.float64)
        return cache[key]

    energies = np.geomspace(energy_min_ev, energy_max_ev, base_points)
    maximum_theta_error = math.inf
    maximum_recoil_error = math.inf
    while True:
        log_energies = np.log(energies)
        interval_widths = np.diff(log_energies)
        probe_logs = log_energies[:-1, None] + (
            interval_widths[:, None] * _PROBE_FRACTIONS
        )
        probes = np.exp(probe_logs)
        endpoint_values = [values(float(energy)) for energy in energies]
        interval_errors = np.zeros(len(energies) - 1)
        maximum_theta_error = 0.0
        maximum_recoil_error = 0.0
        for interval_index, interval_probes in enumerate(probes):
            lower = endpoint_values[interval_index]
            upper = endpoint_values[interval_index + 1]
            for fraction, probe_energy in zip(
                _PROBE_FRACTIONS, interval_probes, strict=True
            ):
                actual = values(float(probe_energy))
                predicted_theta = np.exp(
                    (1.0 - fraction) * np.log(lower[:, 0])
                    + fraction * np.log(upper[:, 0])
                )
                probe_kinematics = pair_kinematics(
                    projectile, target, float(probe_energy)
                )
                _, predicted_recoil = two_body_observables_from_cm_angles(
                    probe_kinematics, predicted_theta
                )
                theta_error = float(
                    np.max(np.abs(predicted_theta / actual[:, 0] - 1.0))
                )
                recoil_error = float(
                    np.max(np.abs(predicted_recoil / actual[:, 1] - 1.0))
                )
                maximum_theta_error = max(maximum_theta_error, theta_error)
                maximum_recoil_error = max(maximum_recoil_error, recoil_error)
                interval_errors[interval_index] = max(
                    interval_errors[interval_index], theta_error, recoil_error
                )

        if max(maximum_theta_error, maximum_recoil_error) <= relative_tolerance:
            return EnergyMesh(
                energies_ev=energies,
                maximum_theta_cm_relative_error=maximum_theta_error,
                maximum_recoil_relative_error=maximum_recoil_error,
                validation_evaluations=len(cache),
            )

        marked = np.flatnonzero(interval_errors > relative_tolerance)
        if len(energies) + len(marked) > max_points:
            raise RuntimeError(
                "Adaptive energy mesh exceeded max_points before reaching "
                f"tolerance {relative_tolerance:g}; current points={len(energies)}, "
                f"theta error={maximum_theta_error:g}, recoil error="
                f"{maximum_recoil_error:g}."
            )
        midpoints = np.sqrt(energies[:-1] * energies[1:])
        energies = np.unique(np.concatenate((energies, midpoints[marked])))
