"""Classical NLH central-potential scattering with exact two-body kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nlh import potential_ev

from .config import (
    ATOMIC_MASS_UNIT_C2_EV,
    DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    DEFAULT_QUADRATURE_ORDER,
    ISOTOPE_MASS_U,
    PUBLISHED_MINIMUM_TURNING_POTENTIAL_EV,
    canonical_element,
)


@dataclass(frozen=True)
class PairKinematics:
    projectile: str
    target: str
    projectile_energy_ev: float
    projectile_mass_c2_ev: float
    target_mass_c2_ev: float
    invariant_mass_c2_ev: float
    relative_kinetic_energy_ev: float
    momentum_cm_ev_c: float
    beta_cm: float
    gamma_cm: float


@dataclass(frozen=True)
class CollisionResult:
    projectile: str
    target: str
    projectile_energy_ev: float
    relative_kinetic_energy_ev: float
    impact_parameter_angstrom: float
    closest_approach_angstrom: float
    turning_potential_ev: float
    theta_cm_rad: float
    theta_projectile_lab_rad: float
    recoil_energy_ev: float
    projectile_out_energy_ev: float
    energy_conservation_error_ev: float


@dataclass(frozen=True)
class TwoBodyOutcome:
    """Lab-frame outcome implied by one CM scattering angle."""

    theta_projectile_lab_rad: float
    recoil_energy_ev: float
    projectile_out_energy_ev: float
    energy_conservation_error_ev: float


class NLHCollisionKernel:
    """Reusable collision solver for one projectile, target, and energy."""

    def __init__(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        *,
        minimum_turning_potential_ev: float = DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
        quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    ) -> None:
        self.minimum_turning_potential_ev = _validate_turning_threshold(
            minimum_turning_potential_ev
        )
        if quadrature_order < 16:
            raise ValueError("quadrature_order must be at least 16.")
        self.quadrature_order = quadrature_order
        self.kinematics = pair_kinematics(projectile, target, projectile_energy_ev)
        relative_energy = self.kinematics.relative_kinetic_energy_ev
        if relative_energy <= self.minimum_turning_potential_ev:
            self.threshold_radius_angstrom = turning_threshold_radius_angstrom(
                self.kinematics.projectile,
                self.kinematics.target,
                minimum_turning_potential_ev=self.minimum_turning_potential_ev,
            )
            self.maximum_impact_parameter_angstrom = 0.0
        else:
            self.threshold_radius_angstrom = _radius_at_potential(
                self.kinematics.projectile,
                self.kinematics.target,
                self.minimum_turning_potential_ev,
            )
            self.maximum_impact_parameter_angstrom = self.threshold_radius_angstrom * math.sqrt(
                1.0 - self.minimum_turning_potential_ev / relative_energy
            )

    @property
    def hard_cross_section_angstrom2(self) -> float:
        return math.pi * self.maximum_impact_parameter_angstrom**2

    def solve(self, impact_parameter_angstrom: float) -> CollisionResult:
        """Solve one impact parameter without rebuilding the pair context."""

        return _solve_with_context(
            self.kinematics,
            self.maximum_impact_parameter_angstrom,
            impact_parameter_angstrom,
            self.minimum_turning_potential_ev,
            self.quadrature_order,
        )


def _validate_pair(projectile: str, target: str) -> tuple[str, str]:
    projectile_symbol = canonical_element(projectile)
    target_symbol = canonical_element(target)
    if target_symbol not in {"H", "O"}:
        raise ValueError("The ice target atom must be H or O.")
    return projectile_symbol, target_symbol


@lru_cache(maxsize=16384)
def pair_kinematics(
    projectile: str, target: str, projectile_energy_ev: float
) -> PairKinematics:
    """Return relativistically exact CM quantities for a stationary target."""

    projectile_symbol, target_symbol = _validate_pair(projectile, target)
    if not math.isfinite(projectile_energy_ev) or projectile_energy_ev <= 0.0:
        raise ValueError("Projectile kinetic energy must be finite and positive.")

    mass_1 = ISOTOPE_MASS_U[projectile_symbol] * ATOMIC_MASS_UNIT_C2_EV
    mass_2 = ISOTOPE_MASS_U[target_symbol] * ATOMIC_MASS_UNIT_C2_EV
    energy_1_lab = mass_1 + projectile_energy_ev
    momentum_lab = math.sqrt(projectile_energy_ev * (projectile_energy_ev + 2.0 * mass_1))
    s = mass_1 * mass_1 + mass_2 * mass_2 + 2.0 * mass_2 * energy_1_lab
    sqrt_s = math.sqrt(s)
    # For a stationary target, the Kallen expression reduces exactly to this
    # form.  It avoids subtracting nearly equal rest-mass-scale quantities at
    # low projectile energy.
    momentum_cm = mass_2 * momentum_lab / sqrt_s
    beta_cm = momentum_lab / (energy_1_lab + mass_2)
    gamma_cm = 1.0 / math.sqrt(1.0 - beta_cm * beta_cm)
    relative_kinetic_energy = (
        2.0 * mass_2 * projectile_energy_ev
        / (sqrt_s + mass_1 + mass_2)
    )
    return PairKinematics(
        projectile=projectile_symbol,
        target=target_symbol,
        projectile_energy_ev=projectile_energy_ev,
        projectile_mass_c2_ev=mass_1,
        target_mass_c2_ev=mass_2,
        invariant_mass_c2_ev=sqrt_s,
        relative_kinetic_energy_ev=relative_kinetic_energy,
        momentum_cm_ev_c=momentum_cm,
        beta_cm=beta_cm,
        gamma_cm=gamma_cm,
    )


def _validate_turning_threshold(value: float) -> float:
    if not math.isfinite(value) or value < PUBLISHED_MINIMUM_TURNING_POTENTIAL_EV:
        raise ValueError(
            "minimum_turning_potential_ev cannot be below the published "
            f"NLH domain ({PUBLISHED_MINIMUM_TURNING_POTENTIAL_EV:g} eV)."
        )
    return float(value)


def _raw_potential(distance: float, projectile: str, target: str) -> float:
    return float(
        potential_ev(
            distance,
            projectile,
            target,
            enforce_fit_domain=False,
        )
    )


@lru_cache(maxsize=None)
def _radius_at_potential(projectile: str, target: str, energy_ev: float) -> float:
    """Invert the monotonic repulsive potential by bisection."""

    lower = 1.0e-12
    upper = 0.05
    while _raw_potential(upper, projectile, target) > energy_ev:
        upper *= 2.0
        if upper > 1.0e6:
            raise RuntimeError("Could not bracket the NLH potential radius.")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if middle == lower or middle == upper:
            break
        if _raw_potential(middle, projectile, target) > energy_ev:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def maximum_impact_parameter_angstrom(
    projectile: str,
    target: str,
    projectile_energy_ev: float,
    *,
    minimum_turning_potential_ev: float = DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
) -> float:
    """Largest impact parameter whose turning point remains in the NLH domain."""

    threshold = _validate_turning_threshold(minimum_turning_potential_ev)
    kinematics = pair_kinematics(projectile, target, projectile_energy_ev)
    relative_energy = kinematics.relative_kinetic_energy_ev
    if relative_energy <= threshold:
        return 0.0
    threshold_radius = turning_threshold_radius_angstrom(
        kinematics.projectile,
        kinematics.target,
        minimum_turning_potential_ev=threshold,
    )
    return threshold_radius * math.sqrt(1.0 - threshold / relative_energy)


def turning_threshold_radius_angstrom(
    projectile: str,
    target: str,
    *,
    minimum_turning_potential_ev: float = DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
) -> float:
    """Radius at which the NLH potential equals the retained-domain boundary."""

    threshold = _validate_turning_threshold(minimum_turning_potential_ev)
    projectile_symbol, target_symbol = _validate_pair(projectile, target)
    return _radius_at_potential(projectile_symbol, target_symbol, threshold)


def hard_cross_section_angstrom2(
    projectile: str,
    target: str,
    projectile_energy_ev: float,
    *,
    minimum_turning_potential_ev: float = DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
) -> float:
    """Geometric cross section of the retained, fit-domain hard collisions."""

    maximum_impact = maximum_impact_parameter_angstrom(
        projectile,
        target,
        projectile_energy_ev,
        minimum_turning_potential_ev=minimum_turning_potential_ev,
    )
    return math.pi * maximum_impact * maximum_impact


def _turning_radius(
    impact_parameter: float,
    relative_energy: float,
    projectile: str,
    target: str,
) -> float:
    def radial_function(radius: float) -> float:
        return (
            1.0
            - (impact_parameter / radius) ** 2
            - _raw_potential(radius, projectile, target) / relative_energy
        )

    lower = max(1.0e-12, impact_parameter)
    upper = max(0.05, 1.01 * lower)
    while radial_function(upper) <= 0.0:
        upper *= 2.0
        if upper > 1.0e6:
            raise RuntimeError("Could not bracket the distance of closest approach.")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if middle == lower or middle == upper:
            break
        if radial_function(middle) <= 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


@lru_cache(maxsize=None)
def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 16:
        raise ValueError("quadrature_order must be at least 16.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angles = 0.25 * math.pi * (nodes + 1.0)
    return angles, 0.25 * math.pi * weights


def _theta_cm(
    impact_parameter: float,
    closest_approach: float,
    relative_energy: float,
    projectile: str,
    target: str,
    quadrature_order: int,
) -> float:
    if impact_parameter == 0.0:
        return math.pi
    angles, weights = _gauss_legendre(quadrature_order)
    cosines = np.cos(angles)
    radii = closest_approach / cosines
    potentials = np.asarray(
        potential_ev(
            radii,
            projectile,
            target,
            enforce_fit_domain=False,
        ),
        dtype=np.float64,
    )
    radicand = (
        1.0
        - (impact_parameter * cosines / closest_approach) ** 2
        - potentials / relative_energy
    )
    if np.any(radicand <= 0.0):
        # A tiny negative value can arise only from roundoff near the turning
        # point.  A material violation indicates a failed root or quadrature.
        if float(np.min(radicand)) < -1.0e-10:
            raise RuntimeError("Scattering quadrature crossed the turning point.")
        radicand = np.maximum(radicand, np.finfo(float).tiny)
    half_orbit = (impact_parameter / closest_approach) * float(
        np.sum(weights * np.sin(angles) / np.sqrt(radicand))
    )
    return min(math.pi, max(0.0, math.pi - 2.0 * half_orbit))


def two_body_outcome_from_cm_angle(
    kinematics: PairKinematics,
    theta_cm_rad: float,
) -> TwoBodyOutcome:
    """Transform a CM angle using exact relativistic two-body kinematics."""

    if not math.isfinite(theta_cm_rad) or not 0.0 <= theta_cm_rad <= math.pi:
        raise ValueError("theta_cm_rad must be finite and lie in [0, pi].")
    mass_1 = kinematics.projectile_mass_c2_ev
    energy_1_cm = math.sqrt(mass_1 * mass_1 + kinematics.momentum_cm_ev_c**2)
    transverse_momentum = kinematics.momentum_cm_ev_c * math.sin(theta_cm_rad)
    longitudinal_momentum = kinematics.momentum_cm_ev_c * math.cos(theta_cm_rad)
    longitudinal_lab = kinematics.gamma_cm * (
        longitudinal_momentum + kinematics.beta_cm * energy_1_cm
    )
    theta_lab = math.atan2(abs(transverse_momentum), longitudinal_lab)
    # For an initially stationary target, t=-2 m_target T_recoil and also
    # t=-2 p_cm^2 (1-cos(theta_cm)). This avoids subtracting rest-mass-scale
    # energies when the recoil is small.
    recoil_energy = (
        kinematics.momentum_cm_ev_c**2
        * (1.0 - math.cos(theta_cm_rad))
        / kinematics.target_mass_c2_ev
    )
    energy_tolerance = 1.0e-12 * max(1.0, kinematics.projectile_energy_ev)
    if not (
        -energy_tolerance
        <= recoil_energy
        <= kinematics.projectile_energy_ev + energy_tolerance
    ):
        raise RuntimeError("Two-body recoil energy lies outside physical bounds.")
    if recoil_energy <= energy_tolerance:
        recoil_energy = 0.0
    elif recoil_energy >= kinematics.projectile_energy_ev - energy_tolerance:
        recoil_energy = kinematics.projectile_energy_ev
    projectile_out_energy = kinematics.projectile_energy_ev - recoil_energy
    conservation_error = kinematics.projectile_energy_ev - (
        projectile_out_energy + recoil_energy
    )
    return TwoBodyOutcome(
        theta_projectile_lab_rad=theta_lab,
        recoil_energy_ev=recoil_energy,
        projectile_out_energy_ev=projectile_out_energy,
        energy_conservation_error_ev=conservation_error,
    )


def two_body_observables_from_cm_angles(
    kinematics: PairKinematics,
    theta_cm_rad: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized lab angles and recoil energies for CM-angle arrays."""

    theta = np.asarray(theta_cm_rad, dtype=np.float64)
    if np.any(~np.isfinite(theta)) or np.any(theta < 0.0) or np.any(theta > math.pi):
        raise ValueError("All CM angles must be finite and lie in [0, pi].")
    mass_1 = kinematics.projectile_mass_c2_ev
    momentum = kinematics.momentum_cm_ev_c
    energy_1_cm = math.sqrt(mass_1 * mass_1 + momentum * momentum)
    transverse_momentum = momentum * np.sin(theta)
    longitudinal_lab = kinematics.gamma_cm * (
        momentum * np.cos(theta) + kinematics.beta_cm * energy_1_cm
    )
    theta_lab = np.arctan2(np.abs(transverse_momentum), longitudinal_lab)
    recoil_energy = (
        momentum * momentum
        * (1.0 - np.cos(theta))
        / kinematics.target_mass_c2_ev
    )
    energy_tolerance = 1.0e-12 * max(1.0, kinematics.projectile_energy_ev)
    if np.any(recoil_energy < -energy_tolerance) or np.any(
        recoil_energy > kinematics.projectile_energy_ev + energy_tolerance
    ):
        raise RuntimeError("Two-body recoil energy lies outside physical bounds.")
    recoil_energy = np.where(
        recoil_energy <= energy_tolerance,
        0.0,
        np.where(
            recoil_energy
            >= kinematics.projectile_energy_ev - energy_tolerance,
            kinematics.projectile_energy_ev,
            recoil_energy,
        ),
    )
    return theta_lab, recoil_energy


def _solve_with_context(
    kinematics: PairKinematics,
    maximum_impact: float,
    impact_parameter_angstrom: float,
    threshold: float,
    quadrature_order: int,
) -> CollisionResult:
    if not math.isfinite(impact_parameter_angstrom) or impact_parameter_angstrom < 0.0:
        raise ValueError("Impact parameter must be finite and non-negative.")
    if maximum_impact == 0.0:
        raise ValueError(
            "The relative kinetic energy does not reach the requested NLH "
            "turning-potential domain."
        )
    tolerance = 1.0e-12 * max(1.0, maximum_impact)
    if impact_parameter_angstrom > maximum_impact + tolerance:
        raise ValueError(
            f"Impact parameter exceeds the retained NLH domain: "
            f"b={impact_parameter_angstrom:.8g} A, b_max={maximum_impact:.8g} A."
        )
    impact_parameter = min(impact_parameter_angstrom, maximum_impact)
    closest_approach = _turning_radius(
        impact_parameter,
        kinematics.relative_kinetic_energy_ev,
        kinematics.projectile,
        kinematics.target,
    )
    turning_potential = _raw_potential(
        closest_approach, kinematics.projectile, kinematics.target
    )
    if turning_potential < threshold * (1.0 - 1.0e-9):
        raise RuntimeError("The collision turning point is outside the retained domain.")
    theta_cm = _theta_cm(
        impact_parameter,
        closest_approach,
        kinematics.relative_kinetic_energy_ev,
        kinematics.projectile,
        kinematics.target,
        quadrature_order,
    )

    outcome = two_body_outcome_from_cm_angle(kinematics, theta_cm)

    return CollisionResult(
        projectile=kinematics.projectile,
        target=kinematics.target,
        projectile_energy_ev=kinematics.projectile_energy_ev,
        relative_kinetic_energy_ev=kinematics.relative_kinetic_energy_ev,
        impact_parameter_angstrom=impact_parameter,
        closest_approach_angstrom=closest_approach,
        turning_potential_ev=turning_potential,
        theta_cm_rad=theta_cm,
        theta_projectile_lab_rad=outcome.theta_projectile_lab_rad,
        recoil_energy_ev=outcome.recoil_energy_ev,
        projectile_out_energy_ev=outcome.projectile_out_energy_ev,
        energy_conservation_error_ev=outcome.energy_conservation_error_ev,
    )


def solve_nlh_collision(
    projectile: str,
    target: str,
    projectile_energy_ev: float,
    impact_parameter_angstrom: float,
    *,
    minimum_turning_potential_ev: float = DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
) -> CollisionResult:
    """Solve one retained NLH binary collision.

    The orbit is evaluated from the central-potential deflection integral.
    The CM angle is then transformed with relativistically exact two-body
    kinematics.  This is an independent-atom collision kernel; it is not yet a
    phase-resolved ice trajectory.
    """

    kernel = NLHCollisionKernel(
        projectile,
        target,
        projectile_energy_ev,
        minimum_turning_potential_ev=minimum_turning_potential_ev,
        quadrature_order=quadrature_order,
    )
    return kernel.solve(impact_parameter_angstrom)
