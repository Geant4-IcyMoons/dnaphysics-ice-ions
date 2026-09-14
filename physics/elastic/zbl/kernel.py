"""Independent universal-ZBL screened binary-collision kernel.

All incident energies are total projectile kinetic energies. Projectile and
target charge arguments are nuclear atomic numbers; ionic charge state is not
part of the ZBL potential. The implementation follows the classical
screening-integral algorithm used by Mendenhall and Weller (2005), with exact
relativistic two-body energy invariants surrounding the classical deflection
integral.

This is a validation reference for the Geant4 runtime. It is not an ice-
specific molecular potential and must not be added to the retained-domain NLH
hard process without a separately validated non-overlap construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


BOHR_RADIUS_ANGSTROM = 0.529177210903
COULOMB_EV_ANGSTROM = 14.3996454784255
AMU_C2_EV = 931.49410242e6
ROOT_ITERATIONS = 64


@dataclass(frozen=True)
class Species:
    symbol: str
    atomic_number: int
    mass_number: int
    mass_u: float

    @property
    def mass_ev(self) -> float:
        return self.mass_u * AMU_C2_EV


PROJECTILES = {
    "H": Species("H", 1, 1, 1.007276466621),
    "He": Species("He", 2, 4, 4.001506179127),
    "C": Species("C", 6, 12, 12.0),
    "O": Species("O", 8, 16, 15.99491461957),
    "S": Species("S", 16, 32, 31.9720711744),
}
TARGETS = {
    "H": PROJECTILES["H"],
    "O": PROJECTILES["O"],
}


def screening(reduced_radius: float) -> float:
    """Universal ZBL screening function."""
    if not math.isfinite(reduced_radius) or reduced_radius < 0.0:
        raise ValueError("Reduced radius must be finite and non-negative.")
    x = reduced_radius
    return (
        0.1818 * math.exp(-3.2 * x)
        + 0.5099 * math.exp(-0.9423 * x)
        + 0.2802 * math.exp(-0.4029 * x)
        + 0.02817 * math.exp(-0.2016 * x)
    )


def _screening_derivative(reduced_radius: float) -> float:
    x = reduced_radius
    return (
        -3.2 * 0.1818 * math.exp(-3.2 * x)
        - 0.9423 * 0.5099 * math.exp(-0.9423 * x)
        - 0.4029 * 0.2802 * math.exp(-0.4029 * x)
        - 0.2016 * 0.02817 * math.exp(-0.2016 * x)
    )


def screening_length_angstrom(projectile_z: int, target_z: int) -> float:
    """Universal ZBL screening length in angstrom."""
    if projectile_z <= 0 or target_z <= 0:
        raise ValueError("Nuclear atomic numbers must be positive.")
    return 0.88534 * BOHR_RADIUS_ANGSTROM / (
        projectile_z**0.23 + target_z**0.23
    )


def _center_of_mass_kinetic_energy_ev(
    projectile_mass_ev: float,
    target_mass_ev: float,
    kinetic_energy_ev: float,
) -> float:
    mass_sum = projectile_mass_ev + target_mass_ev
    invariant_mass = math.sqrt(
        mass_sum * mass_sum + 2.0 * target_mass_ev * kinetic_energy_ev
    )
    return invariant_mass - mass_sum


def maximum_recoil_energy_ev(
    projectile_mass_ev: float,
    target_mass_ev: float,
    kinetic_energy_ev: float,
) -> float:
    """Exact maximum target-recoil energy for stationary-target kinematics."""
    if min(projectile_mass_ev, target_mass_ev, kinetic_energy_ev) <= 0.0:
        raise ValueError("Masses and total kinetic energy must be positive.")
    momentum_squared = kinetic_energy_ev * (
        kinetic_energy_ev + 2.0 * projectile_mass_ev
    )
    invariant_s = (
        projectile_mass_ev**2
        + target_mass_ev**2
        + 2.0 * target_mass_ev * (projectile_mass_ev + kinetic_energy_ev)
    )
    return 2.0 * target_mass_ev * momentum_squared / invariant_s


def cos_theta_cm(
    projectile: Species,
    target: Species,
    kinetic_energy_ev: float,
    impact_parameter_angstrom: float,
) -> float:
    """Classical ZBL center-of-mass scattering cosine for one impact parameter."""
    if kinetic_energy_ev <= 0.0 or impact_parameter_angstrom < 0.0:
        raise ValueError("Energy must be positive and impact parameter non-negative.")
    if impact_parameter_angstrom == 0.0:
        return -1.0
    length = screening_length_angstrom(
        projectile.atomic_number, target.atomic_number
    )
    cm_energy = _center_of_mass_kinetic_energy_ev(
        projectile.mass_ev, target.mass_ev, kinetic_energy_ev
    )
    epsilon = cm_energy / (
        projectile.atomic_number
        * target.atomic_number
        * COULOMB_EV_ANGSTROM
        / length
    )
    beta = impact_parameter_angstrom / length

    def turning(x: float) -> float:
        return x * x - x * screening(x) / epsilon - beta * beta

    lower = 0.0
    upper = max(1.0, beta + 1.0)
    while turning(upper) < 0.0 and upper < 1.0e12:
        upper *= 2.0
    if turning(upper) < 0.0:
        raise RuntimeError("Could not bracket the ZBL turning radius.")
    for _ in range(ROOT_ITERATIONS):
        middle = 0.5 * (lower + upper)
        if turning(middle) < 0.0:
            lower = middle
        else:
            upper = middle
    x0 = 0.5 * (lower + upper)
    denominator = (
        0.5
        + beta * beta / (2.0 * x0 * x0)
        - _screening_derivative(x0) / (2.0 * epsilon)
    )
    if denominator <= 0.0:
        raise RuntimeError("Invalid ZBL scattering-integral denominator.")
    alpha = (1.0 + 1.0 / math.sqrt(denominator)) / 30.0
    for abscissa, weight in zip(
        (0.98302349, 0.84652241, 0.53235309, 0.18347974),
        (0.03472124, 0.14769029, 0.23485003, 0.18602489),
    ):
        x = x0 / abscissa
        radicand = 1.0 - screening(x) / (x * epsilon) - beta * beta / (x * x)
        if radicand <= 0.0:
            raise RuntimeError("Invalid ZBL Lobatto-quadrature radicand.")
        alpha += weight / math.sqrt(radicand)
    complement = math.pi * beta * alpha / x0
    return max(-1.0, min(1.0, -math.cos(complement)))


def pair_cross_section_angstrom2(
    projectile: Species,
    target: Species,
    kinetic_energy_ev: float,
    minimum_transfer_ev: float,
) -> float:
    """Cross section for recoil transfers above a positive cutoff."""
    if minimum_transfer_ev <= 0.0:
        raise ValueError("The transfer cutoff must be positive.")
    maximum_transfer = maximum_recoil_energy_ev(
        projectile.mass_ev, target.mass_ev, kinetic_energy_ev
    )
    if maximum_transfer <= minimum_transfer_ev:
        return 0.0
    target_cosine = 1.0 - 2.0 * minimum_transfer_ev / maximum_transfer
    length = screening_length_angstrom(
        projectile.atomic_number, target.atomic_number
    )
    lower = 0.0
    upper = length
    while (
        cos_theta_cm(projectile, target, kinetic_energy_ev, upper)
        < target_cosine
        and upper < 1.0e8 * length
    ):
        upper *= 2.0
    if upper >= 1.0e8 * length:
        raise RuntimeError("Could not bracket the ZBL transfer cutoff.")
    for _ in range(ROOT_ITERATIONS):
        middle = 0.5 * (lower + upper)
        if (
            cos_theta_cm(projectile, target, kinetic_energy_ev, middle)
            < target_cosine
        ):
            lower = middle
        else:
            upper = middle
    maximum_impact = 0.5 * (lower + upper)
    return math.pi * maximum_impact * maximum_impact
