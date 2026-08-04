"""Pair-specific NLH repulsive potentials for projectiles in water ice.

The coefficients are the published Nordlund--Lehtola--Hobler (NLH) fits for
the nine unique nuclear pairs formed by projectiles H, He, C, O, and S and
target atoms H and O.  The interaction is symmetric, so H--O and O--H use the
same row.

Distances are in angstrom, energies in eV, and radial derivatives/forces in
eV/angstrom.  The potential is a nuclear pair potential and therefore does
not depend on the projectile's instantaneous electronic charge state.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Same constants as the authors' reference nlhpot.c implementation.
_EPSILON_0_F_PER_M = 8.8541878188e-12
_ELEMENTARY_CHARGE_C = 1.602176634e-19
COULOMB_EV_ANGSTROM = (
    1.0
    / (4.0 * np.pi * _EPSILON_0_F_PER_M)
    * _ELEMENTARY_CHARGE_C
    / 1.0e-10
)

# The paper identifies V_rep >= 10 eV as the domain in which a purely
# repulsive pair description can be assumed.  Agreement is best above 30 eV.
MINIMUM_FIT_ENERGY_EV = 10.0

_COEFFICIENT_PATH = Path(__file__).with_name("coefficients.csv")
_ELEMENT_SYMBOLS = {1: "H", 2: "He", 6: "C", 8: "O", 16: "S"}
_ELEMENT_ALIASES = {
    "h": 1,
    "hydrogen": 1,
    "proton": 1,
    "he": 2,
    "helium": 2,
    "alpha": 2,
    "c": 6,
    "carbon": 6,
    "o": 8,
    "oxygen": 8,
    "s": 16,
    "sulfur": 16,
    "sulphur": 16,
}
_SUPPORTED_PROJECTILES = frozenset((1, 2, 6, 8, 16))
_SUPPORTED_TARGETS = frozenset((1, 8))

ScalarOrArray: TypeAlias = float | NDArray[np.float64]


class NLHDomainError(ValueError):
    """Raised when an NLH fit is evaluated outside its physical domain."""


@dataclass(frozen=True)
class NLHCoefficients:
    """Three-exponential NLH screening coefficients for one nuclear pair."""

    z1: int
    z2: int
    a: tuple[float, float, float]
    b_per_angstrom: tuple[float, float, float]
    rms_error_above_30_ev_percent: float
    rms_error_above_10_ev_percent: float

    @property
    def symbols(self) -> tuple[str, str]:
        return _ELEMENT_SYMBOLS[self.z1], _ELEMENT_SYMBOLS[self.z2]


def _atomic_number(element: str | int, role: str) -> int:
    if isinstance(element, (int, np.integer)):
        atomic_number = int(element)
    elif isinstance(element, str):
        try:
            atomic_number = _ELEMENT_ALIASES[element.strip().lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported {role} element {element!r}.") from exc
    else:
        raise TypeError(
            f"{role.capitalize()} must be an element symbol/name or atomic number."
        )
    return atomic_number


@lru_cache(maxsize=1)
def _coefficient_table() -> dict[tuple[int, int], NLHCoefficients]:
    table: dict[tuple[int, int], NLHCoefficients] = {}
    with _COEFFICIENT_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            z1 = int(row["z1"])
            z2 = int(row["z2"])
            table[(z1, z2)] = NLHCoefficients(
                z1=z1,
                z2=z2,
                a=(float(row["a1"]), float(row["a2"]), float(row["a3"])),
                b_per_angstrom=(
                    float(row["b1_per_angstrom"]),
                    float(row["b2_per_angstrom"]),
                    float(row["b3_per_angstrom"]),
                ),
                rms_error_above_30_ev_percent=float(
                    row["rms_error_above_30_ev_percent"]
                ),
                rms_error_above_10_ev_percent=float(
                    row["rms_error_above_10_ev_percent"]
                ),
            )
    return table


def supported_projectile_target_pairs() -> tuple[tuple[str, str], ...]:
    """Return the ten supported directional projectile--target combinations."""

    return tuple(
        (_ELEMENT_SYMBOLS[projectile], _ELEMENT_SYMBOLS[target])
        for projectile in (1, 2, 6, 8, 16)
        for target in (1, 8)
    )


def get_coefficients(
    projectile: str | int, target: str | int
) -> NLHCoefficients:
    """Return published coefficients for a supported projectile--ice pair."""

    projectile_z = _atomic_number(projectile, "projectile")
    target_z = _atomic_number(target, "target")
    if projectile_z not in _SUPPORTED_PROJECTILES:
        raise ValueError("Projectile must be H, He, C, O, or S.")
    if target_z not in _SUPPORTED_TARGETS:
        raise ValueError("Ice target atom must be H or O.")
    key = tuple(sorted((projectile_z, target_z)))
    try:
        return _coefficient_table()[key]
    except KeyError as exc:
        raise RuntimeError(f"Missing NLH coefficient row for atomic numbers {key}.") from exc


def _distance_array(distance_angstrom: ArrayLike) -> NDArray[np.float64]:
    distance = np.asarray(distance_angstrom, dtype=np.float64)
    if distance.size == 0:
        raise ValueError("At least one internuclear distance is required.")
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError(
            "All internuclear distances must be finite and greater than zero."
        )
    return distance


def _return_scalar_or_array(value: NDArray[np.float64]) -> ScalarOrArray:
    if value.ndim == 0:
        return float(value)
    return value


def _raw_components(
    distance_angstrom: ArrayLike,
    projectile: str | int,
    target: str | int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    distance = _distance_array(distance_angstrom)
    coefficients = get_coefficients(projectile, target)
    phi = np.zeros_like(distance)
    dphi = np.zeros_like(distance)
    for a_i, b_i in zip(coefficients.a, coefficients.b_per_angstrom):
        term = a_i * np.exp(-b_i * distance)
        phi += term
        dphi -= b_i * term
    coulomb = COULOMB_EV_ANGSTROM * coefficients.z1 * coefficients.z2
    potential = coulomb * phi / distance
    derivative = coulomb * (distance * dphi - phi) / distance**2
    return phi, dphi, potential, derivative


def _enforce_fit_domain(potential: NDArray[np.float64]) -> None:
    minimum = float(np.min(potential))
    if minimum < MINIMUM_FIT_ENERGY_EV:
        raise NLHDomainError(
            "NLH was evaluated below its published repulsive-fit domain: "
            f"minimum V={minimum:.6g} eV, required V>={MINIMUM_FIT_ENERGY_EV:g} eV. "
            "Use enforce_fit_domain=False only for diagnostics; couple to a "
            "validated near-equilibrium potential at larger separation."
        )


def screening_function(
    distance_angstrom: ArrayLike,
    projectile: str | int,
    target: str | int,
    *,
    enforce_fit_domain: bool = True,
) -> ScalarOrArray:
    """Evaluate the dimensionless NLH screening function phi(r)."""

    phi, _, potential, _ = _raw_components(distance_angstrom, projectile, target)
    if enforce_fit_domain:
        _enforce_fit_domain(potential)
    return _return_scalar_or_array(phi)


def screening_derivative_per_angstrom(
    distance_angstrom: ArrayLike,
    projectile: str | int,
    target: str | int,
    *,
    enforce_fit_domain: bool = True,
) -> ScalarOrArray:
    """Evaluate d phi/dr in inverse angstrom."""

    _, dphi, potential, _ = _raw_components(distance_angstrom, projectile, target)
    if enforce_fit_domain:
        _enforce_fit_domain(potential)
    return _return_scalar_or_array(dphi)


def potential_ev(
    distance_angstrom: ArrayLike,
    projectile: str | int,
    target: str | int,
    *,
    enforce_fit_domain: bool = True,
) -> ScalarOrArray:
    """Evaluate the repulsive pair potential V_NLH(r) in eV."""

    _, _, potential, _ = _raw_components(distance_angstrom, projectile, target)
    if enforce_fit_domain:
        _enforce_fit_domain(potential)
    return _return_scalar_or_array(potential)


def potential_derivative_ev_per_angstrom(
    distance_angstrom: ArrayLike,
    projectile: str | int,
    target: str | int,
    *,
    enforce_fit_domain: bool = True,
) -> ScalarOrArray:
    """Evaluate dV_NLH/dr in eV/angstrom."""

    _, _, potential, derivative = _raw_components(
        distance_angstrom, projectile, target
    )
    if enforce_fit_domain:
        _enforce_fit_domain(potential)
    return _return_scalar_or_array(derivative)


def radial_force_ev_per_angstrom(
    distance_angstrom: ArrayLike,
    projectile: str | int,
    target: str | int,
    *,
    enforce_fit_domain: bool = True,
) -> ScalarOrArray:
    """Evaluate the outward repulsive radial force magnitude, -dV/dr."""

    derivative = potential_derivative_ev_per_angstrom(
        distance_angstrom,
        projectile,
        target,
        enforce_fit_domain=enforce_fit_domain,
    )
    return -derivative
