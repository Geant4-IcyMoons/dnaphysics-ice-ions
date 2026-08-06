"""Unit-explicit Landau--Zener primitives for trajectory post-processing."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


HBAR_EV_S = 6.582_119_569e-16
ANGSTROM_PER_METRE = 1.0e10
ANGSTROM2_TO_CM2 = 1.0e-16


def transition_probability(
    coupling_ev: float,
    diabatic_slope_difference_ev_per_angstrom: float,
    radial_speed_m_per_s: float,
) -> float:
    """Return the one-pass probability of changing diabatic state at a crossing.

    The radial speed must come from the selected nuclear trajectory. This
    function deliberately does not assume a straight-line or Coulomb path.
    """

    coupling = abs(float(coupling_ev))
    slope = abs(float(diabatic_slope_difference_ev_per_angstrom))
    speed = abs(float(radial_speed_m_per_s))
    if not all(math.isfinite(value) for value in (coupling, slope, speed)):
        raise ValueError("Landau--Zener inputs must be finite.")
    if slope <= 0.0 or speed <= 0.0:
        raise ValueError("Crossing slope difference and radial speed must be positive.")
    if coupling == 0.0:
        return 0.0
    radial_speed_angstrom_per_s = speed * ANGSTROM_PER_METRE
    exponent = (
        2.0
        * math.pi
        * coupling**2
        / (HBAR_EV_S * radial_speed_angstrom_per_s * slope)
    )
    return -math.expm1(-exponent)


def impact_parameter_cross_section_cm2(
    impact_parameters_angstrom: Iterable[float],
    capture_probabilities: Iterable[float],
) -> float:
    """Integrate ``2*pi*integral b*P(b) db`` on a supplied trajectory grid."""

    impact = np.asarray(tuple(impact_parameters_angstrom), dtype=float)
    probability = np.asarray(tuple(capture_probabilities), dtype=float)
    if impact.ndim != 1 or probability.ndim != 1 or impact.size != probability.size:
        raise ValueError("Impact parameters and probabilities need equal 1-D grids.")
    if impact.size < 2:
        raise ValueError("At least two impact-parameter points are required.")
    if not np.all(np.isfinite(impact)) or not np.all(np.isfinite(probability)):
        raise ValueError("Impact parameters and probabilities must be finite.")
    if impact[0] < 0.0 or np.any(np.diff(impact) <= 0.0):
        raise ValueError("Impact parameters must be non-negative and increasing.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Capture probabilities must lie in [0, 1].")
    integrand = impact * probability
    integral_angstrom2 = float(
        np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(impact))
    )
    return 2.0 * math.pi * integral_angstrom2 * ANGSTROM2_TO_CM2
