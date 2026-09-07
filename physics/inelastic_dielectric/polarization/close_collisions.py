"""Experimental nonrelativistic close/distant polarization matching.

Schinner & Sigmund, NIMB 164-165 (2000) 220, Sec. 4, Eqs. (17), (24),
DOI 10.1016/S0168-583X(99)01181-7. Their exponential-radius interpolation
is NOT used for our HF densities. We evaluate Eq. (17) for the dispersionless
oscillator dielectric instead; see CLOSE_COLLISIONS.md for the derivation.

This module is a comparison backend, not connected to the table generator.
All internal energies, lengths, and velocities are Hartree atomic units.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma

from . import barkas_dcs as bd
from . import screened_barkas as screened

MODEL = "oscillator-induced-potential-matching-nonrel-v1"
HARTREE_EV = bd.ALPHA_FINE**2 * bd.MEC2_EV
BOHR_CM = bd.RE_CLASSICAL_CM / bd.ALPHA_FINE**2


@dataclass(frozen=True)
class Quadrature:
    impact: int = 64
    time: int = 2049
    tail: float = 32.0
    search: int = 49


def static_core_offset(density):
    """Positive constant C in the actual static potential -Z/r + C + ... ."""
    if density.electrons == 0:
        return 0.0
    if density.electrons == 1:
        return float(density.z)
    l, a, weight = density.terms.T
    return float(np.sum(weight * np.sqrt(a) * gamma(l + 1) / gamma(l + 1.5)))


def induced_shift(kappa, density, *, order=192):
    """Re V_ind(0) from Eq. (17), with epsilon=1-omega^2/(Omega+i0)^2.

    kappa=omega/v; A(k)=Z-F(k). The real angular principal value reduces to
    (kappa/pi) integral_0^infinity A(k)/k log|(k+kappa)/(k-kappa)| dk.
    Fold k>kappa onto t in (0,1). Subtract the logarithmic endpoint value
    analytically, using integral_0^1 log((1+t)/(1-t))/t dt = pi^2/4.
    There is no static screening offset in this *induced* shift.
    """
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa < 0 or order < 8:
        raise ValueError("Require finite kappa >= 0 and quadrature order >= 8.")
    if kappa == 0:
        return 0.0
    if density.electrons == 0:
        return np.pi * density.z * kappa / 2
    nodes, weights = screened._legendre(order)
    # Neutral shifts can be dominated by a narrow t~kappa*r_cloud region.
    # Logarithmic panels resolve it without subtracting a bare Z*kappa term.
    edges = np.r_[0., np.geomspace(1e-10, 1., 11)]
    widths = np.diff(edges)
    t = (edges[:-1,None] + widths[:,None]*(nodes+1)/2).ravel()
    weights = (widths[:,None]*weights/2).ravel()
    # Separate q exactly; electron_deficit is stable even for neutral tails.
    center = float(density.electron_deficit(kappa, direct=True))
    pair = (density.electron_deficit(kappa*t, direct=True)
            + density.electron_deficit(kappa/t, direct=True))
    h = (np.log1p(t) - np.log1p(-t)) / t
    result = kappa / np.pi * (
        np.pi**2 * (density.charge + center) / 2
        + np.dot(weights, h * (pair - 2*center)))
    if not np.isfinite(result) or result < 0:
        raise FloatingPointError("Invalid induced potential; no clipping applied.")
    return float(result)


def close_transfer(b, velocity, z, shift):
    """Eq. (24) with V0=C_static+shift minus the same expression at C_static.

    The static contribution cancels: delta T_close=4*shift*(v^2*b/Z)^2.
    This is a small-impact-parameter asymptote, not a cubic-in-Z DCS and
    not a screened Rutherford law valid at arbitrary impact parameters.
    """
    return 4 * shift * (velocity**2 * np.asarray(b) / z)**2


def _products(b, omega, velocity, density, quadrature):
    b = np.atleast_1d(np.asarray(b, float))
    px, pz = screened._impulse_products(omega*b/velocity, b, density,
                                      quadrature.time, quadrature.tail)
    return px + pz


def distant_transfer(b, omega, velocity, density, quadrature=Quadrature()):
    """Existing force-gradient cubic energy transfer at gamma=1, in Hartree."""
    b = np.atleast_1d(np.asarray(b, float))
    return _products(b, omega, velocity, density, quadrature) / (b**3*velocity**4)


def _distant_integral(lower_x, omega, velocity, density, quadrature):
    """Dimensionless K=(1/2) integral products/x^2 dx above lower_x."""
    if lower_x >= 50:
        raise ValueError("Matching radius lies outside the resolved oscillator domain.")
    nodes, weights = screened._legendre(quadrature.impact)
    span = np.log(50/lower_x)
    x = lower_x * np.exp((nodes + 1)*span/2)
    products = _products(x*velocity/omega, omega, velocity, density, quadrature)
    return float(span/4 * np.dot(weights, products/x))


def _evaluate(loss_eV, velocity, density, quadrature):
    omega = loss_eV/HARTREE_EV
    shift = induced_shift(omega/velocity, density, order=2*quadrature.impact)
    b90 = density.z/velocity**2
    # Enclose the small-p and adiabatic limits; do not prescribe a crossover.
    lower = min(1e-5, omega*b90/velocity*1e-3)
    x = np.geomspace(lower, 12.0, quadrature.search)
    b = x*velocity/omega

    def residual(log_b):
        radius = np.exp(log_b)
        far = distant_transfer(radius, omega, velocity, density, quadrature)[0]
        near = close_transfer(radius, velocity, density.z, shift)
        return float(far/near - 1)

    far = distant_transfer(b, omega, velocity, density, quadrature)
    near = close_transfer(b, velocity, density.z, shift)
    ratio = far/near - 1
    if np.any(~np.isfinite(ratio)):
        raise FloatingPointError("Nonfinite crossover scan.")
    crossings = np.flatnonzero(np.signbit(ratio[:-1]) != np.signbit(ratio[1:]))
    if len(crossings) != 1 or ratio[0] <= 0 or ratio[-1] >= 0:
        raise RuntimeError(f"Expected one close/distant crossover; found {len(crossings)}.")
    j = crossings[0]
    match = float(np.exp(brentq(residual, np.log(b[j]), np.log(b[j+1]), xtol=2e-7)))
    x_match = omega*match/velocity
    k_far = _distant_integral(x_match, omega, velocity, density, quadrature)
    # Exact integral 2*pi*integral_0^bmatch b*deltaT_close db / omega,
    # converted to the same K normalization as the distant oscillator kernel.
    k_close = shift*velocity**9*match**4/(2*omega*density.z**2)
    cutoff = 0.5616*bd.H2O_CB/velocity
    k_cutoff = _distant_integral(omega*cutoff/velocity, omega, velocity, density, quadrature)
    result = {
        "K_cutoff": k_cutoff, "K_matched": k_far+k_close,
        "K_close": k_close, "K_distant": k_far,
        "induced_shift_Ha": shift, "static_core_offset_Ha": static_core_offset(density),
        "b_match_a0": match, "b_cutoff_a0": cutoff,
        "match_over_b90": match/b90, "x_match": x_match,
        "matching_relative_mismatch": abs(residual(np.log(match))),
        "crossovers_found": len(crossings),
    }
    if any(not np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite matched correction.")
    if min(k_close, k_far, k_cutoff) < 0:
        raise FloatingPointError("Negative integrated correction; not clipped.")
    return result


def compare_kernels(loss_eV, velocity_au, density, *, rtol=0.005):
    """Return refined cutoff/matched kernels with estimated numerical errors.

    Nonrelativistic velocity is supplied explicitly; gamma is never inferred
    or added. The full ice DCS generator does not call this function.
    """
    if not np.isfinite(loss_eV) or loss_eV <= 0 or not np.isfinite(velocity_au) or velocity_au <= 1:
        raise ValueError("Require positive loss and velocity above one Bohr velocity.")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be positive.")
    settings = (Quadrature(), Quadrature(96, 4097, 64, 65), Quadrature(144, 8193, 128, 81))
    previous = _evaluate(loss_eV, velocity_au, density, settings[0])
    keys = ("K_cutoff", "K_matched", "K_close", "K_distant", "induced_shift_Ha", "b_match_a0")
    for level, setting in enumerate(settings[1:], start=1):
        current = _evaluate(loss_eV, velocity_au, density, setting)
        errors = {key: abs(current[key]-previous[key])/max(abs(current[key]), 1e-100) for key in keys}
        if max(errors.values()) <= rtol:
            current.update(refinement_level=level, relative_refinement=errors,
                           quadrature_rtol=rtol, numerical_convergence=True)
            fraction = 2*(current["static_core_offset_Ha"]+current["induced_shift_Ha"])/velocity_au**2
            current["shift_over_incident_electron_energy"] = fraction
            current["positive_effective_close_energy"] = bool(fraction < 1)
            current["physical_accuracy_validated"] = False
            return current
        previous = current
    raise RuntimeError(f"Close/cutoff comparison did not converge at W={loss_eV:g} eV: {errors}")


def equivalent_dcs(kernel, velocity_au, df_dW):
    """Same experimental OOS spectral assignment as before, in m^2/eV.

    Integrating W times this spectrum reproduces the oscillator stopping
    moment. It does not establish a channel-resolved scattering DCS.
    """
    return 4*np.pi/velocity_au**5 * np.asarray(kernel) * np.asarray(df_dW) * BOHR_CM**2 * bd.CM2_TO_M2
