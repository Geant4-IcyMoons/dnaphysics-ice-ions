"""Shared ion optical input: partitioned q=0 ELF and per-H2O normalization.

The Born normalization is preserved; oscillator weights use that same spectrum.
This does not equate the oscillator leading response with finite-q PWBA/RPWBA.
"""
from functools import lru_cache

import numpy as np
from scipy.constants import elementary_charge, electron_mass, epsilon_0, hbar
from physics.constants import (
    AVOGADRO, H2O_MOLAR_MASS_G_MOL, ICE_AMORPHOUS_DENSITY_G_CM3,
    ICE_HEXAGONAL_DENSITY_G_CM3, ELF_ROLLOFF_COEF, ELF_ROLLOFF_E0_eV,
)
from physics.inelastic_dielectric.finite_q import emfietzoglou_model_finite_q as model

OPTICAL_INPUT_VERSION = "partitioned-born-optical-per-H2O-v1"
OPTICAL_SUM_POINTS = 240001
OPTICAL_SUM_MAX_EV = 1.0e9
OPTICAL_SUM_UNIT_EV2_M3 = (
    0.5 * np.pi * (hbar / elementary_charge)**2
    * elementary_charge**2 / (electron_mass * epsilon_0)
)


def valence_rolloff(Ei):
    """Existing Born valence tail only; never apply to the hydrogenic core."""
    Ei = np.asarray(Ei, dtype=float)
    factor = np.ones_like(Ei)
    mask = Ei > ELF_ROLLOFF_E0_eV
    if np.any(mask):
        factor[mask] = 1.0 - ELF_ROLLOFF_COEF * np.log10(Ei[mask] / ELF_ROLLOFF_E0_eV)
    return factor


@lru_cache(maxsize=8)
def optical_normalization(Ep, Bmin, excitations, ionizations, density_g_cm3):
    """Normalize the full optical ELF f-sum at the phase's physical density.

    The f-sum identity is integral W*Im[-1/epsilon(W,0)] dW
    = (pi/2)*(hbar*omega_p)^2, with omega_p^2 = n_e*e^2/(m_e*epsilon_0).
    Set N_H2O = rho*N_A/M_H2O, and scale the ELF-derived GOS by
    N_H2O*10*OPTICAL_SUM_UNIT_EV2_M3/integral before dividing by N_H2O.
    This gives microscopic cross sections per H2O with one material density.
    See Dominguez-Munoz et al. (2022), Eqs. (3),(9), for the same ELF/GOS
    conversion. No stopping-power reference enters this normalization.

    The reference moment uses the same joint outer/K allocation and unscaled
    Heredia-Avalos continuum as the production hydrogenic-GOS path. Always
    retain that reference normalization for no-K and old-optical diagnostics:
    omitting or replacing a process must not amplify the remaining channels.
    This global microscopic conversion is distinct from selectively rescaling
    the K continuum and is shared with the nonlinear oscillator optical input.
    """
    s = model.IceOpticalSet(Ep, Bmin, list(excitations), list(ionizations), None)
    energies = np.geomspace(Bmin, OPTICAL_SUM_MAX_EV, OPTICAL_SUM_POINTS)
    # At q=0 the dispersion coefficients drop out. Use the same partitioned
    # response as the ion kernel, not the unpartitioned optical diagnostic.
    C = model.DispersionCoeffs(0.0, 0.0, 0.0)
    q_zero = np.array([0.0])
    k_strength = model.oxygen_K_hydrogenic_gos_continuum_strength(q_zero)
    e1 = model.epsilon1_valence_Eq(
        energies,
        q_zero,
        s,
        C,
        inner_shell_strength_electrons=k_strength,
    )
    e2 = model.epsilon2_valence_Eq(
        energies,
        q_zero,
        s,
        C,
        partitioned=True,
        inner_shell_strength_electrons=k_strength,
    )
    denominator = np.maximum(
        e1["total"][0] ** 2 + e2["total"][0] ** 2,
        np.finfo(float).tiny,
    )
    valence_elf = e2["total"][0] / denominator
    core_energies = np.unique(np.concatenate((
        np.geomspace(
            np.nextafter(model.OXYGEN_K_B_EV, np.inf),
            OPTICAL_SUM_MAX_EV,
            OPTICAL_SUM_POINTS,
        ),
        np.array([model.OXYGEN_K_ZEFF**2 * model.RYD_ELECTRON_VOLT]),
    )))
    core_elf = model.oxygen_K_ion_hydrogenic_gos_elf(
        core_energies,
        0.0,
        B_K_eV=model.OXYGEN_K_B_EV,
        Zeff=model.OXYGEN_K_ZEFF,
        Ep_eV=Ep,
    )
    valence_moment = float(np.trapezoid(energies * valence_elf, energies))
    core_moment = float(np.trapezoid(core_energies * core_elf, core_energies))
    full_moment = valence_moment + core_moment
    if (not np.isfinite(full_moment) or valence_moment <= 0.0
            or np.any(~np.isfinite(valence_elf)) or np.any(valence_elf < 0.0)
            or np.any(~np.isfinite(core_elf)) or np.any(core_elf < 0.0)
            or not np.isfinite(density_g_cm3) or density_g_cm3 <= 0.0):
        raise RuntimeError("Invalid ion optical ELF f-sum; cannot normalize per H2O.")
    density = density_g_cm3 * 1e6 * AVOGADRO / H2O_MOLAR_MASS_G_MOL
    scale = (
        density * OPTICAL_SUM_UNIT_EV2_M3 * model.WATER_TOTAL_OSCILLATOR_STRENGTH
        / full_moment
    )
    return density, scale, valence_moment, core_moment


def normalization(s):
    if s.material not in ("amorphous", "hexagonal"):
        raise ValueError("Shared optical input requires a named ice phase")
    rho = (ICE_AMORPHOUS_DENSITY_G_CM3 if s.material == "amorphous"
           else ICE_HEXAGONAL_DENSITY_G_CM3)
    return optical_normalization(s.Ep, s.Bmin, tuple(s.excitations), tuple(s.ionizations), rho)


def raw_oos_components(W, s):
    """Unscaled per-molecule OOS; anchor partition thresholds on every call.

    Sorted unique energies plus exact thresholds prevent interpolation-dependent
    results on sparse, repeated, or unsorted table grids. No additional
    valence/core renormalization is applied.
    """
    w = np.asarray(W, float)
    if not w.size or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Optical energies must be finite and positive")
    anchors = [s.Bmin, *[o.Bth for o in s.ionizations]]
    energies = np.unique(np.concatenate((w.ravel(), anchors)))
    q, c = np.array([0.]), model.DispersionCoeffs(0., 0., 0.)
    strength = model.oxygen_K_hydrogenic_gos_continuum_strength(q)
    e1 = model.epsilon1_valence_Eq(energies, q, s, c,
        inner_shell_strength_electrons=strength)["total"][0]
    e2 = model.epsilon2_valence_Eq(energies, q, s, c, partitioned=True,
        inner_shell_strength_electrons=strength)["total"][0]
    val = e2 / np.maximum(e1*e1 + e2*e2, np.finfo(float).tiny)
    core = model.oxygen_K_ion_hydrogenic_gos_elf(energies, 0., Ep_eV=s.Ep)
    n = normalization(s)[0]
    factor = energies / (n*OPTICAL_SUM_UNIT_EV2_M3)
    indices = np.searchsorted(energies, w)
    return (factor*val*valence_rolloff(energies))[indices], (factor*core)[indices]


@lru_cache(maxsize=8)
def reference_integrals(Ep, Bmin, excitations, ionizations, material):
    """Actual rolled optical strengths on a fixed, threshold-resolved grid."""
    s = model.IceOpticalSet(Ep, Bmin, list(excitations), list(ionizations), None, material)
    edges = [s.Bmin, *[o.Bth for o in s.ionizations], model.OXYGEN_K_B_EV,
             model.OXYGEN_K_ZEFF**2*model.RYD_ELECTRON_VOLT]
    w = np.unique(np.concatenate((np.geomspace(1., OPTICAL_SUM_MAX_EV, OPTICAL_SUM_POINTS),
                                  edges, np.nextafter(edges, 0.), np.nextafter(edges, np.inf))))
    val, core = raw_oos_components(w, s)
    return float(np.trapezoid(val, w)), float(np.trapezoid(core, w))
