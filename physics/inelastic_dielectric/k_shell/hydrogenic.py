"""Published, unscaled hydrogenic oxygen K continuum; see README.md."""
import numpy as np
from physics.constants import WATER_TOTAL_OSCILLATOR_STRENGTH

OXYGEN_K_B_EV = 543.4

OXYGEN_K_ZEFF = 7.7

OXYGEN_K_OCCUPANCY = 2.0

OXYGEN_K_GOS_VERSION = "heredia-avalos-2005-unshifted-v1"

RYD_ELECTRON_VOLT = 13.605693122994

_OXYGEN_K_STRENGTH_CACHE = {}

def oxygen_K_hydrogenic_gos_df_dE(E_eV, q_au, B_K_eV=OXYGEN_K_B_EV, Zeff=OXYGEN_K_ZEFF):
    """
    Published hydrogenic 1s GOS approximation for the O K shell, in eV^-1.

    Heredia-Avalos et al., Phys. Rev. A 72, 052902 (2005), Table I and
    Appendix A, Eqs. (A.1)-(A.9), (A.13); doi:10.1103/PhysRevA.72.052902.
    For the two occupied 1s states, B_K_eV = 543.4 and Zeff = 7.7:
        Qbar = q_au^2 / Zeff^2
        Wbar = E_eV / (Zeff^2 R_eV)
        kappa^2 = Wbar - 1.0

    The physical threshold gates the spectrum; it does not shift Wbar.
    Eq. (A.8) applies between B_K and Zeff^2 R (~806.68 eV), where kappa^2
    is negative. Eqs. (A.5)-(A.6) apply above that branch boundary. The
    negative branch is the published analytic continuation, not a list of
    molecular bound excitations. Occupancy is included exactly once.

    This ion/projectile helper is not the electron optical K-shell model.
    No optical-area multiplier, ELF rolloff, or valence compensation is
    applied here. The approximate shell spectrum need not integrate to two;
    it does not by itself establish a complete molecular f-sum rule.
    """
    E, q = np.broadcast_arrays(np.asarray(E_eV, float), np.asarray(q_au, float))
    B_K_eV, Zeff = float(B_K_eV), float(Zeff)
    if not np.isfinite(B_K_eV) or B_K_eV <= 0.0:
        raise ValueError("B_K_eV must be finite and positive.")
    if not np.isfinite(Zeff) or Zeff <= 0.0:
        raise ValueError("Zeff must be finite and positive.")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(q)):
        raise ValueError("K-shell energies and momenta must be finite.")
    out = np.zeros_like(E, dtype=float)
    mask = E > B_K_eV
    if not np.any(mask):
        return out

    E_m = E[mask]
    q_m = np.abs(q[mask])
    z2 = Zeff ** 2
    Qbar = q_m * q_m / z2
    Wbar = E_m / (z2 * RYD_ELECTRON_VOLT)
    kappa2 = Wbar - 1.0
    A_1s = Qbar - Wbar + 2.0
    den = ((Qbar - Wbar) ** 2 + 4.0 * Qbar) ** 3

    # The two branches have the same finite limit at kappa^2 = 0.
    coulomb = np.exp(-4.0 / (Qbar + 1.0))
    positive = kappa2 > 0.0
    kappa = np.sqrt(kappa2[positive])
    # atan2 selects the continuous Coulomb phase when A_1s becomes negative.
    theta = np.arctan2(2.0 * kappa, A_1s[positive])
    coulomb[positive] = np.exp(-2.0 * theta / kappa) / (
        -np.expm1(-2.0 * np.pi / kappa)
    )
    negative = kappa2 < 0.0
    eta = np.sqrt(-kappa2[negative])
    # A_1s - 2*eta = Qbar + (1-eta)^2. log1p avoids cancellation
    # in Eq. (A.8) as eta approaches zero from below the branch boundary.
    lower = Qbar[negative] + (1.0 - eta) ** 2
    coulomb[negative] = np.exp(-np.log1p(4.0 * eta / lower) / eta)

    df_dW_per_electron = 128.0 * Wbar * (Qbar + Wbar / 3.0) * coulomb / den
    out[mask] = OXYGEN_K_OCCUPANCY * df_dW_per_electron / (z2 * RYD_ELECTRON_VOLT)
    return out

def _oxygen_K_hydrogenic_strength_table(
    B_K_eV=OXYGEN_K_B_EV,
    Zeff=OXYGEN_K_ZEFF,
):
    """Build the converged continuum-strength lookup used by ion tables."""
    key = (round(float(B_K_eV), 12), round(float(Zeff), 12))
    cached = _OXYGEN_K_STRENGTH_CACHE.get(key)
    if cached is not None:
        return cached

    # log1p(q) resolves the rapid low-q variation while q=40 a0^-1 is
    # already within 1e-6 of the two-electron high-q closure. Integrating in
    # x = log(E-B) resolves the edge, branch boundary, and asymptotic tail.
    q_grid = np.expm1(np.linspace(0.0, np.log1p(40.0), 1025))
    log_offset = np.linspace(np.log(1.0e-12), np.log(1.0e10), 2049)
    offsets = np.exp(log_offset)
    E_grid = float(B_K_eV) + offsets
    dx = float(log_offset[1] - log_offset[0])
    strength = np.empty_like(q_grid)
    for iq, q_value in enumerate(q_grid):
        density = oxygen_K_hydrogenic_gos_df_dE(
            E_grid,
            q_value,
            B_K_eV=B_K_eV,
            Zeff=Zeff,
        ) * offsets
        strength[iq] = dx / 3.0 * (
            density[0]
            + density[-1]
            + 4.0 * np.sum(density[1:-1:2])
            + 2.0 * np.sum(density[2:-1:2])
        )

    if not np.all(np.isfinite(strength)) or np.any(strength <= 0.0):
        raise RuntimeError("Failed to build the oxygen K-continuum strength table.")
    cached = (np.log1p(q_grid), strength)
    _OXYGEN_K_STRENGTH_CACHE[key] = cached
    return cached

def oxygen_K_hydrogenic_gos_continuum_strength(
    q_au,
    B_K_eV=OXYGEN_K_B_EV,
    Zeff=OXYGEN_K_ZEFF,
):
    """Return integral(df_K/dE dE) for the unscaled Heredia K continuum.

    The result is in electrons per H2O molecule. It is evaluated from the
    same threshold-gated, two-branch GOS as the production K-shell DCS. The
    interpolation error is below 1e-5 electron against direct quadrature at
    representative q. Above 40 a0^-1 the converged two-electron limit is used.
    """
    q = np.asarray(q_au, float)
    if not np.all(np.isfinite(q)):
        raise ValueError("K-shell momenta must be finite.")
    log_q_grid, strength_grid = _oxygen_K_hydrogenic_strength_table(
        B_K_eV=B_K_eV,
        Zeff=Zeff,
    )
    result = np.interp(
        np.log1p(np.abs(q)),
        log_q_grid,
        strength_grid,
        right=OXYGEN_K_OCCUPANCY,
    )
    return float(result) if result.ndim == 0 else result

def oxygen_K_hydrogenic_occupancy_remainder(
    q_au,
    B_K_eV=OXYGEN_K_B_EV,
    Zeff=OXYGEN_K_ZEFF,
):
    """Diagnostic occupancy remainder, 2 - integral(df_K/dE dE).

    This is not a bound-excitation strength or an implemented spectral channel.
    Heredia-Avalos supplies an ionization GOS only; its negative-kappa branch is
    an analytic continuation after imposing the experimental edge. A bound
    spectrum requires independent excitation energies and finite-q strengths.
    """
    return OXYGEN_K_OCCUPANCY - oxygen_K_hydrogenic_gos_continuum_strength(
        q_au,
        B_K_eV=B_K_eV,
        Zeff=Zeff,
    )

def oxygen_K_hydrogenic_gos_elf(
    E_eV,
    q_au,
    B_K_eV=OXYGEN_K_B_EV,
    Zeff=OXYGEN_K_ZEFF,
    Ep_eV=20.82,
    total_oscillator_strength=WATER_TOTAL_OSCILLATOR_STRENGTH,
):
    """
    Return the additive O K-shell ELF from the published hydrogenic GOS.

    The GOS-to-ELF conversion uses the same Ep/Z convention as the dielectric
    model:
        Im[-1/epsilon_K] ~= (pi/2) * Ep^2 / Z * (1/E) * df_K(q,E)/dE
    The published GOS amplitude is preserved. This API deliberately exposes no
    independent optical-area normalization: the K continuum participates in
    the joint finite-q molecular oscillator-strength allocation instead.
    """
    E, q = np.broadcast_arrays(np.asarray(E_eV, float), np.asarray(q_au, float))
    df = oxygen_K_hydrogenic_gos_df_dE(E, q, B_K_eV=B_K_eV, Zeff=Zeff)
    elf = (
        0.5
        * np.pi
        * float(Ep_eV) ** 2
        / float(total_oscillator_strength)
        * np.divide(df, E, out=np.zeros_like(df), where=E > 0.0)
    )
    elf = np.where(E > float(B_K_eV), elf, 0.0)
    return elf

def oxygen_K_ion_hydrogenic_gos_elf(
    E_eV,
    q_au,
    B_K_eV=OXYGEN_K_B_EV,
    Zeff=OXYGEN_K_ZEFF,
    Ep_eV=20.82,
):
    """
    Ion/projectile O K-shell ELF from the q-dependent hydrogenic 1s GOS.
    """
    return oxygen_K_hydrogenic_gos_elf(
        E_eV,
        q_au,
        B_K_eV=B_K_eV,
        Zeff=Zeff,
        Ep_eV=Ep_eV,
    )

def oxygen_K_hydrogenic_gos_fsum(
    B_K_eV=OXYGEN_K_B_EV,
    Zeff=OXYGEN_K_ZEFF,
):
    """Return the unscaled optical K-continuum fraction of the H2O f-sum.

    The denominator is the ten-electron molecular oscillator-strength budget.
    This is a diagnostic of the published threshold-gated continuum, not a
    normalization target.
    """
    strength = oxygen_K_hydrogenic_gos_continuum_strength(
        0.0, B_K_eV=B_K_eV, Zeff=Zeff
    )
    return float(strength) / WATER_TOTAL_OSCILLATOR_STRENGTH
