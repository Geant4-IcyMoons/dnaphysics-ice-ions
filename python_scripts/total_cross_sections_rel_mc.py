#!/usr/bin/env python3

import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (
    C_AU,
    EH,
    EV_TO_HA,
    ELF_ROLLOFF_COEF,
    ELF_ROLLOFF_E0_eV,
    MC2_HA,
    MC2_eV,
    N,
    OUTPUT_DIR,
    REGIME_I_MAX_eV,
    REGIME_II_MAX_eV,
    REGIME_III_MAX_eV,
    REGIME_IV_MAX_eV,
    a0,
    mass,
)
import emfietzoglou_model_finite_q as model

# ----------------------------------------------------------------------
# Constants for integration (from constants.py)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Helpers: Relativistic corrections
# ----------------------------------------------------------------------
# Regime correction handles (toggle corrections in regimes II/III/IV).
APPLY_CORRECTIONS_REGIME_II = True
APPLY_CORRECTIONS_REGIME_III = True
APPLY_CORRECTIONS_REGIME_IV = True
APPLY_MOTT_COULOMB = True

def _set_regime_corrections(apply_regime_ii=None, apply_regime_iii=None, apply_regime_iv=None):
    global APPLY_CORRECTIONS_REGIME_II, APPLY_CORRECTIONS_REGIME_III, APPLY_CORRECTIONS_REGIME_IV
    if apply_regime_ii is not None:
        APPLY_CORRECTIONS_REGIME_II = bool(apply_regime_ii)
    if apply_regime_iii is not None:
        APPLY_CORRECTIONS_REGIME_III = bool(apply_regime_iii)
    if apply_regime_iv is not None:
        APPLY_CORRECTIONS_REGIME_IV = bool(apply_regime_iv)

def _set_mc_correction(apply_mc=None):
    global APPLY_MOTT_COULOMB
    if apply_mc is not None:
        APPLY_MOTT_COULOMB = bool(apply_mc)

def _simpson_integrate(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if n < 2:
        return 0.0
    if n != x.size:
        raise ValueError("x and y must have the same length")
    if n == 2:
        return 0.5 * (y[0] + y[1]) * (x[1] - x[0])

    dx = np.diff(x)
    if not np.allclose(dx, dx[0]):
        return float(np.sum(0.5 * dx * (y[:-1] + y[1:])))

    h = dx[0]
    if n % 2 == 1:
        return float(
            (h / 3.0)
            * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
        )

    n1 = n - 1
    simpson_part = (h / 3.0) * (
        y[0]
        + y[n1 - 1]
        + 4.0 * np.sum(y[1 : n1 - 1 : 2])
        + 2.0 * np.sum(y[2 : n1 - 2 : 2])
    )
    trap_part = 0.5 * (y[-2] + y[-1]) * (x[-1] - x[-2])
    return float(simpson_part + trap_part)

def _energy_grid(Emin, Emax, N, use_log=True):
    Emin = float(Emin)
    Emax = float(Emax)
    if (not use_log) or (Emin <= 0.0) or (Emax <= Emin):
        return np.linspace(Emin, Emax, N)
    log_min = np.log(Emin)
    log_max = np.log(Emax)
    if (not np.isfinite(log_min)) or (not np.isfinite(log_max)) or (log_max <= log_min):
        return np.linspace(Emin, Emax, N)
    return np.exp(np.linspace(log_min, log_max, N))

def _regime_flags(Tj):
    Tj = float(Tj)
    if Tj <= REGIME_I_MAX_eV:
        use_mc, use_rel_long, use_rel_trans, use_density_effect = True, False, False, False
        return (use_mc and APPLY_MOTT_COULOMB), use_rel_long, use_rel_trans, use_density_effect
    if Tj <= REGIME_II_MAX_eV:
        if not APPLY_CORRECTIONS_REGIME_II:
            return False, False, False, False
        use_mc, use_rel_long, use_rel_trans, use_density_effect = True, True, False, False
        return (use_mc and APPLY_MOTT_COULOMB), use_rel_long, use_rel_trans, use_density_effect
    if Tj < REGIME_III_MAX_eV:
        if not APPLY_CORRECTIONS_REGIME_III:
            return False, False, False, False
        use_mc, use_rel_long, use_rel_trans, use_density_effect = False, True, True, False
        return (use_mc and APPLY_MOTT_COULOMB), use_rel_long, use_rel_trans, use_density_effect
    if Tj <= REGIME_IV_MAX_eV:
        if not APPLY_CORRECTIONS_REGIME_IV:
            return False, False, False, False
        use_mc, use_rel_long, use_rel_trans, use_density_effect = False, True, True, True
        return (use_mc and APPLY_MOTT_COULOMB), use_rel_long, use_rel_trans, use_density_effect
    if not APPLY_CORRECTIONS_REGIME_IV:
        return False, False, False, False
    use_mc, use_rel_long, use_rel_trans, use_density_effect = False, True, True, True
    return (use_mc and APPLY_MOTT_COULOMB), use_rel_long, use_rel_trans, use_density_effect

def _elf_rolloff_factor(Ei):
    Ei = np.asarray(Ei, dtype=float)
    factor = np.ones_like(Ei)
    mask = Ei > ELF_ROLLOFF_E0_eV
    if np.any(mask):
        factor[mask] = 1.0 - ELF_ROLLOFF_COEF * np.log10(Ei[mask] / ELF_ROLLOFF_E0_eV)
    return factor

def beta2_rel(Tj):
    # Use exactly the form you specified:
    return 1.0 - 1.0 / (Tj / MC2_eV + 1.0)**2

def delta_fermi(T):
    """
    Steinheimer-Fano density effect for liquid water.
    """
    b2 = beta2_rel(T)
    b2 = min(max(float(b2), np.finfo(float).tiny), 1.0 - np.finfo(float).eps)
    beta = np.sqrt(b2)
    X = np.log10(np.sqrt(beta / (1.0 - beta * beta)))

    # liquid water parameters
    alpha = 0.09116
    X1 = 2.8004
    m = 3.4773
    C = -3.5017
    X0 = 0.24

    if X < X0:
        return 0.0
    if X < X1:
        return 4.6052 * X + alpha * (X1 - X) ** m + C
    return 4.6052 * X + C

def Q_q(q_au):
    """
    Q(q) in eV.
    """
    q_au = np.asarray(q_au, dtype=float)
    Q_Ha = np.sqrt((C_AU * q_au)**2 + (MC2_HA)**2) - MC2_HA
    return Q_Ha * EH

# ----------------------------------------------------------------------
# Helper: q-bounds for scalar Ei, Tj (eV)
# ----------------------------------------------------------------------
def _q_bounds_scalar(Ei, Tj, mass=1.0):
    """Return (q_lo, q_hi) for scalar Ei, Tj (both in eV)."""
    Ei_H = Ei * EV_TO_HA
    T_H = Tj * EV_TO_HA
    if Ei_H >= T_H:
        return 0.0, 0.0

    d = T_H - Ei_H
    if d <= 0.0:
        return 0.0, 0.0

    sqrtT = np.sqrt(T_H)
    sqrt_d = np.sqrt(d)
    qlo = np.sqrt(2.0 * mass) * (sqrtT - sqrt_d)
    qhi = np.sqrt(2.0 * mass) * (sqrtT + sqrt_d)

    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        return 0.0, 0.0
    return float(qlo), float(qhi)

def _q_bounds_scalar_rel(Ei, Tj):
    if Ei >= Tj:
        return 0.0, 0.0

    Ei_H = Ei * EV_TO_HA
    T_H  = Tj * EV_TO_HA
    d = T_H - Ei_H
    if d <= 0.0:
        return 0.0, 0.0

    term1 = np.sqrt(T_H * (T_H + 2.0 * MC2_HA))
    term2 = np.sqrt(d   * (d   + 2.0 * MC2_HA))
    s12 = term1 + term2
    if (not np.isfinite(s12)) or (s12 <= 0.0):
        return 0.0, 0.0
    num = Ei_H * (2.0 * T_H - Ei_H + 2.0 * MC2_HA)
    qlo = (num / s12) / C_AU
    qhi = s12 / C_AU
    if (not np.isfinite(qlo)) or (not np.isfinite(qhi)) or (qhi <= qlo) or (qlo <= 0.0):
        return 0.0, 0.0
    return float(qlo), float(qhi)

# ----------------------------------------------------------------------
# Inner q-integral at fixed Ei for channels (excitation / ionization)
# ----------------------------------------------------------------------
def _integrate_channel_single_E(
    Ei, Tj, idx, channel_type, s, C, Nq=400, use_rel_bounds=False
):
    """
    Compute inner integral over q:

        ∫ dq [ ELF_channel(Ei, q) / q ]

    for one excitation or ionization channel, at fixed Ei, Tj.
    """
    if use_rel_bounds:
        qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    else:
        qlo, qhi = _q_bounds_scalar(Ei, Tj, mass)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0

    # q-grid
    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    qvals = np.exp(xi)
    E_arr = np.array([Ei], float)

    # Vectorized dielectric functions at (Ei, qvals)
    e1 = model.epsilon1_valence_Eq(E_arr, qvals, s, C)
    e2 = model.epsilon2_valence_Eq(E_arr, qvals, s, C)

    e1t = e1["total"][:, 0]   # shape (Nq,)
    e2t = e2["total"][:, 0]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

    if channel_type == "excitation":
        vals = e2["excitations"][idx][:, 0] / denom
    elif channel_type == "ionization":
        vals = e2["ionizations"][idx][:, 0] / denom
    else:
        raise ValueError("channel_type must be 'excitation' or 'ionization'")

    vals = vals * _elf_rolloff_factor(Ei)
    accum = float(_simpson_integrate(vals, xi))

    int_cons = 1.0 / (np.pi * a0 * N * Tj)
    return float(int_cons * accum)

def _integrate_channel_single_E_rel(Ei, Tj, idx, channel_type, s, C, Nq=400):
    qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0

    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    qvals = np.exp(xi)
    E_arr = np.array([Ei], float)

    e1 = model.epsilon1_valence_Eq(E_arr, qvals, s, C)
    e2 = model.epsilon2_valence_Eq(E_arr, qvals, s, C)

    e1t = e1["total"][:, 0]
    e2t = e2["total"][:, 0]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

    if channel_type == "excitation":
        vals = e2["excitations"][idx][:, 0] / denom
    else:
        vals = e2["ionizations"][idx][:, 0] / denom

    vals = vals * _elf_rolloff_factor(Ei)
    # Q(q) in eV
    Q_eV = Q_q(qvals)
    Q_eV = np.where(Q_eV == 0.0, np.finfo(float).tiny, Q_eV)

    factor1 = (C_AU**2 * qvals) / np.sqrt((C_AU * qvals)**2 + (MC2_HA**2))
    factor1 *= EH  # dQ/dq in eV per a0^-1
    factor2 = (1.0 + Q_eV / MC2_eV) / (1.0 + Q_eV / (2.0 * MC2_eV))
    factor3 = 1 / Q_eV
    kernel = factor1 * factor2 * factor3

    integrand = vals * kernel
    accum = float(_simpson_integrate(integrand * qvals, xi))

    b2 = beta2_rel(Tj)
    b2 = max(b2, np.finfo(float).tiny)

    int_cons = 1.0 / (np.pi * a0 * N * MC2_eV * b2)
    return float(int_cons * accum)

def _integrate_channel_single_E_trans(
    Ei, Tj, idx, channel_type, s, C, Nq=0, use_density_effect=False
):
    """ Transverse RPWBA term (Fano approximation at q=0)"""
    # beta^2 and transverse bracket
    b2 = beta2_rel(Tj)
    b2 = max(b2, np.finfo(float).tiny)
    bracket = np.log(1.0 / max(1.0 - b2, np.finfo(float).tiny)) - b2
    if use_density_effect:
        bracket = max(bracket - 0.5 * delta_fermi(Tj), 0.0)

    # Optical (q=0) channel-resolved ELF via epsilon2_channel / (epsilon1_total^2 + epsilon2_total^2)
    E_arr = np.array([Ei], float)
    e1 = model.epsilon1_valence_E0(E_arr, s)
    e2 = model.epsilon2_valence_E0(E_arr, s)

    e1t = e1["total"]
    e2t = e2["total"]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

    if channel_type == "excitation":
        elf0 = e2["excitations"][idx] / denom
    elif channel_type == "ionization":
        elf0 = e2["ionizations"][idx] / denom
    else:
        raise ValueError("channel_type must be 'excitation' or 'ionization'")

    elf0 = float(np.asarray(elf0).ravel()[0] * _elf_rolloff_factor(Ei))

    int_cons = 1.0 / (np.pi * a0 * N * MC2_eV * b2)
    return float(int_cons * elf0 * bracket)

# ----------------------------------------------------------------------
# Low-energy Mott–Coulomb (MC) corrections using PWBA kernel evaluations
# ----------------------------------------------------------------------
def _dsigma_pwba_dE(Ei, Tj, idx, channel_type, s, C, Nq=400, use_rel=False):
    """
    Return the differential cross section d sigma/dE at (Ei,Tj) for one channel.
    If use_rel=True, this uses the longitudinal relativistic kernel; otherwise it uses
    the nonrelativistic PWBA kernel.
    """
    if use_rel:
        return _integrate_channel_single_E_rel(Ei, Tj, idx, channel_type, s, C, Nq=Nq)
    return _integrate_channel_single_E(Ei, Tj, idx, channel_type, s, C, Nq=Nq)

def _dsigma_mc_ionization_dE(Ei, Tj, j, s, C, Nq=400, use_rel=False):
    """Mott–Coulomb exchange-style correction for *ionization* shells. """
    osc = s.ionizations[j]
    B = float(getattr(osc, "Bth"))
    U = float(getattr(osc, "U"))

    Tprime = float(Tj + B + U)
    E2 = float(Tj + 2.0*B + U - Ei)

    a = _dsigma_pwba_dE(Ei, Tprime, j, "ionization", s, C, Nq=Nq, use_rel=use_rel)
    b = _dsigma_pwba_dE(E2, Tprime, j, "ionization", s, C, Nq=Nq, use_rel=use_rel)

    # Guard against tiny negative numerical noise
    a = max(a, 0.0)
    b = max(b, 0.0)

    return float(a + b - np.sqrt(a*b))

def _sigma_pwba_excitation_shifted_T(s, C, Tshift, Tj, k, NE=400, Nq=400, use_rel=False):
    """sigma_PWBA for excitation k, with shifted kernel but kinematic E-window from Tj."""
    Emin = float(s.excitations[k].Bth)
    Emax = float(Tj)
    if Emin >= Emax:
        return 0.0

    Egrid = _energy_grid(Emin, Emax, NE)
    vals = np.empty_like(Egrid)
    for i, Ei in enumerate(Egrid):
        vals[i] = _dsigma_pwba_dE(Ei, Tshift, k, "excitation", s, C, Nq=Nq, use_rel=use_rel)

    vals = np.where(vals < 0.0, 0.0, vals)
    return float(_simpson_integrate(vals, Egrid))

def _sigma_mc_ionization(s, C, Tj, j, NE=400, Nq=400, use_rel=False):
    """sigma_MC(T) for ionization shell j: integrate d sigma_MC/dE over E."""
    osc = s.ionizations[j]
    B = float(getattr(osc, "Bth"))
    U = float(getattr(osc, "U"))

    Emin = B
    Emax = 0.5 * (Tj + B)
    if Emin >= Emax:
        return 0.0

    Egrid = _energy_grid(Emin, Emax, NE)
    vals = np.empty_like(Egrid)
    for i, Ei in enumerate(Egrid):
        vals[i] = _dsigma_mc_ionization_dE(Ei, Tj, j, s, C, Nq=Nq, use_rel=use_rel)

    vals = np.where(vals < 0.0, 0.0, vals)
    return float(_simpson_integrate(vals, Egrid))

def _total_transverse_sigma(s, C, Tj, NE=400, use_density_effect=False):
    """Total transverse (valence-only) sigma via the optical-limit kernel."""
    Tj = float(Tj)
    exc_total = 0.0
    ion_total = 0.0

    for k in range(len(s.excitations)):
        Emin = float(s.excitations[k].Bth)
        Emax = Tj
        if Emin >= Emax:
            continue
        Egrid = _energy_grid(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        for i, Ei in enumerate(Egrid):
            vals[i] = _integrate_channel_single_E_trans(
                Ei, Tj, k, "excitation", s, C, use_density_effect=use_density_effect
            )
        exc_total += float(_simpson_integrate(vals, Egrid))

    for j in range(len(s.ionizations)):
        B = float(getattr(s.ionizations[j], "Bth"))
        Emin = B
        Emax = 0.5 * (Tj + B)
        if Emin >= Emax:
            continue
        Egrid = _energy_grid(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        for i, Ei in enumerate(Egrid):
            vals[i] = _integrate_channel_single_E_trans(
                Ei, Tj, j, "ionization", s, C, use_density_effect=use_density_effect
            )
        ion_total += float(_simpson_integrate(vals, Egrid))

    return float(exc_total + ion_total)

# ----------------------------------------------------------------------
# Inner q-integral at fixed Ei for K-shell channel
# ----------------------------------------------------------------------
def _integrate_kshell_single_E(
    Ei, Tj, s, Nq=400, include_kshell=True, use_rel_bounds=False
):
    """
    Inner integral over q for K-shell:
    """
    if not include_kshell or (s.kshell is None):
        return 0.0

    if use_rel_bounds:
        qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    else:
        qlo, qhi = _q_bounds_scalar(Ei, Tj, mass)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0

    ks_arr = model.epsilon2_Kshell_E0(np.array([Ei], float), s)
    ks_val = float(ks_arr[0])
    if ks_val == 0.0:
        return 0.0

    ks_val *= float(_elf_rolloff_factor(Ei))
    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    accum = float(_simpson_integrate(np.full_like(xi, ks_val), xi))

    int_cons = 1.0 / (np.pi * a0 * N * Tj)
    return float(int_cons * accum)

def _integrate_kshell_single_E_rel(Ei, Tj, s, Nq=400, include_kshell=True):
    """
    Inner integral over q for K-shell:
    """
    if not include_kshell or (s.kshell is None):
        return 0.0

    qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0

    # K-shell imaginary part at optical limit, with threshold gating
    ks_arr = model.epsilon2_Kshell_E0(np.array([Ei], float), s)
    ks_val = float(ks_arr[0])
    if ks_val == 0.0:
        return 0.0

    ks_val *= float(_elf_rolloff_factor(Ei))
    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    qvals = np.exp(xi)

    # Q(q) in eV
    Q_eV = Q_q(qvals)
    Q_eV = np.where(Q_eV == 0.0, np.finfo(float).tiny, Q_eV)

    factor1 = (C_AU**2 * qvals) / np.sqrt((C_AU * qvals) ** 2 + (MC2_HA ** 2))
    factor1 *= EH  # dQ/dq in eV per a0^-1
    factor2 = (1.0 + Q_eV / MC2_eV) / (1.0 + Q_eV / (2.0 * MC2_eV))
    factor3 = 1 / Q_eV
    kernel = factor1 * factor2 * factor3

    integrand = ks_val * kernel
    accum = float(_simpson_integrate(integrand * qvals, xi))

    b2 = beta2_rel(Tj)
    b2 = max(b2, np.finfo(float).tiny)

    int_cons = 1.0 / (np.pi * a0 * N * MC2_eV * b2)
    return float(int_cons * accum)

# ----------------------------------------------------------------------
# Q-integrated ELF per channel, on its own E-grid
# ----------------------------------------------------------------------
def integrate_elf_channels_per_channel_q(
    s,
    C,
    T,
    NE=400,
    Nq=400,
    include_kshell=True,
    use_rel_long=False,
    use_rel_trans=False,
    use_density_effect=False,
):
    """
    Integrate ELF(E,q)/q over q for each excitation, ionization, and K-shell,
    and ALSO compute a second "relativistic longitudinal-corrected" version
    of the same inner-q integrals (stored with *_rel keys).

    use_rel_long and use_rel_trans control whether the corresponding arrays
    are computed or filled with zeros.
    """

    # --- Channel energy windows ---
    exc_Emin = np.array([osc.Bth for osc in s.excitations], float)
    exc_Emax = np.array([T for _ in s.excitations], float)

    ion_Emin = np.array([osc.Bth for osc in s.ionizations], float)
    ion_Emax = np.array([(T + osc.Bth) / 2.0 for osc in s.ionizations], float)

    if include_kshell and (s.kshell is not None):
        kshell_Emin = s.kshell.Bth
        kshell_Emax = T
    else:
        kshell_Emin = None
        kshell_Emax = None

    results = {
        "excitation_E": [],
        "excitation_int": [],
        "excitation_int_rel": [],
        "excitation_int_rel_trans": [],

        "ionization_E": [],
        "ionization_int": [],
        "ionization_int_rel": [],
        "ionization_int_rel_trans": [],

        "kshell_E": None,
        "kshell_int": None,
        "kshell_int_rel": None,
    }

    # ------------------- Excitations -------------------
    for k in range(len(s.excitations)):
        Emin, Emax = exc_Emin[k], exc_Emax[k]
        if Emin >= Emax:
            results["excitation_E"].append(np.array([], float))
            results["excitation_int"].append(np.array([], float))
            results["excitation_int_rel"].append(np.array([], float))
            results["excitation_int_rel_trans"].append(np.array([], float))
            continue

        Egrid = _energy_grid(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.zeros_like(Egrid)
        vals_rel_trans = np.zeros_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_channel_single_E(
                Ei, T, k, "excitation", s, C, Nq=Nq, use_rel_bounds=use_rel_long
            )

            # Relativistic-longitudinal corrected inner-q integral
            if use_rel_long:
                vals_rel[i] = _integrate_channel_single_E_rel(Ei, T, k, "excitation", s, C, Nq=Nq)
            if use_rel_trans:
                vals_rel_trans[i] = _integrate_channel_single_E_trans(
                    Ei, T, k, "excitation", s, C, use_density_effect=use_density_effect
                )


        results["excitation_E"].append(Egrid)
        results["excitation_int"].append(vals)
        results["excitation_int_rel"].append(vals_rel)
        results["excitation_int_rel_trans"].append(vals_rel_trans)

    # ------------------- Ionizations -------------------
    for j in range(len(s.ionizations)):
        Emin, Emax = ion_Emin[j], ion_Emax[j]
        if Emin >= Emax:
            results["ionization_E"].append(np.array([], float))
            results["ionization_int"].append(np.array([], float))
            results["ionization_int_rel"].append(np.array([], float))
            results["ionization_int_rel_trans"].append(np.array([], float))
            continue

        Egrid = _energy_grid(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.zeros_like(Egrid)
        vals_rel_trans = np.zeros_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_channel_single_E(
                Ei, T, j, "ionization", s, C, Nq=Nq, use_rel_bounds=use_rel_long
            )

            # Relativistic-longitudinal corrected inner-q integral
            if use_rel_long:
                vals_rel[i] = _integrate_channel_single_E_rel(Ei, T, j, "ionization", s, C, Nq=Nq)
            if use_rel_trans:
                vals_rel_trans[i] = _integrate_channel_single_E_trans(
                    Ei, T, j, "ionization", s, C, use_density_effect=use_density_effect
                )


        results["ionization_E"].append(Egrid)
        results["ionization_int"].append(vals)
        results["ionization_int_rel"].append(vals_rel)
        results["ionization_int_rel_trans"].append(vals_rel_trans)

    # ------------------- K-shell -----------------------
    if (kshell_Emin is not None) and (kshell_Emin < kshell_Emax):
        Egrid = _energy_grid(kshell_Emin, kshell_Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.zeros_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_kshell_single_E(
                Ei, T, s, Nq=Nq, include_kshell=include_kshell, use_rel_bounds=use_rel_long
            )

            # Relativistic-longitudinal corrected inner-q integral
            if use_rel_long:
                vals_rel[i] = _integrate_kshell_single_E_rel(
                    Ei, T, s, Nq=Nq, include_kshell=include_kshell
                )

        results["kshell_E"] = Egrid
        results["kshell_int"] = vals
        results["kshell_int_rel"] = vals_rel if use_rel_long else None

    return results

# ----------------------------------------------------------------------
# Full double integral over E and q: sigma(T) per channel and totals
# ----------------------------------------------------------------------
def integrate_elf_double_integral(
    s,
    C,
    T,
    NE=200,
    Nq=200,
    include_kshell=True,
    use_mott_coulomb=False,
):
    """
    Compute full double integral sigma(T) per channel and totals.
    """

    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(T)
    if (not use_mott_coulomb) or (not APPLY_MOTT_COULOMB):
        use_mc = False

    # 1) Inner q-integrals as functions of E
    integ = integrate_elf_channels_per_channel_q(
        s,
        C,
        T=T,
        NE=NE,
        Nq=Nq,
        include_kshell=include_kshell,
        use_rel_long=use_rel_long,
        use_rel_trans=use_rel_trans,
        use_density_effect=use_density_effect,
    )

    # -------------------- PWBA baseline --------------------
    exc_sigma_pwba = []
    ion_sigma_pwba = []
    kshell_sigma_pwba = None

    for k in range(len(s.excitations)):
        E_k = integ["excitation_E"][k]
        y_k = integ["excitation_int"][k]
        exc_sigma_pwba.append(0.0 if E_k.size == 0 else float(_simpson_integrate(y_k, E_k)))

    for j in range(len(s.ionizations)):
        E_j = integ["ionization_E"][j]
        y_j = integ["ionization_int"][j]
        ion_sigma_pwba.append(0.0 if E_j.size == 0 else float(_simpson_integrate(y_j, E_j)))

    if integ["kshell_E"] is not None:
        E_K = integ["kshell_E"]
        y_K = integ["kshell_int"]
        kshell_sigma_pwba = 0.0 if E_K.size == 0 else float(_simpson_integrate(y_K, E_K))

    valence_sigma_pwba = float(np.sum(exc_sigma_pwba) + np.sum(ion_sigma_pwba))
    total_sigma_pwba = (
        valence_sigma_pwba + kshell_sigma_pwba if kshell_sigma_pwba is not None else valence_sigma_pwba
    )

    # -------------------- Relativistic components --------------------
    exc_sigma_rel = []
    exc_sigma_rel_trans = []
    ion_sigma_rel = []
    ion_sigma_rel_trans = []
    kshell_sigma_rel = None

    for k in range(len(s.excitations)):
        E_k = integ["excitation_E"][k]
        y_k_rel = integ["excitation_int_rel"][k]
        y_k_trans = integ["excitation_int_rel_trans"][k]
        exc_sigma_rel.append(0.0 if E_k.size == 0 else float(_simpson_integrate(y_k_rel, E_k)))
        exc_sigma_rel_trans.append(0.0 if E_k.size == 0 else float(_simpson_integrate(y_k_trans, E_k)))

    for j in range(len(s.ionizations)):
        E_j = integ["ionization_E"][j]
        y_j_rel = integ["ionization_int_rel"][j]
        y_j_trans = integ["ionization_int_rel_trans"][j]
        ion_sigma_rel.append(0.0 if E_j.size == 0 else float(_simpson_integrate(y_j_rel, E_j)))
        ion_sigma_rel_trans.append(0.0 if E_j.size == 0 else float(_simpson_integrate(y_j_trans, E_j)))

    if integ["kshell_E"] is not None:
        E_K = integ["kshell_E"]
        y_K_rel = integ.get("kshell_int_rel", None)
        if (y_K_rel is None) or (E_K.size == 0):
            kshell_sigma_rel = 0.0
        else:
            kshell_sigma_rel = float(_simpson_integrate(y_K_rel, E_K))

    valence_sigma_rel = float(np.sum(exc_sigma_rel) + np.sum(ion_sigma_rel))
    valence_sigma_rel_trans = float(np.sum(exc_sigma_rel_trans) + np.sum(ion_sigma_rel_trans))
    if kshell_sigma_rel is not None:
        total_sigma_rel = valence_sigma_rel + kshell_sigma_rel
    else:
        total_sigma_rel = valence_sigma_rel
    total_sigma_rel_trans = valence_sigma_rel_trans
    total_sigma_rel_total = total_sigma_rel + total_sigma_rel_trans
    valence_sigma_rel_trans_no_density = None
    total_sigma_rel_trans_no_density = None
    if use_rel_trans and use_density_effect:
        valence_sigma_rel_trans_no_density = _total_transverse_sigma(
            s, C, T, NE=NE, use_density_effect=False
        )
        total_sigma_rel_trans_no_density = valence_sigma_rel_trans_no_density

    # -------------------- Mott-Coulomb --------------------
    exc_sigma_mc = None
    ion_sigma_mc = None
    valence_sigma_mc = None
    total_sigma_mc = None

    if use_mc:
        use_rel_in_mc = use_rel_long
        exc_sigma_mc = []
        for k in range(len(s.excitations)):
            Bk = float(s.excitations[k].Bth)
            Tshift = float(T + 2.0 * Bk)
            exc_sigma_mc.append(
                _sigma_pwba_excitation_shifted_T(
                    s, C, Tshift, T, k, NE=NE, Nq=Nq, use_rel=use_rel_in_mc
                )
            )

        ion_sigma_mc = []
        for j in range(len(s.ionizations)):
            ion_sigma_mc.append(
                _sigma_mc_ionization(s, C, T, j, NE=NE, Nq=Nq, use_rel=use_rel_in_mc)
            )

        valence_sigma_mc = float(np.sum(exc_sigma_mc) + np.sum(ion_sigma_mc))
        kshell_for_mc = (
            kshell_sigma_rel if (use_rel_long and kshell_sigma_rel is not None) else kshell_sigma_pwba
        )
        total_sigma_mc = valence_sigma_mc + (kshell_for_mc if kshell_for_mc is not None else 0.0)

    # -------------------- Default output selection --------------------
    exc_sigma = list(exc_sigma_pwba)
    ion_sigma = list(ion_sigma_pwba)
    kshell_sigma = kshell_sigma_pwba
    valence_sigma = float(valence_sigma_pwba)
    total_sigma = float(total_sigma_pwba)

    if use_mc and (exc_sigma_mc is not None) and (ion_sigma_mc is not None):
        exc_sigma = list(exc_sigma_mc)
        ion_sigma = list(ion_sigma_mc)
        kshell_sigma = (
            kshell_sigma_rel if (use_rel_long and kshell_sigma_rel is not None) else kshell_sigma_pwba
        )
        valence_sigma = float(np.sum(exc_sigma) + np.sum(ion_sigma))
        total_sigma = valence_sigma + (kshell_sigma if kshell_sigma is not None else 0.0)
    elif use_rel_long:
        if use_rel_trans:
            exc_sigma = [a + b for a, b in zip(exc_sigma_rel, exc_sigma_rel_trans)]
            ion_sigma = [a + b for a, b in zip(ion_sigma_rel, ion_sigma_rel_trans)]
            valence_sigma = float(valence_sigma_rel + valence_sigma_rel_trans)
        else:
            exc_sigma = list(exc_sigma_rel)
            ion_sigma = list(ion_sigma_rel)
            valence_sigma = float(valence_sigma_rel)
        kshell_sigma = kshell_sigma_rel if kshell_sigma_rel is not None else kshell_sigma_pwba
        total_sigma = valence_sigma + (kshell_sigma if kshell_sigma is not None else 0.0)

    # Optional convenience: combined
    total_sigma_plus_rel = total_sigma_pwba + total_sigma_rel
    total_sigma_plus_rel_total = total_sigma_pwba + total_sigma_rel_total

    return {
        # PWBA Baseline
        "excitation_sigma_pwba": exc_sigma_pwba,
        "ionization_sigma_pwba": ion_sigma_pwba,
        "valence_sigma_pwba": valence_sigma_pwba,
        "total_sigma_pwba": total_sigma_pwba,

        "excitation_sigma": exc_sigma,
        "ionization_sigma": ion_sigma,
        "kshell_sigma": kshell_sigma,
        "valence_sigma": valence_sigma,
        "total_sigma": total_sigma,

        # Low-energy Mott–Coulomb
        "excitation_sigma_mc": exc_sigma_mc,
        "ionization_sigma_mc": ion_sigma_mc,
        "valence_sigma_mc": valence_sigma_mc,
        "total_sigma_mc": total_sigma_mc,

        # Longitudinal relativistic outputs
        "excitation_sigma_rel": exc_sigma_rel,
        "excitation_sigma_rel_trans": exc_sigma_rel_trans,

        "ionization_sigma_rel": ion_sigma_rel,
        "ionization_sigma_rel_trans": ion_sigma_rel_trans,

        "kshell_sigma_rel": kshell_sigma_rel,
        "valence_sigma_rel": valence_sigma_rel,
        "valence_sigma_rel_trans": valence_sigma_rel_trans,
        "valence_sigma_rel_trans_no_density": valence_sigma_rel_trans_no_density,

        "total_sigma_rel": total_sigma_rel,
        "total_sigma_rel_trans": total_sigma_rel_trans,
        "total_sigma_rel_trans_no_density": total_sigma_rel_trans_no_density,
        "total_sigma_rel_total": total_sigma_rel_total,

        "total_sigma_plus_rel": total_sigma_plus_rel,
        "total_sigma_plus_rel_total": total_sigma_plus_rel_total,
    }

# ----------------------------------------------------------------------
# Plotting: full cross sections per channel vs T
# ----------------------------------------------------------------------
def plot_full_cross_sections_per_channel(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """
    Plot a *comparison* of PWBA vs the default model (per channel)

    Baseline (PWBA):
        uses keys '*_sigma_pwba' computed by the integrator.

    Default model uses the regime flags from _regime_flags(T).
    """

    # Allow passing (ax_ion, ax_exc) or None
    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        ax_ion, ax_exc = ax

    T_arr = np.asarray(T_list, dtype=float)
    nT = len(T_arr)

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#7f7f7f"]

    def _get_list(i, key, default):
        return sigma_list[i].get(key, default)

    # ----------------- Ionizations: PWBA vs Default model -----------------
    n_ion = len(s.ionizations)
    for j in range(n_ion):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])

            pwba_list = _get_list(i, "ionization_sigma_pwba", [0.0] * n_ion)
            pwba = pwba_list[j] if j < len(pwba_list) else 0.0

            use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
            if use_mc:
                mc_list = _get_list(i, "ionization_sigma_mc", None)
                model = (mc_list[j] if (mc_list is not None and j < len(mc_list)) else pwba)
            elif use_rel_long:
                relL_list = _get_list(i, "ionization_sigma_rel", [0.0] * n_ion)
                relT_list = _get_list(i, "ionization_sigma_rel_trans", [0.0] * n_ion)
                relL = relL_list[j] if j < len(relL_list) else 0.0
                relT = relT_list[j] if j < len(relT_list) else 0.0
                model = relL + (relT if use_rel_trans else 0.0)
            else:
                model = pwba

            y_pwba.append(pwba)
            y_model.append(model)

        ax_ion.loglog(
            T_arr, y_pwba,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            ls="--",
            label=f"Ion {j+1} PWBA",
        )
        ax_ion.loglog(
            T_arr, y_model,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Ion {j+1} Default model",
        )

    ax_ion.set_xlabel("Incident energy T (eV)")
    ax_ion.set_ylabel("Cross section sigma(T)")
    ax_ion.set_title("Ionizations: PWBA baseline vs Default model")
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)
    ax_ion.legend(loc="best", fontsize=8)

    # ----------------- Excitations: PWBA vs Default model -----------------
    n_exc = len(s.excitations)
    for k in range(n_exc):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])

            pwba_list = _get_list(i, "excitation_sigma_pwba", [0.0] * n_exc)
            pwba = pwba_list[k] if k < len(pwba_list) else 0.0

            use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
            if use_mc:
                mc_list = _get_list(i, "excitation_sigma_mc", None)
                model = (mc_list[k] if (mc_list is not None and k < len(mc_list)) else pwba)
            elif use_rel_long:
                relL_list = _get_list(i, "excitation_sigma_rel", [0.0] * n_exc)
                relT_list = _get_list(i, "excitation_sigma_rel_trans", [0.0] * n_exc)
                relL = relL_list[k] if k < len(relL_list) else 0.0
                relT = relT_list[k] if k < len(relT_list) else 0.0
                model = relL + (relT if use_rel_trans else 0.0)
            else:
                model = pwba

            y_pwba.append(pwba)
            y_model.append(model)

        ax_exc.loglog(
            T_arr, y_pwba,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            ls="--",
            label=f"Exc {k+1} PWBA",
        )
        ax_exc.loglog(
            T_arr, y_model,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Exc {k+1} Default model",
        )

    ax_exc.set_xlabel("Incident energy T (eV)")
    ax_exc.set_ylabel("Cross section sigma(T)")
    ax_exc.set_title("Excitations: PWBA baseline vs Default model")
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)
    ax_exc.legend(loc="best", fontsize=8)

    return (ax_ion, ax_exc)


def plot_relativistic_component_per_channel(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8),
        include_kshell=False, include_totals=False):
    """ Make two plots:
      (1) Ionizations: longitudinal (solid) and transverse (dashed)
      (2) Excitations: longitudinal (solid) and transverse (dashed)

    Uses keys:
      - ionization_sigma_rel (longitudinal)
      - ionization_sigma_rel_trans (transverse)
      - excitation_sigma_rel (longitudinal)
      - excitation_sigma_rel_trans (transverse)

    Plots only for T >= REGIME_II_MAX_eV by default.
    """

    T_arr = np.asarray(T_list, dtype=float)
    mask = T_arr >= REGIME_II_MAX_eV
    if not np.any(mask):
        raise ValueError("No T values >= REGIME_II_MAX_eV found in T_list.")

    Tm = T_arr[mask]
    idxs = np.where(mask)[0]

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#7f7f7f"]

    n_exc = len(s.excitations)
    n_ion = len(s.ionizations)

    # Ax handling: allow ax=None or ax=(ax_ion, ax_exc)
    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        if not (isinstance(ax, (tuple, list)) and len(ax) == 2):
            raise ValueError("ax must be None or a (ax_ion, ax_exc) tuple/list")
        ax_ion, ax_exc = ax

    # ----------------- Ionizations -----------------
    print(f"Plot IONIZATIONS (REL longitudinal solid, transverse dashed; T>={REGIME_II_MAX_eV:.1e} eV)")
    for j in tqdm(range(n_ion)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("ionization_sigma_rel", [0.0] * n_ion)
            trans_list = sigma_list[i].get("ionization_sigma_rel_trans", [0.0] * n_ion)
            y_long.append(long_list[j] if j < len(long_list) else 0.0)
            y_trans.append(trans_list[j] if j < len(trans_list) else 0.0)

        ax_ion.loglog(
            Tm, y_long,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            label=f"Ion {j+1} Long",
        )
        ax_ion.loglog(
            Tm, y_trans,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Ion {j+1} Trans",
        )

    ax_ion.set_xlabel("Incident energy T (eV)")
    ax_ion.set_ylabel("Relativistic component sigma_rel(T)")
    ax_ion.legend(loc="best", fontsize=8)
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)

    # ----------------- Excitations -----------------
    print(f"Plot EXCITATIONS (REL longitudinal solid, transverse dashed; T>={REGIME_II_MAX_eV:.1e} eV)")
    for k in tqdm(range(n_exc)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("excitation_sigma_rel", [0.0] * n_exc)
            trans_list = sigma_list[i].get("excitation_sigma_rel_trans", [0.0] * n_exc)
            y_long.append(long_list[k] if k < len(long_list) else 0.0)
            y_trans.append(trans_list[k] if k < len(trans_list) else 0.0)

        ax_exc.loglog(
            Tm, y_long,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            label=f"Exc {k+1} Long",
        )
        ax_exc.loglog(
            Tm, y_trans,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Exc {k+1} Trans",
        )

    ax_exc.set_xlabel("Incident energy T (eV)")
    ax_exc.set_ylabel("Relativistic component sigma_rel(T)")
    ax_exc.legend(loc="best", fontsize=8)
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)

    return (ax_ion, ax_exc)

def plot_total_cross_section(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """
    Plot TOTAL cross section with *all* corrections (as available in sigma_list).

    The corrected total is chosen per T as:
      * if regime selects MC: use total_sigma_mc when present
      * elif regime selects rel: use total_sigma_rel_total (long+trans) or total_sigma_rel
      * else fall back to PWBA
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)

    y_pwba = []
    y_corr = []

    for i, Tj in enumerate(T_arr):
        pwba = float(sigma_list[i].get("total_sigma_pwba", sigma_list[i].get("total_sigma", 0.0)) or 0.0)

        use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
        if use_mc:
            corr = float(sigma_list[i].get("total_sigma_mc", pwba) or pwba)
        elif use_rel_long:
            if use_rel_trans:
                corr = float(sigma_list[i].get("total_sigma_rel_total", pwba) or pwba)
            else:
                corr = float(sigma_list[i].get("total_sigma_rel", pwba) or pwba)
        else:
            corr = pwba

        y_pwba.append(pwba)
        y_corr.append(corr)

    ax.loglog(T_arr, y_pwba, lw=linewidth, alpha=alpha, ls=":", label="Total PWBA")
    ax.loglog(T_arr, y_corr, lw=linewidth+1, alpha=alpha, ls="-", label="Total (all corrections)")

    ax.set_xlabel("Incident energy T (eV)")
    ax.set_ylabel("Total cross section sigma(T)")
    ax.set_title("Total cross section: PWBA vs all corrections")
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return ax

def plot_corrected_exc_ion_scaled(
        T_list, sigma_list,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8), scale=1e-22):
    """
    Plot corrected excitation and ionization channels (dashed),
    scaled by the provided factor.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)
    T_keV = T_arr * 1e-3
    scale_inv = 1.0 / scale if scale != 0.0 else 0.0
    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#7f7f7f"]

    n_exc = len(sigma_list[0].get("excitation_sigma_pwba", [])) if sigma_list else 0
    n_ion = len(sigma_list[0].get("ionization_sigma_pwba", [])) if sigma_list else 0

    exc_scaled = np.zeros((len(T_arr), n_exc), float)
    ion_scaled = np.zeros((len(T_arr), n_ion), float)

    for i, Tj in enumerate(T_arr):
        sigma = sigma_list[i]
        use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
        if sigma.get("total_sigma_mc", None) is None:
            use_mc = False

        if use_mc:
            exc_vals = sigma.get("excitation_sigma_mc", []) or []
            ion_vals = sigma.get("ionization_sigma_mc", []) or []
        elif use_rel_long:
            exc_vals = sigma.get("excitation_sigma_rel", []) or []
            ion_vals = sigma.get("ionization_sigma_rel", []) or []
            if use_rel_trans:
                exc_vals = [a + b for a, b in zip(exc_vals, (sigma.get("excitation_sigma_rel_trans", []) or []))]
                ion_vals = [a + b for a, b in zip(ion_vals, (sigma.get("ionization_sigma_rel_trans", []) or []))]
        else:
            exc_vals = sigma.get("excitation_sigma_pwba", []) or []
            ion_vals = sigma.get("ionization_sigma_pwba", []) or []

        for j in range(min(n_exc, len(exc_vals))):
            exc_scaled[i, j] = float(exc_vals[j]) / scale
        for j in range(min(n_ion, len(ion_vals))):
            ion_scaled[i, j] = float(ion_vals[j]) / scale

    for j in range(n_exc):
        ax.loglog(
            T_keV,
            exc_scaled[:, j],
            lw=linewidth,
            alpha=alpha,
            ls="--",
            color=exc_colors[j % len(exc_colors)],
            label=f"Exc {j+1} corrected",
        )
    for j in range(n_ion):
        ax.loglog(
            T_keV,
            ion_scaled[:, j],
            lw=linewidth,
            alpha=alpha,
            ls="--",
            color=ion_colors[j % len(ion_colors)],
            label=f"Ion {j+1} corrected",
        )
    ax.set_xlabel("Incident energy T (keV)")
    ax.set_ylabel(f"Cross section × {scale_inv:.0e}")
    ax.set_title("Corrected excitation and ionization channels (scaled)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return ax

# ----------------------------------------------------------------------
# Logging: per-energy corrections
# ----------------------------------------------------------------------
def _compute_correction_row(T, sigma):
    Tj = float(T)
    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(Tj)
    if sigma.get("total_sigma_mc", None) is None:
        use_mc = False

    pwba = float(sigma.get("total_sigma_pwba", 0.0) or 0.0)
    corrected = float(sigma.get("total_sigma", 0.0) or 0.0)

    total_mc = sigma.get("total_sigma_mc", None)
    corr_mc = float(total_mc) - pwba if total_mc is not None else 0.0
    corr_rel_long = float(sigma.get("total_sigma_rel", 0.0) or 0.0)

    corr_density = 0.0
    trans_no_density = sigma.get("total_sigma_rel_trans_no_density", None)
    if use_density_effect and trans_no_density is not None:
        corr_rel_trans = float(trans_no_density)
        corr_density = float(sigma.get("total_sigma_rel_trans", 0.0) or 0.0) - corr_rel_trans
    else:
        corr_rel_trans = float(sigma.get("total_sigma_rel_trans", 0.0) or 0.0)

    return {
        "T_eV": Tj,
        "sigma_pwba": pwba,
        "corr_stage1_mc": corr_mc,
        "corr_stage2_rel_long": corr_rel_long,
        "corr_stage3_rel_trans": corr_rel_trans,
        "corr_stage4_density": corr_density,
        "sigma_corrected": corrected,
        "use_mc": int(use_mc),
        "use_rel_long": int(use_rel_long),
        "use_rel_trans": int(use_rel_trans),
        "use_density_effect": int(use_density_effect),
    }

def save_cross_section_corrections_npz(T_list, sigma_list, out_path=None):
    """
    Save PWBA, per-stage correction terms, corrected totals, and per-channel
    cross sections for each energy to an NPZ file.
    """
    if out_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "cross_section_corrections.npz"

    T_arr = np.asarray(T_list, dtype=float)
    nT = len(T_arr)
    n_exc = len(sigma_list[0].get("excitation_sigma_pwba", [])) if nT else 0
    n_ion = len(sigma_list[0].get("ionization_sigma_pwba", [])) if nT else 0

    def _stack_channel(key, n_chan):
        arr = np.zeros((nT, n_chan), float)
        for i, sigma in enumerate(sigma_list):
            vals = sigma.get(key, None)
            if vals is None:
                continue
            for j in range(min(n_chan, len(vals))):
                arr[i, j] = float(vals[j])
        return arr

    rows = [_compute_correction_row(T, sigma) for T, sigma in zip(T_list, sigma_list)]
    pwba_total = np.array([row["sigma_pwba"] for row in rows], float)
    corr_mc = np.array([row["corr_stage1_mc"] for row in rows], float)
    corr_rel_long = np.array([row["corr_stage2_rel_long"] for row in rows], float)
    corr_rel_trans = np.array([row["corr_stage3_rel_trans"] for row in rows], float)
    corr_density = np.array([row["corr_stage4_density"] for row in rows], float)
    corrected_total = np.array([row["sigma_corrected"] for row in rows], float)
    use_mc = np.array([row["use_mc"] for row in rows], int)
    use_rel_long = np.array([row["use_rel_long"] for row in rows], int)
    use_rel_trans = np.array([row["use_rel_trans"] for row in rows], int)
    use_density_effect = np.array([row["use_density_effect"] for row in rows], int)

    exc_pwba = _stack_channel("excitation_sigma_pwba", n_exc)
    ion_pwba = _stack_channel("ionization_sigma_pwba", n_ion)
    exc_rel_long = _stack_channel("excitation_sigma_rel", n_exc)
    ion_rel_long = _stack_channel("ionization_sigma_rel", n_ion)
    exc_rel_trans = _stack_channel("excitation_sigma_rel_trans", n_exc)
    ion_rel_trans = _stack_channel("ionization_sigma_rel_trans", n_ion)
    exc_mc = _stack_channel("excitation_sigma_mc", n_exc)
    ion_mc = _stack_channel("ionization_sigma_mc", n_ion)

    exc_selected = np.zeros((nT, n_exc), float)
    ion_selected = np.zeros((nT, n_ion), float)
    for i, (Tj, sigma) in enumerate(zip(T_arr, sigma_list)):
        use_mc_i, use_rel_long_i, use_rel_trans_i, _ = _regime_flags(Tj)
        if sigma.get("total_sigma_mc", None) is None:
            use_mc_i = False
        if use_mc_i:
            exc_selected[i, :] = exc_mc[i, :]
            ion_selected[i, :] = ion_mc[i, :]
        elif use_rel_long_i:
            if use_rel_trans_i:
                exc_selected[i, :] = exc_rel_long[i, :] + exc_rel_trans[i, :]
                ion_selected[i, :] = ion_rel_long[i, :] + ion_rel_trans[i, :]
            else:
                exc_selected[i, :] = exc_rel_long[i, :]
                ion_selected[i, :] = ion_rel_long[i, :]
        else:
            exc_selected[i, :] = exc_pwba[i, :]
            ion_selected[i, :] = ion_pwba[i, :]

    np.savez(
        out_path,
        T_eV=T_arr,
        total_sigma_pwba=pwba_total,
        total_sigma_corrected=corrected_total,
        corr_stage1_mc=corr_mc,
        corr_stage2_rel_long=corr_rel_long,
        corr_stage3_rel_trans=corr_rel_trans,
        corr_stage4_density=corr_density,
        use_mc=use_mc,
        use_rel_long=use_rel_long,
        use_rel_trans=use_rel_trans,
        use_density_effect=use_density_effect,
        excitation_sigma_pwba=exc_pwba,
        ionization_sigma_pwba=ion_pwba,
        excitation_sigma_rel_long=exc_rel_long,
        ionization_sigma_rel_long=ion_rel_long,
        excitation_sigma_rel_trans=exc_rel_trans,
        ionization_sigma_rel_trans=ion_rel_trans,
        excitation_sigma_mc=exc_mc,
        ionization_sigma_mc=ion_mc,
        excitation_sigma_selected=exc_selected,
        ionization_sigma_selected=ion_selected,
    )

    print(f"Saved correction log to {out_path}")

def plot_total_cross_section_corrections(T_list, sigma_list, out_path=None):
    """
    Plot PWBA vs corrected totals and per-stage correction terms vs energy.
    """
    if out_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "cross_section_corrections.png"

    rows = [_compute_correction_row(T, sigma) for T, sigma in zip(T_list, sigma_list)]
    T_arr = np.array([row["T_eV"] for row in rows], float)
    pwba = np.array([row["sigma_pwba"] for row in rows], float)
    corrected = np.array([row["sigma_corrected"] for row in rows], float)
    corr_mc = np.array([row["corr_stage1_mc"] for row in rows], float)
    corr_rel_long = np.array([row["corr_stage2_rel_long"] for row in rows], float)
    corr_rel_trans = np.array([row["corr_stage3_rel_trans"] for row in rows], float)
    corr_density = np.array([row["corr_stage4_density"] for row in rows], float)

    abs_vals = np.abs(
        np.concatenate([corr_mc, corr_rel_long, corr_rel_trans, corr_density])
    )
    abs_vals = abs_vals[abs_vals > 0.0]
    linthresh = float(np.min(abs_vals)) if abs_vals.size else 1e-40

    fig, (ax_tot, ax_corr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax_tot.loglog(T_arr, pwba, lw=2, ls=":", label="PWBA total")
    ax_tot.loglog(T_arr, corrected, lw=2, ls="-", label="Corrected total")
    ax_tot.set_ylabel("Total cross section sigma(T)")
    ax_tot.grid(True, which="both", ls="--", alpha=0.3)
    ax_tot.legend(loc="best", fontsize=9)

    ax_corr.set_xscale("log")
    ax_corr.set_yscale("symlog", linthresh=linthresh)
    ax_corr.plot(T_arr, corr_mc, lw=1.8, label="Stage 1: MC")
    ax_corr.plot(T_arr, corr_rel_long, lw=1.8, label="Stage 2: Rel long")
    ax_corr.plot(T_arr, corr_rel_trans, lw=1.8, label="Stage 3: Rel trans")
    ax_corr.plot(T_arr, corr_density, lw=1.8, label="Stage 4: Density effect")
    ax_corr.set_xlabel("Incident energy T (eV)")
    ax_corr.set_ylabel("Correction term")
    ax_corr.grid(True, which="both", ls="--", alpha=0.3)
    ax_corr.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved correction diagnostics to {out_path}")

# ----------------------------------------------------------------------
# Workers for parallel computing
# ----------------------------------------------------------------------
# Globals set once per worker process
_WORKER_S = None
_WORKER_C = None
_WORKER_KW = None

def _init_worker(
    a_vec,
    b_vec,
    c_vec,
    NE,
    Nq,
    include_kshell,
    use_mott_coulomb,
    apply_regime_ii,
    apply_regime_iii,
    apply_regime_iv,
    apply_mc,
):
    """
    Runs once inside each worker process. Builds s and C once to avoid repeated pickling.
    """
    global _WORKER_S, _WORKER_C, _WORKER_KW

    import emfietzoglou_model_finite_q as model
    _set_regime_corrections(
        apply_regime_ii=apply_regime_ii,
        apply_regime_iii=apply_regime_iii,
        apply_regime_iv=apply_regime_iv,
    )
    _set_mc_correction(apply_mc=apply_mc)
    _WORKER_S = model.epsilon_optical('amorphous')
    _WORKER_C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)
    _WORKER_KW = dict(
        NE=NE,
        Nq=Nq,
        include_kshell=include_kshell,
        use_mott_coulomb=use_mott_coulomb,
    )

def _compute_for_T(T):
    """
    Compute sigma dict for one T.
    Returns (T, sigma_dict)
    """
    sigma = integrate_elf_double_integral(_WORKER_S, _WORKER_C, float(T), **_WORKER_KW)
    return float(T), sigma

def main():
    # Optical model / dispersion coefficients
    s = model.epsilon_optical('amorphous')
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])

    # RR2017 defaults for c_disp, d_disp, b1, b2
    C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)

    # Energy grid (eV)
    T_list = np.logspace(-1, 7, 400)

    print("Computing double-integrated cross sections (parallel over T)...")

    # Computing Choices
    NE = 200
    Nq = 200
    include_kshell = True
    use_mott_coulomb = True
    apply_mc = True
    apply_regime_ii = True
    apply_regime_iii = True
    apply_regime_iv = True

    _set_regime_corrections(
        apply_regime_ii=apply_regime_ii,
        apply_regime_iii=apply_regime_iii,
        apply_regime_iv=apply_regime_iv,
    )
    _set_mc_correction(apply_mc=apply_mc)
    # Use ~ (CPU cores) workers;
    max_workers = 12
    print("Using %i workers" %(max_workers))

    results_by_T = {}

    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(
                a_vec,
                b_vec,
                c_vec,
                NE,
                Nq,
                include_kshell,
                use_mott_coulomb,
                apply_regime_ii,
                apply_regime_iii,
                apply_regime_iv,
                apply_mc,
            ),
    ) as ex:
        futures = [ex.submit(_compute_for_T, T) for T in T_list]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            T_val, sigma = fut.result()
            results_by_T[T_val] = sigma

    # Restore original T order
    sigma_list = [results_by_T[float(T)] for T in T_list]

    # ----------------- Log corrections per energy -----------------
    save_cross_section_corrections_npz(T_list, sigma_list)
    plot_total_cross_section_corrections(T_list, sigma_list)

    # ----------------- Plot 0: corrected excitation/ionization (scaled) -----------------
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_corrected_exc_ion_scaled(T_list, sigma_list, ax=ax, scale=1e-22)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "corrected_excitation_ionization_scaled.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved corrected excitation/ionization plot to {out_path}")

    # ----------------- Plot 1: comparison per channel (PWBA vs selected corrections) -----------------
    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    ax_ion, ax_exc = plot_full_cross_sections_per_channel(
        T_list, sigma_list, s
    )

    ax_ion.figure.tight_layout()
    ax_ion.figure.savefig("comparison_ionizations_pwba_vs_model.png", dpi=300)
    plt.close(ax_ion.figure)

    ax_exc.figure.tight_layout()
    ax_exc.figure.savefig("comparison_excitations_pwba_vs_model.png", dpi=300)
    plt.close(ax_exc.figure)
    print("Saved comparison plots to comparison_ionizations_pwba_vs_model.png and comparison_excitations_pwba_vs_model.png")


    # ----------------- Plot 2: REL components per channel (L solid, T dashed) -----------------
    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    plot_relativistic_component_per_channel(T_list, sigma_list, s, ax=(ax_ion, ax_exc))
    fig_ion.tight_layout()
    fig_exc.tight_layout()
    fig_ion.savefig("rel_cross_sections_ionizations_LT.png", dpi=300)
    fig_exc.savefig("rel_cross_sections_excitations_LT.png", dpi=300)
    plt.close(fig_ion)
    plt.close(fig_exc)
    print("Saved REL component plots to rel_cross_sections_ionizations_LT.png and rel_cross_sections_excitations_LT.png")

    # ----------------- Plot 3: TOTAL cross section with all corrections -----------------
    fig, ax = plt.subplots()
    plot_total_cross_section(T_list, sigma_list, s, ax=ax)
    fig.tight_layout()
    fig.savefig("total_cross_section_all_corrections.png", dpi=300)
    plt.close(fig)
    print("Saved total plot to total_cross_section_all_corrections.png")

if __name__ == "__main__":
    main()
