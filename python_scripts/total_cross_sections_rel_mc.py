#!/usr/bin/env python3

import os, sys
import shutil
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
    CUSTOM_DATA_ROOT_GEANT4,
    CUSTOM_DATA_ROOT_PROJECT,
    CROSS_SECTIONS_DIR,
    EH,
    EV_TO_HA,
    ELF_ROLLOFF_COEF,
    ELF_ROLLOFF_E0_eV,
    FONT_COURIER,
    FONTSIZE_24,
    MC2_HA,
    MC2_eV,
    N,
    N_REFERENCE_DENSITY_G_CM3,
    ICE_AMORPHOUS_DENSITY_G_CM3,
    ICE_HEXAGONAL_DENSITY_G_CM3,
    OUTPUT_DIR,
    RC_BASE_ELASTIC,
    REGIME_I_MAX_eV,
    REGIME_II_MAX_eV,
    REGIME_III_MAX_eV,
    REGIME_IV_MAX_eV,
    a0,
    mass,
    rcparams_with_fontsize,
)
import emfietzoglou_model_finite_q as model

# Select ice structure: "amorphous" or "hexagonal"
ICE_TYPE = "hexagonal"
ICE_LABEL = f"{ICE_TYPE}_ice"

# Density used for rel_mc cross-section scaling in this script.
# Set to 1.0 to plot/reference cross sections at rho = 1 g/cm^3.
# Set to None to use phase-specific nominal densities.
REL_MC_TARGET_DENSITY_G_CM3 = 1.0

# Extend DCS grid beyond Born table using a linear T grid.
DCS_T_MAX_EEV = 1.0e7
DCS_T_STEP_EEV = 2.0e5

# Geant4 Emfietzoglou DCS table scale: file values * scale -> m^2
EMFI_DCS_SCALE_M2 = 1.0e-22 / 3.343

# ----------------------------------------------------------------------
# Constants for integration (from constants.py)
# ----------------------------------------------------------------------
def _normalize_ice_type(name):
    if not name:
        return None
    key = str(name).strip().lower()
    if "amorph" in key:
        return "amorphous"
    if "hex" in key:
        return "hexagonal"
    return None

def _assumed_density_g_cm3(ice_type):
    norm = _normalize_ice_type(ice_type)
    if norm == "amorphous":
        return float(ICE_AMORPHOUS_DENSITY_G_CM3)
    if norm == "hexagonal":
        return float(ICE_HEXAGONAL_DENSITY_G_CM3)
    raise ValueError(f"Unsupported ice type for density scaling: {ice_type}")

def _target_density_g_cm3(ice_type):
    norm = _normalize_ice_type(ice_type)
    if norm is None:
        raise ValueError(f"Unsupported ice type for density scaling: {ice_type}")
    if REL_MC_TARGET_DENSITY_G_CM3 is None:
        return _assumed_density_g_cm3(norm)
    return float(REL_MC_TARGET_DENSITY_G_CM3)

def _density_scale_factor_for_ice(ice_type):
    rho_ref = float(N_REFERENCE_DENSITY_G_CM3)
    if rho_ref <= 0.0:
        return 1.0
    return _target_density_g_cm3(ice_type) / rho_ref

def _infer_ice_type_from_path(path_like):
    text = str(path_like).lower()
    if "amorph" in text:
        return "amorphous"
    if "hex" in text:
        return "hexagonal"
    return None

def _density_scale_factor_from_npz(npz_data):
    if "density_scale_factor" not in npz_data:
        return 1.0
    try:
        val = float(np.asarray(npz_data["density_scale_factor"]).reshape(-1)[0])
    except Exception:
        return 1.0
    if not np.isfinite(val) or val <= 0.0:
        return 1.0
    return val

def _scale_sigma_entry(sigma, factor):
    if (not np.isfinite(factor)) or factor <= 0.0 or np.isclose(factor, 1.0):
        return dict(sigma)
    out = {}
    for key, val in sigma.items():
        if "sigma" not in key or val is None:
            out[key] = val
            continue
        if np.isscalar(val):
            out[key] = float(val) * factor
        elif isinstance(val, np.ndarray):
            out[key] = np.asarray(val, float) * factor
        else:
            out[key] = [float(x) * factor for x in val]
    return out

def _scale_sigma_list(sigma_list, factor):
    if (not np.isfinite(factor)) or factor <= 0.0 or np.isclose(factor, 1.0):
        return [dict(s) for s in sigma_list]
    return [_scale_sigma_entry(sigma, factor) for sigma in sigma_list]

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

    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

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
            label=f"Ion. {j+1} PWBA",
        )
        ax_ion.loglog(
            T_arr, y_model,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Ion. {j+1} Default model",
        )

    ax_ion.set_xlabel("Electron Energy (T; eV)")
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
            label=f"Exc. {k+1} PWBA",
        )
        ax_exc.loglog(
            T_arr, y_model,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Exc. {k+1} Default model",
        )

    ax_exc.set_xlabel("Electron Energy (T; eV)")
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

    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

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
            label=f"Ion. {j+1} Long",
        )
        ax_ion.loglog(
            Tm, y_trans,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Ion. {j+1} Trans",
        )

    ax_ion.set_xlabel("Electron energy (T; eV)")
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
            label=f"Exc. {k+1} Long",
        )
        ax_exc.loglog(
            Tm, y_trans,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Exc. {k+1} Trans",
        )

    ax_exc.set_xlabel("Electron energy (T; eV)")
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

    ax.set_xlabel("Electron energy (T; eV)")
    ax.set_ylabel("Total cross section sigma(T)")
    ax.set_title("Total cross section: PWBA vs all corrections")
    ax.legend(loc="best", fontsize=9)
    return ax

def _load_total_sigma_npz(npz_path):
    with np.load(npz_path) as data:
        T = np.asarray(data["T_eV"], float)
        pwba = np.asarray(data.get("total_sigma_pwba", []), float)
        corrected = np.asarray(
            data.get("total_sigma_corrected", data.get("total_sigma", [])), float
        )
        stored = _density_scale_factor_from_npz(data)
        inferred = _infer_ice_type_from_path(npz_path)
        if inferred is not None:
            target = _density_scale_factor_for_ice(inferred)
            adjust = target / stored if stored > 0.0 else target
            if np.isfinite(adjust) and adjust > 0.0 and (not np.isclose(adjust, 1.0)):
                pwba = pwba * adjust
                corrected = corrected * adjust
    return T, pwba, corrected

def plot_total_cross_section_two_panel(
    amorphous_npz,
    hexagonal_npz,
    out_path=None,
):
    """
    Two-panel comparison (amorphous vs hexagonal) with a shared legend row below.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    T_a, pwba_a, corr_a = _load_total_sigma_npz(amorphous_npz)
    T_h, pwba_h, corr_h = _load_total_sigma_npz(hexagonal_npz)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, height_ratios=[3.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    ln1 = ax_a.loglog(T_a, pwba_a, "k:", linewidth=2, label="Total PWBA")[0]
    ln2 = ax_a.loglog(T_a, corr_a, "k-", linewidth=2.5, label="Total (all corrections)")[0]
    ax_a.set_xlabel("Electron energy ($T$; eV)")
    ax_a.set_ylabel("Total cross section sigma(T)")
    ax_a.set_title("Amorphous ice")

    ax_h.loglog(T_h, pwba_h, "k:", linewidth=2, label="Total PWBA")
    ax_h.loglog(T_h, corr_h, "k-", linewidth=2.5, label="Total (all corrections)")
    ax_h.set_xlabel("Electron energy ($T$; eV)")
    ax_h.set_ylabel("")
    ax_h.set_title("Hexagonal ice")

    # Enforce common y-range and y-ticks across both panels.
    y_lo = min(ax_a.get_ylim()[0], ax_h.get_ylim()[0])
    y_hi = max(ax_a.get_ylim()[1], ax_h.get_ylim()[1])
    if not np.isfinite(y_lo) or y_lo <= 0.0:
        y_lo = 1e-30
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = y_lo * 10.0
    ax_a.set_ylim(y_lo, y_hi)
    ax_h.set_ylim(y_lo, y_hi)
    for ax in (ax_a, ax_h):
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(NullLocator())

    legend_ax.legend(
        [ln1, ln2],
        ["Total PWBA", "Total (all corrections)"],
        loc="center",
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
        labelspacing=0.8,
    )

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved two-panel total cross section plot to {out_path}")
    plt.close(fig)

def plot_channel_cross_sections_two_panel(
    amorphous_npz,
    hexagonal_npz,
    out_path=None,
):
    """
    Two-panel comparison (amorphous vs hexagonal) of channel-resolved cross sections
    with a shared legend row below (excitation top, ionization bottom).
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    with np.load(amorphous_npz) as data_a:
        stored_a = _density_scale_factor_from_npz(data_a)
        target_a = _density_scale_factor_for_ice("amorphous")
        adjust_a = target_a / stored_a if stored_a > 0.0 else target_a
        T_a, sigma_a = _sigma_list_from_npz(data_a, scale_factor=adjust_a)

    with np.load(hexagonal_npz) as data_h:
        stored_h = _density_scale_factor_from_npz(data_h)
        target_h = _density_scale_factor_for_ice("hexagonal")
        adjust_h = target_h / stored_h if stored_h > 0.0 else target_h
        T_h, sigma_h = _sigma_list_from_npz(data_h, scale_factor=adjust_h)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, height_ratios=[3.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    plot_corrected_exc_ion_scaled(T_a, sigma_a, ax=ax_a)
    plot_corrected_exc_ion_scaled(T_h, sigma_h, ax=ax_h)
    ax_a.set_title("Amorphous ice")
    ax_h.set_title("Hexagonal ice")
    ax_a.set_xlabel("Electron energy ($T$; eV)")
    ax_h.set_xlabel("Electron energy ($T$; eV)")
    ax_a.set_ylabel(r"Total cross-section (cm$^2$)")
    ax_h.set_ylabel("")

    max_x = max(float(np.max(T_a)), float(np.max(T_h)))
    ax_a.set_xlim(1.0, max_x)
    ax_h.set_xlim(1.0, max_x)
    ax_a.set_ylim(bottom=1e-25)
    ax_h.set_ylim(bottom=1e-25)

    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    handles, labels = ax_a.get_legend_handles_labels()
    n_exc = len(sigma_a[0].get("excitation_sigma_pwba", [])) if sigma_a else 0
    n_ion = len(sigma_a[0].get("ionization_sigma_pwba", [])) if sigma_a else 0
    exc_handles = handles[:n_exc]
    ion_handles = handles[n_exc:n_exc + n_ion]
    kshell_handle = None
    if "K-shell" in labels:
        try:
            kshell_handle = handles[labels.index("K-shell")]
        except Exception:
            kshell_handle = None

    gs_leg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], height_ratios=[1.0, 1.0], hspace=0.30)
    ax_leg_exc = fig.add_subplot(gs_leg[0, 0])
    ax_leg_ion = fig.add_subplot(gs_leg[1, 0])
    ax_leg_exc.axis("off")
    ax_leg_ion.axis("off")

    if exc_handles:
        exc_labels = ["Excitation"] + [str(i + 1) for i in range(n_exc)]
        exc_handles = [Line2D([], [], color="none", linestyle="none")] + exc_handles
        exc_ncol = max(1, len(exc_labels))
        ax_leg_exc.legend(
            exc_handles,
            exc_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=exc_ncol,
            frameon=False,
            columnspacing=0.9,
            handlelength=2.2,
            handletextpad=0.6,
            borderaxespad=0.0,
        )
    if ion_handles:
        ion_labels = ["Ionization"] + [str(i + 1) for i in range(n_ion)]
        ion_handles = [Line2D([], [], color="none", linestyle="none")] + ion_handles
        if kshell_handle is not None:
            ion_labels.append("K-shell")
            ion_handles.append(kshell_handle)
        ion_ncol = max(1, len(ion_labels))
        ax_leg_ion.legend(
            ion_handles,
            ion_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=ion_ncol,
            frameon=False,
            columnspacing=0.9,
            handlelength=2.2,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

    for ax in (ax_a, ax_h):
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(NullLocator())

    # Enforce common y-range and y-ticks across both panels.
    y_lo = min(ax_a.get_ylim()[0], ax_h.get_ylim()[0])
    y_hi = max(ax_a.get_ylim()[1], ax_h.get_ylim()[1])
    if not np.isfinite(y_lo) or y_lo <= 0.0:
        y_lo = 1e-25
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = y_lo * 10.0
    ax_a.set_ylim(y_lo, y_hi)
    ax_h.set_ylim(y_lo, y_hi)
    for ax in (ax_a, ax_h):
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved two-panel channel cross section plot to {out_path}")
    plt.close(fig)

def plot_corrected_exc_ion_scaled(
        T_list, sigma_list,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """
    Plot corrected excitation and ionization channels (dashed),
    scaled by the provided factor.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)
    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

    n_exc = len(sigma_list[0].get("excitation_sigma_pwba", [])) if sigma_list else 0
    n_ion = len(sigma_list[0].get("ionization_sigma_pwba", [])) if sigma_list else 0

    exc_scaled = np.zeros((len(T_arr), n_exc), float)
    ion_scaled = np.zeros((len(T_arr), n_ion), float)
    kshell_scaled = np.full(len(T_arr), np.nan, float)

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
            exc_scaled[i, j] = float(exc_vals[j])
        for j in range(min(n_ion, len(ion_vals))):
            ion_scaled[i, j] = float(ion_vals[j])
        kshell_val = None
        if use_rel_long and sigma.get("kshell_sigma_rel", None) is not None:
            kshell_val = sigma.get("kshell_sigma_rel", None)
        else:
            kshell_val = sigma.get("kshell_sigma", None)
        if kshell_val is not None:
            try:
                kshell_scaled[i] = float(kshell_val)
            except Exception:
                kshell_scaled[i] = np.nan

    for j in range(n_exc):
        ax.loglog(
            T_arr,
            exc_scaled[:, j],
            lw=linewidth,
            alpha=alpha,
            ls="-",
            color=exc_colors[j % len(exc_colors)],
            label=f"{j+1}",
        )
    for j in range(n_ion):
        ax.loglog(
            T_arr,
            ion_scaled[:, j],
            lw=linewidth,
            alpha=alpha,
            ls="-",
            color=ion_colors[j % len(ion_colors)],
            label=f"{j+1}",
        )
    if np.any(np.isfinite(kshell_scaled)):
        ax.loglog(
            T_arr,
            kshell_scaled,
            lw=linewidth,
            alpha=alpha,
            ls="-",
            color="purple",
            label="K-shell",
        )
    ax.set_xlabel("Electron Energy ($T$; eV)", labelpad=1)
    ax.set_ylabel(r"Cross-Section (cm$^2$)")
    ax.set_xlim(1.0, np.max(T_arr))
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())
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

def save_cross_section_corrections_npz(
    T_list,
    sigma_list,
    out_path=None,
    NE=None,
    Nq=None,
    dcs_data=None,
    ice_label=None,
    density_scale_factor=1.0,
    density_ref_g_cm3=None,
    density_assumed_g_cm3=None,
):
    """
    Save PWBA, per-stage correction terms, corrected totals, and per-channel
    cross sections for each energy to an NPZ file.
    """
    if out_path is None:
        if ice_label is None:
            ice_label = ICE_LABEL
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"cross_section_corrections_{ice_label}.npz"

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

    total_sigma = np.array([float(s.get("total_sigma", 0.0) or 0.0) for s in sigma_list], float)
    total_sigma_mc = np.array(
        [float(s.get("total_sigma_mc", np.nan)) if s.get("total_sigma_mc", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_rel = np.array(
        [float(s.get("total_sigma_rel", np.nan)) if s.get("total_sigma_rel", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_rel_trans = np.array(
        [float(s.get("total_sigma_rel_trans", np.nan)) if s.get("total_sigma_rel_trans", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_rel_trans_no_density = np.array(
        [
            float(s.get("total_sigma_rel_trans_no_density", np.nan))
            if s.get("total_sigma_rel_trans_no_density", None) is not None
            else np.nan
            for s in sigma_list
        ],
        float,
    )
    total_sigma_rel_total = np.array(
        [float(s.get("total_sigma_rel_total", np.nan)) if s.get("total_sigma_rel_total", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_plus_rel = np.array(
        [float(s.get("total_sigma_plus_rel", np.nan)) if s.get("total_sigma_plus_rel", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_plus_rel_total = np.array(
        [float(s.get("total_sigma_plus_rel_total", np.nan)) if s.get("total_sigma_plus_rel_total", None) is not None else np.nan for s in sigma_list],
        float,
    )
    kshell_sigma = np.array(
        [float(s.get("kshell_sigma", np.nan)) if s.get("kshell_sigma", None) is not None else np.nan for s in sigma_list],
        float,
    )
    kshell_sigma_rel = np.array(
        [float(s.get("kshell_sigma_rel", np.nan)) if s.get("kshell_sigma_rel", None) is not None else np.nan for s in sigma_list],
        float,
    )

    np_save_args = dict(
        T_eV=T_arr,
        density_scale_factor=float(density_scale_factor),
        total_sigma_pwba=pwba_total,
        total_sigma_corrected=corrected_total,
        total_sigma=total_sigma,
        total_sigma_mc=total_sigma_mc,
        total_sigma_rel=total_sigma_rel,
        total_sigma_rel_trans=total_sigma_rel_trans,
        total_sigma_rel_trans_no_density=total_sigma_rel_trans_no_density,
        total_sigma_rel_total=total_sigma_rel_total,
        total_sigma_plus_rel=total_sigma_plus_rel,
        total_sigma_plus_rel_total=total_sigma_plus_rel_total,
        kshell_sigma=kshell_sigma,
        kshell_sigma_rel=kshell_sigma_rel,
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

    if density_ref_g_cm3 is not None:
        np_save_args["density_ref_g_cm3"] = float(density_ref_g_cm3)
    if density_assumed_g_cm3 is not None:
        np_save_args["density_assumed_g_cm3"] = float(density_assumed_g_cm3)

    if NE is not None:
        np_save_args["NE"] = int(NE)
    if Nq is not None:
        np_save_args["Nq"] = int(Nq)

    if dcs_data is not None:
        np_save_args["dcs_T_line"] = np.asarray(dcs_data.get("T_line", []), float)
        np_save_args["dcs_E_line"] = np.asarray(dcs_data.get("E_line", []), float)
        np_save_args["dcs_exc_vals"] = np.asarray(dcs_data.get("exc_vals", []), float)
        np_save_args["dcs_ion_vals"] = np.asarray(dcs_data.get("ion_vals", []), float)

    np.savez(
        out_path,
        **np_save_args,
    )

    print(f"Saved correction log to {out_path}")

def _npz_int_value(npz_data, key):
    if key not in npz_data:
        return None
    try:
        val = np.asarray(npz_data[key]).reshape(-1)[0]
    except Exception:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def _npz_matches_params(npz_data, NE, Nq, T_list=None):
    ne = _npz_int_value(npz_data, "NE")
    nq = _npz_int_value(npz_data, "Nq")
    if ne is None or nq is None:
        return False
    if int(ne) != int(NE) or int(nq) != int(Nq):
        return False
    if T_list is not None and "T_eV" in npz_data:
        T_arr = np.asarray(npz_data["T_eV"], float)
        if len(T_arr) != len(T_list):
            return False
        if not np.allclose(T_arr, np.asarray(T_list, float)):
            return False
    return True

def _dcs_data_from_npz(npz_data):
    keys = ("dcs_T_line", "dcs_E_line", "dcs_exc_vals", "dcs_ion_vals")
    if not all(key in npz_data for key in keys):
        return None
    T_line = np.asarray(npz_data["dcs_T_line"], float)
    E_line = np.asarray(npz_data["dcs_E_line"], float)
    if T_line.size == 0 or E_line.size == 0:
        return None
    return {
        "T_line": T_line,
        "E_line": E_line,
        "exc_vals": np.asarray(npz_data["dcs_exc_vals"], float),
        "ion_vals": np.asarray(npz_data["dcs_ion_vals"], float),
    }

def _sigma_list_from_npz(npz_data, scale_factor=1.0):
    if "T_eV" not in npz_data:
        raise KeyError("Missing T_eV in NPZ cache.")
    T_arr = np.asarray(npz_data["T_eV"], float)
    nT = len(T_arr)

    def _array(key):
        if key not in npz_data:
            return None
        return np.asarray(npz_data[key], float)

    def _row_vals(arr, i):
        if arr is None:
            return []
        row = np.asarray(arr[i], float)
        if row.ndim == 0:
            return [float(row)]
        return [float(val) for val in row]

    def _scalar(arr, i):
        if arr is None:
            return None
        val = float(arr[i])
        if not np.isfinite(val):
            return None
        return val

    exc_pwba = _array("excitation_sigma_pwba")
    ion_pwba = _array("ionization_sigma_pwba")
    exc_rel_long = _array("excitation_sigma_rel_long")
    ion_rel_long = _array("ionization_sigma_rel_long")
    exc_rel_trans = _array("excitation_sigma_rel_trans")
    ion_rel_trans = _array("ionization_sigma_rel_trans")
    exc_mc = _array("excitation_sigma_mc")
    ion_mc = _array("ionization_sigma_mc")

    total_sigma = _array("total_sigma")
    total_sigma_pwba = _array("total_sigma_pwba")
    total_sigma_mc = _array("total_sigma_mc")
    total_sigma_rel = _array("total_sigma_rel")
    total_sigma_rel_trans = _array("total_sigma_rel_trans")
    total_sigma_rel_trans_no_density = _array("total_sigma_rel_trans_no_density")
    total_sigma_rel_total = _array("total_sigma_rel_total")
    total_sigma_plus_rel = _array("total_sigma_plus_rel")
    total_sigma_plus_rel_total = _array("total_sigma_plus_rel_total")
    kshell_sigma = _array("kshell_sigma")
    kshell_sigma_rel = _array("kshell_sigma_rel")

    sigma_list = []
    for i in range(nT):
        sigma = {
            "excitation_sigma_pwba": _row_vals(exc_pwba, i),
            "ionization_sigma_pwba": _row_vals(ion_pwba, i),
            "excitation_sigma_rel": _row_vals(exc_rel_long, i),
            "ionization_sigma_rel": _row_vals(ion_rel_long, i),
            "excitation_sigma_rel_trans": _row_vals(exc_rel_trans, i),
            "ionization_sigma_rel_trans": _row_vals(ion_rel_trans, i),
            "excitation_sigma_mc": _row_vals(exc_mc, i),
            "ionization_sigma_mc": _row_vals(ion_mc, i),
            "total_sigma": _scalar(total_sigma, i),
            "total_sigma_pwba": _scalar(total_sigma_pwba, i),
            "total_sigma_mc": _scalar(total_sigma_mc, i),
            "total_sigma_rel": _scalar(total_sigma_rel, i),
            "total_sigma_rel_trans": _scalar(total_sigma_rel_trans, i),
            "total_sigma_rel_trans_no_density": _scalar(total_sigma_rel_trans_no_density, i),
            "total_sigma_rel_total": _scalar(total_sigma_rel_total, i),
            "total_sigma_plus_rel": _scalar(total_sigma_plus_rel, i),
            "total_sigma_plus_rel_total": _scalar(total_sigma_plus_rel_total, i),
            "kshell_sigma": _scalar(kshell_sigma, i),
            "kshell_sigma_rel": _scalar(kshell_sigma_rel, i),
        }
        sigma_list.append(_scale_sigma_entry(sigma, scale_factor))
    return T_arr.tolist(), sigma_list

def load_cross_section_corrections_npz(
    npz_path,
    NE,
    Nq,
    T_list=None,
    require_dcs=False,
    target_density_scale_factor=None,
):
    if npz_path is None or not os.path.exists(npz_path):
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as npz_data:
            if not _npz_matches_params(npz_data, NE, Nq, T_list=T_list):
                return None
            dcs_data = _dcs_data_from_npz(npz_data)
            if require_dcs and dcs_data is None:
                return None
            stored_scale = _density_scale_factor_from_npz(npz_data)
            if target_density_scale_factor is None:
                scale_factor = stored_scale
            else:
                if (not np.isfinite(stored_scale)) or stored_scale <= 0.0:
                    stored_scale = 1.0
                scale_factor = float(target_density_scale_factor) / stored_scale
            T_loaded, sigma_list = _sigma_list_from_npz(npz_data, scale_factor=scale_factor)
    except Exception as exc:
        print(f"Failed to load cached NPZ {npz_path}: {exc}")
        return None
    return T_loaded, sigma_list, dcs_data

def _geant4_dna_dir():
    for root in (CUSTOM_DATA_ROOT_GEANT4, CUSTOM_DATA_ROOT_PROJECT):
        if root is None:
            continue
        if not root.exists():
            continue
        dna_dir = root / "G4EMLOW8.6.1" / "dna"
        if dna_dir.exists():
            return dna_dir
    return None

def _load_dcs_template_grid(path, t_min=None, t_max=None, include_min=True, include_max=True):
    from collections import OrderedDict

    grid = OrderedDict()
    with open(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                T = float(parts[0])
                E = float(parts[1])
            except ValueError:
                continue
            if t_min is not None:
                if T < t_min or (not include_min and T == t_min):
                    continue
            if t_max is not None:
                if T > t_max or (not include_max and T == t_max):
                    continue
            grid.setdefault(T, []).append(E)
    return grid

def _merge_dcs_template_grids(*grids):
    from collections import OrderedDict

    merged = OrderedDict()
    for grid in grids:
        for T, E_list in grid.items():
            if T in merged:
                continue
            merged[T] = E_list
    return merged

def _extend_dcs_grid(grid, t_max, t_step):
    from collections import OrderedDict

    if not grid:
        return grid
    t_max = float(t_max)
    t_step = float(t_step)
    if t_step <= 0.0:
        return grid

    grid_sorted = OrderedDict(sorted(grid.items(), key=lambda kv: kv[0]))
    last_T = max(grid_sorted.keys())
    if t_max <= last_T:
        return grid_sorted

    template_E = list(grid_sorted[last_T])
    t = last_T + t_step
    while t <= t_max + 0.5 * t_step:
        grid_sorted[float(t)] = list(template_E)
        t += t_step
    return grid_sorted

def _format_dcs_row(T, E, vals):
    fields = [f"{T:.9E}", f"{E:.9E}"]
    fields.extend(f"{val:.9E}" for val in vals)
    return " ".join(fields) + "\n"

def _write_dcs_tables_from_data(dcs_data, exc_out, ion_out):
    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = np.asarray(dcs_data.get("exc_vals", []), float)
    ion_vals = np.asarray(dcs_data.get("ion_vals", []), float)

    if T_line.size == 0 or E_line.size == 0:
        raise ValueError("DCS data is empty.")
    if T_line.shape != E_line.shape:
        raise ValueError("DCS T/E arrays must have the same shape.")
    if exc_vals.ndim == 1:
        exc_vals = exc_vals.reshape(-1, 1)
    if ion_vals.ndim == 1:
        ion_vals = ion_vals.reshape(-1, 1)
    if exc_vals.shape[0] != T_line.size or ion_vals.shape[0] != T_line.size:
        raise ValueError("DCS value arrays must align with T/E lines.")

    with open(exc_out, "w") as exc_handle, open(ion_out, "w") as ion_handle:
        for i in range(T_line.size):
            exc_handle.write(_format_dcs_row(T_line[i], E_line[i], exc_vals[i]))
            ion_handle.write(_format_dcs_row(T_line[i], E_line[i], ion_vals[i]))

def _load_dcs_table(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"DCS table {path} has fewer than 3 columns.")
    T_line = np.asarray(data[:, 0], float)
    E_line = np.asarray(data[:, 1], float)
    vals = np.asarray(data[:, 2:], float)
    return T_line, E_line, vals

def _load_dcs_pair(exc_path, ion_path):
    T_exc, E_exc, exc_vals = _load_dcs_table(exc_path)
    T_ion, E_ion, ion_vals = _load_dcs_table(ion_path)
    if (T_exc.shape != T_ion.shape) or (E_exc.shape != E_ion.shape):
        raise ValueError("Excitation and ionization DCS grids have different shapes.")
    if not np.allclose(T_exc, T_ion) or not np.allclose(E_exc, E_ion):
        raise ValueError("Excitation and ionization DCS grids do not match.")
    return {
        "T_line": T_exc,
        "E_line": E_exc,
        "exc_vals": exc_vals,
        "ion_vals": ion_vals,
    }

def _integrate_dcs_to_totals(T_line, E_line, vals):
    T_line = np.asarray(T_line, float)
    E_line = np.asarray(E_line, float)
    vals = np.asarray(vals, float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if (T_line.size != E_line.size) or (T_line.size != vals.shape[0]):
        raise ValueError("DCS arrays must have matching lengths.")

    unique_T = []
    totals = []
    n = T_line.size
    start = 0
    for i in range(1, n + 1):
        if i == n or T_line[i] != T_line[start]:
            Tval = float(T_line[start])
            E_seg = np.asarray(E_line[start:i], float)
            V_seg = np.asarray(vals[start:i], float)
            if E_seg.size == 0:
                start = i
                continue
            order = np.argsort(E_seg)
            E_sorted = E_seg[order]
            V_sorted = V_seg[order]
            row = [_simpson_integrate(V_sorted[:, j], E_sorted) for j in range(V_sorted.shape[1])]
            unique_T.append(Tval)
            totals.append(row)
            start = i
    return np.asarray(unique_T, float), np.asarray(totals, float)

def _write_total_table(T_vals, totals, out_path):
    with open(out_path, "w") as handle:
        for Tval, row in zip(T_vals, totals):
            fields = [f"{Tval:.9E}"]
            fields.extend(f"{val:.9E}" for val in row)
            handle.write(" ".join(fields) + "\n")

def _write_total_tables_from_dcs(dcs_data, exc_out, ion_out):
    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = np.asarray(dcs_data.get("exc_vals", []), float)
    ion_vals = np.asarray(dcs_data.get("ion_vals", []), float)
    if T_line.size == 0 or E_line.size == 0:
        raise ValueError("DCS data is empty; cannot compute totals.")
    T_exc, exc_totals = _integrate_dcs_to_totals(T_line, E_line, exc_vals)
    T_ion, ion_totals = _integrate_dcs_to_totals(T_line, E_line, ion_vals)
    if not np.allclose(T_exc, T_ion):
        raise ValueError("Excitation and ionization totals have mismatched T grids.")
    _write_total_table(T_exc, exc_totals, exc_out)
    _write_total_table(T_ion, ion_totals, ion_out)

def _export_to_custom_geant4(paths):
    roots = []
    for root in (CUSTOM_DATA_ROOT_GEANT4, CUSTOM_DATA_ROOT_PROJECT):
        if root is None:
            continue
        dna_dir = root / "G4EMLOW8.6.1" / "dna"
        if dna_dir.exists():
            roots.append(dna_dir)
    if not roots:
        print("No custom Geant4 DNA data dir found; skipping export.")
        return
    for dna_dir in roots:
        for path in paths:
            if path is None:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Missing file for export: {path}")
            dest = dna_dir / path.name
            shutil.copy2(path, dest)
        print(f"Exported {len(paths)} files to {dna_dir}")

_DCS_WORKER_S = None
_DCS_WORKER_C = None
_DCS_WORKER_T_LINE = None
_DCS_WORKER_E_LINE = None
_DCS_WORKER_NQ = None
_DCS_WORKER_EXC_B = None
_DCS_WORKER_ION_B = None
_DCS_WORKER_KSHELL_B = None

def _compute_dcs_channel_values(
    channel_type,
    idx,
    s,
    C,
    Nq,
    T_line,
    E_line,
    exc_B,
    ion_B,
    kshell_B,
):
    T_line = np.asarray(T_line, float)
    E_line = np.asarray(E_line, float)
    n_lines = T_line.size
    vals = np.zeros(n_lines, float)

    for i in range(n_lines):
        Tj = float(T_line[i])
        Ei = float(E_line[i])

        if channel_type == "excitation":
            Bk = exc_B[idx]
            if Ei < Bk or Ei > Tj:
                val = 0.0
            else:
                val = _selected_dsigma_excitation(Ei, Tj, idx, s, C, Nq)
        elif channel_type == "ionization":
            Bj = ion_B[idx]
            if Ei < Bj or Ei > 0.5 * (Tj + Bj):
                val = 0.0
            else:
                val = _selected_dsigma_ionization(Ei, Tj, idx, s, C, Nq)
        elif channel_type == "kshell":
            if kshell_B is None:
                val = 0.0
            elif Ei < kshell_B or Ei > 0.5 * (Tj + kshell_B):
                val = 0.0
            else:
                val = _selected_dsigma_kshell(Ei, Tj, s, Nq)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")

        if (not np.isfinite(val)) or (val < 0.0):
            val = 0.0
        vals[i] = val / EMFI_DCS_SCALE_M2

    return vals

def _init_dcs_worker(
    a_vec,
    b_vec,
    c_vec,
    T_line,
    E_line,
    Nq,
    apply_mc,
    apply_regime_ii,
    apply_regime_iii,
    apply_regime_iv,
    ice_type,
):
    global _DCS_WORKER_S, _DCS_WORKER_C
    global _DCS_WORKER_T_LINE, _DCS_WORKER_E_LINE, _DCS_WORKER_NQ
    global _DCS_WORKER_EXC_B, _DCS_WORKER_ION_B, _DCS_WORKER_KSHELL_B

    _set_regime_corrections(
        apply_regime_ii=apply_regime_ii,
        apply_regime_iii=apply_regime_iii,
        apply_regime_iv=apply_regime_iv,
    )
    _set_mc_correction(apply_mc=apply_mc)

    _DCS_WORKER_S = model.epsilon_optical(ice_type)
    _DCS_WORKER_C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)
    _DCS_WORKER_T_LINE = np.asarray(T_line, float)
    _DCS_WORKER_E_LINE = np.asarray(E_line, float)
    _DCS_WORKER_NQ = int(Nq)
    _DCS_WORKER_EXC_B = [float(osc.Bth) for osc in _DCS_WORKER_S.excitations]
    _DCS_WORKER_ION_B = [float(osc.Bth) for osc in _DCS_WORKER_S.ionizations]
    _DCS_WORKER_KSHELL_B = (
        float(_DCS_WORKER_S.kshell.Bth) if _DCS_WORKER_S.kshell is not None else None
    )

def _compute_dcs_channel_worker(args):
    channel_type, idx = args
    vals = _compute_dcs_channel_values(
        channel_type,
        idx,
        _DCS_WORKER_S,
        _DCS_WORKER_C,
        _DCS_WORKER_NQ,
        _DCS_WORKER_T_LINE,
        _DCS_WORKER_E_LINE,
        _DCS_WORKER_EXC_B,
        _DCS_WORKER_ION_B,
        _DCS_WORKER_KSHELL_B,
    )
    return channel_type, idx, vals

def _selected_dsigma_excitation(Ei, Tj, k, s, C, Nq):
    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(Tj)
    if use_mc:
        Bk = float(s.excitations[k].Bth)
        Tshift = float(Tj + 2.0 * Bk)
        return _dsigma_pwba_dE(Ei, Tshift, k, "excitation", s, C, Nq=Nq, use_rel=use_rel_long)
    if use_rel_long:
        val = _integrate_channel_single_E_rel(Ei, Tj, k, "excitation", s, C, Nq=Nq)
        if use_rel_trans:
            val += _integrate_channel_single_E_trans(
                Ei, Tj, k, "excitation", s, C, use_density_effect=use_density_effect
            )
        return val
    return _integrate_channel_single_E(Ei, Tj, k, "excitation", s, C, Nq=Nq, use_rel_bounds=False)

def _selected_dsigma_ionization(Ei, Tj, j, s, C, Nq):
    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(Tj)
    if use_mc:
        return _dsigma_mc_ionization_dE(Ei, Tj, j, s, C, Nq=Nq, use_rel=use_rel_long)
    if use_rel_long:
        val = _integrate_channel_single_E_rel(Ei, Tj, j, "ionization", s, C, Nq=Nq)
        if use_rel_trans:
            val += _integrate_channel_single_E_trans(
                Ei, Tj, j, "ionization", s, C, use_density_effect=use_density_effect
            )
        return val
    return _integrate_channel_single_E(Ei, Tj, j, "ionization", s, C, Nq=Nq, use_rel_bounds=False)

def _selected_dsigma_kshell(Ei, Tj, s, Nq):
    if s.kshell is None:
        return 0.0
    _, use_rel_long, _, _ = _regime_flags(Tj)
    if use_rel_long:
        return _integrate_kshell_single_E_rel(Ei, Tj, s, Nq=Nq, include_kshell=True)
    return _integrate_kshell_single_E(Ei, Tj, s, Nq=Nq, include_kshell=True, use_rel_bounds=False)

def write_emfietzoglou_dcs_tables(
    s,
    C,
    Nq=200,
    out_dir=None,
    template_path=None,
    dcs_data=None,
    return_data=False,
    parallel_channels=True,
    max_workers=None,
    a_vec=None,
    b_vec=None,
    c_vec=None,
    ice_label=None,
    ice_type=None,
    apply_mc=None,
    apply_regime_ii=None,
    apply_regime_iii=None,
    apply_regime_iv=None,
):
    if out_dir is None:
        out_dir = CROSS_SECTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if ice_label is None:
        ice_label = ICE_LABEL
    exc_out = out_dir / f"sigmadiff_excitation_e_{ice_label}_emfietzoglou_kyriakou.dat"
    ion_out = out_dir / f"sigmadiff_ionisation_e_{ice_label}_emfietzoglou_kyriakou.dat"
    exc_total_out = out_dir / f"sigma_excitation_e_{ice_label}_emfietzoglou_kyriakou.dat"
    ion_total_out = out_dir / f"sigma_ionisation_e_{ice_label}_emfietzoglou_kyriakou.dat"

    if dcs_data is None and exc_out.exists() and ion_out.exists():
        try:
            dcs_data = _load_dcs_pair(exc_out, ion_out)
            print(f"Loaded existing DCS from {exc_out} and {ion_out}")
        except Exception as exc:
            print(f"Failed to load existing DCS tables: {exc}")
            dcs_data = None

    if dcs_data is None:
        if template_path is None:
            dna_dir = _geant4_dna_dir()
            if dna_dir is None:
                raise FileNotFoundError("Could not locate Geant4 DNA data directory.")
            emfi_path = dna_dir / "sigmadiff_ionisation_e_emfietzoglou.dat"
            if not os.path.exists(emfi_path):
                raise FileNotFoundError(f"Missing DCS template file: {emfi_path}")
            grid_low = _load_dcs_template_grid(emfi_path)

            grid = grid_low
            born_path = dna_dir / "sigmadiff_ionisation_e_born.dat"
            if os.path.exists(born_path) and grid_low:
                t_switch = max(grid_low.keys())
                grid_high = _load_dcs_template_grid(
                    born_path, t_min=t_switch, include_min=False
                )
                if grid_high:
                    grid = _merge_dcs_template_grids(grid_low, grid_high)
            if DCS_T_MAX_EEV > max(grid.keys()):
                grid = _extend_dcs_grid(grid, DCS_T_MAX_EEV, DCS_T_STEP_EEV)
        else:
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Missing DCS template file: {template_path}")
            grid = _load_dcs_template_grid(template_path)

        T_line = []
        E_line = []
        for T, E_list in grid.items():
            Tj = float(T)
            for Ei in E_list:
                T_line.append(Tj)
                E_line.append(float(Ei))

        T_line = np.asarray(T_line, float)
        E_line = np.asarray(E_line, float)

        exc_B = [float(osc.Bth) for osc in s.excitations]
        ion_B = [float(osc.Bth) for osc in s.ionizations]
        kshell_B = float(s.kshell.Bth) if s.kshell is not None else None

        n_lines = T_line.size
        n_exc = len(exc_B)
        n_ion = len(ion_B)

        exc_vals = np.zeros((n_lines, n_exc), float)
        ion_vals = np.zeros((n_lines, n_ion + 1), float)

        if apply_mc is None:
            apply_mc = APPLY_MOTT_COULOMB
        if apply_regime_ii is None:
            apply_regime_ii = APPLY_CORRECTIONS_REGIME_II
        if apply_regime_iii is None:
            apply_regime_iii = APPLY_CORRECTIONS_REGIME_III
        if apply_regime_iv is None:
            apply_regime_iv = APPLY_CORRECTIONS_REGIME_IV
        if ice_type is None:
            ice_type = ICE_TYPE

        tasks = [("excitation", k) for k in range(n_exc)]
        tasks.extend(("ionization", j) for j in range(n_ion))
        if kshell_B is not None:
            tasks.append(("kshell", 0))

        do_parallel = (
            parallel_channels
            and len(tasks) > 1
            and a_vec is not None
            and b_vec is not None
            and c_vec is not None
        )

        if do_parallel:
            if max_workers is None:
                max_workers = os.cpu_count() or 1
            max_workers = max(1, min(int(max_workers), len(tasks)))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_dcs_worker,
                initargs=(
                    a_vec,
                    b_vec,
                    c_vec,
                    T_line,
                    E_line,
                    Nq,
                    apply_mc,
                    apply_regime_ii,
                    apply_regime_iii,
                    apply_regime_iv,
                    ice_type,
                ),
            ) as ex:
                futures = [ex.submit(_compute_dcs_channel_worker, task) for task in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="DCS channels"):
                    channel_type, idx, vals = fut.result()
                    if channel_type == "excitation":
                        exc_vals[:, idx] = vals
                    elif channel_type == "ionization":
                        ion_vals[:, idx] = vals
                    elif channel_type == "kshell":
                        ion_vals[:, -1] = vals
        else:
            for k in tqdm(range(n_exc), desc="DCS excitation"):
                exc_vals[:, k] = _compute_dcs_channel_values(
                    "excitation", k, s, C, Nq, T_line, E_line, exc_B, ion_B, kshell_B
                )
            for j in tqdm(range(n_ion), desc="DCS ionization"):
                ion_vals[:, j] = _compute_dcs_channel_values(
                    "ionization", j, s, C, Nq, T_line, E_line, exc_B, ion_B, kshell_B
                )
            if kshell_B is not None:
                ion_vals[:, -1] = _compute_dcs_channel_values(
                    "kshell", 0, s, C, Nq, T_line, E_line, exc_B, ion_B, kshell_B
                )

        dcs_data = {
            "T_line": T_line,
            "E_line": E_line,
            "exc_vals": exc_vals,
            "ion_vals": ion_vals,
        }

    _write_dcs_tables_from_data(dcs_data, exc_out, ion_out)
    _write_total_tables_from_dcs(dcs_data, exc_total_out, ion_total_out)
    _export_to_custom_geant4([exc_out, ion_out, exc_total_out, ion_total_out])

    print(f"Saved excitation DCS to {exc_out}")
    print(f"Saved ionization DCS to {ion_out}")
    print(f"Saved excitation total to {exc_total_out}")
    print(f"Saved ionization total to {ion_total_out}")
    if return_data:
        return dcs_data
    return None

def plot_total_cross_section_corrections(T_list, sigma_list, out_path=None, ice_label=None):
    """
    Plot PWBA vs corrected totals and per-stage correction terms vs energy.
    """
    if out_path is None:
        if ice_label is None:
            ice_label = ICE_LABEL
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"cross_section_corrections_{ice_label}.png"

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
    ax_corr.set_xlabel("Electron energy (T; eV)")
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
    ice_type,
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
    _WORKER_S = model.epsilon_optical(ice_type)
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
    if ICE_TYPE not in ("amorphous", "hexagonal"):
        raise ValueError(f"Unsupported ICE_TYPE: {ICE_TYPE}")
    # Optical model / dispersion coefficients
    s = model.epsilon_optical(ICE_TYPE)
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])

    # RR2017 defaults for c_disp, d_disp, b1, b2
    C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)

    # Energy grid (eV)
    T_list = np.logspace(-1, 7, 1000)

    # Computing Choices
    NE = 300
    Nq = 300
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

    rho_ref = float(N_REFERENCE_DENSITY_G_CM3)
    rho_phase_nominal = _assumed_density_g_cm3(ICE_TYPE)
    rho_target = _target_density_g_cm3(ICE_TYPE)
    density_scale_factor = _density_scale_factor_for_ice(ICE_TYPE)
    print(
        f"Density scaling for {ICE_TYPE}: "
        f"rho_ref={rho_ref:.6g} g/cm^3, rho_target={rho_target:.6g} g/cm^3, "
        f"rho_phase_nominal={rho_phase_nominal:.6g} g/cm^3, "
        f"scale={density_scale_factor:.6g}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / f"cross_section_corrections_{ICE_LABEL}.npz"
    cached = load_cross_section_corrections_npz(
        cache_path,
        NE=NE,
        Nq=Nq,
        T_list=T_list,
        target_density_scale_factor=density_scale_factor,
    )

    sigma_list = None
    dcs_data = None
    dcs_written = False

    if cached is not None:
        T_list, sigma_list, dcs_data = cached
        print(f"Loaded cached cross sections from {cache_path}")
    else:
        print("Computing double-integrated cross sections (parallel over T)...")
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
                    ICE_TYPE,
                ),
        ) as ex:
            futures = [ex.submit(_compute_for_T, T) for T in T_list]

            for fut in tqdm(as_completed(futures), total=len(futures)):
                T_val, sigma = fut.result()
                results_by_T[T_val] = sigma

        # Restore original T order
        sigma_list = [results_by_T[float(T)] for T in T_list]
        sigma_list = _scale_sigma_list(sigma_list, density_scale_factor)

        dcs_data = write_emfietzoglou_dcs_tables(
            s,
            C,
            Nq=Nq,
            return_data=True,
            ice_label=ICE_LABEL,
            ice_type=ICE_TYPE,
            a_vec=a_vec,
            b_vec=b_vec,
            c_vec=c_vec,
            apply_mc=apply_mc,
            apply_regime_ii=apply_regime_ii,
            apply_regime_iii=apply_regime_iii,
            apply_regime_iv=apply_regime_iv,
        )
        dcs_written = True
        save_cross_section_corrections_npz(
            T_list,
            sigma_list,
            out_path=cache_path,
            NE=NE,
            Nq=Nq,
            dcs_data=dcs_data,
            ice_label=ICE_LABEL,
            density_scale_factor=density_scale_factor,
            density_ref_g_cm3=rho_ref,
            density_assumed_g_cm3=rho_target,
        )

    if not dcs_written:
        if dcs_data is None:
            dcs_data = write_emfietzoglou_dcs_tables(
                s,
                C,
                Nq=Nq,
                return_data=True,
                ice_label=ICE_LABEL,
                ice_type=ICE_TYPE,
                a_vec=a_vec,
                b_vec=b_vec,
                c_vec=c_vec,
                apply_mc=apply_mc,
                apply_regime_ii=apply_regime_ii,
                apply_regime_iii=apply_regime_iii,
                apply_regime_iv=apply_regime_iv,
            )
            dcs_written = True
            save_cross_section_corrections_npz(
                T_list,
                sigma_list,
                out_path=cache_path,
                NE=NE,
                Nq=Nq,
                dcs_data=dcs_data,
                ice_label=ICE_LABEL,
                density_scale_factor=density_scale_factor,
                density_ref_g_cm3=rho_ref,
                density_assumed_g_cm3=rho_target,
            )
        else:
            write_emfietzoglou_dcs_tables(
                s,
                C,
                Nq=Nq,
                dcs_data=dcs_data,
                ice_label=ICE_LABEL,
                ice_type=ICE_TYPE,
            )
            dcs_written = True

    # ----------------- Log corrections per energy -----------------
    plot_total_cross_section_corrections(T_list, sigma_list, ice_label=ICE_LABEL)

    # ----------------- Plot 0: corrected excitation/ionization (scaled) -----------------
    style = rcparams_with_fontsize(
        RC_BASE_ELASTIC,
        FONTSIZE_24,
        overrides={
            "font.family": FONT_COURIER,
            "mathtext.rm": FONT_COURIER,
            "mathtext.fontset": "custom",
        },
    )
    with plt.rc_context(style):
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        fig = plt.figure(figsize=(9, 7.8))
        gs = GridSpec(2, 1, height_ratios=[4.7, 1.3], hspace=0.30)
        ax = fig.add_subplot(gs[0, 0])
        plot_corrected_exc_ion_scaled(T_list, sigma_list, ax=ax)

        gs_leg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[1.0, 1.0], hspace=0.30)
        ax_leg_exc = fig.add_subplot(gs_leg[0, 0])
        ax_leg_ion = fig.add_subplot(gs_leg[1, 0])
        ax_leg_exc.axis("off")
        ax_leg_ion.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            n_exc = len(s.excitations)
            n_ion = len(s.ionizations)
            exc_handles = handles[:n_exc]
            ion_handles = handles[n_exc:n_exc + n_ion]
            if exc_handles:
                exc_labels = [str(i + 1) for i in range(n_exc)]
                ax_leg_exc.legend(
                    exc_handles,
                    exc_labels,
                    loc="center left",
                    bbox_to_anchor=(0.0, 0.5),
                    ncol=max(1, n_exc),
                    frameon=False,
                    columnspacing=0.9,
                    handlelength=2.2,
                    handletextpad=0.6,
                    borderaxespad=0.0,
                )
            if ion_handles:
                ion_labels = [str(i + 1) for i in range(n_ion)]
                ax_leg_ion.legend(
                    ion_handles,
                    ion_labels,
                    loc="center left",
                    bbox_to_anchor=(0.0, 0.5),
                    ncol=max(1, n_ion),
                    frameon=False,
                    columnspacing=0.9,
                    handlelength=2.2,
                    handletextpad=0.6,
                    borderaxespad=0.0,
                )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"corrected_excitation_ionization_scaled_{ICE_LABEL}.png"
        fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.06)
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
    ax_ion.figure.savefig(f"comparison_ionizations_pwba_vs_model_{ICE_LABEL}.png", dpi=300)
    plt.close(ax_ion.figure)

    ax_exc.figure.tight_layout()
    ax_exc.figure.savefig(f"comparison_excitations_pwba_vs_model_{ICE_LABEL}.png", dpi=300)
    plt.close(ax_exc.figure)
    print(
        f"Saved comparison plots to comparison_ionizations_pwba_vs_model_{ICE_LABEL}.png "
        f"and comparison_excitations_pwba_vs_model_{ICE_LABEL}.png"
    )


    # ----------------- Plot 2: REL components per channel (L solid, T dashed) -----------------
    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    plot_relativistic_component_per_channel(T_list, sigma_list, s, ax=(ax_ion, ax_exc))
    fig_ion.tight_layout()
    fig_exc.tight_layout()
    fig_ion.savefig(f"rel_cross_sections_ionizations_LT_{ICE_LABEL}.png", dpi=300)
    fig_exc.savefig(f"rel_cross_sections_excitations_LT_{ICE_LABEL}.png", dpi=300)
    plt.close(fig_ion)
    plt.close(fig_exc)
    print(
        f"Saved REL component plots to rel_cross_sections_ionizations_LT_{ICE_LABEL}.png "
        f"and rel_cross_sections_excitations_LT_{ICE_LABEL}.png"
    )

    # ----------------- Plot 3: TOTAL cross section with all corrections -----------------
    fig, ax = plt.subplots()
    plot_total_cross_section(T_list, sigma_list, s, ax=ax)
    fig.tight_layout()
    fig.savefig(f"total_cross_section_all_corrections_{ICE_LABEL}.png", dpi=300)
    plt.close(fig)
    print(f"Saved total plot to total_cross_section_all_corrections_{ICE_LABEL}.png")

    # ----------------- Plot 4: two-panel amorphous vs hexagonal totals -----------------
    amorphous_npz = OUTPUT_DIR / "cross_section_corrections_amorphous_ice.npz"
    hexagonal_npz = OUTPUT_DIR / "cross_section_corrections_hexagonal_ice.npz"
    if amorphous_npz.exists() and hexagonal_npz.exists():
        out_path = OUTPUT_DIR / "channel_cross_section_two_panel_amorphous_hexagonal.png"
        plot_channel_cross_sections_two_panel(amorphous_npz, hexagonal_npz, out_path=out_path)
    else:
        missing = []
        if not amorphous_npz.exists():
            missing.append(amorphous_npz.name)
        if not hexagonal_npz.exists():
            missing.append(hexagonal_npz.name)
        if missing:
            print(f"Skipping two-panel total plot (missing {', '.join(missing)}).")

if __name__ == "__main__":
    main()
