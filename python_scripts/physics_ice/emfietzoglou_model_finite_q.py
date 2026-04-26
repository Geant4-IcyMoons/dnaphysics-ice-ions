r"""
Dielectric model for water ice (amorphous, hexagonal) with q-dispersion.

What this implements
- 5 excitation bands with per-band dispersion parameters (a_j, b_j, c_j) applied to f_j(q)
- 4 ionization shells with global dispersion for E_i(q) and gamma_i(q)
- Optional O K-shell kept optical
- Returns \epsilon_1(E,q), \epsilon_2(E,q), and ELF(E,q) = Im[-1/\epsilon(E,q)]

Units
- Energies in eV
- q in a0^{-1}
"""

from dataclasses import dataclass
from typing import List, Literal, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from constants import (
    EH,
    FONT_COURIER,
    FONTSIZE_16,
    HFONT_COURIER,
    ICE_DATA_XLSX_PATH,
    PROJECT_ROOT,
    RC_BASE_STANDARD,
    RY,
    rcparams_with_fontsize,
)

font = FONT_COURIER
hfont = HFONT_COURIER
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'

FONTSIZE = FONTSIZE_16
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_STANDARD, FONTSIZE))

Material = Literal["amorphous", "hexagonal"]
ArrayLike = Union[float, np.ndarray]

# ---- partitioning helpers (Kyriakou et al.) ----
def _Theta(x):
    # Θ(0)=1 per Appendix
    x = np.asarray(x)
    out = np.ones_like(x)
    out[x < 0] = 0.0
    return out

def _H(x):
    # H(0)=0 per Appendix
    x = np.asarray(x)
    out = np.zeros_like(x)
    out[x > 0] = 1.0
    return out

def _safe_div(num, den):
    eps = 1e-300
    return num / np.where(np.abs(den) < eps, np.sign(den) * eps + eps, den)

def _apply_partitioning_optical(E, exc_arrays, exc_Ek, ion_arrays, ion_B, Bmin):
    """
    Apply Kyriakou et al. redistribution at q=0 only (optical limit).
    Indices:
      - ionization shells: i
      - excitation channels: j
    Returns modified lists in the same order as input.
    """
    E = np.asarray(E, float)
    nE = E.size

    IonIm = np.stack(ion_arrays, axis=0) if ion_arrays else np.zeros((0, nE))
    IonB  = np.array(list(ion_B), dtype=float) if ion_arrays else np.zeros((0,))

    ExcIm = np.stack(exc_arrays, axis=0) if exc_arrays else np.zeros((0, nE))
    ExcEk = np.array(list(exc_Ek), dtype=float) if exc_arrays else np.zeros((0,))

    n_ion = IonIm.shape[0]
    n_exc = ExcIm.shape[0]

    # Sort ion shells by ascending Bi
    if n_ion > 0:
        order = np.argsort(IonB)
        IonIm = IonIm[order]
        IonB  = IonB[order]

    B1 = IonB[0] if n_ion > 0 else np.inf

    # -------------------------
    # Ionizations (Eqs. A1–A5)
    # -------------------------
    IonIm_mod = np.zeros_like(IonIm)

    for idx_i in range(n_ion):
        Im_i = IonIm[idx_i]
        Bi   = IonB[idx_i]

        # C_i(E) = H(E - Bi)
        C_i = _H(E - Bi)

        # S_i(E) = -Im_i(Bi) * exp(Bi - E)
        Im_i_Bi = np.interp(Bi, E, Im_i, left=0.0, right=0.0)
        S_i = - Im_i_Bi * np.exp(Bi - E)

        C_greater = np.zeros_like(E)
        S_greater = np.zeros_like(E)

        # Redistribution from higher shells j > i
        for idx_j in range(idx_i + 1, n_ion):
            Im_j = IonIm[idx_j]
            Bj   = IonB[idx_j]

            # denominator: sum_{m <= j-1} Im_m(E) H(E - B_m)
            den = np.zeros_like(E)
            for m in range(0, idx_j):
                den += IonIm[m] * _H(E - IonB[m])

            # C>i term
            C_term = Im_j * _Theta(Bj - E) * _safe_div(Im_i, den)

            # S>i term
            Im_j_Bj = np.interp(Bj, E, Im_j, left=0.0, right=0.0)
            S_term  = Im_j_Bj * np.exp(Bj - E) * _H(E - Bj) * _safe_div(Im_i, den)

            C_greater += C_term
            S_greater += S_term

        IonIm_mod[idx_i] = (Im_i + S_i + S_greater + C_greater) * C_i

    # -------------------------
    # Excitations (Eqs. A6–A9)
    # -------------------------
    ExcIm_mod = np.zeros_like(ExcIm)
    B_gap = Bmin

    if n_exc > 0:
        # For C_j: denominator sum_j Im_j(E) Θ(B1 − E_j)
        if np.isfinite(B1):
            Theta_B1_minus_Ej = np.array(
                [1.0 if (B1 - Ej) >= 0.0 else 0.0 for Ej in ExcEk],
                float
            )
        else:
            Theta_B1_minus_Ej = np.zeros((n_exc,), float)

        Exc_den_Cj = np.sum(
            ExcIm * Theta_B1_minus_Ej[:, None],
            axis=0
        ) if n_exc > 0 else np.zeros_like(E)

        # For S_{1j}: denominator sum_j Im_j(E)
        Exc_den_S1 = np.sum(ExcIm, axis=0)

        # Ion sum for C_j
        if n_ion > 0:
            Ion_sum = np.sum(IonIm, axis=0)
        else:
            Ion_sum = np.zeros_like(E)

        Theta_B1_minus_E = _Theta(B1 - E) if np.isfinite(B1) else np.zeros_like(E)

        # Im_1(B1) for S_{1j}
        if n_ion > 0 and np.isfinite(B1):
            Im_1_B1 = np.interp(B1, E, IonIm[0], left=0.0, right=0.0)
        else:
            Im_1_B1 = 0.0

        for j in range(n_exc):
            Im_j = ExcIm[j]
            Ej   = ExcEk[j]

            # Gate at B_min (valence onset)
            if np.isfinite(B_gap):
                C_j_gate = _Theta(E - B_gap)
            else:
                C_j_gate = np.ones_like(E)

            # S_{1j}(E): redistribution of the low-energy tail of shell 1
            if n_ion > 0 and np.isfinite(B1):
                S1_weight = _safe_div(Im_j, Exc_den_S1)
                S_1j = Im_1_B1 * np.exp(B1 - E) * _H(E - B1) * S1_weight
            else:
                S_1j = np.zeros_like(E)

            # C_j(E): redistribution of ionization strength below B1 into excitations with Ej <= B1
            if n_ion > 0 and np.isfinite(B1):
                top_weight_j = Im_j * (1.0 if (B1 - Ej) >= 0.0 else 0.0)
                weight_j     = _safe_div(top_weight_j, Exc_den_Cj)
                C_j = Ion_sum * Theta_B1_minus_E * weight_j
            else:
                C_j = np.zeros_like(E)

            ExcIm_mod[j] = (Im_j + S_1j + C_j) * C_j_gate

    exc_mod_list = [ExcIm_mod[j] if n_exc > 0 else np.zeros_like(E) for j in range(n_exc)]
    ion_mod_list = [IonIm_mod[i] if n_ion > 0 else np.zeros_like(E) for i in range(n_ion)]

    return exc_mod_list, ion_mod_list

# ---- data containers ----
@dataclass(frozen=True)
class Osc:
    E0: float       # resonance energy (eV)
    gamma: float    # damping width (eV)
    f: float        # oscillator strength (dimensionless)
    kind: Literal["ionization", "excitation", "k_shell"]
    Bth: float = 0.0  # threshold (eV), used for ionization channels
    U: float = 0.0  # kinetic energy (eV), in each shell


@dataclass(frozen=True)
class IceOpticalSet:
    Ep: float
    Bmin: float
    excitations: List[Osc]   # 5 derivative-Drude terms
    ionizations: List[Osc]   # 4 Drude terms with thresholds
    kshell: Osc              # optional Drude (kept optical)


@dataclass(frozen=True)
class DispersionCoeffs:
    """
    Per-excitation dispersion (length 5 each, ordered as in the excitations list):
        a_fj, b_fj, c_fj  -> f_j(q) = f_j*exp(-a_j q^2) + b_j q^2 exp(-c_j q^2)
    Global ionization dispersion:
        c_disp, d_disp    -> E_i(q) = E_i + [1 - exp(-c_disp * q**d_disp)] * RY * q^2
        b1, b2            -> gamma(q) = gamma_0 + b1*(RY*q) + b2*(RY*q)**2

    Notes:
    - q is in a0^{-1}. RY is used to express the kinematic q^2 term in eV.
    - If scalars are provided for a_fj, b_fj, c_fj they will be broadcast to length 5.
    """
    a_fj: ArrayLike   # scalar or len 5
    b_fj: ArrayLike   # scalar or len 5
    c_fj: ArrayLike   # scalar or len 5
    c_disp: float = 1.5   # RR2017 default
    d_disp: float = 0.4   # RR2017 default
    b1: float = 0.735     # RR2017 default
    b2: float = 0.441     # RR2017 default

# ---- Drude kernels (optical) ----
def _drude_e2(E: np.ndarray, f: float, E0: float, gamma: float) -> np.ndarray:
    num = f * gamma * E
    den = (E0**2 - E**2)**2 + (gamma * E)**2
    return num / den


def _drude_e1(E: np.ndarray, f: float, E0: float, gamma: float) -> np.ndarray:
    num = f * (E0**2 - E**2)
    den = (E0**2 - E**2)**2 + (gamma * E)**2
    return num / den


def _d_drude_e2(E: np.ndarray, f: float, E0: float, gamma: float) -> np.ndarray:
    num = 2.0 * f * (gamma**3) * (E**3)
    den = ((E0**2 - E**2)**2 + (gamma * E)**2)**2
    return num / den


def _d_drude_e1(E: np.ndarray, f: float, E0: float, gamma: float) -> np.ndarray:
    base = (E0**2 - E**2)
    den1 = (E0**2 - E**2)**2 + (gamma * E)**2
    num = f * base * ((E0**2 - E**2)**2 + 3.0 * (gamma * E)**2)
    den = den1**2
    return num / den


# ---- q=0 material parameters ----
def epsilon_optical(material: Material) -> IceOpticalSet:
    if material == "amorphous":
        Ep = 20.82
        Bmin = 7.0
        excit = [
            Osc(8.65,  1.6, 0.0090, "excitation"),
            Osc(10.50, 2.5, 0.0096, "excitation"),
            Osc(12.60, 3.5, 0.0210, "excitation"),
            Osc(14.10, 3.0, 0.0040, "excitation"),
            Osc(14.50, 2.5, 0.0030, "excitation"),
        ]
        ioniz = [
            Osc(15.40, 5.7, 0.1250, "ionization", Bth=10.0, U=61.91),
            Osc(18.60, 7.1, 0.1300, "ionization", Bth=13.0, U=59.52),
            Osc(24.50, 15.0, 0.1100, "ionization", Bth=17.0, U=48.36),
            Osc(38.00, 30.0, 0.4110, "ionization", Bth=32.0, U=70.71),
        ]
        kshell = Osc(450.0, 360.0, 0.3143, "k_shell", Bth=539.7, U=794.75)
    elif material == "hexagonal":
        Ep = 20.59
        Bmin = 7.0
        excit = [
            Osc(8.65,  1.6, 0.0168, "excitation"),
            Osc(10.50, 1.5, 0.0065, "excitation"),
            Osc(12.60, 3.0, 0.0190, "excitation"),
            Osc(14.10, 2.7, 0.0110, "excitation"),
            Osc(14.50, 1.5, 0.0044, "excitation"),
        ]
        ioniz = [
            Osc(15.80, 4.6, 0.1000, "ionization", Bth=10.0, U=61.91),
            Osc(18.00, 7.5, 0.2000, "ionization", Bth=13.0, U=59.52),
            Osc(24.50, 14.0, 0.1100, "ionization", Bth=17.0, U=48.36),
            Osc(35.00, 30.0, 0.3580, "ionization", Bth=32.0, U=70.71),
        ]
        kshell = Osc(450.0, 360.0, 0.3143, "k_shell", Bth=539.7, U=794.75)
    else:
        raise ValueError("material must be 'amorphous' or 'hexagonal'")

    return IceOpticalSet(Ep=Ep, Bmin=Bmin, excitations=excit, ionizations=ioniz, kshell=kshell)

# =====================================================================
# q = 0: VALENCE-ONLY ε2, ε1, and separate K-shell ε2
# =====================================================================

def epsilon2_valence_E0(E: np.ndarray, s: IceOpticalSet) -> dict:
    """
    Imaginary dielectric, optical limit (q=0), valence only.
    Returns a dict with per-channel arrays and the total:
      {
        "excitations": [array(...), ...],
        "ionizations": [array(...), ...],
        "total": array(...)
      }
    This is the unpartitioned optical ε2; Kyriakou partition is applied
    separately in the q-dependent code to derive correction factors.
    """
    E = np.asarray(E, float)

    # Gating (optical): excitations above Bmin; ionizations above individual Bth
    g_exc = (E >= s.Bmin).astype(float)

    exc = [(s.Ep**2) * g_exc * _d_drude_e2(E, o.f, o.E0, o.gamma) for o in s.excitations]
    ion = []
    for o in s.ionizations:
        g = (E >= o.Bth).astype(float)
        ion.append((s.Ep**2) * g * _drude_e2(E, o.f, o.E0, o.gamma))

    total = np.zeros_like(E)
    for y in exc:
        total += y
    for y in ion:
        total += y

    return {"excitations": exc, "ionizations": ion, "total": total}

def epsilon1_valence_E0(E: np.ndarray, s: IceOpticalSet) -> dict:
    """
    Real dielectric, optical limit (q=0), valence only.
    No threshold gating. Baseline +1 is included only in 'total'.
    Returns:
      {
        "excitations": [array(...), ...],
        "ionizations": [array(...), ...],
        "total": array(...)
      }
    """
    E = np.asarray(E, float)

    exc = [(s.Ep**2) * _d_drude_e1(E, o.f, o.E0, o.gamma) for o in s.excitations]
    ion = [(s.Ep**2) * _drude_e1(E, o.f, o.E0, o.gamma) for o in s.ionizations]

    total = np.ones_like(E)
    for y in exc:
        total += y
    for y in ion:
        total += y

    return {"excitations": exc, "ionizations": ion, "total": total}

def epsilon2_Kshell_E0(E: np.ndarray, s: IceOpticalSet) -> np.ndarray:
    E = np.asarray(E, float)
    o = s.kshell
    y = (s.Ep**2) * _drude_e2(E, o.f, o.E0, o.gamma)
    return np.where(E >= o.Bth, y, 0.0)

def epsilon2_Kshell_E0_fsum_corrected(E, s):
    r"""
    Oxygen K-shell ε2^(K)(E, q=0) normalized by the f-sum so that
        ∫_0^∞ E * ε2^(K)(E) dE = (π/2) * Ep^2 * N_K,   with N_K = 0.178.
    This enforces Neff^(K) = 0.178 and makes Neff_total → 1, Neff_valence → 0.822.

    Uses a single (normal) Drude shape with onset at the O K edge.
    Requires in 's.kshell' at least: E0, gamma, Bth. Uses s.Ep for Ep.
    """
    import numpy as np

    E   = np.asarray(E, dtype=float)
    Ep  = float(s.Ep)
    ks  = getattr(s, "kshell", None)
    if ks is None:
        # No K-shell parameters present
        return np.zeros_like(E)

    E0   = float(getattr(ks, "E0",   450.0))
    gamma= float(getattr(ks, "gamma",360.0))
    Bth  = float(getattr(ks, "Bth",  540.0))
    N_K  = 0.1788  # Emfietzoglou et al., fixed atomic fraction

    # Unit-amplitude Drude *shape* for ε2:
    # ε2_shape(E) = (γ E) / [(E0^2 - E^2)^2 + (γ E)^2], zeroed below the edge
    num   = gamma * E
    den   = (E0*E0 - E*E)**2 + (gamma * E)**2
    shape = np.where(E >= Bth, np.where(den > 0.0, num / den, 0.0), 0.0)

    # Target area for ∫ E * ε2^(K)(E) dE
    target = 0.5 * np.pi * (Ep**2) * N_K

    # Cumulative trapezoid; last value is the area
    dE     = np.diff(E)
    midE   = 0.5 * (E[1:] + E[:-1])
    area_shape = np.sum(0.5 * (shape[1:] + shape[:-1]) * dE * midE)
    A = target / area_shape if np.isfinite(area_shape) and area_shape > 0.0 else 0.0

    return A * shape

def elf_E0(E: np.ndarray, s: IceOpticalSet, include_kshell: bool = True) -> np.ndarray:
    """ELF at q=0 using valence ε plus optional additive K-shell ε2."""
    E = np.asarray(E, float)
    e1v = epsilon1_valence_E0(E, s)  # dict
    e2v = epsilon2_valence_E0(E, s)  # dict
    e1t, e2t = e1v["total"], e2v["total"]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)
    elf_val = e2t / denom
    if include_kshell:
        elf_val = elf_val + epsilon2_Kshell_E0(E, s)
    return elf_val

def _ensure_1d(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x.ravel()

def _lenN(val, N: int) -> np.ndarray:
    """Broadcast scalar/array to length N (for 5 excitations or 4 ionizations)."""
    arr = np.asarray(val, float)
    if arr.ndim == 0:
        return np.full(N, float(arr))
    if arr.size == N:
        return arr.astype(float)
    raise ValueError(f"Expected length {N} or scalar, got shape {arr.shape}")

def _fj_q(fj0: np.ndarray, q: np.ndarray, C: DispersionCoeffs) -> np.ndarray:
    """
    Excitation strengths vs q.
    f_j(q) = f_j * exp(-a_j q^2) + b_j q^2 * exp(-c_j q^2)
    Returns shape (nq, n_exc).
    """
    n_exc = fj0.size
    a = _lenN(C.a_fj, n_exc)
    b = _lenN(C.b_fj, n_exc)
    c = _lenN(C.c_fj, n_exc)

    q2 = q[:, None]**2
    term1 = np.exp(-a[None, :] * q2)
    term2 = (b[None, :] * q2) * np.exp(-c[None, :] * q2)
    return fj0[None, :] * (term1 + term2)

def _renorm_fi_q(fi0: np.ndarray, fj0_sum0: float, fjq_sum: np.ndarray) -> np.ndarray:
    """
    Ionization strengths vs q by f-sum normalization:
      f_i(q) = f_i * (1 - sum_j f_j(q)) / (1 - sum_j f_j(0))
    Returns shape (nq, n_ion).
    """

    scale = (1.0 - fjq_sum) / (1.0 - fj0_sum0)   # shape (nq,)
    return fi0[None, :] * scale[:, None]

def _Ei_q(Ei0: np.ndarray, q: np.ndarray, C: DispersionCoeffs) -> np.ndarray:
    """
    Ionization resonance shift vs q:
      E_i(q) = E_i + [1 - exp(-c_disp * q^d_disp)] * RY * q^2
    Returns shape (nq, n_ion).
    """
    Ei_q = Ei0[None, :] + (RY * (q ** 2.))[:, None] * (1. - np.exp(-C.c_disp * (q ** C.d_disp)))[:, None]
    return Ei_q

def _gamma_q(g0, q, C):
    q = _ensure_1d(q)
    lin = (RY * q)[:, None]            # b1 * (Ry q)
    quad = (RY * (q ** 2))[:, None]    # b2 * (Ry q^2)
    return g0[None, :] + C.b1 * lin + C.b2 * quad

# ============================================================
# Finite-q ε₂ and ε₁ (valence only), channel-resolved
# ============================================================
def epsilon2_valence_Eq(E: np.ndarray, q: ArrayLike, s: IceOpticalSet, C: DispersionCoeffs, partitioned: bool = True) -> dict:
    """
    Imaginary dielectric at finite q, valence only.
    CORRECTED: Applies Kyriakou partitioning dynamically to the q-dispersed
    Drude functions, ensuring redistribution respects the shifted peaks E_i(q).
    """
    E = np.asarray(E, float)
    q = _ensure_1d(q)
    nE, nq = E.size, q.size

    # 1. Pre-calculate dispersed parameters for all q
    # Excitations: f, gamma disperse; E0 fixed
    fj0 = np.array([o.f for o in s.excitations], float)
    Ej0 = np.array([o.E0 for o in s.excitations], float)
    gj0 = np.array([o.gamma for o in s.excitations], float)
    fjq = _fj_q(fj0, q, C)                        # (nq, n_exc)
    gjq = _gamma_q(gj0, q, C)                     # (nq, n_exc)

    # Ionizations: f by renorm, E and gamma disperse
    fi0 = np.array([o.f for o in s.ionizations], float)
    Ei0 = np.array([o.E0 for o in s.ionizations], float)
    gi0 = np.array([o.gamma for o in s.ionizations], float)
    fj0_sum0 = np.sum(fj0)
    fjq_sum = np.sum(fjq, axis=1)                 # (nq,)
    fiq = _renorm_fi_q(fi0, fj0_sum0, fjq_sum)    # (nq, n_ion)
    Eiq = _Ei_q(Ei0, q, C)                        # (nq, n_ion)
    giq = _gamma_q(gi0, q, C)                     # (nq, n_ion)

    # Ionization thresholds are fixed physical constants.
    ion_B = [o.Bth for o in s.ionizations]

    # Containers for results
    # We build them as (n_channels, nq, nE) first for easier slicing
    exc_results = np.zeros((len(s.excitations), nq, nE))
    ion_results = np.zeros((len(s.ionizations), nq, nE))

    # 2. Loop over q to apply partitioning at each step
    for i in range(nq):
        # A. Calculate Raw Drude for this q
        # Excitations (Derivative Drude)
        exc_raw = []
        for j in range(len(s.excitations)):
            y = (s.Ep**2) * _d_drude_e2(E, fjq[i, j], Ej0[j], gjq[i, j])
            exc_raw.append(y)

        # Ionizations (Normal Drude)
        ion_raw = []
        for k in range(len(s.ionizations)):
            y = (s.Ep**2) * _drude_e2(E, fiq[i, k], Eiq[i, k], giq[i, k])
            ion_raw.append(y)

        # B. Apply Partitioning (Dynamic)
        if partitioned:
            # Pass the DISPERSED functions to the algorithm
            # Note: Ej0 is used for excitation step functions (fixed for exc)
            part_exc, part_ion = _apply_partitioning_optical(
                E, exc_raw, Ej0, ion_raw, ion_B, s.Bmin
            )
        else:
            # Fallback: Just apply standard gating if not partitioned
            part_exc = [(y * (E >= s.Bmin).astype(float)) for y in exc_raw]
            part_ion = [(y * (E >= ion_B[k]).astype(float)) for k, y in enumerate(ion_raw)]

        # Store results
        for j, y in enumerate(part_exc):
            exc_results[j, i, :] = y
        for k, y in enumerate(part_ion):
            ion_results[k, i, :] = y

    # 3. Format output as list of (nq, nE) arrays
    exc_list = [exc_results[j] for j in range(len(s.excitations))]
    ion_list = [ion_results[k] for k in range(len(s.ionizations))]

    total = np.zeros((nq, nE), float)
    for y in exc_list:
        total += y
    for y in ion_list:
        total += y

    return {"excitations": exc_list, "ionizations": ion_list, "total": total}


def epsilon1_valence_Eq(E: np.ndarray, q: ArrayLike, s: IceOpticalSet, C: DispersionCoeffs) -> dict:
    """
    Real dielectric at finite q, valence only, channel-resolved.
    No gating; include +1 baseline only in 'total'.
    Returns dict with arrays shaped like epsilon2_valence_Eq.
    """
    E = np.asarray(E, float)
    q = _ensure_1d(q)
    nE, nq = E.size, q.size

    # Params @ q
    fj0 = np.array([o.f for o in s.excitations], float)
    Ej0 = np.array([o.E0 for o in s.excitations], float)
    gj0 = np.array([o.gamma for o in s.excitations], float)
    fjq = _fj_q(fj0, q, C)                        # (nq, n_exc)
    gjq = _gamma_q(gj0, q, C)                     # (nq, n_exc)

    fi0 = np.array([o.f for o in s.ionizations], float)
    Ei0 = np.array([o.E0 for o in s.ionizations], float)
    gi0 = np.array([o.gamma for o in s.ionizations], float)
    fj0_sum0 = np.sum(fj0)
    fjq_sum = np.sum(fjq, axis=1)                 # (nq,)
    fiq = _renorm_fi_q(fi0, fj0_sum0, fjq_sum)    # (nq, n_ion)
    Eiq = _Ei_q(Ei0, q, C)                        # (nq, n_ion)
    giq = _gamma_q(gi0, q, C)                     # (nq, n_ion)

    exc_list = []
    for j in range(fj0.size):
        y = np.empty((nq, nE), float)
        for iq in range(nq):
            y[iq, :] = (s.Ep**2) * _d_drude_e1(E, fjq[iq, j], Ej0[j], gjq[iq, j])
        exc_list.append(y)

    ion_list = []
    for k in range(fi0.size):
        y = np.empty((nq, nE), float)
        for iq in range(nq):
            y[iq, :] = (s.Ep**2) * _drude_e1(E, fiq[iq, k], Eiq[iq, k], giq[iq, k])
        ion_list.append(y)

    total = np.ones((nq, nE), float)
    for y in exc_list:
        total += y
    for y in ion_list:
        total += y

    return {"excitations": exc_list, "ionizations": ion_list, "total": total}


# ============================================================
# Finite-q ELF (valence + optional optical K-shell)
# ============================================================
def elf_Eq(E: np.ndarray, q: ArrayLike, s: IceOpticalSet, C: DispersionCoeffs, include_kshell: bool = True, partitioned: bool = True) -> np.ndarray:
    """
    ELF(E,q) = ε2^(m)/(ε1^(m)^2 + ε2^(m)^2)  +  ε2^(K)(E,0).
    Returns array (nq, nE). K-shell is optical and hard-gated at its edge.

    If partitioned=True, the Kyriakou optical partition is used to define
    energy-dependent correction factors that are applied to the finite-q
    oscillator contributions.
    """
    E = np.asarray(E, float)
    q = _ensure_1d(q)
    e1 = epsilon1_valence_Eq(E, q, s, C)["total"]  # (nq, nE)
    e2 = epsilon2_valence_Eq(E, q, s, C, partitioned=partitioned)["total"]  # (nq, nE)
    denom = e1**2 + e2**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)
    elf = e2 / denom
    if include_kshell:
        ks = epsilon2_Kshell_E0_fsum_corrected(E, s)             # (nE,)
        elf = elf + ks[None, :]
    return elf


# ============================================================
# Plotting helpers for finite-q
# ============================================================
def plot_Im_epsilon_channel_resolved_Eq(
    E: np.ndarray,
    s: IceOpticalSet,
    C: DispersionCoeffs,
    q: float,
    ax=None,
    alpha: float = 0.95,
    linewidth: float = 1.8,
    legend: bool = False,
    also_plot_composite: bool = True,
    partitioned: bool = True,
):
    """Per-channel ε₂ at a single finite q."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)
    res = epsilon2_valence_Eq(E, np.array([q], float), s, C, partitioned=partitioned)
    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2"]
    for j, y in enumerate(res["excitations"]):
        ax.plot(E, y[0], color=exc_colors[j % len(exc_colors)], alpha=alpha, lw=linewidth, label=f"Exc {j+1}")
    for k, y in enumerate(res["ionizations"]):
        ax.plot(E, y[0], color=ion_colors[k % len(ion_colors)], alpha=alpha, lw=linewidth, label=f"Ion {k+1}")
    if also_plot_composite:
        ax.plot(E, res["total"][0], color="k", lw=2.2, label=fr"Total $Im(\epsilon)$ (q={q:.2f})")
    if legend:
        ax.legend(fontsize=10, loc='upper right')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\mathrm{Im}(\epsilon)(E,q)$")

def plot_Re_epsilon_channel_resolved_Eq(
    E: np.ndarray,
    s: IceOpticalSet,
    C: DispersionCoeffs,
    q: float,
    ax=None,
    alpha: float = 0.95,
    linewidth: float = 1.8,
    legend: bool = False,
    also_plot_composite: bool = True,
    include_baseline_one: bool = False,
):
    """Per-channel ε₁ at a single finite q."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)
    res = epsilon1_valence_Eq(E, np.array([q], float), s, C)
    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2"]
    for j, y in enumerate(res["excitations"]):
        ax.plot(E, y[0], color=exc_colors[j % len(exc_colors)], alpha=alpha, lw=linewidth, label=f"Exc {j+1}")
    for k, y in enumerate(res["ionizations"]):
        ax.plot(E, y[0], color=ion_colors[k % len(ion_colors)], alpha=alpha, lw=linewidth, label=f"Ion {k+1}")
    if include_baseline_one:
        ax.axhline(1.0, color="0.6", ls=":", lw=1.5)
    if also_plot_composite:
        ax.plot(E, res["total"][0], color="k", lw=2.2, label=fr"Total $Re(\epsilon)$ (q={q:.2f})")
    if legend:
        ax.legend(fontsize=10, loc='upper right')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\mathrm{Re}(\epsilon)(E,q)$")

def plot_ELF_channel_resolved_Eq(
    E: np.ndarray,
    s: IceOpticalSet,
    C: DispersionCoeffs,
    q: float,
    include_kshell: bool = True,
    partitioned: bool = True,
    ax=None,
    alpha: float = 0.95,
    linewidth: float = 1.8,
    legend: bool = False,
    also_plot_composite: bool = True,
):
    """ELF(E,q) with optional optical K-shell added."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)
    elf = elf_Eq(E, np.array([q], float), s, C, include_kshell=include_kshell, partitioned=partitioned)[0]
    ax.plot(E, elf, color="k", lw=2.2, alpha=alpha, label=fr"ELF (q={q:.2f})")
    if include_kshell:
        ks = epsilon2_Kshell_E0(E, s)
        m = E >= s.kshell.Bth
        if np.any(m):
            ax.plot(E[m], ks[m], color="0.3", ls="--", lw=linewidth, alpha=alpha, label="O K-shell")
    if legend:
        ax.legend(fontsize=10, loc='upper right')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$ELF(E,q)$")

def plot_ELF_totals_multiq(
    E: np.ndarray,
    qvals: np.ndarray,
    s: IceOpticalSet,
    C: DispersionCoeffs,
    include_kshell: bool = True,
    partitioned: bool = True,
    ax=None,
    y_log: bool = False,
    linewidth: float = 1.8,
):
    """Overlay ELF totals for multiple q on one energy axis."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)
    qvals = _ensure_1d(qvals)
    elf = elf_Eq(E, qvals, s, C, include_kshell=include_kshell, partitioned=partitioned)  # (nq, nE)
    for i, q in enumerate(qvals):
        ax.plot(E, elf[i], lw=linewidth, label=fr"q = {q:.2f}")
    if y_log:
        ax.set_yscale("log")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"ELF (E,q)")
    ax.legend(fontsize=10, loc='upper right')


# ============================================================
# Consistency check (q→0) — optional but recommended
# ============================================================
def check_q0_convergence(E: np.ndarray, s: IceOpticalSet, C: DispersionCoeffs,
                         rtol: float = 1e-10, atol: float = 1e-12, partitioned: bool = False) -> None:
    """
    Verifies that ε(E,q→0) reproduces the optical ε(E,0).

    If partitioned=False, uses the unpartitioned optical ε(E,0) as reference.
    If partitioned=True, the q→0 limit of the finite-q model is already built
    from the partitioned optical decomposition, so the strict comparison to
    epsilon*_valence_E0 is skipped.
    """
    if partitioned:
        return
    E = np.asarray(E, float)
    q0 = np.array([0.0], float)
    e1_q0 = epsilon1_valence_Eq(E, q0, s, C)["total"][0]
    e2_q0 = epsilon2_valence_Eq(E, q0, s, C, partitioned=partitioned)["total"][0]
    e1_opt = epsilon1_valence_E0(E, s)["total"]
    e2_opt = epsilon2_valence_E0(E, s)["total"]
    if not np.allclose(e1_q0, e1_opt, rtol=rtol, atol=atol):
        raise AssertionError("ε1(E,q=0) does not match optical ε1.")
    if not np.allclose(e2_q0, e2_opt, rtol=rtol, atol=atol):
        raise AssertionError("ε2(E,q=0) does not match optical ε2.")



def elf_channels_Eq(E, s, C, q, include_kshell=True):
    """Channel-resolved ELF at a single finite q. Returns dict."""
    E = np.asarray(E, float)
    q = float(q)
    e1 = epsilon1_valence_Eq(E, np.array([q], float), s, C)
    e2 = epsilon2_valence_Eq(E, np.array([q], float), s, C)
    e1t = e1["total"][0]
    e2t = e2["total"][0]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)
    exc_elf = [y[0] / denom for y in e2["excitations"]]
    ion_elf = [y[0] / denom for y in e2["ionizations"]]
    val_total = e2t / denom
    ks = epsilon2_Kshell_E0(E, s) if include_kshell else None
    total = val_total + (ks if ks is not None else 0.0)
    return {"excitations": exc_elf, "ionizations": ion_elf, "valence_total": val_total, "kshell": ks, "total": total}


def plot_ELF_channel_resolved_multiq(E, qvals, s, C, include_kshell=True, ncols=2, figsize=None, alpha=0.95, linewidth=1.8):
    """Multi-panel channel-separated and cumulative ELF at each q in qvals."""
    E = np.asarray(E, float)
    qvals = _ensure_1d(qvals)
    nG = qvals.size
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(nG / ncols))
    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)
    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2"]
    legend_handles = None
    legend_labels = None
    # Determine a common y-maximum across all panels from the total ELF curves
    y_max = 0.0
    for q in qvals:
        _res = elf_channels_Eq(E, s, C, float(q), include_kshell=include_kshell)
        if _res["total"].size:
            y_max = max(y_max, float(np.nanmax(_res["total"])))
    y_max = y_max + 0.1
    for i, q in enumerate(qvals):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        res = elf_channels_Eq(E, s, C, float(q), include_kshell=include_kshell)
        for j, y in enumerate(res["excitations"]):
            ax.plot(E, y, color=exc_colors[j % len(exc_colors)], alpha=alpha, lw=linewidth, label=f"Exc {j+1}")
        for k, y in enumerate(res["ionizations"]):
            ax.plot(E, y, color=ion_colors[k % len(ion_colors)], alpha=alpha, lw=linewidth, label=f"Ion {k+1}")
        if res["kshell"] is not None:
            m = E >= s.kshell.Bth
            if np.any(m):
                ax.plot(E[m], res["kshell"][m], color="0.3", ls="--", lw=linewidth, alpha=alpha, label="O K-shell")
        ax.plot(E, res["total"], color="k", lw=2.2, label="Total ELF")
        ax.set_title(fr"q={q:.2f}")
        # Capture legend from the first (top-left) panel only
        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if r == nrows - 1:
            ax.set_xlabel("Energy (eV)")
        if c == 0:
            ax.set_ylabel(r"$ELF(E,q)$")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_ylim(0.0, y_max)
    for j in range(nG, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    if legend_handles and legend_labels:
        fig.subplots_adjust(bottom=0.16)
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(5, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    return fig


def plot_Im_epsilon_channel_resolved_multiq(E, qvals, s, C, ncols=2, figsize=None, alpha=0.95, linewidth=1.8):
    """Multi-panel channel-separated Im(ε) at each q in qvals."""
    E = np.asarray(E, float)
    qvals = _ensure_1d(qvals)
    nG = qvals.size
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(nG / ncols))
    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)
    legend_handles = legend_labels = None
    # Common y-maximum from total Im(ε)
    y_max = 0.0
    for q in qvals:
        _tot = epsilon2_valence_Eq(E, np.array([float(q)], float), s, C)["total"][0]
        if _tot.size:
            y_max = max(y_max, float(np.nanmax(_tot)))
    y_max = y_max + 0.1
    for i, q in enumerate(qvals):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        plot_Im_epsilon_channel_resolved_Eq(E, s, C, q=float(q), ax=ax, legend=False, alpha=alpha, linewidth=linewidth)
        ax.set_title(fr"q={float(q):.2f}")
        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if r == nrows - 1:
            ax.set_xlabel("Energy (eV)")
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Im}(\epsilon)(E,q)$")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_ylim(0.0, y_max)
    for j in range(nG, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    if legend_handles and legend_labels:
        fig.subplots_adjust(bottom=0.16)
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(5, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    return fig


def plot_Re_epsilon_channel_resolved_multiq(E, qvals, s, C, ncols=2, figsize=None, alpha=0.95, linewidth=1.8, include_baseline_one=False):
    """Multi-panel channel-separated Re(ε) at each q in qvals."""
    E = np.asarray(E, float)
    qvals = _ensure_1d(qvals)
    nG = qvals.size
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(nG / ncols))
    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)
    legend_handles = legend_labels = None
    # Common y-maximum from total Re(ε)
    y_max = 0.0
    for q in qvals:
        _tot = epsilon1_valence_Eq(E, np.array([float(q)], float), s, C)["total"][0]
        if _tot.size:
            y_max = max(y_max, float(np.nanmax(_tot)))
    y_max = y_max + 0.1
    for i, q in enumerate(qvals):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        plot_Re_epsilon_channel_resolved_Eq(E, s, C, q=float(q), ax=ax, legend=False, alpha=alpha, linewidth=linewidth, include_baseline_one=include_baseline_one)
        ax.set_title(fr"q={float(q):.2f}")
        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if r == nrows - 1:
            ax.set_xlabel("Energy (eV)")
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Re}(\epsilon)(E,q)$")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_ylim(0.0, y_max)
    for j in range(nG, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    if legend_handles and legend_labels:
        fig.subplots_adjust(bottom=0.16)
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(5, len(legend_labels)), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    return fig


def plot_model_vs_experiment_multiq(
    E: np.ndarray | None,
    qvals: ArrayLike,
    s: IceOpticalSet,
    C: DispersionCoeffs,
    ice: str,
    use_partitioning: bool = True,
    overlay_optical_q0: bool = False,
    savepath: str | Path | None = "output/Model_vs_Experiment_multiq.pdf",
) -> Path | None:
    """
    Multi-panel comparison of model ELF vs. q-resolved tabular data for
    q in qvals (one panel per q). Each integer-q panel overlays its matching
    ELFmodel_*_q#.dat file (q=0 -> q0.dat, q=1 -> q1.dat, etc.) labeled as
    "Experimental data".
    """
    from matplotlib.gridspec import GridSpec
    import pandas as pd

    E = np.asarray(E if E is not None else np.linspace(1.0, 60.0, 20000), float)
    qvals = np.asarray(qvals, float).ravel()
    ice_data_file = ICE_DATA_XLSX_PATH
    sheet_name = "Hexagonal" if ice == "hexagonal" else "Amorphous"
    df_exp = pd.read_excel(ice_data_file, sheet_name=sheet_name)
    mask_e2 = df_exp["eV"].notna() & df_exp["e2"].notna()
    mask_e1 = df_exp["eV.1"].notna() & df_exp["e1"].notna()
    mask_elf = df_exp["eV.2"].notna() & df_exp["ELF"].notna()
    exp_e2_E = df_exp.loc[mask_e2, "eV"].values
    exp_e2 = df_exp.loc[mask_e2, "e2"].values
    exp_e1_E = df_exp.loc[mask_e1, "eV.1"].values
    exp_e1 = df_exp.loc[mask_e1, "e1"].values
    exp_elf_E = df_exp.loc[mask_elf, "eV.2"].values
    exp_elf = df_exp.loc[mask_elf, "ELF"].values

    ncols = 2
    nrows = int(np.ceil(qvals.size / ncols))
    fig = plt.figure(figsize=(7 * ncols, 3.6 * nrows + 1.2))
    gs = GridSpec(nrows + 1, ncols, height_ratios=[*([3.0] * nrows), 0.9], hspace=0.32, wspace=0.25)
    fig.suptitle(f"{ice.capitalize()} ice", y=0.95)
    legend_ax = fig.add_subplot(gs[-1, :])
    legend_ax.axis("off")

    # Preload tabular ELF by integer q for quick lookup
    tabular_by_q: dict[int, np.ndarray] = {}
    for q in qvals:
        q_int = int(round(q))
        if np.isclose(q, q_int):
            tag = "amo" if ice == "amorphous" else ice
            dat_path = PROJECT_ROOT / "tabular" / f"ELFmodel_{tag}_ice_q{q_int}.dat"
            if dat_path.exists():
                try:
                    dat = np.loadtxt(dat_path)
                    if dat.ndim == 2 and dat.shape[1] >= 2:
                        tabular_by_q[q_int] = dat
                except OSError:
                    pass

    handles_all: list = []
    labels_all: list = []
    for i, q in enumerate(qvals):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])

        elf_model = elf_Eq(E, np.array([q], float), s, C, include_kshell=True, partitioned=use_partitioning)[0]
        ln_model = ax.plot(E, elf_model, "k-", linewidth=2, label="Model: ELF", zorder=5)[0]

        # Overlay tabulated ELF model data for this specific integer q (q0->q0.dat, etc.)
        q_int = int(round(q))
        if np.isclose(q, q_int) and q_int in tabular_by_q:
            dat = tabular_by_q[q_int]
            tab_handle = ax.plot(
                dat[:, 0],
                dat[:, 1],
                "s-",
                color="gray",
                linewidth=1.4,
                markersize=6,
                markerfacecolor="gray",
                markeredgecolor="gray",
                markevery=50,  # show a diamond every 8th point (tune as needed)
                label="Experimental data",
                zorder=11,
            )[0]
        else:
            tab_handle = None

        opt_handle = None
        if overlay_optical_q0 and np.isclose(q, 0.0) and exp_elf.size:
            opt_handle = ax.plot(
                exp_elf_E,
                exp_elf,
                color="red",
                linestyle=":",
                linewidth=1.8,
                label="Optical data (q=0)",
                zorder=12,
            )[0]

        ax.set_xlim(0, E[-1])
        if r == nrows - 1:
            ax.set_xlabel("Electron Energy (eV)")
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("ELF")
        else:
            ax.set_ylabel("")
        ax.text(
            0.98,
            0.95,
            rf"q = {q:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

        handles_all.append(ln_model)
        labels_all.append("Model")
        if tab_handle is not None:
            handles_all.append(tab_handle)
            labels_all.append("Experimental data")
        if opt_handle is not None:
            handles_all.append(opt_handle)
            labels_all.append("Optical data (q=0)")

    for j in range(qvals.size, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    unique = []
    seen = set()
    for h, lbl in zip(handles_all, labels_all):
        if lbl not in seen:
            unique.append((h, lbl))
            seen.add(lbl)
    handles_u, labels_u = zip(*unique)
    legend_ax.legend(
        handles_u,
        labels_u,
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
        borderpad=1.5,
        labelspacing=0.8,
    )

    fig.tight_layout()
    out_path = Path(savepath) if savepath is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[saved] {out_path}")
    plt.show()
    return out_path

if __name__ == "__main__":

    # ice = "hexagonal"
    ice = "amorphous"
    s = epsilon_optical(ice)
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])
    C = DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)  # RR2017 defaults for c_disp,d_disp,b1,b2
    E = np.linspace(1.0, 200.0, 20000)
    qvals = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 2.0])  # a0^{-1}
    use_partitioning = True
    # Finite-q: ensure optical limit is recovered when partitioning is off
    check_q0_convergence(E, s, C, partitioned=False)

    # Model vs. experiment: optical data overlaid for q = 0, 1, 2, 3
    qvals_compare = np.array([0.0, 1.0, 2.0, 3.0])
    plot_model_vs_experiment_multiq(
        E,
        qvals_compare,
        s,
        C,
        ice="amorphous",
        use_partitioning=use_partitioning,
        savepath=f"output/Model_vs_Experiment_multiq_amorphous.pdf",
    )
    exit()

    # Pick a q to show channel-resolved finite-q curves
    q_sel = 0.0  # a0^{-1}
    figA, axA = plt.subplots()
    plot_Im_epsilon_channel_resolved_Eq(E, s, C, q=q_sel, legend=True, ax=axA, partitioned=use_partitioning)
    axA.set_title(f"Channel-resolved Im($\\epsilon$) at q = {q_sel:.2f}; {ice} ice")
    plt.savefig(f"output/Channel_resolved_Im_finiteq_{ice}_q{q_sel:.2f}.pdf", bbox_inches="tight")
    plt.show()

    figB, axB = plt.subplots()
    plot_Re_epsilon_channel_resolved_Eq(E, s, C, q=q_sel, legend=True, include_baseline_one=True, ax=axB)
    axB.set_title(f"Channel-resolved Re($\\epsilon$) at q = {q_sel:.2f}; {ice} ice")
    plt.savefig(f"output/Channel_resolved_Re_finiteq_{ice}_q{q_sel:.2f}.pdf", bbox_inches="tight")
    plt.show()

    # ELF multi-panel: channel-resolved per q
    figC = plot_ELF_channel_resolved_multiq(E, qvals, s, C, include_kshell=True, ncols=2)
    figC.suptitle(f"Channel-resolved ELF across q; {ice} ice", y=0.99)
    plt.savefig(f"output/ELF_channel_resolved_multipanel_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # Im(epsilon) multi-panel: channel-resolved per q
    figE = plot_Im_epsilon_channel_resolved_multiq(E, qvals, s, C, ncols=2)
    figE.suptitle(rf"Channel-resolved Im($\epsilon$) across q; {ice} ice", y=0.99)
    plt.savefig(f"output/Im_epsilon_channel_resolved_multipanel_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # Re(epsilon) multi-panel: channel-resolved per q
    figF = plot_Re_epsilon_channel_resolved_multiq(E, qvals, s, C, ncols=2, include_baseline_one=True)
    figF.suptitle(rf"Channel-resolved Re($\epsilon$) across q; {ice} ice", y=0.99)
    plt.savefig(f"output/Re_epsilon_channel_resolved_multipanel_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # Single-q ELF (with K-shell overlay)
    figD, axD = plt.subplots()
    plot_ELF_channel_resolved_Eq(E, s, C, q=q_sel, include_kshell=True, legend=True, ax=axD, partitioned=use_partitioning)
    axD.set_title(f"ELF at q = {q_sel:.2f} with O K-shell; {ice} ice")
    plt.savefig(f"output/ELF_singleq_{ice}_q{q_sel:.2f}.pdf", bbox_inches="tight")
    plt.show()
