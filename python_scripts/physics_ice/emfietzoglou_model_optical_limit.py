"""
Dielectric model for water ice (amorphous, hexagonal) with q-dispersion.

What this implements
- 5 excitation bands with per-band dispersion parameters (a_j, b_j, c_j) applied to f_j(q)
- 4 ionization shells with global dispersion for E_i(q) and gamma_i(q)
- Optional O K-shell kept optical
- Returns \epsilon_1(E,q), \epsilon_2(E,q), and ELF(E,q) = Im[-1/\\epsilon(E,q)]

Units
- Energies in eV
- q in a0^{-1}
"""

from dataclasses import dataclass
from typing import List, Literal, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from constants import (
    EH,
    FONT_COURIER,
    FONTSIZE_18,
    HFONT_COURIER,
    ICE_DATA_XLSX_PATH,
    RC_BASE_STANDARD,
    RY,
    rcparams_with_fontsize,
)

font = FONT_COURIER
hfont = HFONT_COURIER
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'

FONTSIZE = FONTSIZE_18
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_STANDARD, FONTSIZE))

Material = Literal["amorphous", "hexagonal"]
ArrayLike = Union[float, np.ndarray]

# ---- data containers ----
@dataclass(frozen=True)
class Osc:
    E0: float       # resonance energy (eV)
    gamma: float    # damping width (eV)
    f: float        # oscillator strength (dimensionless)
    kind: Literal["ionization", "excitation", "k_shell"]
    Bth: float = 0.0  # threshold (eV), used for ionization channels


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
            Osc(15.40, 5.7, 0.1250, "ionization", Bth=10.0),
            Osc(18.60, 7.1, 0.1300, "ionization", Bth=13.0),
            Osc(24.50, 15.0, 0.1100, "ionization", Bth=17.0),
            Osc(38.00, 30.0, 0.4110, "ionization", Bth=32.2),
        ]
        kshell = Osc(450.0, 360.0, 0.3143, "k_shell", Bth=540.0)
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
            Osc(15.80, 4.6, 0.1000, "ionization", Bth=10.0),
            Osc(18.00, 7.5, 0.2000, "ionization", Bth=13.0),
            Osc(24.50, 14.0, 0.1100, "ionization", Bth=17.0),
            Osc(35.00, 30.0, 0.3580, "ionization", Bth=32.2),
        ]
        kshell = Osc(450.0, 360.0, 0.3143, "k_shell", Bth=540.0)
    else:
        raise ValueError("material must be 'amorphous' or 'hexagonal'")

    return IceOpticalSet(Ep=Ep, Bmin=Bmin, excitations=excit, ionizations=ioniz, kshell=kshell)

# ===== Partitioning / truncation algorithm (Kyriakou et al., MedPhys 2015, Appendix) =====

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

# def _apply_partitioning_optical(E, exc_arrays, exc_Ek, ion_arrays, ion_B, Bmin):
#     """
#     Parameters
#     ----------
#     E : (N,) energy grid (ascending)
#     exc_arrays : list of (N,) arrays, Im[ε_k(E)] for excitations
#     exc_Ek : list of Ek (use oscillator E0 for each excitation)
#     ion_arrays : list of (N,) arrays, Im[ε_n(E)] for ionizations
#     ion_B : list of Bn thresholds for ionizations
#     Bmin : float, minimum valence onset (for gating excitations)
#
#     Returns
#     -------
#     exc_mod, ion_mod : lists of arrays after partitioning
#     """
#     E = np.asarray(E, float)
#     nE = E.size
#
#     IonIm = np.stack(ion_arrays, axis=0) if ion_arrays else np.zeros((0, nE))
#     IonB  = np.array(list(ion_B), dtype=float) if ion_arrays else np.zeros((0,))
#
#     ExcIm = np.stack(exc_arrays, axis=0) if exc_arrays else np.zeros((0, nE))
#     ExcEk = np.array(list(exc_Ek), dtype=float) if exc_arrays else np.zeros((0,))
#
#     n_ion = IonIm.shape[0]
#     n_exc = ExcIm.shape[0]
#
#     # Sort ion shells by ascending B
#     if n_ion > 0:
#         order = np.argsort(IonB)
#         IonIm = IonIm[order]
#         IonB  = IonB[order]
#
#     B1 = IonB[0] if n_ion > 0 else np.inf
#
#     # ---- Ionizations (A1)-(A5) ----
#     IonIm_mod = np.zeros_like(IonIm)
#     for n in range(n_ion):
#         Im_n = IonIm[n]
#         Bn   = IonB[n]
#
#         Cn = _H(E - Bn)
#
#         Im_n_Bn = np.interp(Bn, E, Im_n)
#         Sn = - Im_n_Bn * np.exp(Bn - E)
#
#         C_greater = np.zeros_like(E)
#         S_greater = np.zeros_like(E)
#
#         for j in range(n+1, n_ion):
#             Im_j = IonIm[j]; Bj = IonB[j]
#             den = np.zeros_like(E)
#             for i in range(0, j):
#                 den += IonIm[i] * _H(E - IonB[i])
#
#             C_term = Im_j * _Theta(Bj - E) * _safe_div(Im_n, den)
#             Im_j_Bj = np.interp(Bj, E, Im_j)
#             S_term = Im_j_Bj * np.exp(Bj - E) * _H(E - Bj) * _safe_div(Im_n, den)
#
#             C_greater += C_term
#             S_greater += S_term
#
#         IonIm_mod[n] = (Im_n + Sn + S_greater + C_greater) * Cn
#
#     # ---- Excitations (A6)-(A9) ----
#     ExcIm_mod = np.zeros_like(ExcIm)
#
#     B_gate_exc = Bmin
#
#     if n_exc > 0:
#         Theta_B_gate_minus_Ek = np.array([1.0 if (B_gate_exc - Ek) >= 0 else 0.0 for Ek in ExcEk], float) if np.isfinite(B_gate_exc) else np.zeros((n_exc,), float)
#         Exc_den_Cn = np.sum(ExcIm * Theta_B_gate_minus_Ek[:, None], axis=0) if n_exc > 0 else np.zeros_like(E)
#         Exc_den_Sn1 = np.sum(ExcIm, axis=0)
#
#         Im_n1_B_gate = np.interp(B_gate_exc, E, IonIm[0]) if n_ion > 0 else 0.0
#
#         Ion_sum = np.sum(IonIm, axis=0) if n_ion > 0 else np.zeros_like(E)
#
#         for k in range(n_exc):
#             Im_k = ExcIm[k]; Ek = ExcEk[k]
#
#             # Gate excitations with Bmin instead of B1 so features below the first
#             # ionization threshold (down to Bmin) are retained after partitioning.
#             Ck = _Theta(E - B_gate_exc) if np.isfinite(B_gate_exc) else np.ones_like(E)
#
#             Sn1_weight = _safe_div(Im_k, Exc_den_Sn1) if n_exc > 0 else np.zeros_like(E)
#             Sn1 = Im_n1_B_gate * np.exp(B_gate_exc - E) * _H(E - B_gate_exc) * Sn1_weight if np.isfinite(B_gate_exc) else np.zeros_like(E)
#
#             top_weight = Im_k * (1.0 if (B_gate_exc - Ek) >= 0 else 0.0)
#             weight = _safe_div(top_weight, Exc_den_Cn) if n_exc > 0 else np.zeros_like(E)
#             Cn_term = Ion_sum * _Theta(B_gate_exc - E) * weight if np.isfinite(B_gate_exc) else np.zeros_like(E)
#
#             ExcIm_mod[k] = (Im_k + Sn1 + Cn_term) * Ck
#
#     # Convert back to lists in original ion order
#     exc_mod_list = [ExcIm_mod[i] if n_exc>0 else np.zeros_like(E) for i in range(n_exc)]
#     ion_mod_list = [IonIm_mod[i] if n_ion>0 else np.zeros_like(E) for i in range(n_ion)]
#
#     # Conservation check (pointwise)
#     orig = (np.sum(IonIm, axis=0) if n_ion>0 else 0.0) + (np.sum(ExcIm, axis=0) if n_exc>0 else 0.0)
#     mod  = (np.sum(IonIm_mod, axis=0) if n_ion>0 else 0.0) + (np.sum(ExcIm_mod, axis=0) if n_exc>0 else 0.0)
#     if not np.allclose(orig, mod, rtol=1e-10, atol=1e-10):
#         # Don't crash production runs; warn via print
#         _max_abs = float(np.max(np.abs(orig - mod)))
#         print(f"[partition] WARNING: conservation drift max_abs={_max_abs:.3e}")
#
#     return exc_mod_list, ion_mod_list

def _apply_partitioning_optical(E, exc_arrays, exc_Ek, ion_arrays, ion_B, Bmin):
    """
    Parameters
    ----------
    E : (N,) energy grid (ascending)
    exc_arrays : list of (N,) arrays, Im[ε_k(E)] for excitations
    exc_Ek : list of Ek (oscillator energies for each excitation)
    ion_arrays : list of (N,) arrays, Im[ε_n(E)] for ionizations
    ion_B : list of Bn thresholds for ionizations
    Bmin : float, minimum valence onset (used as B_gap for gating excitations)

    Returns
    -------
    exc_mod, ion_mod : lists of arrays after partitioning
    """
    E = np.asarray(E, float)
    nE = E.size

    # Stack inputs
    IonIm = np.stack(ion_arrays, axis=0) if ion_arrays else np.zeros((0, nE))
    IonB  = np.array(list(ion_B), dtype=float) if ion_arrays else np.zeros((0,))

    ExcIm = np.stack(exc_arrays, axis=0) if exc_arrays else np.zeros((0, nE))
    ExcEk = np.array(list(exc_Ek), dtype=float) if exc_arrays else np.zeros((0,))

    n_ion = IonIm.shape[0]
    n_exc = ExcIm.shape[0]

    # Sort ion shells by ascending B
    if n_ion > 0:
        order = np.argsort(IonB)
        IonIm = IonIm[order]
        IonB  = IonB[order]

    B1 = IonB[0] if n_ion > 0 else np.inf   # minimum binding energy

    # ------------------------------------------------------------------
    # Ionizations: Eqs. (A1)–(A5) Kyriakou et al.
    # ------------------------------------------------------------------
    IonIm_mod = np.zeros_like(IonIm)

    for n in range(n_ion):
        Im_n = IonIm[n]
        Bn   = IonB[n]

        # Cn = H(E - Bn)
        Cn = _H(E - Bn)

        # Sn = - Im_n(Bn) * exp(Bn - E)
        Im_n_Bn = np.interp(Bn, E, Im_n, left=0.0, right=0.0)
        Sn = - Im_n_Bn * np.exp(Bn - E)

        C_greater = np.zeros_like(E)
        S_greater = np.zeros_like(E)

        for j in range(n + 1, n_ion):
            Im_j = IonIm[j]
            Bj   = IonB[j]

            # denominator: sum_{i=1..j-1} Im_i(E) H(E - Bi)
            den = np.zeros_like(E)
            for i in range(0, j):
                den += IonIm[i] * _H(E - IonB[i])

            # C>n term: Im_j(E) Θ(Bj - E) * Im_n(E) / den
            C_term = Im_j * _Theta(Bj - E) * _safe_div(Im_n, den)

            # S>n term: Im_j(Bj) exp(Bj - E) H(E - Bj) * Im_n(E) / den
            Im_j_Bj = np.interp(Bj, E, Im_j, left=0.0, right=0.0)
            S_term  = Im_j_Bj * np.exp(Bj - E) * _H(E - Bj) * _safe_div(Im_n, den)

            C_greater += C_term
            S_greater += S_term

        IonIm_mod[n] = (Im_n + Sn + S_greater + C_greater) * Cn

    # ------------------------------------------------------------------
    # Excitations: Eqs. (A6)–(A9) Kyriakou et al.
    # Here Bmin is used as B_gap in Ck, while B1 is used in Sn=1 and Cn.
    # ------------------------------------------------------------------
    ExcIm_mod = np.zeros_like(ExcIm)
    B_gap = Bmin

    if n_exc > 0:
        # Denominator for Cn: sum_j Im_j(E) Θ(B1 - Ej)
        if np.isfinite(B1):
            Theta_B1_minus_Ek = np.array(
                [1.0 if (B1 - Ek) >= 0.0 else 0.0 for Ek in ExcEk],
                float
            )
        else:
            Theta_B1_minus_Ek = np.zeros((n_exc,), float)

        Exc_den_Cn = np.sum(ExcIm * Theta_B1_minus_Ek[:, None], axis=0) if n_exc > 0 else np.zeros_like(E)

        # Denominator for Sn=1: sum_j Im_j(E)
        Exc_den_Sn1 = np.sum(ExcIm, axis=0)

        # Ionization contributions
        if n_ion > 0 and np.isfinite(B1):
            # Im_{n=1}(B1)
            Im_n1_B1 = np.interp(B1, E, IonIm[0], left=0.0, right=0.0)
        else:
            Im_n1_B1 = 0.0

        if n_ion > 0:
            Ion_sum = np.sum(IonIm, axis=0)
        else:
            Ion_sum = np.zeros_like(E)

        # Θ(B1 - E), used in Cn term
        Theta_B1_minus_E = _Theta(B1 - E) if np.isfinite(B1) else np.zeros_like(E)

        for k in range(n_exc):
            Im_k = ExcIm[k]
            Ek   = ExcEk[k]

            # Ck = Θ(E - B_gap), excitation gate
            if np.isfinite(B_gap):
                Ck = _Theta(E - B_gap)
            else:
                Ck = np.ones_like(E)

            # Sn=1 term: Im_{n=1}(B1) exp(B1 - E) H(E - B1) * Im_k(E) / sum_j Im_j(E)
            if n_ion > 0 and np.isfinite(B1):
                Sn1_weight = _safe_div(Im_k, Exc_den_Sn1)
                Sn1 = Im_n1_B1 * np.exp(B1 - E) * _H(E - B1) * Sn1_weight
            else:
                Sn1 = np.zeros_like(E)

            # Cn term: redistribution of ionization strength below B1 to excitations with Ek <= B1
            if np.isfinite(B1) and n_ion > 0:
                top_weight_k = Im_k * (1.0 if (B1 - Ek) >= 0.0 else 0.0)
                weight_k     = _safe_div(top_weight_k, Exc_den_Cn)
                Cn_term      = Ion_sum * Theta_B1_minus_E * weight_k
            else:
                Cn_term = np.zeros_like(E)

            # Im_k(mod)(E) = {Im_k + Sn1 + Cn} * Ck
            ExcIm_mod[k] = (Im_k + Sn1 + Cn_term) * Ck

    # ------------------------------------------------------------------
    # Convert back to lists in original ion order
    # ------------------------------------------------------------------
    exc_mod_list = [ExcIm_mod[i] if n_exc > 0 else np.zeros_like(E) for i in range(n_exc)]
    ion_mod_list = [IonIm_mod[i] if n_ion > 0 else np.zeros_like(E) for i in range(n_ion)]

    # Conservation check (pointwise)
    orig = (np.sum(IonIm, axis=0) if n_ion > 0 else 0.0) + (np.sum(ExcIm, axis=0) if n_exc > 0 else 0.0)
    mod  = (np.sum(IonIm_mod, axis=0) if n_ion > 0 else 0.0) + (np.sum(ExcIm_mod, axis=0) if n_exc > 0 else 0.0)

    if not np.allclose(orig, mod, rtol=1e-10, atol=1e-10):
        max_abs = float(np.max(np.abs(orig - mod)))
        print(f"[partition] WARNING: conservation drift max_abs={max_abs:.3e}")

    return exc_mod_list, ion_mod_list

# =====================================================================
# q = 0: VALENCE-ONLY ε2, ε1, and separate K-shell ε2
# =====================================================================

def epsilon2_valence_E0(E: np.ndarray, s: IceOpticalSet, partitioned: bool = False) -> dict:
    """
    Imaginary dielectric, optical limit (q=0), valence only.
    Returns a dict with per-channel arrays and the total:
      {
        "excitations": [array(...), ...],
        "ionizations": [array(...), ...],
        "total": array(...)
      }
    If partitioned=True, applies the Kyriakou et al. (2015) redistribution algorithm.
    """
    E = np.asarray(E, float)

    if partitioned:
        # 1. Generate raw, ungated arrays for the partitioning algorithm
        # Excitations use derivative Drude (_d_drude_e2)
        exc_arrays = [(s.Ep**2) * _d_drude_e2(E, o.f, o.E0, o.gamma) for o in s.excitations]
        exc_Ek = [o.E0 for o in s.excitations]

        # Ionizations use normal Drude (_drude_e2) - NO threshold gating here
        ion_arrays = [(s.Ep**2) * _drude_e2(E, o.f, o.E0, o.gamma) for o in s.ionizations]
        ion_B = [o.Bth for o in s.ionizations]

        # 2. Apply the algorithm
        exc_mod, ion_mod = _apply_partitioning_optical(E, exc_arrays, exc_Ek, ion_arrays, ion_B, s.Bmin)

        # 3. Sum up
        total = np.zeros_like(E)
        for y in exc_mod:
            total += y
        for y in ion_mod:
            total += y

        return {"excitations": exc_mod, "ionizations": ion_mod, "total": total}

    else:
        # Original Logic: Gating (optical): excitations above Bmin; ionizations above individual Bth
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
    Note: The partitioning algorithm conserves the total Im[eps], so the total Re[eps]
    calculated here using the original Drude forms remains valid for the total.
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

    # Cumulative trapezoid to avoid warnings; last value is the area
    dE     = np.diff(E)
    midE   = 0.5 * (E[1:] + E[:-1])
    area_shape = np.sum(0.5 * (shape[1:] + shape[:-1]) * dE * midE)
    A = target / area_shape if np.isfinite(area_shape) and area_shape > 0.0 else 0.0

    return A * shape

def elf_E0(E: np.ndarray, s: IceOpticalSet, include_kshell: bool = True, partitioned: bool = False) -> np.ndarray:
    """ELF at q=0 using valence ε plus optional additive K-shell ε2."""
    E = np.asarray(E, float)
    e1v = epsilon1_valence_E0(E, s)  # dict
    e2v = epsilon2_valence_E0(E, s, partitioned=partitioned)  # dict
    e1t, e2t = e1v["total"], e2v["total"]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)
    elf_val = e2t / denom
    if include_kshell:
        # elf_val = elf_val + epsilon2_Kshell_E0(E, s)
        elf_val = elf_val + epsilon2_Kshell_E0_fsum_corrected(E, s)
    return elf_val

def plot_Im_epsilon_channel_resolved(
    E: np.ndarray,
    s: IceOpticalSet,
    C: DispersionCoeffs,  # unused; kept for API compatibility
    q: float = 0.0,       # unused; always optical (q=0)
    include_kshell: bool = False,  # unused here
    ax=None,
    alpha: float = 0.95,
    linewidth: float = 1.8,
    legend: bool = False,
    also_plot_composite: bool = False,
    partitioned: bool = False
):
    """Overlay per-channel ε₂ at q=0 (valence only)."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)

    res = epsilon2_valence_E0(E, s, partitioned=partitioned)

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2"]

    for j, y in enumerate(res["excitations"]):
        ax.plot(E, y, color=exc_colors[j % len(exc_colors)], alpha=alpha, lw=linewidth, label=f"Exc {j+1}")
    for k, y in enumerate(res["ionizations"]):
        ax.plot(E, y, color=ion_colors[k % len(ion_colors)], alpha=alpha, lw=linewidth, label=f"Ion {k+1}")

    if also_plot_composite:
        ax.plot(E, res["total"], color="k", lw=2.2, label=r"Total $\epsilon_2$ (valence, q=0)")

    if legend:
        ax.legend(loc='upper right')

    ax.set_xlabel("Energy Transfer ($E$; eV)")
    ax.set_ylabel(r"$\epsilon_2(E, q{=}0)$")

def plot_Re_epsilon_channel_resolved(
    E: np.ndarray,
    s: IceOpticalSet,
    C: DispersionCoeffs,  # unused; kept for API compatibility
    q: float = 0.0,       # unused; always optical (q=0)
    include_kshell: bool = False,  # unused here
    ax=None,
    alpha: float = 0.95,
    linewidth: float = 1.8,
    legend: bool = False,
    also_plot_composite: bool = False,
    include_baseline_one: bool = False,
    partitioned: bool = False  # Added for compatibility, though Re is not partitioned per se
):
    """Overlay per-channel ε₁ at q=0 (valence only)."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)

    res = epsilon1_valence_E0(E, s)

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2"]

    for j, y in enumerate(res["excitations"]):
        ax.plot(E, y, color=exc_colors[j % len(exc_colors)], alpha=alpha, lw=linewidth, label=f"Exc {j+1}")
    for k, y in enumerate(res["ionizations"]):
        ax.plot(E, y, color=ion_colors[k % len(ion_colors)], alpha=alpha, lw=linewidth, label=f"Ion {k+1}")

    if also_plot_composite:
        ax.plot(E, res["total"], color="k", lw=2.2, label=r"Total $\epsilon_1$ (valence, q=0)")

    if legend:
        ax.legend(loc='upper right')

    ax.set_xlabel("Energy Transfer ($E$; eV)")
    ax.set_ylabel(r"$\epsilon_1(E, q{=}0)$")

def plot_ELF_channel_resolved(
    E: np.ndarray,
    s: IceOpticalSet,
    C: DispersionCoeffs,  # unused; kept for API compatibility
    q: float = 0.0,       # unused; always optical (q=0)
    include_kshell: bool = True,
    ax=None,
    alpha: float = 0.95,
    linewidth: float = 1.8,
    also_plot_composite: bool = True,
    legend: bool = True,
    partitioned: bool = False
):
    """Channel-resolved ELF at q=0 from channel-resolved ε1, ε2."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)

    # Channel-resolved ε at q=0 (must already be implemented as dicts)
    e1 = epsilon1_valence_E0(E, s)     # {"excitations":[...], "ionizations":[...], "total":...}
    e2 = epsilon2_valence_E0(E, s, partitioned=partitioned)     # {"excitations":[...], "ionizations":[...], "total":...}

    denom = e1["total"]**2 + e2["total"]**2
    if np.any(denom <= 0):
        raise ValueError("Non-positive |ε|^2 encountered; check ε construction.")

    # Per-channel ELF contributions: each valence channel divided by |ε^(m)|^2
    exc_elf = [y / denom for y in e2["excitations"]]
    ion_elf = [y / denom for y in e2["ionizations"]]

    # K-shell (optical) added only to ELF, not to ε
    kshell = epsilon2_Kshell_E0(E, s) if include_kshell else None

    # Totals
    elf_valence = e2["total"] / denom
    elf_total = elf_valence + (kshell if kshell is not None else 0.0)

    # Plotting
    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2"]

    for j, y in enumerate(exc_elf):
        ax.plot(E, y, color=exc_colors[j % len(exc_colors)], alpha=alpha, lw=linewidth, label=f"Exc {j+1}")
    for k, y in enumerate(ion_elf):
        ax.plot(E, y, color=ion_colors[k % len(ion_colors)], alpha=alpha, lw=linewidth, label=f"Ion {k+1}")

    if kshell is not None:
        ax.plot(E, kshell, color="0.3", ls="--", lw=linewidth, alpha=alpha, label="O K-shell")

    if also_plot_composite:
        ax.plot(E, elf_total, color="k", lw=2.2, label=r"Total ELF (q=0)")

    if legend:
        ax.legend(loc='upper right')

    ax.set_xlabel("Energy Transfer ($E$; eV)")
    ax.set_ylabel(r"$\mathrm{ELF}=\mathrm{Im}\!\left[\frac{1}{\epsilon}\right]$ at $q=0$")

def plot_Kshell_channel_resolved(
    E: np.ndarray,
    s: IceOpticalSet,
    include_kshell: bool = True,
    ax=None,
    alpha: float = 0.95,
    linewidth: float = 1.8,
    legend: bool = False,
    also_plot_composite: bool = False,  # kept for API symmetry; no effect here
):
    """
    Plot only the optical K-shell ε₂ curve with the same interface and style
    as the other plot_*_channel_resolved functions.

    Notes:
      - Uses epsilon2_Kshell_E0(E, s) which must be HARD-gated at s.kshell.Bth.
      - This curve is NOT included in ε plots; it is a reference for ELF.
    """
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)

    if not include_kshell:
        # Keep interface behavior consistent: do nothing but keep axes labeled.
        ax.set_xlabel("Energy Transfer ($E$; eV)")
        ax.set_ylabel(r"$\epsilon_2(E, q{=}0)$")
        if legend:
            ax.legend(loc='upper right')
        return

    y = epsilon2_Kshell_E0(E, s)  # should be gated at Bth internally

    # Same visual language as others
    ax.plot(E, y, color="0.3", ls="--", lw=linewidth, alpha=alpha, label="O K-shell")

    if legend:
        ax.legend(loc='upper right')

    ax.set_xlabel("Energy Transfer ($E$; eV)")
    ax.set_ylabel(r"$\epsilon_2(E, q{=}0)$")

def plot_neff_and_I(E_min=0.1, E_max=1.0e6, npts=50000, savepath=None, partitioned=False):
    r"""
    Plot N_eff(E_max) and I(E_max) at q=0 for amorphous and hexagonal ice,
    reproducing Fig. 3: valence vs total (valence + K-shell).

    Definitions:
      N_eff(E_max) = (2 / (π Ep^2)) * ∫_0^{E_max} E * Im[1/ε](E,0) dE
      ln I(E_max)  = [∫_0^{E_max} E ln(E) Im[1/ε](E,0) dE] / [∫_0^{E_max} E Im[1/ε](E,0) dE]
    """

    def _cumtrapz(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = np.zeros_like(x)
        dx  = np.diff(x)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
        return out

    # energy grid
    E = np.logspace(np.log10(E_min), np.log10(E_max), int(npts))

    # get parameter sets
    sets = {
        "amorphous": epsilon_optical("amorphous"),
        "hexagonal": epsilon_optical("hexagonal"),
    }

    results = {}
    for name, s in sets.items():
        # Valence ELF from the model (q=0), WITHOUT K shell
        elf_val = elf_E0(E, s, include_kshell=False, partitioned=partitioned)

        # K-shell ε2 added directly to Im[1/ε] per Eq. (7)
        e2k = epsilon2_Kshell_E0_fsum_corrected(E, s)

        # Total optical ELF
        elf_tot = elf_val + e2k

        # Running integrals
        S_val = _cumtrapz(E, E * elf_val)
        S_tot = _cumtrapz(E, E * elf_tot)

        # N_eff(E_max)
        neff_val = (2.0 / (np.pi * s.Ep**2)) * S_val
        neff_tot = (2.0 / (np.pi * s.Ep**2)) * S_tot

        # I(E_max)
        num_val = _cumtrapz(E, E * np.log(E) * elf_val)
        num_tot = _cumtrapz(E, E * np.log(E) * elf_tot)

        # Safe ratios; undefined at index 0 where S=0
        with np.errstate(divide="ignore", invalid="ignore"):
            logI_val = np.where(S_val > 0.0, num_val / S_val, np.nan)
            logI_tot = np.where(S_tot > 0.0, num_tot / S_tot, np.nan)
        I_val = np.exp(logI_val)
        I_tot = np.exp(logI_tot)

        results[name] = dict(E=E, neff_val=neff_val, neff_tot=neff_tot,
                             I_val=I_val, I_tot=I_tot, s=s)

    # Report maximum Neff values for each material
    for name, res in results.items():
        neff_val_max = float(np.nanmax(res["neff_val"]))
        neff_tot_max = float(np.nanmax(res["neff_tot"]))
        print(f"[Neff] {name}: max valence = {neff_val_max:.4f}, max total = {neff_tot_max:.4f}")

    # Plot with legend bars underneath each panel
    from matplotlib.gridspec import GridSpec
    # Use a taller figure and relatively larger legend row to avoid overlap
    fig = plt.figure(figsize=(10.5, 6.0))
    gs = GridSpec(2, 2, height_ratios=[3.5, 2.0], hspace=0.25, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel (a): N_eff
    ax1.set_xscale("log")
    ax1.plot(results["amorphous"]["E"], results["amorphous"]["neff_val"], lw=1.9, label="amorphous — valence")
    ax1.plot(results["amorphous"]["E"], results["amorphous"]["neff_tot"], lw=1.9, ls="--", label="amorphous — total")
    ax1.plot(results["hexagonal"]["E"], results["hexagonal"]["neff_val"], lw=1.9, label="hexagonal — valence")
    ax1.plot(results["hexagonal"]["E"], results["hexagonal"]["neff_tot"], lw=1.9, ls="--", label="hexagonal — total")
    # mark the K edge (use amorphous Bth)
    ks = getattr(sets["amorphous"], "kshell", None)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"$E_{\max}$ (eV)")
    ax1.set_ylabel(r"$N_{\mathrm{eff}}$")
    ax1.set_title(r"$N_{\mathrm{eff}}$ vs $E_{\max}$")

    # Legend bar for panel (a) — stack entries vertically to avoid any overlap
    ax1_leg = fig.add_subplot(gs[1, 0])
    ax1_leg.axis("off")
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1_leg.legend(handles1, labels1, loc="center left", ncol=1, frameon=False)

    # Panel (b): I(E_max)
    ax2.set_xscale("log")
    ax2.plot(results["amorphous"]["E"], results["amorphous"]["I_val"], lw=1.9, label="amorphous — valence")
    ax2.plot(results["amorphous"]["E"], results["amorphous"]["I_tot"], lw=1.9, ls="--", label="amorphous — total")
    ax2.plot(results["hexagonal"]["E"], results["hexagonal"]["I_val"], lw=1.9, label="hexagonal — valence")
    ax2.plot(results["hexagonal"]["E"], results["hexagonal"]["I_tot"], lw=1.9, ls="--", label="hexagonal — total")
    ax2.set_xlabel(r"$E_{\max}$ (eV)")
    ax2.set_ylabel("I-value (eV)")
    ax2.set_title(r"$I(E_{\max})$")

    # Legend bar for panel (b) — stack entries vertically to avoid any overlap
    ax2_leg = fig.add_subplot(gs[1, 1])
    ax2_leg.axis("off")
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2_leg.legend(handles2, labels2, loc="center left", ncol=1, frameon=False)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()

    return results

def plot_model_vs_experiment_two_panel(use_partitioning: bool = True, savepath: str | Path | None = "output/Model_vs_Experiment_both.pdf") -> Path:
    """
    Create a two-panel horizontal comparison (amorphous, hexagonal) of model vs
    experimental dielectric properties (Im eps, Re eps, ELF) at q=0. Legend is
    centered below both panels. Formatting matches the single-panel plots; grid
    is omitted.
    """
    from matplotlib.gridspec import GridSpec

    ice_data_file = ICE_DATA_XLSX_PATH
    E = np.linspace(1.0, 60.0, 20000)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, height_ratios=[3.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    handles_all = []
    labels_all = []

    for ax, ice in zip(axes, ["amorphous", "hexagonal"]):
        s = epsilon_optical(ice)

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

        e1_dict = epsilon1_valence_E0(E, s)
        e2_dict = epsilon2_valence_E0(E, s, partitioned=use_partitioning)
        e1_model = e1_dict["total"]
        e2_model = e2_dict["total"] + epsilon2_Kshell_E0_fsum_corrected(E, s)
        elf_model = elf_E0(E, s, include_kshell=True, partitioned=use_partitioning)

        # Model predictions
        ln1 = ax.plot(E, e2_model, "k-", linewidth=2, label=r"Model: Im($\epsilon$)", zorder=5)[0]
        ln2 = ax.plot(E, e1_model, "k--", linewidth=2, label=r"Model: Re($\epsilon$)", zorder=5)[0]
        ln3 = ax.plot(E, elf_model, "k-.", linewidth=2, label="Model: ELF", zorder=5)[0]

        # Experimental
        ln4 = ax.plot(exp_e2_E, exp_e2, "d-", color="darkgray", linewidth=1.5, markersize=5,
                      markerfacecolor="darkgray", markeredgecolor="darkgray",
                      label=r"Exp: Im($\epsilon$)", zorder=10)[0]
        ln5 = ax.plot(exp_e1_E, exp_e1, "s-", color="darkgray", linewidth=1.5, markersize=5,
                      markerfacecolor="darkgray", markeredgecolor="darkgray",
                      label=r"Exp: Re($\epsilon$)", zorder=10)[0]
        ln6 = ax.plot(exp_elf_E, exp_elf, "^-", color="darkgray", linewidth=1.5, markersize=5,
                      markerfacecolor="darkgray", markeredgecolor="darkgray",
                      label="Exp: ELF", zorder=10)[0]

        ax.set_xlim(0, 30)
        ax.set_ylim(0, 2.9)
        ax.set_xlabel("Energy transfer ($E$; eV)")
        if ice == "amorphous":
            ax.set_ylabel("Dielectric properties")
        else:
            ax.set_ylabel("")
        ax.set_title(f"{ice.capitalize()} ice")
        # No grid as requested

        handles_all.extend([ln1, ln2, ln3, ln4, ln5, ln6])
        labels_all.extend([
            r"Model: Im($\epsilon$)", r"Model: Re($\epsilon$)", "Model: ELF",
            r"Exp: Im($\epsilon$)", r"Exp: Re($\epsilon$)", "Exp: ELF"
        ])

    # Legend centered below panels
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
    E = np.linspace(1.0, 60.0, 20000)
    qvals = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 2.0])  # a0^{-1}

    # Enable Partitioning for corrected plots
    use_partitioning = True

    # Valence ε and ELF(+K) at multiple q
    e1q_val = epsilon1_valence_E0(E, s)
    e2q_val = epsilon2_valence_E0(E, s, partitioned=use_partitioning)

    q = 0.0

    # Load experimental data from ice data.xlsx

    ice_data_file = Path(__file__).parent.parent / 'tabular' / 'ice data.xlsx'
    sheet_name = 'Hexagonal' if ice == "hexagonal" else 'Amorphous'
    df_exp = pd.read_excel(ice_data_file, sheet_name=sheet_name)
    # Extract experimental data with valid values
    mask_e2 = df_exp['eV'].notna() & df_exp['e2'].notna()
    mask_e1 = df_exp['eV.1'].notna() & df_exp['e1'].notna()
    mask_elf = df_exp['eV.2'].notna() & df_exp['ELF'].notna()
    exp_e2_E = df_exp.loc[mask_e2, 'eV'].values
    exp_e2 = df_exp.loc[mask_e2, 'e2'].values
    exp_e1_E = df_exp.loc[mask_e1, 'eV.1'].values
    exp_e1 = df_exp.loc[mask_e1, 'e1'].values
    exp_elf_E = df_exp.loc[mask_elf, 'eV.2'].values
    exp_elf = df_exp.loc[mask_elf, 'ELF'].values

    from matplotlib.gridspec import GridSpec

    # Separate figure: channel-resolved Im(epsilon) at desired q, with legend bar underneath
    fig3 = plt.figure(figsize=(8.0, 7.5))
    gs3 = GridSpec(2, 1, height_ratios=[3.0, 2.0], hspace=0.30)
    ax3 = fig3.add_subplot(gs3[0, 0])
    plot_Im_epsilon_channel_resolved(E, s, C, q=q, include_kshell=True, legend=False,
                                     also_plot_composite=True, ax=ax3, partitioned=use_partitioning)
    # Add experimental data
    ax3.plot(exp_e2_E, exp_e2, 'd-', color='lightgray', linewidth=1.5, markersize=5,
             label='Experimental data', zorder=10)
    ax3.set_xlabel("Energy Transfer ($E$; eV)")
    ax3.set_ylabel(r"$\operatorname{Im}(\epsilon)$")
    ax3.set_title(f"Channel-resolved Im($\\epsilon$) at q = {q}; {ice} ice")
    # Legend bar below main plot (multi-row/column, fully separated from panel)
    ax3_leg = fig3.add_subplot(gs3[1, 0])
    ax3_leg.axis('off')
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax3_leg.legend(handles3, labels3, loc='center', ncol=3, frameon=False)
    plt.savefig(f"output/Channel_resolved_Im_optical_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # Separate figure: channel-resolved Re(epsilon) at desired q, with legend bar underneath
    fig4 = plt.figure(figsize=(8.0, 7.5))
    gs4 = GridSpec(2, 1, height_ratios=[3.0, 2.0], hspace=0.30)
    ax4 = fig4.add_subplot(gs4[0, 0])
    plot_Re_epsilon_channel_resolved(E, s, C, q=q, include_kshell=True, legend=False,
                                     also_plot_composite=True, ax=ax4, partitioned=use_partitioning)
    # Add experimental data
    ax4.plot(exp_e1_E, exp_e1, 'd-', color='lightgray', linewidth=1.5, markersize=5,
             label='Experimental data', zorder=10)
    ax4.set_xlabel("Energy Transfer ($E$; eV)")
    ax4.set_ylabel(r"$\operatorname{Re}(\epsilon)$")
    ax4.set_title(f"Channel-resolved Re($\\epsilon$) at q = {q}; {ice} ice")
    # Legend bar below main plot (multi-row/column, fully separated from panel)
    ax4_leg = fig4.add_subplot(gs4[1, 0])
    ax4_leg.axis('off')
    handles4, labels4 = ax4.get_legend_handles_labels()
    ax4_leg.legend(handles4, labels4, loc='center', ncol=3, frameon=False)
    plt.savefig(f"output/Channel_resolved_Re_optical_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # Separate figure: channel-resolved ELF at desired q (optical limit), with legend bar underneath
    fig5 = plt.figure(figsize=(8.0, 7.5))
    gs5 = GridSpec(2, 1, height_ratios=[3.0, 2.0], hspace=0.30)
    ax5 = fig5.add_subplot(gs5[0, 0])
    plot_ELF_channel_resolved(E, s, C, q=q, include_kshell=True,
                              legend=False, also_plot_composite=True, ax=ax5, partitioned=use_partitioning)
    # Add experimental data
    ax5.plot(exp_elf_E, exp_elf, 'd-', color='lightgray', linewidth=1.5, markersize=5,
             label='Experimental data', zorder=10)
    ax5.set_title(f"Channel-resolved ELF at q = {q}; {ice} ice")
    # Legend bar below main plot (multi-row/column, fully separated from panel)
    ax5_leg = fig5.add_subplot(gs5[1, 0])
    ax5_leg.axis('off')
    handles5, labels5 = ax5.get_legend_handles_labels()
    ax5_leg.legend(handles5, labels5, loc='center', ncol=3, frameon=False)
    plt.savefig(f"output/Channel_resolved_ELF_optical_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # K-shell only, same look-and-feel; window 400–600 eV
    E_k = np.linspace(400.0, 600.0, 4000)
    figK, axK = plt.subplots()
    plot_Kshell_channel_resolved(E_k, s, include_kshell=True,
                                 legend=True, also_plot_composite=False, ax=axK)
    axK.set_title(f"O K-shell $\\epsilon_2$ (optical, gated) 400–600 eV; {ice} ice")
    plt.savefig(f"output/Kshell_channel_resolved_optical_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # New figure: Model vs Experimental comparison (single panel, all three quantities)
    fig_comp, ax_comp = plt.subplots(figsize=(10, 7))

    # Get total model predictions at q=0 (optical limit)
    e1_dict = epsilon1_valence_E0(E, s)
    e2_dict = epsilon2_valence_E0(E, s, partitioned=use_partitioning)
    e1_model = e1_dict["total"]
    e2_model = e2_dict["total"]
    # Add K-shell contribution to epsilon2
    e2_model = e2_model + epsilon2_Kshell_E0_fsum_corrected(E, s)
    # Compute ELF with K-shell
    elf_model = elf_E0(E, s, include_kshell=True, partitioned=use_partitioning)

    # Plot all quantities on the same panel
    # Model predictions (black lines)
    ax_comp.plot(E, e2_model, 'k-', linewidth=2, label=r'Model: Im($\epsilon$)', zorder=5)
    ax_comp.plot(E, e1_model, 'k--', linewidth=2, label=r'Model: Re($\epsilon$)', zorder=5)
    ax_comp.plot(E, elf_model, 'k-.', linewidth=2, label='Model: ELF', zorder=5)

    # Experimental data (dark gray with filled shapes)
    ax_comp.plot(exp_e2_E, exp_e2, 'd-', color='darkgray', linewidth=1.5, markersize=5,
                markerfacecolor='darkgray', markeredgecolor='darkgray',
                label=r'Exp: Im($\epsilon$)', zorder=10)
    ax_comp.plot(exp_e1_E, exp_e1, 's-', color='darkgray', linewidth=1.5, markersize=5,
                markerfacecolor='darkgray', markeredgecolor='darkgray',
                label=r'Exp: Re($\epsilon$)', zorder=10)
    ax_comp.plot(exp_elf_E, exp_elf, '^-', color='darkgray', linewidth=1.5, markersize=5,
                markerfacecolor='darkgray', markeredgecolor='darkgray',
                label='Exp: ELF', zorder=10)

    ax_comp.set_xlabel("Energy Transfer ($E$; eV)")
    ax_comp.set_ylabel("Dielectric Properties")
    ax_comp.set_title(f"{ice.capitalize()} Ice")
    ax_comp.legend(ncol=2)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.set_xlim(0, 30)

    fig_comp.tight_layout()
    plt.savefig(f"output/Model_vs_Experiment_{ice}.pdf", bbox_inches="tight")
    plt.show()

    plot_neff_and_I(savepath="output/Neff_I_Fig3_like.pdf", partitioned=use_partitioning)

    plot_model_vs_experiment_two_panel(savepath="output/dielectric_both.pdf", use_partitioning=use_partitioning)
