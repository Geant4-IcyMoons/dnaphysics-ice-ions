"""
Dielectric model for water ice (amorphous, hexagonal) with q-dispersion.

What this implements
- 5 excitation bands with per-band dispersion parameters (a_j, b_j, c_j) applied to f_j(q)
- 4 ionization shells with global dispersion for E_i(q) and gamma_i(q)
- Optional O K-shell kept optical
- Returns ε1(E,q), ε2(E,q), and ELF(E,q) = Im[-1/ε(E,q)]

Units
- Energies in eV
- q in a0^{-1}
"""

from dataclasses import dataclass
from typing import List, Literal, Union
import numpy as np
import matplotlib.pyplot as plt

font = 'Gill Sans'
hfont = {'fontname': font}
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'

FONTSIZE = 16
plt.rcParams.update({
    'axes.linewidth': 1.5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'lines.markerfacecolor': 'white',
    'lines.markeredgecolor': 'k',
    'xtick.major.size': 0,
    'xtick.major.width': 1.5,
    'xtick.minor.size': 0,
    'xtick.minor.width': 1.5,
    'xtick.direction': 'in',
    'xtick.major.pad': 5,
    'ytick.major.size': 0,
    'ytick.major.width': 1.5,
    'ytick.minor.size': 0,
    'ytick.minor.width': 1.5,
    'ytick.direction': 'in',
    'axes.titleweight': 'normal',
    'axes.titlepad': 20,
    'font.size': FONTSIZE,
    'axes.titlesize': FONTSIZE,
    'axes.labelsize': FONTSIZE,
    'xtick.labelsize': FONTSIZE,
    'ytick.labelsize': FONTSIZE,
    'legend.fontsize': FONTSIZE,
})

# ---- constants ----
EH = 27.211386245988  # eV, Hartree
RY = 13.605693009     # eV, Rydberg

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
        Bmin = 7.5
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
            Osc(38.00, 30.0, 0.4110, "ionization", Bth=32.0),
        ]
        kshell = Osc(450.0, 360.0, 0.3143, "k_shell", Bth=532.0)
    elif material == "hexagonal":
        Ep = 20.59
        Bmin = 7.5
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
            Osc(35.00, 30.0, 0.3580, "ionization", Bth=32.0),
        ]
        kshell = Osc(450.0, 360.0, 0.3143, "k_shell", Bth=532.0)
    else:
        raise ValueError("material must be 'amorphous' or 'hexagonal'")

    return IceOpticalSet(Ep=Ep, Bmin=Bmin, excitations=excit, ionizations=ioniz, kshell=kshell)


# ---- optical ε1, ε2, ELF ----
def epsilon2_E0(E: np.ndarray, s: IceOpticalSet, include_kshell: bool = True) -> np.ndarray:
    E = np.asarray(E, float)
    e2 = np.zeros_like(E)

    # excitations above band gap
    mask_exc = E >= s.Bmin
    if np.any(mask_exc):
        acc = np.zeros_like(E)
        for o in s.excitations:
            acc += _d_drude_e2(E, o.f, o.E0, o.gamma)
        e2[mask_exc] += (s.Ep**2) * acc[mask_exc]

    # ionizations above thresholds
    acc_ion = np.zeros_like(E)
    for o in s.ionizations:
        acc_ion += np.where(E >= o.Bth, _drude_e2(E, o.f, o.E0, o.gamma), 0.0)
    e2 += (s.Ep**2) * acc_ion

    # optional K shell
    if include_kshell:
        o = s.kshell
        e2 += _drude_e2(E, o.f, o.E0, o.gamma)

    return e2


def epsilon1_E0(E: np.ndarray, s: IceOpticalSet, include_kshell: bool = True) -> np.ndarray:
    E = np.asarray(E, float)
    e1 = np.ones_like(E)

    acc_ion = np.zeros_like(E)
    for o in s.ionizations:
        acc_ion += np.where(E >= o.Bth, _drude_e1(E, o.f, o.E0, o.gamma), 0.0)

    acc_exc = np.zeros_like(E)
    mask_exc = E >= s.Bmin
    if np.any(mask_exc):
        for o in s.excitations:
            acc_exc += _d_drude_e1(E, o.f, o.E0, o.gamma)

    e1 += (s.Ep**2) * (acc_ion + acc_exc)

    if include_kshell:
        o = s.kshell
        e1 += _drude_e1(E, o.f, o.E0, o.gamma)

    return e1


def elf_E0(E: np.ndarray, s: IceOpticalSet, include_kshell: bool = True) -> np.ndarray:
    e1 = epsilon1_E0(E, s, include_kshell)
    e2 = epsilon2_E0(E, s, include_kshell)
    denom = e1**2 + e2**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)
    return e2 / denom


# ---- q-dispersion helpers ----
def _as_vec(x: ArrayLike, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.full(n, float(x))
    if x.size != n:
        raise ValueError(f"Expected length {n}, got {x.size}")
    return x


def _fj_q_vec(fj0_vec: np.ndarray, q: np.ndarray, C: DispersionCoeffs) -> np.ndarray:
    """
    f_j(q) = f_j(0) * exp(-a_j q^2) + b_j q^2 * exp(-c_j q^2)
    Returns shape (len(q), n_exc).
    """
    n_exc = fj0_vec.size
    a = _as_vec(C.a_fj, n_exc)[None, :]
    b = _as_vec(C.b_fj, n_exc)[None, :]
    c = _as_vec(C.c_fj, n_exc)[None, :]
    q2 = q[:, None] ** 2
    return fj0_vec[None, :] * np.exp(-a * q2) + b * q2 * np.exp(-c * q2)


def _gamma_q(gamma0: float, q: np.ndarray, C: DispersionCoeffs) -> np.ndarray:
    """ Width dispersion in eV using RR2017 coefficients and Ry scaling. """
    return gamma0 + C.b1 * (RY * q) + C.b2 * (RY * q) ** 2


def _Ei_q(Ei0: float, q: np.ndarray, C: DispersionCoeffs) -> np.ndarray:
    """
    E_i(q) = E_i(0) + a_emp(q) * RY * q^2
    with a_emp(q) = 1 - exp(-c_disp * q**d_disp)
    """
    aemp = 1.0 - np.exp(-C.c_disp * (q ** C.d_disp))
    return Ei0 + aemp * RY * (q ** 2)


def _renormalize_fi_q(fi0: np.ndarray, fj0_sum: float, fjq_sum: np.ndarray) -> np.ndarray:
    """
    f-sum closure across ionization shells at each q:
      sum_i f_i(q) = (1 - sum_j f_j(q)) * [sum_i f_i(0)] / (1 - sum_j f_j(0))
    Redistribute proportionally to fi0.
    """
    headroom0 = 1.0 - fj0_sum
    if headroom0 <= 0.0:
        raise ValueError("Sum of excitation strengths at q=0 exceeds unity.")
    scale = (1.0 - fjq_sum) / headroom0  # shape (len(q),)
    return fi0[None, :] * scale[:, None]


# ---- ε1(E,q), ε2(E,q), ELF(E,q) with dispersion ----
def epsilon2_Eq(E: np.ndarray, q: np.ndarray, s: IceOpticalSet, C: DispersionCoeffs,
                include_kshell: bool = True, enforce_headroom: float = 0.98) -> np.ndarray:
    E = np.asarray(E, float)
    q = np.atleast_1d(q).astype(float)

    # excitations: per-band fj(q), gamma_j(q); Ej fixed
    fj0 = np.array([o.f for o in s.excitations])
    Ej0 = np.array([o.E0 for o in s.excitations])
    gj0 = np.array([o.gamma for o in s.excitations])

    fjq = _fj_q_vec(fj0, q, C)                         # (len(q), 5)
    gjq = np.stack([_gamma_q(g, q, C) for g in gj0], axis=1)

    # quick guard against violating the sum rule headroom
    fjq_sum = np.sum(fjq, axis=1)                      # (len(q),)
    if np.any(fjq_sum >= enforce_headroom):
        raise ValueError("Σ f_j(q) too large at some q; check exponent signs or q units.")

    # ionizations: fi(q) by sum rule; Ei(q), gamma_i(q)
    fi0 = np.array([o.f for o in s.ionizations])
    Ei0 = np.array([o.E0 for o in s.ionizations])
    gi0 = np.array([o.gamma for o in s.ionizations])
    Bi  = np.array([o.Bth for o in s.ionizations])

    fiq = _renormalize_fi_q(fi0, np.sum(fj0), fjq_sum) # (len(q), 4)
    Eiq = np.stack([_Ei_q(Ei, q, C) for Ei in Ei0], axis=1)
    giq = np.stack([_gamma_q(g, q, C) for g in gi0], axis=1)

    e2 = np.zeros((q.size, E.size), float)

    # excitations above band gap
    mask_exc = E >= s.Bmin
    if np.any(mask_exc):
        Ee = E[None, :]
        acc_exc = np.zeros_like(e2)
        for j in range(fj0.size):
            num = 2.0 * fjq[:, j][:, None] * (gjq[:, j][:, None]**3) * (Ee**3)
            den = ((Ej0[j]**2 - Ee**2)**2 + (gjq[:, j][:, None] * Ee)**2)**2
            acc_exc += num / den
        e2[:, mask_exc] += (s.Ep**2) * acc_exc[:, mask_exc]

    # ionizations above thresholds
    Ee = E[None, :]
    for k in range(fi0.size):
        num = fiq[:, k][:, None] * giq[:, k][:, None] * Ee
        den = ((Eiq[:, k][:, None]**2 - Ee**2)**2) + (giq[:, k][:, None] * Ee)**2
        contrib = num / den
        gate = (E >= Bi[k])[None, :].astype(float)
        e2 += (s.Ep**2) * (contrib * gate)

    if include_kshell:
        o = s.kshell
        e2 += _drude_e2(E[None, :], o.f, o.E0, o.gamma)

    return e2


def epsilon1_Eq(E: np.ndarray, q: np.ndarray, s: IceOpticalSet, C: DispersionCoeffs,
                include_kshell: bool = True, enforce_headroom: float = 0.98) -> np.ndarray:
    E = np.asarray(E, float)
    q = np.atleast_1d(q).astype(float)

    fj0 = np.array([o.f for o in s.excitations])
    Ej0 = np.array([o.E0 for o in s.excitations])
    gj0 = np.array([o.gamma for o in s.excitations])

    fjq = _fj_q_vec(fj0, q, C)
    gjq = np.stack([_gamma_q(g, q, C) for g in gj0], axis=1)

    fjq_sum = np.sum(fjq, axis=1)
    if np.any(fjq_sum >= enforce_headroom):
        raise ValueError("Σ f_j(q) too large at some q; check exponent signs or q units.")

    fi0 = np.array([o.f for o in s.ionizations])
    Ei0 = np.array([o.E0 for o in s.ionizations])
    gi0 = np.array([o.gamma for o in s.ionizations])
    Bi  = np.array([o.Bth for o in s.ionizations])

    fiq = _renormalize_fi_q(fi0, np.sum(fj0), fjq_sum)
    Eiq = np.stack([_Ei_q(Ei, q, C) for Ei in Ei0], axis=1)
    giq = np.stack([_gamma_q(g, q, C) for g in gi0], axis=1)

    e1 = np.ones((q.size, E.size), float)

    # ionizations
    Ee = E[None, :]
    acc_ion = np.zeros_like(e1)
    for k in range(fi0.size):
        num = (Eiq[:, k][:, None]**2 - Ee**2)
        den = ((Eiq[:, k][:, None]**2 - Ee**2)**2) + (giq[:, k][:, None] * Ee)**2
        term = num / den
        gate = (E >= Bi[k])[None, :].astype(float)
        acc_ion += fiq[:, k][:, None] * term * gate
    e1 += (s.Ep**2) * acc_ion

    # excitations (above gap)
    mask_exc = E >= s.Bmin
    if np.any(mask_exc):
        acc_exc = np.zeros_like(e1)
        for j in range(fj0.size):
            base = (Ej0[j]**2 - Ee**2)
            den1 = (Ej0[j]**2 - Ee**2)**2 + (gjq[:, j][:, None] * Ee)**2
            num = base * ((Ej0[j]**2 - Ee**2)**2 + 3.0 * (gjq[:, j][:, None] * Ee)**2)
            acc_exc += fjq[:, j][:, None] * (num / (den1**2))
        e1[:, mask_exc] += (s.Ep**2) * acc_exc[:, mask_exc]

    if include_kshell:
        o = s.kshell
        e1 += _drude_e1(E[None, :], o.f, o.E0, o.gamma)

    return e1


def elf_Eq(E: np.ndarray, q: np.ndarray, s: IceOpticalSet, C: DispersionCoeffs,
           include_kshell: bool = True) -> np.ndarray:
    e1 = epsilon1_Eq(E, q, s, C, include_kshell)
    e2 = epsilon2_Eq(E, q, s, C, include_kshell)
    denom = e1**2 + e2**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)
    return e2 / denom

# ---- example template (fill with your per-band a_j, b_j, c_j) ----
if __name__ == "__main__":
    s = epsilon_optical("amorphous")
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])
    C = DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)  # RR2017 defaults for c_disp,d_disp,b1,b2
    E = np.linspace(1.0, 60.0, 2000)
    qvals = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 2.])  # a0^{-1}
    e1q = epsilon1_Eq(E, qvals, s, C)
    e2q = epsilon2_Eq(E, qvals, s, C)
    elfq = elf_Eq(E, qvals, s, C)

for i, elf_q in enumerate(elfq):
    plt.plot(E, elf_q, label=f"q={qvals[i]:.2f}")
plt.legend()
plt.xlabel("Energy (eV)")
plt.show()


exit()