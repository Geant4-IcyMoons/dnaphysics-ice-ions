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
    y = _drude_e2(E, o.f, o.E0, o.gamma)
    return np.where(E >= o.Bth, y, 0.0)

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
    q = _ensure_1d(q)
    n_exc = fj0.size
    a = _lenN(C.a_fj, n_exc)
    b = _lenN(C.b_fj, n_exc)
    c = _lenN(C.c_fj, n_exc)

    q2 = q[:, None]**2
    term1 = fj0[None, :] * np.exp(-a[None, :] * q2)
    term2 = (b[None, :] * q2) * np.exp(-c[None, :] * q2)
    fjq = term1 + term2
    # Headroom check: sum_j f_j(q) < 1
    s = np.sum(fjq, axis=1)
    if np.any(s >= 0.995):
        raise ValueError("Sum_j f_j(q) too large (>= 0.995). Revisit a_fj,b_fj,c_fj.")
    return fjq

def _renorm_fi_q(fi0: np.ndarray, fj0_sum0: float, fjq_sum: np.ndarray) -> np.ndarray:
    """
    Ionization strengths vs q by f-sum normalization:
      f_i(q) = f_i * (1 - sum_j f_j(q)) / (1 - sum_j f_j(0))
    Returns shape (nq, n_ion).
    """
    if not (fj0_sum0 < 1.0):
        raise ValueError("sum_j f_j(0) must be < 1 for normalization.")
    scale = (1.0 - fjq_sum) / (1.0 - fj0_sum0)   # shape (nq,)
    return fi0[None, :] * scale[:, None]

def _Ei_q(Ei0: np.ndarray, q: np.ndarray, C: DispersionCoeffs) -> np.ndarray:
    """
    Ionization resonance shift vs q:
      E_i(q) = E_i + [1 - exp(-c_disp * q^d_disp)] * (RY) * q^2
    Returns shape (nq, n_ion).
    """
    q = _ensure_1d(q)
    lift = (1.0 - np.exp(-C.c_disp * (q**C.d_disp))) * (RY * q**2)  # shape (nq,)
    return Ei0[None, :] + lift[:, None]

def _gamma_q(g0: np.ndarray, q: np.ndarray, C: DispersionCoeffs) -> np.ndarray:
    """
    Width dispersion for excitations and ionizations:
      gamma(q) = g0 + b1*(RY*q) + b2*(RY*q)^2
    Returns shape (nq, n_chan).
    """
    q = _ensure_1d(q)
    dq1 = (RY * q)[:, None]
    dq2 = (RY * q)**2
    return g0[None, :] + C.b1 * dq1 + C.b2 * dq2[:, None]

# ============================================================
# Finite-q ε₂ and ε₁ (valence only), channel-resolved
# ============================================================
def epsilon2_valence_Eq(E: np.ndarray, q: ArrayLike, s: IceOpticalSet, C: DispersionCoeffs) -> dict:
    """
    Imaginary dielectric at finite q, valence only (excitations + ionizations), channel-resolved.
    Gating: excitations for E >= Bmin; ionizations for E >= Bth per channel.
    Returns dict with arrays shaped:
      'excitations': [ (nq, nE), ... n_exc ],
      'ionizations': [ (nq, nE), ... n_ion ],
      'total': (nq, nE)
    """
    E = np.asarray(E, float)
    q = _ensure_1d(q)
    nE, nq = E.size, q.size
    # Params @ q
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

    # Build channels
    exc_list = []
    for j in range(fj0.size):
        y = np.empty((nq, nE), float)
        for iq in range(nq):
            y[iq, :] = (s.Ep**2) * _d_drude_e2(E, fjq[iq, j], Ej0[j], gjq[iq, j])
        # gate by Bmin
        y[:, E < s.Bmin] = 0.0
        exc_list.append(y)

    ion_list = []
    for k, ok in enumerate(s.ionizations):
        y = np.empty((nq, nE), float)
        for iq in range(nq):
            y[iq, :] = (s.Ep**2) * _drude_e2(E, fiq[iq, k], Eiq[iq, k], giq[iq, k])
        # gate by Bth per ionization
        y[:, E < ok.Bth] = 0.0
        ion_list.append(y)

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
def elf_Eq(E: np.ndarray, q: ArrayLike, s: IceOpticalSet, C: DispersionCoeffs, include_kshell: bool = True) -> np.ndarray:
    """
    ELF(E,q) = ε2^(m)/(ε1^(m)^2 + ε2^(m)^2)  +  ε2^(K)(E,0).
    Returns array (nq, nE). K-shell is optical and hard-gated at its edge.
    """
    E = np.asarray(E, float)
    q = _ensure_1d(q)
    e1 = epsilon1_valence_Eq(E, q, s, C)["total"]  # (nq, nE)
    e2 = epsilon2_valence_Eq(E, q, s, C)["total"]  # (nq, nE)
    denom = e1**2 + e2**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)
    elf = e2 / denom
    if include_kshell:
        ks = epsilon2_Kshell_E0(E, s)             # (nE,)
        elf = elf + ks[None, :]
    return elf


# ============================================================
# Plotting helpers for finite-q (new; existing ones untouched)
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
):
    """Per-channel ε₂ at a single finite q."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)
    res = epsilon2_valence_Eq(E, np.array([q], float), s, C)
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
    elf = elf_Eq(E, np.array([q], float), s, C, include_kshell=include_kshell)[0]
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
    ax=None,
    y_log: bool = False,
    linewidth: float = 1.8,
):
    """Overlay ELF totals for multiple q on one energy axis."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)
    qvals = _ensure_1d(qvals)
    elf = elf_Eq(E, qvals, s, C, include_kshell=include_kshell)  # (nq, nE)
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
                         rtol: float = 1e-10, atol: float = 1e-12) -> None:
    """
    Verifies that ε(E,q→0) reproduces the optical ε(E,0).
    Raises if mismatch exceeds tolerances.
    """
    E = np.asarray(E, float)
    q0 = np.array([0.0], float)
    e1_q0 = epsilon1_valence_Eq(E, q0, s, C)["total"][0]
    e2_q0 = epsilon2_valence_Eq(E, q0, s, C)["total"][0]
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
        # Remove per-panel legends
        # ax.legend(fontsize=10, loc='upper right')
        if r == nrows - 1:
            ax.set_xlabel("Energy (eV)")
        # Only leftmost column gets a y-label; hide y tick labels for others
        if c == 0:
            ax.set_ylabel(r"$ELF(E,q)$")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    for j in range(nG, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    # Add a single shared legend below all panels if we captured handles
    if legend_handles and legend_labels:
        # Make space at bottom for legend bar
        fig.subplots_adjust(bottom=0.16)
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(5, len(legend_labels)), frameon=False)
    # Leave extra headroom for a super-title above panels
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
    for j in range(nG, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    if legend_handles and legend_labels:
        fig.subplots_adjust(bottom=0.16)
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(5, len(legend_labels)), frameon=False)
    # Leave extra headroom for a super-title above panels
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
    for j in range(nG, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    if legend_handles and legend_labels:
        fig.subplots_adjust(bottom=0.16)
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(5, len(legend_labels)), frameon=False)
    # Leave extra headroom for a super-title above panels
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    return fig

if __name__ == "__main__":

    ice = "hexagonal"
    # ice = "amorphous"
    s = epsilon_optical(ice)
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])
    C = DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)  # RR2017 defaults for c_disp,d_disp,b1,b2
    E = np.linspace(1.0, 60.0, 20000)
    qvals = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 2.0])  # a0^{-1}
    # Finite-q: ensure optical limit is recovered
    check_q0_convergence(E, s, C)

    # Pick a q to show channel-resolved finite-q curves
    q_sel = 0. # a0^{-1}
    figA, axA = plt.subplots()
    plot_Im_epsilon_channel_resolved_Eq(E, s, C, q=q_sel, legend=True, ax=axA)
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
    plot_ELF_channel_resolved_Eq(E, s, C, q=q_sel, include_kshell=True, legend=True, ax=axD)
    axD.set_title(f"ELF at q = {q_sel:.2f} with O K-shell; {ice} ice")
    plt.savefig(f"output/ELF_singleq_{ice}_q{q_sel:.2f}.pdf", bbox_inches="tight")
    plt.show()
