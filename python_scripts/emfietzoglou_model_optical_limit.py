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
    Bth  = float(getattr(ks, "Bth",  532.0))
    N_K  = 0.178  # Emfietzoglou et al., fixed atomic fraction

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
):
    """Overlay per-channel ε₂ at q=0 (valence only)."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)

    res = epsilon2_valence_E0(E, s)

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2"]

    for j, y in enumerate(res["excitations"]):
        ax.plot(E, y, color=exc_colors[j % len(exc_colors)], alpha=alpha, lw=linewidth, label=f"Exc {j+1}")
    for k, y in enumerate(res["ionizations"]):
        ax.plot(E, y, color=ion_colors[k % len(ion_colors)], alpha=alpha, lw=linewidth, label=f"Ion {k+1}")

    if also_plot_composite:
        ax.plot(E, res["total"], color="k", lw=2.2, label=r"Total $\epsilon_2$ (valence, q=0)")

    if legend:
        ax.legend(fontsize=10, loc='upper right')

    ax.set_xlabel("Energy (eV)")
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
        ax.legend(fontsize=10, loc='upper right')

    ax.set_xlabel("Energy (eV)")
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
):
    """Channel-resolved ELF at q=0 from channel-resolved ε1, ε2."""
    if ax is None:
        ax = plt.gca()
    E = np.asarray(E, float)

    # Channel-resolved ε at q=0 (must already be implemented as dicts)
    e1 = epsilon1_valence_E0(E, s)     # {"excitations":[...], "ionizations":[...], "total":...}
    e2 = epsilon2_valence_E0(E, s)     # {"excitations":[...], "ionizations":[...], "total":...}

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
        ax.legend(fontsize=10, loc='upper right')

    ax.set_xlabel("Energy (eV)")
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
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"$\epsilon_2(E, q{=}0)$")
        if legend:
            ax.legend(fontsize=10, loc='upper right')
        return

    y = epsilon2_Kshell_E0(E, s)  # should be gated at Bth internally

    # Same visual language as others
    ax.plot(E, y, color="0.3", ls="--", lw=linewidth, alpha=alpha, label="O K-shell")

    if legend:
        ax.legend(fontsize=10, loc='upper right')

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\epsilon_2(E, q{=}0)$")

def plot_neff_and_I(E_min=0.1, E_max=1.0e5, npts=5000, savepath=None):
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
        elf_val = elf_E0(E, s, include_kshell=False)

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

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

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
    ax1.legend(frameon=False, fontsize=9)

    # Panel (b): I(E_max)
    ax2.set_xscale("log")
    ax2.plot(results["amorphous"]["E"], results["amorphous"]["I_val"], lw=1.9, label="amorphous — valence")
    ax2.plot(results["amorphous"]["E"], results["amorphous"]["I_tot"], lw=1.9, ls="--", label="amorphous — total")
    ax2.plot(results["hexagonal"]["E"], results["hexagonal"]["I_val"], lw=1.9, label="hexagonal — valence")
    ax2.plot(results["hexagonal"]["E"], results["hexagonal"]["I_tot"], lw=1.9, ls="--", label="hexagonal — total")
    ax2.set_xlabel(r"$E_{\max}$ (eV)")
    ax2.set_ylabel("I-value (eV)")
    ax2.set_title(r"$I(E_{\max})$")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()

    return results

# ---- example template (fill with your per-band a_j, b_j, c_j) ----
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

    # Valence ε and ELF(+K) at multiple q
    e1q_val = epsilon1_valence_E0(E, s)
    e2q_val = epsilon2_valence_E0(E, s)

    q = 0.0

    # Separate figure: channel-resolved Im(epsilon) at desired q
    fig3, ax3 = plt.subplots()
    plot_Im_epsilon_channel_resolved(E, s, C, q=q, include_kshell=True, legend=True,
                                     also_plot_composite=True, ax=ax3)
    ax3.set_xlabel("Energy (eV)")
    ax3.set_ylabel(r"$\operatorname{Im}(\epsilon)$")
    ax3.set_title(f"Channel-resolved Im($\\epsilon$) at q = {q}; {ice} ice")
    plt.savefig(f"output/Channel_resolved_Im_optical_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # Separate figure: channel-resolved Re(epsilon) at desired q
    fig4, ax4 = plt.subplots()
    plot_Re_epsilon_channel_resolved(E, s, C, q=q, include_kshell=True, legend=True,
                                     also_plot_composite=True, ax=ax4)
    ax4.set_xlabel("Energy (eV)")
    ax4.set_ylabel(r"$\operatorname{Re}(\epsilon)$")
    ax4.set_title(f"Channel-resolved Re($\\epsilon$) at q = {q}; {ice} ice")
    plt.savefig(f"output/Channel_resolved_Re_optical_{ice}.pdf", bbox_inches="tight")
    plt.show()

    # Separate figure: channel-resolved ELF at desired q (optical limit)
    fig5, ax5 = plt.subplots()
    plot_ELF_channel_resolved(E, s, C, q=q, include_kshell=True,
                              legend=True, also_plot_composite=True, ax=ax5)
    ax5.set_title(f"Channel-resolved ELF at q = {q}; {ice} ice")
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

    plot_neff_and_I(savepath="output/Neff_I_Fig3_like.pdf")
