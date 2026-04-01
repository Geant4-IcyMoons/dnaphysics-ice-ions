#!/usr/bin/env python3
"""
Generate the six .dat files needed for two models:
  Michaud/ELSEPA (muffin potential): low (1.7–200 eV), high (200 eV–10 MeV), and high-energy DCS/CDF.
  Michaud/SR:                      low (1.7–200 eV), high (200 eV–10 MeV), and high-energy DCS/CDF.

Diagnostics are also produced for both models.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogFormatterMathtext, LogLocator, NullLocator

from constants import (
    BLEND_E_MAX,
    BLEND_E_MIN_LOW as E_MIN_LOW,
    BLEND_E_SPLIT as E_SPLIT,
    CROSS_SECTIONS_DIR,
    ELASTIC_BLEND_E0,
    ELASTIC_BLEND_T,
    ELSEPA_MUFFIN_CDF,
    ELSEPA_MUFFIN_TOTAL,
    EV_TO_MEV,
    FM2_TO_CM2,
    FONT_COURIER,
    FONTSIZE_24,
    HFONT_COURIER,
    MICHAUD_SIGMA_SCALE_CM2,
    MICHAUD_TABLE2_PATH,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RC_BASE_STANDARD,
    SR_ALPHA_1,
    SR_BETA_1,
    SR_CONST_K,
    SR_E_SQUARED_MEV_FM,
    SR_ELECTRON_MASS_MEV,
    SR_Z_WATER,
    rcparams_with_fontsize,
)

font = FONT_COURIER
hfont = HFONT_COURIER
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'

FONTSIZE = FONTSIZE_24
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_STANDARD, FONTSIZE))

ROOT = PROJECT_ROOT
OUTDIR = OUTPUT_DIR  # diagnostics
OUTDIR.mkdir(parents=True, exist_ok=True)
# Save final .dat files in the shared cross_sections folder at project root
DATADIR = CROSS_SECTIONS_DIR
DATADIR.mkdir(parents=True, exist_ok=True)
E_MAX = BLEND_E_MAX  # extend a hair above 10 MeV


def _apply_log_ticks(ax: plt.Axes, tick_size: int = FONTSIZE + 1) -> None:
    """Force visible major/minor ticks on log-log plots."""
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.tick_params(axis="both", which="major", labelsize=tick_size, length=8, width=1.2, direction="in")
    ax.tick_params(axis="both", which="minor", length=4, width=1.0, direction="in")


def _apply_semilog_ticks(ax: plt.Axes, tick_size: int = FONTSIZE + 1) -> None:
    """Force visible major/minor ticks on semi-log-x plots."""
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", labelsize=tick_size, length=8, width=1.2, direction="in")
    ax.tick_params(axis="both", which="minor", length=4, width=1.0, direction="in")


def load_michaud():
    df = pd.read_csv(MICHAUD_TABLE2_PATH, skiprows=3, header=None)
    e = pd.to_numeric(df[0], errors="coerce").to_numpy()
    sigma_raw = pd.to_numeric(df[1], errors="coerce").to_numpy()
    mask = np.isfinite(e) & np.isfinite(sigma_raw)
    return e[mask], sigma_raw[mask] * MICHAUD_SIGMA_SCALE_CM2  # cm^2


def load_elsepa_muffin_total():
    data = np.loadtxt(ELSEPA_MUFFIN_TOTAL)
    return data[:, 0], data[:, 1]


def load_elsepa_muffin_cdf():
    return np.loadtxt(ELSEPA_MUFFIN_CDF)


def _prepare_cdf_block(theta_deg: np.ndarray, cdf_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort/clean one [theta, CDF] block and normalize CDF to [0, 1]."""
    theta = np.asarray(theta_deg, float)
    cdf = np.asarray(cdf_vals, float)
    m = np.isfinite(theta) & np.isfinite(cdf)
    theta = theta[m]
    cdf = cdf[m]
    if theta.size < 3:
        raise RuntimeError("CDF block has too few valid points.")

    order = np.argsort(theta)
    theta = theta[order]
    cdf = cdf[order]

    # Merge duplicate theta values by taking the largest cumulative value.
    theta_u, inv = np.unique(theta, return_inverse=True)
    cdf_u = np.zeros_like(theta_u, dtype=float)
    np.maximum.at(cdf_u, inv, cdf)

    # Bound and enforce monotonicity.
    cdf_u = np.clip(cdf_u, 0.0, 1.0)
    cdf_u = np.maximum.accumulate(cdf_u)

    # Ensure [0, 180] support.
    if theta_u[0] > 0.0:
        theta_u = np.insert(theta_u, 0, 0.0)
        cdf_u = np.insert(cdf_u, 0, 0.0)
    if theta_u[-1] < 180.0:
        theta_u = np.append(theta_u, 180.0)
        cdf_u = np.append(cdf_u, 1.0)

    if cdf_u[-1] <= cdf_u[0]:
        raise RuntimeError("Degenerate CDF block (no dynamic range).")
    cdf_u = (cdf_u - cdf_u[0]) / (cdf_u[-1] - cdf_u[0])
    cdf_u[0] = 0.0
    cdf_u[-1] = 1.0
    return theta_u, cdf_u


def gamma_from_cdf_block(theta_deg: np.ndarray, cdf_vals: np.ndarray) -> float:
    """
    Compute g(E)=<cos(theta)> from a tabulated cumulative CDF(theta).
    The CDF is differentiated numerically to get p(theta) over theta in radians.
    """
    theta_u, cdf_u = _prepare_cdf_block(theta_deg, cdf_vals)
    theta_rad = np.deg2rad(theta_u)
    pdf = np.gradient(cdf_u, theta_rad, edge_order=1)
    pdf = np.clip(pdf, 0.0, None)
    area = np.trapezoid(pdf, theta_rad)
    if area <= 0.0:
        raise RuntimeError("Non-positive PDF normalization while extracting gamma.")
    pdf /= area
    g = float(np.trapezoid(np.cos(theta_rad) * pdf, theta_rad))
    return float(np.clip(g, 0.0, 0.999999))


def build_elsepa_gamma_table(cdf_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return [E_eV, gamma(E)] for each ELSEPA CDF energy block."""
    energies = np.unique(cdf_table[:, 0])
    g_list = []
    e_list = []
    for energy in energies:
        block = cdf_table[np.isclose(cdf_table[:, 0], energy)]
        if block.size == 0:
            continue
        g_val = gamma_from_cdf_block(block[:, 2], block[:, 1])
        e_list.append(float(energy))
        g_list.append(g_val)
    if not e_list:
        raise RuntimeError("Failed to extract any gamma values from ELSEPA CDF table.")
    e_arr = np.asarray(e_list, float)
    g_arr = np.asarray(g_list, float)
    order = np.argsort(e_arr)
    return e_arr[order], g_arr[order]


def reconstruct_sigma_total_from_transport(
    energy_eV: np.ndarray,
    sigma_transport_cm2: np.ndarray,
    elsepa_energy_eV: np.ndarray,
    elsepa_gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct total elastic sigma from transport-like sigma using:
      sigma_tot = sigma_tr / (1 - g)
    with g interpolated from ELSEPA gamma(E) in log-energy.
    """
    e = np.asarray(energy_eV, float)
    sigma_tr = np.asarray(sigma_transport_cm2, float)
    e_ref = np.asarray(elsepa_energy_eV, float)
    g_ref = np.asarray(elsepa_gamma, float)

    if np.any(e <= 0.0) or np.any(e_ref <= 0.0):
        raise RuntimeError("Energies for gamma interpolation must be positive.")
    g_interp = np.interp(np.log(e), np.log(e_ref), g_ref, left=g_ref[0], right=g_ref[-1])
    g_interp = np.clip(g_interp, 0.0, 0.999999)

    sigma_tot = sigma_tr / np.maximum(1.0 - g_interp, 1e-12)
    if np.any(sigma_tot + 1e-300 < sigma_tr):
        raise RuntimeError("sigma_tot must be >= sigma_transport for all energies.")
    return sigma_tot, g_interp


def plot_michaud_transport_vs_total(E_m: np.ndarray, sigma_tr: np.ndarray, sigma_tot: np.ndarray) -> None:
    """Plot reported Michaud and corrected (transport->total) Michaud cross sections."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(E_m, sigma_tr, color="lightgray", linewidth=5, label="Michaud+03 (reported)", zorder=3)
    ax.loglog(E_m, sigma_tot, "k-.", linewidth=2, label="Michaud corrected (transport->total)", zorder=3)
    label_size = FONTSIZE + 2
    tick_size = FONTSIZE + 1
    legend_size = FONTSIZE + 1
    ax.set_xlabel(r"Electron Energy ($T$; eV)", fontsize=label_size)
    ax.set_ylabel(r"Cross-Section (cm$^2$)", fontsize=label_size)
    ax.set_xlim(1.0, E_MAX)
    x_ticks = [10.0**k for k in range(0, 8)]  # 1e0 ... 1e7
    ax.set_xticks(x_ticks)
    _apply_log_ticks(ax, tick_size=tick_size)
    ax.legend(loc="best", fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(OUTDIR / "diagnostic_sigma_michaud_transport_vs_total.png", dpi=200)
    fig.savefig(OUTDIR / "michaud_corrected_cross_sections.png", dpi=200)
    plt.show()


def plot_gamma_vs_energy(E_gamma: np.ndarray, gamma: np.ndarray, E_m: np.ndarray, gamma_m: np.ndarray) -> None:
    """Plot gamma(E)=<cos(theta)> extracted from ELSEPA CDF and mapped to Michaud grid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(E_gamma, gamma, color="k", lw=2, label=r"ELSEPA-derived $\gamma(E)$")
    ax.semilogx(E_m, gamma_m, color="tab:blue", lw=1.8, ls="--", label=r"$\gamma(E)$ mapped to Michaud grid")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\gamma(E)=\langle\cos\theta\rangle$")
    ax.set_xlim(1.0, E_MAX)
    ax.set_ylim(0.0, 1.0)
    x_ticks = [10.0**k for k in range(0, 8)]  # 1e0 ... 1e7
    ax.set_xticks(x_ticks)
    _apply_semilog_ticks(ax)
    ax.set_title(r"ELSEPA-Derived Elastic Anisotropy $\gamma(E)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "diagnostic_gamma_elsepa.png", dpi=200)
    plt.show()


def build_hg_cdf_rows(energies_eV: np.ndarray, gamma_vals: np.ndarray, n_theta: int = 361) -> np.ndarray:
    """
    Build [E, CDF(theta), theta_deg] rows using Henyey-Greenstein phase function
    with anisotropy gamma=<cos(theta)>.
    """
    theta_deg = np.linspace(0.0, 180.0, n_theta)
    theta_rad = np.deg2rad(theta_deg)
    mu = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    rows = []
    for E, g in zip(energies_eV, gamma_vals):
        g = float(np.clip(g, 0.0, 0.999999))
        denom = np.power(1.0 + g * g - 2.0 * g * mu, 1.5)
        p_mu = (1.0 - g * g) / (2.0 * np.maximum(denom, 1e-30))
        p_theta = np.clip(p_mu * sin_theta, 0.0, None)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p_theta[1:] + p_theta[:-1]) * np.diff(theta_rad))])
        if cdf[-1] <= 0.0:
            raise RuntimeError("HG CDF normalization failed.")
        cdf /= cdf[-1]
        cdf = np.maximum.accumulate(cdf)
        cdf[0] = 0.0
        cdf[-1] = 1.0
        if np.any(np.diff(cdf) < -1e-12):
            raise RuntimeError("HG CDF is not monotonic.")
        for t_deg, cprob in zip(theta_deg, cdf):
            rows.append([float(E), float(cprob), float(t_deg)])
    return np.asarray(rows, dtype=float)


def sanitize_cdf_rows(rows: np.ndarray) -> np.ndarray:
    """
    Sanitize [E, CDF, theta_deg] rows per energy block:
      1) sort by theta ascending,
      2) enforce monotonic non-decreasing CDF,
      3) renormalize to [0, 1],
      4) force endpoints near theta min/max to 0 and 1.
    """
    arr = np.asarray(rows, float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise RuntimeError("CDF rows must be Nx3: [E, CDF, theta_deg].")
    out = []
    for E in np.unique(arr[:, 0]):
        block = arr[arr[:, 0] == E]
        theta = block[:, 2]
        cdf = block[:, 1]
        order = np.argsort(theta)
        theta = theta[order]
        cdf = cdf[order]
        cdf = np.maximum.accumulate(cdf)
        c0 = cdf[0]
        c1 = cdf[-1]
        if c1 <= c0 + 1e-20:
            cdf = np.linspace(0.0, 1.0, len(cdf))
        else:
            cdf = (cdf - c0) / (c1 - c0)
        cdf[0] = 0.0
        cdf[-1] = 1.0
        out.append(np.column_stack([np.full_like(theta, E), cdf, theta]))
    return np.vstack(out)


def _sr_sigma_and_n(energy_eV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Screened Rutherford elastic cross-section for electrons in water/ice.
    Returns:
        sigma_cm2 : total cross section [cm^2]
        n         : screening parameter used in angular distribution
    """
    e_squared = SR_E_SQUARED_MEV_FM
    electron_mass_c2 = SR_ELECTRON_MASS_MEV
    z = SR_Z_WATER
    k_MeV = np.asarray(energy_eV, float) * EV_TO_MEV
    length_fm = (e_squared * (k_MeV + electron_mass_c2)) / (k_MeV * (k_MeV + 2 * electron_mass_c2))
    sigma_ruth_fm2 = z * (z + 1) * length_fm**2
    alpha_1 = SR_ALPHA_1
    beta_1 = SR_BETA_1
    constK = SR_CONST_K
    numerator = (alpha_1 + beta_1 * np.log(energy_eV)) * constK * (z ** (2.0 / 3.0))
    k_ratio = k_MeV / electron_mass_c2
    denominator = k_ratio * (2 + k_ratio)
    n = np.where(denominator > 0, numerator / denominator, 0)
    sigma_fm2 = np.pi * sigma_ruth_fm2 / (n * (n + 1.0))
    return sigma_fm2 * FM2_TO_CM2, n  # fm^2 -> cm^2


def screened_rutherford_cross_section(energy_eV: np.ndarray) -> np.ndarray:
    sigma, _ = _sr_sigma_and_n(energy_eV)
    return sigma

def blend_sigma(E_grid, E_mich, s_mich, E_elsepa, s_elsepa, E0=ELASTIC_BLEND_E0, t=ELASTIC_BLEND_T):
    s_bl = np.zeros_like(E_grid)
    target_100 = s_mich[-1]
    for i, E in enumerate(E_grid):
        if E < E0:
            s_bl[i] = np.interp(E, E_mich, s_mich)
        elif E >= t:
            s_bl[i] = float(np.interp(E, E_elsepa, s_elsepa, left=s_elsepa[0], right=s_elsepa[-1]))
        else:
            s = (E - E0) / (t - E0)
            w = s * s * (3 - 2 * s)
            s_hi = float(np.interp(E, E_elsepa, s_elsepa, left=s_elsepa[0], right=s_elsepa[-1]))
            s_bl[i] = (1 - w) * target_100 + w * s_hi
    return s_bl


def _diag_set(tag: str, E_grid: np.ndarray, s_bl: np.ndarray, s_ref: np.ndarray, ref_label: str,
              E_m: np.ndarray, s_m: np.ndarray, E_ref_tab: np.ndarray,
              s_ref_tab: np.ndarray, cdf: np.ndarray | None,
              michaud_label: str = "Michaud+03", blended_label: str = "Blended"):
    """Emit a family of diagnostic plots for one high-energy reference (ELSEPA or SR)."""
    mask_low = E_grid <= 200.0
    mask_high = E_grid >= 200.0

    # Total σ plot
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    if tag.startswith("elsepa_muffin"):
        # Match legacy elastic_cross_sections style.
        ax0.loglog(E_m, s_m, color="lightgray", linewidth=5, label=michaud_label, zorder=3)
        ax0.loglog(E_ref_tab, s_ref_tab, color="slategray", linewidth=5, label="ELSEPA (muffin)", zorder=3)
        ax0.loglog(E_grid, s_bl, "k-.", linewidth=2, label=blended_label, zorder=3)
        ax0.axvspan(ELASTIC_BLEND_E0, ELASTIC_BLEND_T, color="lightgray", alpha=0.3, zorder=0)
        ax0.set_xlabel(r"Electron Energy ($T$; eV)")
        ax0.set_ylabel(r"Cross-Section (cm$^2$)")
        ax0.legend(loc="best")
    else:
        ax0.loglog(E_ref_tab, s_ref_tab, label=ref_label, color="slategray", lw=2.0, ls="-")
        ax0.loglog(E_m, s_m, label="Michaud (reported)", color="lightgray", lw=2.0, ls="-")
        ax0.loglog(E_grid, s_bl, label="Blended full (2 eV–10 MeV)", color="k", lw=2.0, ls="-.")
        ax0.axvline(200.0, color="0.6", ls=":", lw=1.5)
        ax0.set_xlabel("Energy (eV)")
        ax0.set_ylabel(r"$\sigma_{\mathrm{elastic}}$ (cm$^2$)")
        ax0.set_title(rf"Total $\sigma_{{\mathrm{{elastic}}}}$ (Michaud/{ref_label})")
        ax0.legend()

    ax0.set_xlim(1.0, E_MAX)
    x_ticks = [10.0**k for k in range(0, 8)]  # 1e0 ... 1e7
    ax0.set_xticks(x_ticks)
    _apply_log_ticks(ax0)
    fig0.tight_layout()
    fig0.savefig(OUTDIR / f"diagnostic_sigma_models_{tag}.png", dpi=200)
    plt.show()

    # Window diagnostics
    mask_win = (E_grid >= 100.0) & (E_grid <= 494.0)
    sigma_ref_win = np.interp(E_grid[mask_win], E_ref_tab, s_ref_tab)
    sigma_mich_win = np.interp(E_grid[mask_win], E_m, s_m)

    fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(E_grid[mask_win], sigma_ref_win, label=f"{ref_label} total", color="slategray")
    ax1.plot(E_grid[mask_win], s_bl[mask_win], label="Blended", color="k", ls="-.")
    ax1.plot(E_grid[mask_win], sigma_mich_win, label="Michaud (interp.)", color="lightgray", ls=":")
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel(r"$\sigma$ (cm$^2$)")
    ax1.set_title(rf"Elastic $\sigma$ in transition window (100–494 eV) – {ref_label}")
    ax1.minorticks_on()
    ax1.tick_params(axis="both", which="major", length=8, width=1.2, direction="in")
    ax1.tick_params(axis="both", which="minor", length=4, width=1.0, direction="in")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(OUTDIR / f"diagnostic_sigma_window_{tag}.png", dpi=200)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7.5, 3.5))
    ax2.plot(E_grid[mask_win], s_bl[mask_win] / np.maximum(sigma_ref_win, np.finfo(float).tiny), color="tab:blue")
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel(rf"$\sigma_{{\mathrm{{blend}}}} / \sigma_{{{ref_label}}}$")
    ax2.set_title(rf"Scaling factor vs {ref_label} DCS (100–494 eV)")
    ax2.minorticks_on()
    ax2.tick_params(axis="both", which="major", length=8, width=1.2, direction="in")
    ax2.tick_params(axis="both", which="minor", length=4, width=1.0, direction="in")
    fig2.tight_layout()
    fig2.savefig(OUTDIR / f"diagnostic_scaling_window_{tag}.png", dpi=200)
    plt.show()


def diagnostics(E_grid, s_bl_elsepa, s_bl_sr, E_m, s_m, E_e, s_e, s_sr, cdf):
    _diag_set("elsepa_muffin", E_grid, s_bl_elsepa, s_e, "ELSEPA (muffin)", E_m, s_m, E_e, s_e, cdf)
    _diag_set("sr", E_grid, s_bl_sr, s_sr, "SR", E_m, s_m, E_grid, s_sr, None)


def main():
    E_m, s_m = load_michaud()
    E_e, s_e = load_elsepa_muffin_total()
    # Extend ELSEPA total to E_MAX with a flat tail if needed
    if E_e[-1] < E_MAX:
        E_e = np.append(E_e, E_MAX)
        s_e = np.append(s_e, s_e[-1])

    cdf_raw = load_elsepa_muffin_cdf()
    # Keep only high branch energies >= split and extend to E_MAX
    cdf_hi = cdf_raw[cdf_raw[:, 0] >= E_SPLIT]
    if cdf_hi.size == 0:
        raise RuntimeError("ELSEPA muffin CDF has no energies >= split energy.")
    # Ensure the high-branch CDF starts exactly at E_SPLIT.
    uniq_E = np.unique(cdf_hi[:, 0])
    if not np.isclose(uniq_E[0], E_SPLIT):
        block_first = cdf_hi[cdf_hi[:, 0] == uniq_E[0]]
        if block_first.size == 0:
            raise RuntimeError("ELSEPA muffin CDF is missing its first energy block.")
        block_split = block_first.copy()
        block_split[:, 0] = E_SPLIT
        cdf_hi = np.vstack([block_split, cdf_hi])
        # Sort by (energy, theta) explicitly; sorting by energy alone can scramble
        # equal-energy rows if an unstable sort is used.
        idx = np.lexsort((cdf_hi[:, 2], cdf_hi[:, 0]))
        cdf_hi = cdf_hi[idx]
    last_E = cdf_hi[-1, 0]
    if last_E < E_MAX:
        block_last = cdf_hi[cdf_hi[:, 0] == last_E]
        block_extended = block_last.copy()
        block_extended[:, 0] = E_MAX
        cdf_hi = np.vstack([cdf_hi, block_extended])
    cdf_hi = sanitize_cdf_rows(cdf_hi)

    # Reconstruct total elastic sigma from Michaud transport-like sigma:
    #   sigma_tot = sigma_tr / (1 - gamma(E))
    # gamma(E) is extracted from ELSEPA muffin angular CDF at all available ELSEPA energies,
    # then sampled onto each Michaud energy point.
    E_gamma, gamma_elsepa = build_elsepa_gamma_table(cdf_raw)
    sigma_tot_michaud, gamma_on_michaud_grid = reconstruct_sigma_total_from_transport(
        E_m, s_m, E_gamma, gamma_elsepa
    )
    plot_michaud_transport_vs_total(E_m, s_m, sigma_tot_michaud)
    plot_gamma_vs_energy(E_gamma, gamma_elsepa, E_m, gamma_on_michaud_grid)

    # Save auxiliary reconstruction tables (same sigma table convention where applicable).
    scale_tab = 1.0e16
    out_sigma_tot = DATADIR / "sigma_elastic_e_michaud_elsepa_low_transport_to_total.dat"
    np.savetxt(out_sigma_tot, np.column_stack([E_m, sigma_tot_michaud * scale_tab]), fmt="%.8e")
    print(f"[saved] {out_sigma_tot} (reconstructed total from transport using ELSEPA gamma)")

    out_gamma = DATADIR / "gamma_elastic_e_elsepa_muffin.dat"
    np.savetxt(out_gamma, np.column_stack([E_gamma, gamma_elsepa]), fmt="%.8e")
    print(f"[saved] {out_gamma}")

    # Ensure the grid hits the boundaries exactly: 1.7, 200 eV, and E_MAX without duplicates.
    E_base = np.logspace(np.log10(E_MIN_LOW), np.log10(E_MAX), 500)
    E_base[0] = E_MIN_LOW
    E_base[-1] = E_MAX
    E_grid = np.unique(np.sort(np.concatenate([E_base, [E_SPLIT]])))
    s_sr, n_sr = _sr_sigma_and_n(E_grid)
    s_bl_elsepa = blend_sigma(E_grid, E_m, s_m, E_e, s_e)
    s_bl_elsepa_corrected = blend_sigma(E_grid, E_m, sigma_tot_michaud, E_e, s_e)
    s_bl_sr = blend_sigma(E_grid, E_m, s_m, E_grid, s_sr)

    diagnostics(E_grid, s_bl_elsepa, s_bl_sr, E_m, s_m, E_e, s_e, s_sr, cdf_hi)
    _diag_set(
        "elsepa_muffin_corrected",
        E_grid, s_bl_elsepa_corrected, s_e, "ELSEPA (muffin)",
        E_m, sigma_tot_michaud, E_e, s_e, cdf_hi,
        michaud_label="Michaud+03 ($\sigma_{\\rm{tot}}$)",
        blended_label="Blended"
    )

    # Save ONLY the six requested files:
    mask_low = E_grid <= E_SPLIT
    mask_high = E_grid >= E_SPLIT

    # Save in units of 1e-16 cm^2 (like Michaud tables)
    scale_tab = 1.0e16
    out_low_elsepa = DATADIR / "sigma_elastic_e_michaud_elsepa_low.dat"
    np.savetxt(out_low_elsepa, np.column_stack([E_grid[mask_low], s_bl_elsepa[mask_low] * scale_tab]), fmt="%.8e")
    print(f"[saved] {out_low_elsepa} (isotropic angles expected; ELSEPA muffin high branch)")

    out_high_elsepa = DATADIR / "sigma_elastic_e_michaud_elsepa_high.dat"
    np.savetxt(out_high_elsepa, np.column_stack([E_grid[mask_high], s_bl_elsepa[mask_high] * scale_tab]), fmt="%.8e")
    print(f"[saved] {out_high_elsepa} (use ELSEPA muffin angular CDF)")

    out_cdf_elsepa = DATADIR / "sigmadiff_cumulated_elastic_e_michaud_elsepa_high.dat"
    np.savetxt(out_cdf_elsepa, cdf_hi, fmt="%.10e")
    print(f"[saved] {out_cdf_elsepa} (ELSEPA muffin angular CDF)")

    # Isotropic low-energy CDF for Michaud-ELSEPA low branch
    theta_iso = np.linspace(0.0, 180.0, 181)
    cos_t = np.cos(np.deg2rad(theta_iso))
    cdf_iso = 0.5 * (1.0 - cos_t)
    rows_iso = []
    for E in E_grid[mask_low]:
        for cprob, tdeg in zip(cdf_iso, theta_iso):
            rows_iso.append([E, cprob, tdeg])
    out_cdf_elsepa_low = DATADIR / "sigmadiff_cumulated_elastic_e_michaud_elsepa_low.dat"
    np.savetxt(out_cdf_elsepa_low, np.array(rows_iso), fmt="%.10e")
    print(f"[saved] {out_cdf_elsepa_low} (isotropic CDF)")

    # Additional LOW variant: corrected sigma_tot + non-isotropic (forward-peaked) CDF.
    gamma_on_grid = np.interp(np.log(E_grid[mask_low]), np.log(E_gamma), gamma_elsepa, left=gamma_elsepa[0], right=gamma_elsepa[-1])
    gamma_on_grid = np.clip(gamma_on_grid, 0.0, 0.999999)
    rows_hg_low = build_hg_cdf_rows(E_grid[mask_low], gamma_on_grid, n_theta=361)

    out_low_elsepa_corrected = DATADIR / "sigma_elastic_e_michaud_elsepa_low_corrected_forward.dat"
    np.savetxt(
        out_low_elsepa_corrected,
        np.column_stack([E_grid[mask_low], s_bl_elsepa_corrected[mask_low] * scale_tab]),
        fmt="%.8e"
    )
    print(f"[saved] {out_low_elsepa_corrected} (corrected LOW sigma; transport->total)")

    out_cdf_elsepa_low_corrected = DATADIR / "sigmadiff_cumulated_elastic_e_michaud_elsepa_low_corrected_forward.dat"
    np.savetxt(out_cdf_elsepa_low_corrected, rows_hg_low, fmt="%.10e")
    print(f"[saved] {out_cdf_elsepa_low_corrected} (forward-peaked LOW CDF)")

    out_low_sr = DATADIR / "sigma_elastic_e_michaud_sr_low.dat"
    np.savetxt(out_low_sr, np.column_stack([E_grid[mask_low], s_bl_sr[mask_low] * scale_tab]), fmt="%.8e")
    print(f"[saved] {out_low_sr}")

    out_high_sr = DATADIR / "sigma_elastic_e_michaud_sr_high.dat"
    np.savetxt(out_high_sr, np.column_stack([E_grid[mask_high], s_bl_sr[mask_high] * scale_tab]), fmt="%.8e")
    print(f"[saved] {out_high_sr}")

    # SR angular CDF derived from screened Rutherford differential cross-section (high)
    theta_grid = np.linspace(0.0, 180.0, 361)
    cos_t = np.cos(np.deg2rad(theta_grid))
    rows = []
    for E, n_val in zip(E_grid[mask_high], n_sr[mask_high]):
        A = 1.0 + 2.0 * n_val
        denom0 = 2.0 + 2.0 * n_val
        cdf_theta = 2.0 * n_val * (n_val + 1.0) * (1.0 / (A - cos_t) - 1.0 / denom0)
        cdf_theta = np.clip(cdf_theta, 0.0, 1.0)
        # Normalize per energy block to a proper cumulative CDF over [0, 180] deg.
        c0 = cdf_theta[0]
        c1 = cdf_theta[-1]
        if c1 <= c0 + 1e-20:
            cdf_theta = np.linspace(0.0, 1.0, len(cdf_theta))
        else:
            cdf_theta = (cdf_theta - c0) / (c1 - c0)
        cdf_theta[0] = 0.0
        cdf_theta[-1] = 1.0
        for t_deg, cprob in zip(theta_grid, cdf_theta):
            rows.append([E, cprob, t_deg])
    out_cdf_sr = DATADIR / "sigmadiff_cumulated_elastic_e_michaud_sr_high.dat"
    rows_sr = sanitize_cdf_rows(np.array(rows, dtype=float))
    np.savetxt(out_cdf_sr, rows_sr, fmt="%.10e")
    print(f"[saved] {out_cdf_sr} (SR angular CDF)")

    # Isotropic low-energy CDF for Michaud-SR low branch
    rows_iso_sr = []
    for E in E_grid[mask_low]:
        for cprob, tdeg in zip(cdf_iso, theta_iso):
            rows_iso_sr.append([E, cprob, tdeg])
    out_cdf_sr_low = DATADIR / "sigmadiff_cumulated_elastic_e_michaud_sr_low.dat"
    np.savetxt(out_cdf_sr_low, np.array(rows_iso_sr), fmt="%.10e")
    print(f"[saved] {out_cdf_sr_low} (isotropic CDF)")


if __name__ == "__main__":
    main()
