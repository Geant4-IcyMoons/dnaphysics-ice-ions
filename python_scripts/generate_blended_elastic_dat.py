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
from matplotlib.ticker import LogFormatterMathtext, NullLocator

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
    FONTSIZE_16,
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

FONTSIZE = FONTSIZE_16
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_STANDARD, FONTSIZE))

ROOT = PROJECT_ROOT
OUTDIR = OUTPUT_DIR  # diagnostics
OUTDIR.mkdir(parents=True, exist_ok=True)
# Save final .dat files in the shared cross_sections folder at project root
DATADIR = CROSS_SECTIONS_DIR
DATADIR.mkdir(parents=True, exist_ok=True)
E_MAX = BLEND_E_MAX  # extend a hair above 10 MeV


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
              s_ref_tab: np.ndarray, cdf: np.ndarray | None):
    """Emit a family of diagnostic plots for one high-energy reference (ELSEPA or SR)."""
    mask_low = E_grid <= 200.0
    mask_high = E_grid >= 200.0

    # Total σ plot
    fig0, ax0 = plt.subplots(figsize=(8, 5))
    ax0.loglog(E_grid, s_bl, label="Blended (2 eV–10 MeV)", color="k", lw=2)
    ax0.loglog(E_grid[mask_low], s_bl[mask_low], label="Model low (2–200 eV)", color="tab:blue", lw=2, ls="--")
    ax0.loglog(E_grid[mask_high], s_bl[mask_high], label=f"Model high ({ref_label})", color="tab:red", lw=2, ls="-.")
    ax0.loglog(E_ref_tab, s_ref_tab, label=ref_label, color="tab:orange", lw=1.8, ls=":")
    ax0.axvline(200.0, color="0.6", ls=":", lw=1.5)
    ax0.set_xlabel("Energy (eV)")
    ax0.set_ylabel(r"$\sigma_{\mathrm{elastic}}$ (cm$^2$)")
    ax0.set_xlim(1.0, E_MAX)
    x_ticks = [10.0**k for k in range(0, 8)]  # 1e0 ... 1e7
    ax0.set_xticks(x_ticks)
    ax0.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax0.xaxis.set_minor_locator(NullLocator())
    ax0.set_title(rf"Total $\sigma_{{\mathrm{{elastic}}}}$ (Michaud/{ref_label})")
    ax0.legend()
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
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(OUTDIR / f"diagnostic_sigma_window_{tag}.png", dpi=200)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7.5, 3.5))
    ax2.plot(E_grid[mask_win], s_bl[mask_win] / np.maximum(sigma_ref_win, np.finfo(float).tiny), color="tab:blue")
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel(rf"$\sigma_{{\mathrm{{blend}}}} / \sigma_{{{ref_label}}}$")
    ax2.set_title(rf"Scaling factor vs {ref_label} DCS (100–494 eV)")
    fig2.tight_layout()
    fig2.savefig(OUTDIR / f"diagnostic_scaling_window_{tag}.png", dpi=200)
    plt.show()

    # CDF overlays only for ELSEPA (SR has analytic angles, no tabulated CDF)
    if cdf is not None:
        energies = [200.0, 300.0, 494.0]
        uniq_E = np.unique(cdf[:, 0])
        fig3, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, E0 in zip(axes, energies):
            nearest_E = uniq_E[np.argmin(np.abs(uniq_E - E0))]
            block = cdf[np.isclose(cdf[:, 0], nearest_E)]
            if block.size == 0:
                ax.set_title(f"No CDF near {E0:.0f} eV")
                continue
            ax.plot(block[:, 2], block[:, 1], color="k", label="ELSEPA CDF")
            ax.plot(block[:, 2], block[:, 1], color="0.5", ls="--", label="Blended CDF (same)")
            ax.set_title(rf"CDF at {nearest_E:.0f} eV\nmax $\Delta=0.00e+00$")
            ax.set_xlabel("Theta (deg)")
            ax.legend()
        axes[0].set_ylabel("Cumulative probability")
        fig3.tight_layout()
        fig3.savefig(OUTDIR / f"diagnostic_cdf_overlay_{tag}.png", dpi=200)
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
        cdf_hi = cdf_hi[np.argsort(cdf_hi[:, 0])]
    last_E = cdf_hi[-1, 0]
    if last_E < E_MAX:
        block_last = cdf_hi[cdf_hi[:, 0] == last_E]
        block_extended = block_last.copy()
        block_extended[:, 0] = E_MAX
        cdf_hi = np.vstack([cdf_hi, block_extended])

    # Ensure the grid hits the boundaries exactly: 1.7, 200 eV, and E_MAX without duplicates.
    E_base = np.logspace(np.log10(E_MIN_LOW), np.log10(E_MAX), 500)
    E_base[0] = E_MIN_LOW
    E_base[-1] = E_MAX
    E_grid = np.unique(np.sort(np.concatenate([E_base, [E_SPLIT]])))
    s_sr, n_sr = _sr_sigma_and_n(E_grid)
    s_bl_elsepa = blend_sigma(E_grid, E_m, s_m, E_e, s_e)
    s_bl_sr = blend_sigma(E_grid, E_m, s_m, E_grid, s_sr)

    diagnostics(E_grid, s_bl_elsepa, s_bl_sr, E_m, s_m, E_e, s_e, s_sr, cdf_hi)

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
        for t_deg, cprob in zip(theta_grid, cdf_theta):
            rows.append([E, cprob, t_deg])
    out_cdf_sr = DATADIR / "sigmadiff_cumulated_elastic_e_michaud_sr_high.dat"
    np.savetxt(out_cdf_sr, np.array(rows), fmt="%.10e")
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
