#!/usr/bin/env python3
"""
Quick helper to plot Henyey–Greenstein angular PDFs for selected
vibrational-excitation channels at specified energies.

Function:
  plot_hg_for_channel(channel_idx: int, energies_eV: list[float]) -> matplotlib.figure.Figure

Channel index semantics:
- channel_idx here is identical to the integer `channelIndex` stored in the ROOT file
  by the simulation (0–7). Use values from `channelIndex` directly with this function.

Channels (0–7) follow the TARGETS order used in this repo:
  0: vT2       (Table 2: v''(T))
  1: vL1       (Table 2: v'(L))
  2: vL2       (Table 2: v''(L))
  3: v2        (Table 3: v2)
  4: v1,3      (Table 3: v1,3)
  5: v3        (Table 3: v3)
  6: v1,3+vL   (Table 3: v1,3+vL)
  7: 2(v1,3)   (Table 3: 2(v1,3))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import sys
import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics_ice.constants import (
    CUSTOM_DATA_ROOT_PROJECT,
    FONTSIZE_12,
    MICHAUD_TABLE2_PATH,
    MICHAUD_TABLE3_PATH,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RC_BASE_MINIMAL,
    VIB_CHANNEL_MAPPING,
    VIB_TARGET_NAMES,
    rcparams_with_fontsize,
)
from root_utils import resolve_root_paths, resolve_first_root

# Absolute paths used elsewhere in this repo
# Resolve paths relative to project root (dnaphysics-ice)
# This script will live under dnaphysics-ice/python_scripts
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_DATA_ROOT = CUSTOM_DATA_ROOT_PROJECT

TABLE2_PATH = str(MICHAUD_TABLE2_PATH)
TABLE3_PATH = str(MICHAUD_TABLE3_PATH)

# Unified font size for all plots
fontsize = FONTSIZE_12
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_MINIMAL, fontsize))

# Mapping from channel index to (which table, sigma_col, gamma_col)
CHANNEL_MAPPING = VIB_CHANNEL_MAPPING

TARGET_NAMES = VIB_TARGET_NAMES

# Consistent typography across plots is set above


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    df2 = pd.read_csv(TABLE2_PATH, header=None)
    df3 = pd.read_csv(TABLE3_PATH, header=None)
    return df2, df3


def _extract_channel(df2: pd.DataFrame, df3: pd.DataFrame, channel_idx: int):
    if not (0 <= channel_idx <= 7):
        raise ValueError("channel_idx must be in 0..7")

    table_name, col_sigma, col_gamma = CHANNEL_MAPPING[channel_idx]
    df = df2 if table_name == "table2" else df3

    energy = pd.to_numeric(df.iloc[3:, 0], errors="coerce").values.astype(float)
    sigma = pd.to_numeric(df.iloc[3:, col_sigma], errors="coerce").values.astype(float)
    gamma = pd.to_numeric(df.iloc[3:, col_gamma], errors="coerce").values.astype(float)

    mask = ~(np.isnan(energy) | np.isnan(sigma) | np.isnan(gamma))
    e = energy[mask]
    s = sigma[mask]
    g = gamma[mask]
    return e, s, g


def channel_idx_to_name(idx: int) -> str:
    """
    Map ROOT `channelIndex` (0–7) to human-readable channel name.
    This is a 1:1 mapping with the simulation’s logged channelIndex.
    """
    if not (0 <= idx < len(TARGET_NAMES)):
        raise ValueError("channel index out of range (0..7)")
    return TARGET_NAMES[idx]


def channel_name_to_idx(name: str) -> int:
    """
    Map channel name to ROOT channelIndex (0–7).
    Accepts exact names from TARGET_NAMES.
    """
    try:
        return TARGET_NAMES.index(name)
    except ValueError:
        raise ValueError(f"Unknown channel name: {name}. Valid: {TARGET_NAMES}")

def _hg_pdf_theta(theta_deg: np.ndarray, g: float) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    g = float(np.clip(g, -0.9999, 0.9999))
    denom = np.power(1.0 + g*g - 2.0*g*cos_t, 1.5)
    p_cos = (1.0 - g*g) / (4.0 * np.pi * denom)
    p_theta_per_rad = p_cos * sin_t
    p_theta_per_deg = p_theta_per_rad / (np.pi / 180.0)
    return p_theta_per_deg


def _hg_forward_fraction(g: float) -> float:
    if not np.isfinite(g):
        return 0.5
    g = float(np.clip(g, -0.999999, 0.999999))
    if abs(g) < 1e-12:
        return 0.5
    A0 = 1.0 + g*g
    term = (1.0 - g) / np.sqrt(A0)
    return (1.0 + g) / (2.0 * g) * (1.0 - term)


def _invert_forward_fraction_to_g(Y: float, tol: float = 1e-10, maxit: int = 100) -> float:
    Y = float(np.clip(Y, 0.0, 1.0))
    if Y <= 0.0:
        return -0.999999
    if Y >= 1.0:
        return 0.999999
    lo, hi = -0.999999, 0.999999
    f_lo = _hg_forward_fraction(lo) - Y
    f_hi = _hg_forward_fraction(hi) - Y
    if f_lo * f_hi > 0:
        return 0.0
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid = _hg_forward_fraction(mid) - Y
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return float(np.clip(mid, -0.999999, 0.999999))
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(np.clip(0.5 * (lo + hi), -0.999999, 0.999999))


def plot_hg_for_channel(channel_idx: int, energies_eV: list[float]):
    df2, df3 = _load_tables()
    e_tab, s_tab, g_tab = _extract_channel(df2, df3, channel_idx)

    if e_tab.size == 0:
        raise RuntimeError("No data found for requested channel")

    theta = np.linspace(0.0, 180.0, 721)

    n = len(energies_eV)
    ncols = 3 if n >= 3 else n
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for idx, E in enumerate(energies_eV):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        j = int(np.argmin(np.abs(e_tab - E)))
        E_closest = float(e_tab[j])
        # Treat Michaud γ as hemisphere-based anisotropy: γ = 2Y - 1
        # Map to HG parameter g by solving Y(g) = (1+γ)/2
        gamma = float(g_tab[j])
        g = _invert_forward_fraction_to_g(0.5 * (1.0 + gamma))
        p = _hg_pdf_theta(theta, g)
        # Normalize to peak = 1 for visibility
        p_max = float(np.nanmax(p)) if np.isfinite(p).any() else 0.0
        if p_max > 0:
            p = p / p_max

        ax.plot(theta, p, lw=2)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel("Normalized density (peak = 1)")
        ax.set_title(f"Channel {channel_idx} ({TARGET_NAMES[channel_idx]}): E={E_closest:.2f} eV, "+
                     f"g={g:.3f}")

    for k in range(n, nrows*ncols):
        r = k // ncols
        c = k % ncols
        axes[r][c].axis('off')

    fig.tight_layout()
    return fig


def plot_hg_for_channel_with_gamma(channel_idx: int, energies_eV: list[float]):
    """
    Like plot_hg_for_channel, but adds an adjacent panel showing the full
    anisotropy curve γ(E) for the selected channel with colored points at the
    chosen energies (closest table energies).
    """
    df2, df3 = _load_tables()
    e_tab, s_tab, g_tab = _extract_channel(df2, df3, channel_idx)

    if e_tab.size == 0:
        raise RuntimeError("No data found for requested channel")

    theta = np.linspace(0.0, 180.0, 721)

    n = len(energies_eV)
    ncols = 3 if n >= 3 else n
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(5*ncols + 5, 4*nrows))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[*(1 for _ in range(ncols)), 1.3], figure=fig)

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, n)))
    energy_points = []

    # Try to locate and load ROOT output (optional overlay)
    arrs = None
    try:
        import uproot  # optional dependency
        root_paths = resolve_root_paths(PROJECT_ROOT / "build" / "dna.root")
        if root_paths:
            with uproot.open(root_paths[0]) as f:
                tree = f["step"]
                arrs = tree.arrays(["cosTheta", "channelIndex", "kineticEnergy", "flagProcess"], library="np")
        else:
            arrs = None
    except Exception:
        arrs = None

    # Left grid: angular PDFs
    for idx, E in enumerate(energies_eV):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])

        j = int(np.argmin(np.abs(e_tab - E)))
        E_closest = float(e_tab[j])
        gamma = float(g_tab[j])
        g = _invert_forward_fraction_to_g(0.5 * (1.0 + gamma))
        p = _hg_pdf_theta(theta, g)
        # Normalize to peak = 1 for visibility
        p_max = float(np.nanmax(p)) if np.isfinite(p).any() else 0.0
        if p_max > 0:
            p = p / p_max

        color = colors[idx]
        ax.plot(theta, p, lw=2, color=color, label=f"HG (g={g:.3f})")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel("Normalized density (peak = 1)")
        ax.set_title(f"{TARGET_NAMES[channel_idx]} | E={E_closest:.2f} eV, g={g:.3f}")

        # Overlay simulation histogram if ROOT data is available (±1 eV window)
        if arrs is not None:
            try:
                proc_mask = (arrs["flagProcess"] == 15) & (arrs["channelIndex"] == channel_idx)
                cos_all = np.asarray(arrs["cosTheta"][proc_mask], dtype=float)
                e_all = np.asarray(arrs["kineticEnergy"][proc_mask], dtype=float)
                if e_all.size:
                    tol = 0.001  # eV
                    sel = np.abs(e_all - E_closest) <= tol
                    if np.any(sel):
                        theta_sel = np.degrees(np.arccos(np.clip(cos_all[sel], -1.0, 1.0)))
                        # Histogram normalized so peak = 1
                        counts, bins = np.histogram(theta_sel, bins=90, range=(0, 180))
                        if counts.size and np.nanmax(counts) > 0:
                            counts = counts.astype(float) / float(np.nanmax(counts))
                            ax.step(bins[:-1], counts, where='post', color=color, alpha=0.8,
                                    label=f"Sim (±{tol:.0f} eV, N={np.count_nonzero(sel)})")
                            ax.fill_between(bins[:-1], counts, step='post', alpha=0.20, color=color)
                        ax.legend(loc='best')
            except Exception:
                pass

        # For the anisotropy panel we must plot γ(E), not g, against the γ(E) curve
        energy_points.append((E_closest, gamma, color))

    # Blank any unused left-grid panels
    for k in range(n, nrows*ncols):
        r = k // ncols
        c = k % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.axis('off')

    # Right panel: anisotropy curve with highlighted points
    axg = fig.add_subplot(gs[:, -1])
    axg.plot(e_tab, g_tab, 'k-', lw=2, label='γ(E)')
    for (Ee, gamma_point, col) in energy_points:
        axg.plot(Ee, gamma_point, 'o', ms=8, color=col, label=f'{Ee:.2f} eV')
    axg.set_xlabel('Kinetic energy (eV)')
    axg.set_ylabel('Anisotropy γ')
    axg.set_title(f'{TARGET_NAMES[channel_idx]} anisotropy')
    axg.set_ylim(-0.05, 1.05)
    axg.grid(True, alpha=0.2, linestyle=':')
    # De-duplicate legend
    handles, labels = axg.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axg.legend(uniq.values(), uniq.keys(), fontsize=16, loc='best')

    fig.tight_layout()
    return fig


def compare_at_nearest_event_energy(root_file: str,
                                    channel_idx: int,
                                    desired_energy_eV: float,
                                    nbins: int = 90,
                                    show_gamma_panel: bool = True):
    """
    Find the event energy in ROOT nearest to the desired value (for the given
    channel and vibrational process), select exactly that energy (no margins),
    plot its angular histogram, and overlay the HG theoretical curve computed
    at that exact energy (γ from Michaud, mapped via hemisphere inversion).
    """
    # Load tables
    df2, df3 = _load_tables()
    e_tab, s_tab, g_tab = _extract_channel(df2, df3, channel_idx)
    if e_tab.size == 0:
        raise RuntimeError("No data found for requested channel")

    # Load ROOT
    try:
        import uproot
    except Exception as exc:
        raise ImportError("compare_at_nearest_event_energy requires uproot. pip install uproot") from exc

    root_path = resolve_first_root(root_file)
    with uproot.open(root_path) as f:
        tree = f["step"]
        arrs = tree.arrays(["cosTheta", "channelIndex", "kineticEnergy", "flagProcess"], library="np")

    # Filter to vib process and channel
    m = (arrs["flagProcess"] == 15) & (arrs["channelIndex"] == channel_idx)
    if not np.any(m):
        raise RuntimeError("No events for this channel/process in ROOT file")
    cos_all = np.asarray(arrs["cosTheta"][m], dtype=float)
    e_all = np.asarray(arrs["kineticEnergy"][m], dtype=float)

    # Find nearest event energy to the desired value
    j = int(np.argmin(np.abs(e_all - desired_energy_eV)))
    E_star = float(e_all[j])

    # Select energy with tight tolerance to avoid floating equality traps
    sel = np.isclose(e_all, E_star, rtol=0.0, atol=max(1e-9, 1e-6*max(1.0, abs(E_star))))
    if not np.any(sel):
        raise RuntimeError("No events exactly at the nearest energy (unexpected)")

    theta_sel = np.degrees(np.arccos(np.clip(cos_all[sel], -1.0, 1.0)))

    # Theoretical curve at this exact energy: map γ(E_star) -> g via hemisphere inversion
    gamma_star = float(np.interp(E_star, e_tab, g_tab))
    g_star = _invert_forward_fraction_to_g(0.5 * (1.0 + gamma_star))
    theta_grid = np.linspace(0, 180, 721)
    p = _hg_pdf_theta(theta_grid, g_star)
    # peak normalize for visual shape comparison
    if np.nanmax(p) > 0:
        p = p / np.nanmax(p)

    # Plot
    if show_gamma_panel:
        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1.0], figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        axg = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        axg = None

    # Histogram: peak-normalized
    counts, bins = np.histogram(theta_sel, bins=nbins, range=(0, 180))
    if counts.size and np.nanmax(counts) > 0:
        counts = counts.astype(float) / float(np.nanmax(counts))
        ax.step(bins[:-1], counts, where='post', color='#1f77b4', alpha=0.8, label='Simulation (peak=1)')
        ax.fill_between(bins[:-1], counts, step='post', alpha=0.20, color='#1f77b4')

    ax.plot(theta_grid, p, '-', lw=2, color='crimson', label=f'HG (g={g_star:.3f})')
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Normalized density (peak = 1)")
    ax.set_title(f"ch {channel_idx} @ E*={E_star:.3f} eV (nearest to {desired_energy_eV:.3f} eV)")
    ax.legend(loc='best')

    if axg is not None:
        axg.plot(e_tab, g_tab, 'k-', lw=2)
        axg.plot(E_star, gamma_star, 'o', ms=8, color='crimson')
        axg.set_xlabel('Kinetic energy (eV)')
        axg.set_ylabel('Anisotropy γ')
        axg.set_title('γ(E) with selected E*')
        axg.set_ylim(-0.05, 1.05)
        axg.grid(True, alpha=0.2, linestyle=':')

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage: save default plot to output folder
    try:
        fig = plot_hg_for_channel_with_gamma(1, [3., 10., 30.])
        out = OUTPUT_DIR / "hg_vib_channel_1.png"
        plt.show()
        fig.savefig(out, bbox_inches='tight')
        print(f"Wrote {out}")
    except Exception as e:
        print(f"plot_vibExcitation_channelwise_angular_distributions: skipped example save: {e}")
