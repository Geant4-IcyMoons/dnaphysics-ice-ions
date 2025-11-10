#!/usr/bin/env python3
"""
Diagnostics plotting utilities for dnaphysics-ice.

Features
- Per-process, per-channel cross-section plots (ROOT scatter; optional reference overlay).
- Vibrational energy-loss histograms per channel (for process 15).
- Deflection-angle distributions for all (process, channel) pairs.
- Compact 4-panel simulation summary.

Inputs
- ROOT: Expects a tree named "step" with columns:
  flagParticle, kineticEnergy, flagProcess, vibCrossSection, channelIndex,
  channelMicroXS, kineticEnergyDifference, cosTheta, x, y, z.
- Reference .dat (optional): First column energy (eV) followed by M partial XS
  columns (10^-16 cm^2). For vib, M=8 (channels 0..7). For elastic, M=1.

Usage
- python plot_root_vibExcitation.py --root build/dna.root --processes 15 --out xs_channels.png
- python plot_root_vibExcitation.py --root build/dna.root --processes all --out xs_channels.png
- python plot_root_vibExcitation.py --dat /abs/path/to/sigma_excitationvib_e_michaud.dat --out xs_channels.png

Notes
- If --dat is not provided, the script tries to locate reference files under
  $G4LEDATA/dna or a local g4_custom_ice install.
- Designed to be non-interactive (saves figures; no GUI required).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import uproot

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

# -------- Paths and data locations --------
# This script lives under dnaphysics-ice/python_scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Roots for locating data
GEANT4_PROJECTS_ROOT = PROJECT_ROOT.parent  # .../geant4_projects
CUSTOM_DATA_ROOT = PROJECT_ROOT / "g4_custom_ice" / "install" / "share" / "Geant4" / "data"
TABULAR_DIR = PROJECT_ROOT / "tabular"

MICHAUD_TABLE2 = str(TABULAR_DIR / "michaud_table2.csv")
MICHAUD_TABLE3 = str(TABULAR_DIR / "michaud_table3.csv")

# -------- Small helpers --------
def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return str(PROJECT_ROOT / p)

def _resolve_output(name: str) -> str:
    base = os.path.basename(name) if name else "output.png"
    return str(OUTPUT_DIR / base)


def _find_g4ledata_file(basename: str) -> str | None:
    """Find a file under G4LEDATA/dna or in the local g4_custom_ice install.

    Order:
      1) $G4LEDATA/dna/<basename>
      2) <TOP_ROOT>/g4_custom_ice/install/share/Geant4/data/G4EMLOW*/dna/<basename>
    Returns a string path if found, else None.
    """
    led = os.environ.get("G4LEDATA")
    if led:
        p = Path(led) / "dna" / basename
        if p.exists():
            return str(p)
    # 2) Local custom install with versioned G4EMLOW dir under project tree
    data_root = CUSTOM_DATA_ROOT
    if data_root.exists():
        for sub in sorted(data_root.iterdir()):
            if sub.is_dir() and sub.name.startswith("G4EMLOW"):
                candidate = sub / "dna" / basename
                if candidate.exists():
                    return str(candidate)
    return None

def _find_backup_dat(basename: str) -> str | None:
    """Fallback: look for reference .dat in backup/geant4_icyMoons."""
    p = GEANT4_PROJECTS_ROOT / "backup" / "geant4_icyMoons" / basename
    return str(p) if p.exists() else None

def _process_name_map() -> dict:
    return {
        10: "Solvation", 11: "Elastic", 12: "Excitation", 13: "Ionisation", 14: "Attachment", 15: "VibExc",
        21: "pElastic", 22: "pExcitation", 23: "pIonisation", 24: "pChgDec",
        31: "HElastic", 32: "HExcitation", 33: "HIonisation", 35: "HChgInc",
        41: "aElastic", 42: "aExcitation", 43: "aIonisation", 44: "aChgDec",
        51: "a+Elastic", 52: "a+Excitation", 53: "a+Ionisation", 54: "a+ChgDec", 55: "a+ChgInc",
        61: "HeElastic", 62: "HeExcitation", 63: "HeIonisation", 65: "HeChgInc",
        73: "GIonIonis", 110: "mscEl", 710: "msc", 720: "Coulomb", 730: "ionIoni", 740: "nucStop"
    }


# -------- Data I/O --------
def load_arrays(path: str, tree_name: str = "step"):
    """Load ROOT ntuple arrays from the given file, selecting known columns."""
    path = _resolve_path(path)
    with uproot.open(path) as f:
        if tree_name not in f:
            raise RuntimeError(f"Tree '{tree_name}' not found in {path}")
        t = f[tree_name]
        wanted = [
            "flagParticle",
            "kineticEnergy",
            "flagProcess",
            "vibCrossSection",
            "channelIndex",
            "channelMicroXS",
            "kineticEnergyDifference",
            "cosTheta",
            "x","y","z",
        ]
        available = [name for name in wanted if name in t.keys()]
        return t.arrays(available, library="np")


def load_reference_from_path(fpath: str):
    """Load reference partial XS from a .dat file.

    Format: first column is energy (eV), followed by M partial XS columns
    in 1e-16 cm^2. Returns (E_eV, ref_by_channel) where len(ref_by_channel)=M.
    """
    E: list[float] = []
    rows: list[list[float]] = []
    max_cols = 0
    with open(fpath, "r") as fin:
        for line in fin:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            e = float(parts[0])
            vals = [float(x) for x in parts[1:]]
            if not vals:
                continue
            E.append(e)
            rows.append(vals)
            if len(vals) > max_cols:
                max_cols = len(vals)
    E_arr = np.asarray(E, dtype=float)
    ref_by_ch: list[np.ndarray] = []
    for j in range(max_cols):
        col = np.array([row[j] if j < len(row) else 0.0 for row in rows], dtype=float)
        ref_by_ch.append(col)
    return E_arr, ref_by_ch


# -------- Physics helpers --------
def _to_micro_cm2(xs_macro_mm_inv_subset: np.ndarray, nH2O_cm3: float) -> np.ndarray:
    """Convert macroscopic mm^-1 to microscopic cm^2 using number density."""
    return (xs_macro_mm_inv_subset * 10.0) / float(nH2O_cm3)


# HG helpers for deflection overlays
def _hg_p_per_deg(theta_deg: np.ndarray, g: np.ndarray) -> np.ndarray:
    th = np.deg2rad(theta_deg)[:, None]
    cos_t = np.cos(th)
    sin_t = np.sin(th)
    g = np.clip(np.asarray(g, dtype=float)[None, :], -0.9999, 0.9999)
    denom = np.power(1.0 + g*g - 2.0*g*cos_t, 1.5)
    p_cos = (1.0 - g*g) / (4.0 * np.pi * denom)
    p_per_rad = p_cos * sin_t
    return p_per_rad / (np.pi / 180.0)


def _hg_forward_fraction(g: float) -> float:
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
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return float(np.clip(0.5 * (lo + hi), -0.999999, 0.999999))


def _load_michaud_gamma_for_channel(idx: int):
    import pandas as pd
    mapping = [
        ("table2", 5, 6), ("table2", 7, 8), ("table2", 9, 10),
        ("table3", 1, 2), ("table3", 3, 4), ("table3", 5, 6),
        ("table3", 7, 8), ("table3", 9, 10)
    ]
    if not (0 <= idx <= 7):
        return None, None
    table, _, col_g = mapping[idx]
    path = MICHAUD_TABLE2 if table == "table2" else MICHAUD_TABLE3
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        return None, None
    e = pd.to_numeric(df.iloc[3:, 0], errors='coerce').values.astype(float)
    g = pd.to_numeric(df.iloc[3:, col_g], errors='coerce').values.astype(float)
    m = ~(np.isnan(e) | np.isnan(g))
    e = e[m]; g = g[m]
    if e.size == 0:
        return None, None
    order = np.argsort(e)
    return e[order], g[order]


# -------- Plotters --------
def plot_cross_sections_for_process(arrs, pcode: int, ncols: int, scale: float,
                                    nH2O_cm3: float, out_path: str, dat_path: str | None = None):
    """Plot per-channel XS vs KE for a given process code and save to out_path."""
    # Extract arrays for this process
    if arrs is None:
        ke = xs_macro_mm_inv = chan_idx = chan_micro = None
    else:
        mproc = (np.asarray(arrs["flagProcess"], dtype=float) == float(pcode))
        if not np.any(mproc):
            return None
        ke = np.asarray(arrs["kineticEnergy"], dtype=float)[mproc]
        # Try to find a macroscopic cross-section field (mm^-1)
        macro_keys = [
            "vibCrossSection",  # used in this project for multiple processes
            "macroCrossSection",
            "processCrossSection",
        ]
        xs_macro_mm_inv = None
        for k in macro_keys:
            if k in arrs:
                xs_macro_mm_inv = np.asarray(arrs[k], dtype=float)[mproc]
                break
        if xs_macro_mm_inv is None:
            xs_macro_mm_inv = np.zeros_like(ke)
        chan_idx = (np.asarray(arrs["channelIndex"], dtype=int)[mproc]
                    if "channelIndex" in arrs else None)
        chan_micro = (np.asarray(arrs["channelMicroXS"], dtype=float)[mproc]
                      if "channelMicroXS" in arrs else None)

    # Reference data
    ref_E = None; ref_by_ch = None
    if dat_path:
        ref_E, ref_by_ch = load_reference_from_path(dat_path)
    else:
        if int(pcode) == 15:
            p = _find_g4ledata_file("sigma_excitationvib_e_michaud.dat")
            if not p:
                p = _find_backup_dat("sigma_excitationvib_e_michaud.dat")
            if p:
                ref_E, ref_by_ch = load_reference_from_path(p)
        elif int(pcode) == 11:
            p = _find_g4ledata_file("sigma_elastic_e_michaud.dat")
            if not p:
                p = _find_backup_dat("sigma_elastic_e_michaud.dat")
            if p:
                ref_E, ref_by_ch = load_reference_from_path(p)
        elif int(pcode) == 14:
            # Attachment (Michaud fit of 'Others' near ~4 eV), custom .dat
            p = _find_g4ledata_file("sigma_attachment_e_michaud.dat")
            if not p:
                p = _find_backup_dat("sigma_attachment_e_michaud.dat")
            if p:
                ref_E, ref_by_ch = load_reference_from_path(p)

    # Channels
    channels: List[int] = []
    if chan_idx is not None:
        pos = chan_idx[chan_idx >= 0]
        if pos.size:
            channels = np.unique(pos).tolist()
    # If process has no explicit channels, plot a single panel as channel 0
    if not channels:
        channels = [0]

    n = len(channels)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows),
        squeeze=False, sharex=True, sharey='row', constrained_layout=True
    )

    pname = _process_name_map().get(int(pcode), f"proc{int(pcode)}")
    legend_added = False
    for i, ch in enumerate(channels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        line_sim = None
        line_ref = None
        if ke is not None:
            if chan_idx is not None and ch is not None and np.any(chan_idx >= 0):
                m = (chan_idx == ch)
            else:
                m = np.ones_like(ke, dtype=bool)
            x = ke[m]
            if x.size:
                y_conv = _to_micro_cm2(xs_macro_mm_inv[m], nH2O_cm3)
                if chan_micro is not None:
                    y_micro = np.asarray(chan_micro[m], dtype=float)
                    y = np.where(y_micro > 0.0, y_micro, y_conv)
                else:
                    y = y_conv
                order = np.argsort(x)
                line_sim, = ax.plot(x[order], y[order] * scale, "--", linewidth=2, c="dodgerblue", zorder=2, label="Simulation")

        if ref_E is not None and ref_by_ch is not None:
            if ch >= 0 and ch < len(ref_by_ch):
                y_ref = ref_by_ch[ch]
                if y_ref is not None and y_ref.size == ref_E.size:
                    line_ref, = ax.plot(ref_E, y_ref, color="black", linewidth=2, alpha=0.9, zorder=1, label="Reference")
            elif len(ref_by_ch) > 0:
                y_total = np.sum(np.vstack(ref_by_ch), axis=0)
                if y_total.size == ref_E.size:
                    line_ref, = ax.plot(ref_E, y_total, color="black", linewidth=2, alpha=0.9, zorder=1, label="Reference (total)")

        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
        panel_label = f"{pname}_ch{ch}" if (chan_idx is not None and np.any(chan_idx >= 0)) else pname
        ax.set_title(panel_label)

        if not legend_added and (ch == 0 or i == 0):
            handles = []
            labels = []
            if line_ref is not None:
                handles.append(line_ref); labels.append(line_ref.get_label())
            if line_sim is not None:
                handles.append(line_sim); labels.append(line_sim.get_label())
            if handles:
                ax.legend(handles, labels, loc="best", frameon=True)
                legend_added = True

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if not ax.get_visible():
                continue
            if c != 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    for r in range(nrows - 1):
        for c in range(ncols):
            ax = axes[r][c]
            if not ax.get_visible():
                continue
            ax.set_xlabel("")
            ax.tick_params(labelbottom=True)

    fig.align_ylabels([axes[r][0] for r in range(nrows) if axes[r][0].get_visible()])
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath


def plot_vib_energy_loss_hist(arrs, ncols: int, out_path: str):
    """Plot energy-loss histograms per vib channel using kineticEnergyDifference."""
    if arrs is None or "kineticEnergyDifference" not in arrs or "channelIndex" not in arrs:
        return None
    m_vib = (np.asarray(arrs["flagProcess"], dtype=float) == 15.0)
    if not np.any(m_vib):
        return None
    dE = np.asarray(arrs["kineticEnergyDifference"], dtype=float)[m_vib]
    ch = np.asarray(arrs["channelIndex"], dtype=int)[m_vib]
    chans = np.unique(ch[ch >= 0])
    if chans.size == 0:
        return None

    n = chans.size
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), squeeze=False, sharex=True)
    vib_centers = np.array([0.024, 0.061, 0.092, 0.205, 0.417, 0.460, 0.510, 0.834], dtype=float)
    vib_b = np.array([0.025/1.665, 0.030/1.665, 0.040/1.665, 0.016/1.665, 0.050/1.665, 0.005/1.665, 0.040/1.665, 0.075/1.665], dtype=float)

    for i, cidx in enumerate(chans):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        vals = dE[ch == cidx]
        if vals.size:
            vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
            nbins = 80
            h, be, _ = ax.hist(vals, bins=nbins, range=(0.0, vmax), color='lightgray', alpha=0.8)
            if int(cidx) < vib_centers.size:
                omega = vib_centers[int(cidx)]; b = vib_b[int(cidx)]
                xg = np.linspace(0.0, vmax, 500)
                g = (1.0/(np.sqrt(np.pi)*b)) * np.exp(-((xg - omega)**2)/(b*b))
                binw = (be[1]-be[0]) if len(be) > 1 else (vmax/nbins if nbins>0 else 1.0)
                scale = float(vals.size) * binw
                ax.plot(xg, g*scale, color='black', linewidth=2.0, alpha=0.9, label='Gaussian')
        ax.set_title(f"vib_{int(cidx)}")
        ax.set_ylabel("Counts")
        ax.tick_params(axis='x', which='both', length=4, width=1.5, labelbottom=True)

    total_axes = nrows * ncols
    for j in range(chans.size, total_axes):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    for c in range(ncols):
        bottom_r = None
        for r in range(nrows-1, -1, -1):
            if axes[r][c].get_visible():
                bottom_r = r; break
        if bottom_r is not None:
            axb = axes[bottom_r][c]
            axb.set_xlabel("Energy loss dE (eV)")
            axb.tick_params(axis='x', which='both', labelbottom=True)

    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath


def plot_summary(arrs, out_path: str, fontsize: float):
    """Create a compact 4-panel summary of the simulation arrays."""
    if arrs is None:
        return None
    fp   = arrs.get("flagParticle"); fl_p = arrs.get("flagProcess")
    kinE = arrs.get("kineticEnergy"); x_nm = arrs.get("x"); y_nm = arrs.get("y"); z_nm = arrs.get("z")
    if fp is None or fl_p is None or kinE is None or x_nm is None or y_nm is None or z_nm is None:
        return None
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), squeeze=True, constrained_layout=True)

    # Panel 1: histogram of flagProcess with category overlays
    ax1 = axs[0]
    vals = np.asarray(fl_p, dtype=float)
    uniq, counts = np.unique(vals, return_counts=True)
    ax1.bar(uniq, counts, width=0.9, color="#dddddd", edgecolor="none", label="All")
    cats = {
        "Excitation": [12,15,22,32,42,52,62],
        "Elastic": [11,21,31,41,51,61,110,210,410,510,710,120,220,420,520,720],
        "Ionisation": [13,23,33,43,53,63,73,130,230,430,530,730],
    }
    colors = {"Excitation": "#2ca02c", "Elastic": "#1f77b4", "Ionisation": "#d62728"}
    for name, ids in cats.items():
        mask = np.isin(uniq, ids)
        ax1.bar(uniq[mask], counts[mask], width=0.9, color=colors[name], alpha=0.7, label=name)
    ax1.set_yscale('log'); ax1.set_xlabel("flagProcess"); ax1.set_ylabel("Counts")

    # Panel 2: 3D scatter x:y:z for electrons (downsample)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    idx = np.where(np.asarray(fp)==1)[0]
    step = max(1, idx.size // 50000) if idx.size > 50000 else 1
    idx = idx[::step]
    ax2.scatter(x_nm[idx], y_nm[idx], z_nm[idx], s=1, c='black', alpha=0.6)
    ax2.set_xlabel("x (nm)"); ax2.set_ylabel("y (nm)"); ax2.set_zlabel("z (nm)")

    # Panel 3: position histogram along x by process types
    ax3 = axs[2]
    bins = np.linspace(0, 2000, 101)
    proc_sets = {10:("Solv",'#9467bd'),11:("Elastic",'#d62728'),12:("Excit",'#2ca02c'),13:("Ionis",'#1f77b4'),14:("Attach",'#8c564b'),15:("Vib",'#e377c2')}
    for pid, (lab, col) in proc_sets.items():
        m = (np.asarray(fl_p)==pid)
        h, be = np.histogram(x_nm[m], bins=bins)
        centers = 0.5*(be[1:]+be[:-1])
        ax3.plot(centers, h, label=lab, color=col)
    ax3.set_xlabel("x (nm)"); ax3.set_yscale('log'); ax3.set_ylabel("Counts")

    # Panel 4: kinetic energy histogram for electrons
    ax4 = axs[3]
    m = (np.asarray(fp)==1)
    kmax = np.nanmax(np.asarray(kinE)[m]) if np.any(m) else np.nanmax(np.asarray(kinE))
    rng = (0, float(kmax) if np.isfinite(kmax) and kmax>0 else 2000)
    ax4.hist(np.asarray(kinE)[m], bins=100, range=rng, histtype='stepfilled', alpha=0.7, color='#d62728')
    ax4.set_yscale('log'); ax4.set_xlabel("Kinetic Energy (eV)"); ax4.set_ylabel("Counts")

    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath


def plot_deflection_angles_all(arrs, out_path: str, fontsize: float = FONTSIZE):
    """Plot deflection-angle histograms for all (process, channel) pairs."""
    if out_path is None:
        return None
    if arrs is None or "cosTheta" not in arrs or "flagProcess" not in arrs:
        return None

    cos_theta = np.asarray(arrs["cosTheta"], dtype=float)
    flag_process = np.asarray(arrs["flagProcess"], dtype=int)
    channel_index = np.asarray(arrs.get("channelIndex", np.full(flag_process.shape, -1)), dtype=int)
    kin_energy = np.asarray(arrs["kineticEnergy"], dtype=float) if "kineticEnergy" in arrs else None

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    pairs = np.column_stack((flag_process, channel_index))
    unique_pairs, first_indices = np.unique(pairs, axis=0, return_index=True)
    order = np.argsort(first_indices)
    unique_pairs = unique_pairs[order]

    n_groups = len(unique_pairs)
    if n_groups == 0:
        return None

    ncols = min(4, n_groups)
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    for i, (proc_code, ch_code) in enumerate(unique_pairs):
        ax = axes[i]
        mask = (flag_process == proc_code) & (channel_index == ch_code)
        thetas = theta_deg[mask]
        if thetas.size == 0:
            ax.set_visible(False)
            continue
        counts, bins, _ = ax.hist(thetas, bins=90, range=(0, 180), histtype="stepfilled", alpha=0.8, color='lightgray', label='Simulation')

        if int(proc_code) == 15 and 0 <= int(ch_code) <= 7 and kin_energy is not None:
            E_tab, gamma_tab = _load_michaud_gamma_for_channel(int(ch_code))
            if E_tab is not None and gamma_tab is not None and E_tab.size > 0:
                e_sel = kin_energy[mask]
                if e_sel.size > 0:
                    if e_sel.size > 5000:
                        step = max(1, e_sel.size // 5000)
                        e_sel = e_sel[::step]
                    gamma_sel = np.interp(e_sel, E_tab, gamma_tab)
                    Y_sel = 0.5 * (1.0 + gamma_sel)
                    g_sel = np.array([_invert_forward_fraction_to_g(float(Y)) for Y in Y_sel], dtype=float)
                    theta_grid = np.linspace(0, 180, 721)
                    p_mat = _hg_p_per_deg(theta_grid, g_sel)
                    p_int = np.nanmean(p_mat, axis=1)
                    if np.nanmax(p_int) > 0:
                        p_int = p_int / np.nanmax(p_int)
                        scale = np.nanmax(counts) if counts.size else 1.0
                        ax.plot(theta_grid, p_int * scale, 'k-', lw=2.0, label='HG (integrated)')

        # mean_angle = float(np.mean(thetas)); median_angle = float(np.median(thetas))
        # ax.axvline(median_angle, color="k", linestyle="--", linewidth=1.5, label=f"median={median_angle:.1f}°")
        # ax.axvline(mean_angle, color="r", linestyle=":", linewidth=1.5, label=f"mean={mean_angle:.1f}°")
        ax.set_xlim(0, 180)
        ax.set_xlabel(r"$\theta$ (deg)", fontsize=fontsize)
        ax.set_ylabel("Counts", fontsize=fontsize)
        ax.set_title(f"proc={int(proc_code)} ch={int(ch_code)} (N={thetas.size:,})", fontsize=fontsize*0.9)
        ax.legend(fontsize=fontsize*0.7, frameon=True)

    for j in range(n_groups, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Deflection angle distributions (all processes & channels)", fontsize=fontsize*1.2)
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Plot cross-sections, deflection angles, and summaries from dna.root and optional reference .dat")
    ap.add_argument("--root", default="build/dna.root", help="Path to ROOT file (default: build/dna.root)")
    ap.add_argument("--process", type=int, default=15, help="Single process code to plot when --processes is not given (default: 15)")
    ap.add_argument("--processes", default=None, help="Comma-separated process codes or 'all' to iterate over all present in ROOT")
    ap.add_argument("--dat", default=None, help="Absolute path to a reference .dat file (energy + partial XS columns). Overrides auto lookup")
    ap.add_argument("--out", default="xs_channels.png", help="Base output filename for XS plots (suffixes per process)")
    ap.add_argument("--summary_out", default="summary_panels.png", help="Output image for 4-panel summary")
    ap.add_argument("--deflection_out", default="deflection_angles.png", help="Output image for deflection-angle distributions")
    ap.add_argument("--de_hist_out", default="de_hist_vib.png", help="Output image for vib energy-loss histograms")
    ap.add_argument("--ncols", type=int, default=5, help="Max panels per row (default: 5)")
    ap.add_argument("--scale", type=float, default=1e16, help="Y-scale multiplier for microscopic XS (default: 1e16)")
    ap.add_argument("--nH2O_cm3", type=float, default=3.343e22, help="Number density (cm^-3) for macro→micro conversion")
    ap.add_argument("--fontsize", type=float, default=18, help="Base font size for ticks, labels, titles, legend")
    ap.add_argument("--summary", action="store_true", help="Also write the 4-panel summary figure")
    args = ap.parse_args()

    # Font sizes are controlled globally via FONTSIZE

    arrs = None
    root_path = _resolve_path(args.root)
    if os.path.exists(root_path):
        arrs = load_arrays(root_path, "step")

    # Determine processes to plot
    if arrs is not None and (args.processes is None or str(args.processes).strip().lower() == 'all'):
        proc_list = np.unique(np.asarray(arrs["flagProcess"], dtype=float)).astype(int).tolist()
    elif args.processes:
        proc_list = [int(x.strip()) for x in str(args.processes).split(',') if x.strip()]
    else:
        proc_list = [int(args.process)]

    # Plot per-process XS
    base, ext = os.path.splitext(args.out)
    pname_map = _process_name_map()
    for pcode in proc_list:
        suffix = pname_map.get(int(pcode), str(int(pcode)))
        outname = f"{base}_{suffix}{ext or '.png'}"
        plot_cross_sections_for_process(
            arrs=arrs,
            pcode=int(pcode),
            ncols=args.ncols,
            scale=args.scale,
            nH2O_cm3=args.nH2O_cm3,
            out_path=outname,
            dat_path=args.dat,
        )

    # Vib energy-loss histograms (if available)
    plot_vib_energy_loss_hist(arrs, args.ncols, args.de_hist_out)

    # Summary and deflection angles
    plot_summary(arrs, args.summary_out, fontsize=FONTSIZE)
    plot_deflection_angles_all(arrs, args.deflection_out, fontsize=FONTSIZE)


if __name__ == "__main__":
    main()
