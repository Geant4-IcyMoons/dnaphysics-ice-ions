#!/usr/bin/env python3
"""
Diagnostics plotting utilities for dnaphysics-ice.

Features
- Per-process cross-section plots (one panel per process; optional reference overlay).
- Elastic XS reference vs simulation multi-panel (one panel per elastic process found).
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
import warnings

import numpy as np
import matplotlib.pyplot as plt
import uproot

font = 'Courier'
hfont = {'fontname': font}
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'
FONTSIZE = 16

plt.rcParams.update({
    'axes.linewidth': 1.5,
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


def _clean_string(val: object) -> str:
    """Decode bytes, drop null-terminated tail, and strip whitespace."""
    if val is None:
        return ""
    if isinstance(val, (bytes, bytearray)):
        s = val.decode(errors="ignore")
    else:
        s = str(val)
    if "\x00" in s:
        s = s.split("\x00", 1)[0]
    return s.strip()


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


def _decode_to_str_array(arr: np.ndarray | None) -> np.ndarray | None:
    """Convert a numpy array of bytes/strings into a str array (None -> None)."""
    if arr is None:
        return None
    out: list[str] = []
    for v in np.asarray(arr, dtype=object):
        out.append(_clean_string(v))
    return np.asarray(out, dtype=object)


def _resolve_elastic_reference_path(hints: list[str], fallback: str | None = None) -> str | None:
    """Pick the best-matching elastic reference .dat based on model/process hints."""
    if fallback:
        candidate = _resolve_path(fallback)
        if os.path.exists(candidate):
            return candidate
    for h in hints:
        if not h:
            continue
        m = h.lower()
        if "elsepa_low" in m:
            fname = "sigma_elastic_e_michaud_elsepa_low.dat"
        elif "elsepa_high" in m:
            fname = "sigma_elastic_e_michaud_elsepa_high.dat"
        elif "michaud" in m:
            fname = "sigma_elastic_e_michaud.dat"
        else:
            fname = None
        if not fname:
            continue
        p = _find_g4ledata_file(fname)
        if not p:
            p = _find_backup_dat(fname)
        if p:
            return p
    return None


def _sanitize_label(label: str) -> str:
    """Return a filesystem-friendly label."""
    if not label:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(label))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _select_sim_cross_section(arrs, mask, nH2O_cm3: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (ke, xs_micro_cm2) for a given boolean mask."""
    ke = np.asarray(arrs["kineticEnergy"], dtype=float)[mask]
    macro_keys = ["vibCrossSection", "macroCrossSection", "processCrossSection"]
    xs_macro = None
    for k in macro_keys:
        if k in arrs:
            xs_macro = np.asarray(arrs[k], dtype=float)[mask]
            break
    if xs_macro is None:
        xs_macro = np.zeros_like(ke)
    xs_micro_cm2 = _to_micro_cm2(xs_macro, nH2O_cm3)
    if "channelMicroXS" in arrs:
        xs_micro = np.asarray(arrs["channelMicroXS"], dtype=float)[mask]
        xs_micro_cm2 = np.where(xs_micro > 0.0, xs_micro, xs_micro_cm2)
    return ke, xs_micro_cm2


def print_root_processes(arrs) -> None:
    """Print unique processes present in the ROOT file."""
    if arrs is None:
        print("No ROOT data loaded; nothing to list.")
        return
    if "flagProcess" not in arrs:
        print("flagProcess column not found; cannot list processes.")
        return

    flag_proc = np.asarray(arrs["flagProcess"], dtype=int)
    proc_names = _decode_to_str_array(arrs.get("processName"))
    model_names = _decode_to_str_array(arrs.get("modelName"))
    uniq, counts = np.unique(flag_proc, return_counts=True)

    print(f"Found {uniq.size} unique flagProcess entries in ROOT:")
    for code, cnt in zip(uniq, counts):
        label = _process_name_map().get(int(code), f"proc{int(code)}")
        mask = (flag_proc == code)
        names = None
        models = None
        if proc_names is not None:
            names = sorted([n for n in np.unique(proc_names[mask]) if n])
        if model_names is not None:
            models = sorted([m for m in np.unique(model_names[mask]) if m])
        names_str = f" | processName: {', '.join(names)}" if names else ""
        models_str = f" | modelName: {', '.join(models)}" if models else ""
        print(f"  {int(code):>4d}  {label:<12} steps={cnt}{names_str}{models_str}")


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
            "processName",
            "modelName",
        ]
        available = [name for name in wanted if name in t.keys()]
        # uproot string interpretation can emit harmless overflow warnings; silence them for cleaner CLI output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in scalar add", category=RuntimeWarning)
            return t.arrays(available, library="np")


def load_reference_from_path(fpath: str):
    """Load reference partial XS from a .dat file.

    Format: first column is energy (eV), followed by M partial XS columns.

    By default the columns are interpreted as being in 1e-16 cm^2 (Michaud
    vibrational / elastic tables).  For some data sets (notably the
    Emfietzoglou ionisation table used by G4DNAEmfietzoglouIonisationModel),
    the caller applies an extra scale factor to convert the raw table units
    to 1e-16 cm^2 before plotting.

    Returns (E_eV, ref_by_channel) where len(ref_by_channel)=M and each
    element is a NumPy array of the same length as E_eV.
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
                                    nH2O_cm3: float, out_path: str, dat_path: str | None = None,
                                    model_filter: str | None = None):
    """Plot XS vs KE per channel for a given process (optionally filtered by modelName)."""
    if arrs is None:
        return None
    mask_proc = (np.asarray(arrs["flagProcess"], dtype=float) == float(pcode))
    if model_filter and "modelName" in arrs:
        models = _decode_to_str_array(arrs["modelName"])
        if models is not None:
            mask_proc = mask_proc & (models == model_filter)
    if not np.any(mask_proc):
        return None

    ke_all = np.asarray(arrs["kineticEnergy"], dtype=float)[mask_proc]
    chan_idx = (np.asarray(arrs["channelIndex"], dtype=int)[mask_proc]
                if "channelIndex" in arrs else None)
    chan_micro = (np.asarray(arrs["channelMicroXS"], dtype=float)[mask_proc]
                  if "channelMicroXS" in arrs else None)

    macro_keys = ["vibCrossSection", "macroCrossSection", "processCrossSection"]

    # Reference data
    model_hints: list[str] = []
    if "modelName" in arrs and np.any(mask_proc):
        models = np.asarray(arrs["modelName"], dtype=object)[mask_proc]
        unique = np.unique(models[models != b""])
        for u in unique:
            model_hints.append(u.decode(errors="ignore") if isinstance(u, (bytes, bytearray)) else str(u))
    ref_E = None
    ref_by_ch = None
    ref_label = None
    if dat_path:
        ref_E, ref_by_ch = load_reference_from_path(dat_path)
        ref_label = "Reference"
    elif int(pcode) == 11 and model_hints:
        p = _resolve_elastic_reference_path(model_hints)
        if p:
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_label = Path(p).name
    elif int(pcode) == 15:
        p = _find_g4ledata_file("sigma_excitationvib_e_michaud.dat")
        if not p:
            p = _find_backup_dat("sigma_excitationvib_e_michaud.dat")
        if p:
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_label = Path(p).name

    channels: List[int] = []
    if chan_idx is not None:
        pos = chan_idx[chan_idx >= 0]
        if pos.size:
            channels = np.unique(pos).tolist()
    if not channels:
        channels = [0]

    n = len(channels)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    figsize = (10, 6) if n == 1 else (5 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize,
        squeeze=False, sharex=True, sharey='row', constrained_layout=False
    )

    pname = _process_name_map().get(int(pcode), f"proc{int(pcode)}")
    legend_added = False

    for i, ch in enumerate(channels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if chan_idx is not None and np.any(chan_idx >= 0):
            m = (chan_idx == ch)
        else:
            m = np.ones_like(ke_all, dtype=bool)
        if not np.any(m):
            ax.set_visible(False)
            continue

        ke = ke_all[m]
        if ke.size == 0:
            ax.set_visible(False)
            continue
        ke_order = np.argsort(ke)

        xs_macro = None
        for k in macro_keys:
            if k in arrs:
                xs_macro = np.asarray(arrs[k], dtype=float)[mask_proc][m]
                break
        if xs_macro is None:
            xs_macro = np.zeros_like(ke)
        xs_micro_cm2 = _to_micro_cm2(xs_macro, nH2O_cm3)
        if chan_micro is not None:
            xs_micro_sel = chan_micro[m]
            xs_micro_cm2 = np.where(xs_micro_sel > 0.0, xs_micro_sel, xs_micro_cm2)

        line_sim, = ax.plot(
            ke[ke_order],
            xs_micro_cm2[ke_order] * scale,
            "--",
            linewidth=2,
            c="dodgerblue",
            label="Simulation",
            zorder=2,
        )

        lines_ref = []
        if ref_E is not None and ref_by_ch:
            if 0 <= ch < len(ref_by_ch):
                y_ref = ref_by_ch[ch]
            else:
                y_ref = np.sum(np.vstack(ref_by_ch), axis=0)
            lr, = ax.plot(ref_E, y_ref, color="black", linewidth=2.5, alpha=0.9, label=ref_label or "Reference", zorder=1)
            lines_ref.append(lr)

        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
        panel_label = f"{pname}_ch{ch}" if (chan_idx is not None and np.any(chan_idx >= 0)) else pname
        ax.set_title(panel_label)
        span_vals = []
        if ref_E is not None and ref_E.size:
            pos_ref = ref_E[ref_E > 0]
            if pos_ref.size:
                span_vals.append(np.nanmin(pos_ref))
                span_vals.append(np.nanmax(pos_ref))
        if ke.size:
            kpos = ke[ke > 0]
            if kpos.size:
                span_vals.append(np.nanmin(kpos))
                span_vals.append(np.nanmax(kpos))
        if span_vals:
            e_min = float(np.nanmin(span_vals))
            e_max = float(np.nanmax(span_vals))
            if e_max / max(e_min, 1e-30) > 1e3:
                ax.set_xscale("log")

        if not legend_added and (lines_ref or line_sim is not None):
            handles = []
            labels = []
            if lines_ref:
                handles.extend(lines_ref)
                labels.extend([lr.get_label() for lr in lines_ref])
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
    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath


def plot_elastic_reference_vs_sim(arrs, ncols: int, scale: float, nH2O_cm3: float,
                                  out_path: str, dat_path: str | None = None):
    """Multi-panel elastic XS comparison: reference (black) vs simulation (blue dashed)."""
    if arrs is None or "kineticEnergy" not in arrs:
        return None

    proc_names = _decode_to_str_array(arrs.get("processName"))
    model_names = _decode_to_str_array(arrs.get("modelName"))
    flag_proc = np.asarray(arrs["flagProcess"], dtype=int) if "flagProcess" in arrs else None

    macro_keys = [
        "vibCrossSection",
        "macroCrossSection",
        "processCrossSection",
    ]

    groups: list[dict] = []
    if proc_names is not None:
        elastic_mask = np.array([("elastic" in str(p).lower()) for p in proc_names], dtype=bool)
        uniq_names = np.unique(proc_names[elastic_mask])
        for name in uniq_names:
            if not name:
                continue
            mask = (proc_names == name)
            groups.append({"label": name, "mask": mask})
    if not groups and flag_proc is not None:
        elastic_codes = [11, 21, 31, 41, 51, 61, 110, 210, 410, 510, 710]
        uniq_codes = np.unique(flag_proc[np.isin(flag_proc, elastic_codes)])
        for code in uniq_codes:
            mask = (flag_proc == code)
            label = _process_name_map().get(int(code), f"proc{int(code)}")
            groups.append({"label": label, "mask": mask})
    if not groups:
        return None

    n = len(groups)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows),
        squeeze=False, sharex=True, sharey='row', constrained_layout=False
    )

    legend_added = False
    for i, g in enumerate(groups):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        mask = g["mask"]

        ke = np.asarray(arrs["kineticEnergy"], dtype=float)[mask]
        if ke.size == 0:
            ax.set_visible(False)
            continue

        xs_macro = None
        for k in macro_keys:
            if k in arrs:
                xs_macro = np.asarray(arrs[k], dtype=float)[mask]
                break
        if xs_macro is None:
            ax.set_visible(False)
            continue

        xs_micro_cm2 = _to_micro_cm2(xs_macro, nH2O_cm3)
        if "channelMicroXS" in arrs:
            xs_micro = np.asarray(arrs["channelMicroXS"], dtype=float)[mask]
            xs_micro_cm2 = np.where(xs_micro > 0.0, xs_micro, xs_micro_cm2)

        order = np.argsort(ke)
        line_sim, = ax.plot(
            ke[order],
            xs_micro_cm2[order] * scale,
            "--",
            linewidth=2,
            c="dodgerblue",
            zorder=2,
            label="Simulation",
        )

        model_hints: list[str] = []
        if model_names is not None:
            uniq_models = np.unique(model_names[mask])
            model_hints.extend([m for m in uniq_models if m])
        model_hints.append(g.get("label", ""))

        ref_line = None
        ref_E = None
        ref_path = _resolve_elastic_reference_path(model_hints, fallback=dat_path)
        if ref_path:
            try:
                ref_E, ref_by_ch = load_reference_from_path(ref_path)
                if ref_by_ch:
                    y_ref = ref_by_ch[0] if len(ref_by_ch) else None
                    if y_ref is not None and y_ref.size:
                        ref_line, = ax.plot(
                            ref_E,
                            y_ref,
                            color="black",
                            linewidth=2,
                            alpha=0.9,
                            zorder=1,
                            label=Path(ref_path).name if ref_path else "Reference",
                        )
            except Exception:
                ref_line = None

        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
        ax.set_title(g["label"])

        span_vals = []
        if ref_E is not None and ref_E.size:
            pos_ref = ref_E[ref_E > 0]
            if pos_ref.size:
                span_vals.append(np.nanmin(pos_ref))
                span_vals.append(np.nanmax(ref_E))
        if ke.size:
            kpos = ke[ke > 0]
            if kpos.size:
                span_vals.append(np.nanmin(kpos))
                span_vals.append(np.nanmax(kpos))
        if span_vals:
            e_min = float(np.nanmin(span_vals))
            e_max = float(np.nanmax(span_vals))
            if e_max / max(e_min, 1e-30) > 1e3:
                ax.set_xscale("log")

        if not legend_added and (ref_line is not None or line_sim is not None):
            handles = []
            labels = []
            if ref_line is not None:
                handles.append(ref_line); labels.append(ref_line.get_label())
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
    fig.tight_layout()
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


def _born_angular_distribution(theta_deg, E_kin_eV, E_sec_eV):
    """
    Compute theoretical Born approximation angular distribution for ionisation.
    
    Based on G4DNABornAngle::SampleDirectionForShell:
    - E_sec < 50 eV: Isotropic (uniform in cos(theta))
    - 50 eV <= E_sec <= 200 eV: 90% forward peaked (0-45°), 10% isotropic
    - E_sec > 200 eV: Born approximation formula
    
    Args:
        theta_deg: array of angles in degrees
        E_kin_eV: incident kinetic energy in eV
        E_sec_eV: secondary electron energy in eV
    
    Returns:
        Normalized PDF values at each theta
    """
    theta_rad = np.deg2rad(theta_deg)
    
    if E_sec_eV < 50:
        # Isotropic: uniform in cos(theta)
        pdf = np.sin(theta_rad) / 2.0  # d(cos)/dtheta = sin(theta), integral over hemisphere = 2
    elif E_sec_eV <= 200:
        # Mixed forward/isotropic
        # 90% in forward cone (0-45°), 10% isotropic
        forward_mask = theta_deg <= 45
        pdf = np.zeros_like(theta_rad)
        pdf[forward_mask] = 0.9 / (2 * np.pi * (1 - np.cos(np.pi/4))) * np.sin(theta_rad[forward_mask])
        pdf[~forward_mask] = 0.1 * np.sin(theta_rad[~forward_mask]) / 2.0
    else:
        # Born approximation: P(theta) ~ sin^3(theta) / (1 + E_sec/(2*m_e*c^2) - cos(theta))^2
        # where sin^2(theta) = (1 - E_sec/E_kin) / (1 + E_sec/(2*m_e*c^2))
        m_e_c2 = 510998.95  # electron rest mass in eV
        sin2_max = (1 - E_sec_eV/E_kin_eV) / (1 + E_sec_eV/(2*m_e_c2))
        sin2 = np.sin(theta_rad)**2
        
        # Born formula (approximate)
        pdf = (np.sin(theta_rad) * sin2) / (1 + E_sec_eV/(2*m_e_c2) - np.cos(theta_rad))**2
        
        # Suppress unphysical angles where sin^2 > sin2_max
        pdf[sin2 > sin2_max] = 0
    
    # Normalize
    integral = np.trapz(pdf, theta_rad)
    if integral > 0:
        pdf /= integral
    
    return pdf


def plot_deflection_angles_all(arrs, out_path: str, fontsize: float = FONTSIZE):
    """Plot deflection-angle histograms per (process, model) pair (channels aggregated), one figure each."""
    if out_path is None:
        return None
    if arrs is None or "cosTheta" not in arrs or "flagProcess" not in arrs:
        return None

    cos_theta = np.asarray(arrs["cosTheta"], dtype=float)
    flag_process = np.asarray(arrs["flagProcess"], dtype=int)
    model_names = _decode_to_str_array(arrs.get("modelName"))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    # Build groups keyed by (proc, model)
    keys_list = []
    if model_names is not None:
        for p, m in zip(flag_process, model_names):
            keys_list.append((int(p), m if m else ""))
    else:
        for p in flag_process:
            keys_list.append((int(p), ""))
    unique_keys = []
    seen = set()
    for k in keys_list:
        if k in seen:
            continue
        seen.add(k)
        unique_keys.append(k)

    if not unique_keys:
        return None

    base, ext = os.path.splitext(out_path)
    for proc_code, model_name in unique_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        if model_names is not None:
            mask = (flag_process == int(proc_code)) & (model_names == model_name)
        else:
            mask = (flag_process == int(proc_code))
        thetas = theta_deg[mask]
        if thetas.size == 0:
            plt.close(fig)
            continue
        ax.hist(thetas, bins=90, range=(0, 180), histtype="stepfilled", alpha=0.8, color='lightgray')
        ax.set_xlim(0, 180)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel("Counts")
        title = model_name if model_name else f"proc {int(proc_code)}"
        ax.set_title(title)
        fig.tight_layout()
        suffix = f"proc{int(proc_code)}"
        if model_name:
            suffix += f"_{_sanitize_label(model_name)}"
        outname = f"{base}_{suffix}{ext or '.png'}"
        out_resolved = _resolve_output(outname)
        plt.show()
        fig.savefig(out_resolved, bbox_inches="tight")
        print(f"Wrote {out_resolved}")
        plt.close(fig)
    return out_path


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

    print_root_processes(arrs)

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
        # If elastic has multiple models, emit one plot per model
        if int(pcode) == 11 and arrs is not None and "modelName" in arrs:
            models_all = _decode_to_str_array(arrs["modelName"])
            if models_all is not None:
                mproc = (np.asarray(arrs["flagProcess"], dtype=float) == float(pcode))
                uniq_models = np.unique(models_all[mproc])
                for mname in uniq_models:
                    if not mname:
                        continue
                    msafe = _sanitize_label(mname)
                    outname = f"{base}_{suffix}_{msafe}{ext or '.png'}"
                    plot_cross_sections_for_process(
                        arrs=arrs,
                        pcode=int(pcode),
                        ncols=args.ncols,
                        scale=args.scale,
                        nH2O_cm3=args.nH2O_cm3,
                        out_path=outname,
                        dat_path=args.dat,
                        model_filter=mname,
                    )
                continue

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
