#!/usr/bin/env python3
"""
Generate differential cross-section .dat files for the 8 vibrational-excitation
channels using Michaud et al. (2003) Tables 2 & 3, with angular anisotropy via a
von Mises–Fisher kernel.

Inputs:
  - michaud_table2.csv  # intermolecular: v''(T), v'(L), v''(L)
  - michaud_table3.csv  # intramolecular: v2, v1,3, v3, v1,3+vL, 2(v1,3)

Outputs (single file with all 8 channels):
  - sigmadiff_excitationvib_e_michaud.dat
    Format: E[eV]  θ[deg]  dσ/dΩ_vT2  dσ/dΩ_vL1  dσ/dΩ_vL2  dσ/dΩ_v2  dσ/dΩ_v1,3  dσ/dΩ_v3  dσ/dΩ_v1,3+vL  dσ/dΩ_2(v1,3)
    Units: [eV, deg, 1e-16 cm^2/sr × 8 channels]

  - sigmadiff_cumulated_excitationvib_e_michaud_hp.dat
    Format: E[eV]  CDF[0-1]  θ[deg]  (repeated for cumulative sampling)
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

from constants import (
    BACKUP_MICHAUD_TABLE2_PATH,
    BACKUP_MICHAUD_TABLE3_PATH,
    FMT_C,
    FMT_E,
    FMT_T,
    FMT_XS,
    FONT_COURIER,
    FONTSIZE_12,
    HFONT_COURIER,
    OUTPUT_DIR as OUTPUT_DIR_PATH,
    RC_BASE_STANDARD,
    VIB_E_MAX,
    VIB_E_MIN,
    VIB_N_E,
    VIB_N_TH,
    VIB_TARGETS,
    VIB_THETA_MAX,
    VIB_THETA_MIN,
    rcparams_with_fontsize,
)

font = FONT_COURIER
hfont = HFONT_COURIER
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'

# Resolve paths; always write outputs under python_scripts/output
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
TABLE2_PATH = str(BACKUP_MICHAUD_TABLE2_PATH)
TABLE3_PATH = str(BACKUP_MICHAUD_TABLE3_PATH)
OUTPUT_DIR  = str(OUTPUT_DIR_PATH)

# Unified font size
fontsize = FONTSIZE_12
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_STANDARD, fontsize))

E_MIN, E_MAX, N_E = VIB_E_MIN, VIB_E_MAX, VIB_N_E
THETA_MIN, THETA_MAX, N_TH = VIB_THETA_MIN, VIB_THETA_MAX, VIB_N_TH

# 8 vibrational channels in order - note we use what's actually in Michaud tables
# Table 2: v''(T), v'(L), v''(L)  
# Table 3: v2, v1,3, v3, v1,3+vL, 2(v1,3)
# Note: v'(r) has mostly zero/empty values, so we skip it
TARGETS = VIB_TARGETS

# ---------- I/O helpers ----------

def read_table(path: str) -> pd.DataFrame:
    """
    Read CSV with three header rows:
      Row 0: "E_(0)(eV)", "Excitation modes", ...
      Row 1: "", "Elastic", "", "v^(')_(r)", "", "v^('')_(T)", ...
      Row 2: "", "sigma", "gamma", "sigma", "gamma", ...
    Returns a tidy DataFrame with columns:
        E0_eV, <mode>_a, <mode>_y   for each recognized mode
    """
    # Read first to get column structure
    raw = pd.read_csv(path, header=[0,1,2])
    
    df = pd.DataFrame()
    
    # First column is energy - find it
    energy_col = None
    for col in raw.columns:
        top = col[0] if isinstance(col, tuple) else col
        if isinstance(top, str) and ("E_(0)" in top or "e_0" in top.lower()):
            energy_col = col
            break
    if energy_col is None:
        energy_col = raw.columns[0]
    
    df["E0_eV"] = pd.to_numeric(raw[energy_col], errors="coerce")
    
    # Parse mode columns - sigma and gamma come in pairs
    # sigma column has the mode label, gamma column (next one) has "Unnamed" in middle
    accum: Dict[str, Dict[str, tuple]] = {}
    
    col_list = list(raw.columns)
    i = 0
    while i < len(col_list):
        col = col_list[i]
        
        if col == energy_col:
            i += 1
            continue
        
        if not isinstance(col, tuple) or len(col) < 3:
            i += 1
            continue
        
        top, middle, bottom = col[0], col[1], col[2]
        
        # Check if this is a sigma column
        if str(bottom).strip().lower() == "sigma":
            label_str = str(middle).strip() if pd.notna(middle) and "Unnamed" not in str(middle) else ""
            
            if label_str:
                canon = canonize_mode_label(label_str)
                
                if canon is not None:
                    # Add sigma column
                    accum.setdefault(canon, {})
                    accum[canon]['a'] = col
                    print(f"  Found sigma: {label_str} -> {canon}")
                    
                    # Check if next column is gamma for the same mode
                    if i + 1 < len(col_list):
                        next_col = col_list[i + 1]
                        if isinstance(next_col, tuple) and len(next_col) >= 3:
                            _, _, next_bottom = next_col[0], next_col[1], next_col[2]
                            if str(next_bottom).strip().lower() == "gamma":
                                accum[canon]['y'] = next_col
                                print(f"  Found gamma: {label_str} -> {canon}")
                                i += 1  # Skip the gamma column
        
        i += 1
    
    # Build columns for modes with both sigma & gamma
    for canon, d in accum.items():
        if 'a' in d and 'y' in d:
            df[f"{canon}_a"] = pd.to_numeric(raw[d['a']], errors="coerce")
            df[f"{canon}_y"] = pd.to_numeric(raw[d['y']], errors="coerce")
    
    # Drop non-numeric energy rows and sort
    df = df.dropna(subset=["E0_eV"]).sort_values("E0_eV").reset_index(drop=True)
    return df

def canonize_mode_label(s: str) -> str | None:
    """
    Map messy LaTeX-ish labels to canonical keys.
    Table 2 examples seen in CSV:
      'Elastic', "v^(')_(r)"  [~10 meV translational], "v^('')_(T)",
      "v^(')_(L)", "v^('')_(L)"
    Table 3 examples:
      "v_(2)", "v_(1,3)", "v_(3)", "v_(t,3)+v_(L)", "2(v_(1,3))", "Others"
    Returns one of:
      vT1, vT2, vL1, vL2, v2, v1,3, v3, v1,3+vL, 2(v1,3), Others, Elastic
    or None to ignore.
    """
    t = s.strip()
    t = t.replace("’","'").replace("`","'").replace("“", '"').replace("”", '"')
    t_l = re.sub(r"\s+", "", t.lower())

    # Common direct mappings
    if "elastic" in t_l:
        return "Elastic"
    if "others" in t_l:
        return "Others"

    # Table 3 intramolecular vibrations
    if re.fullmatch(r"v_?\(?2\)?", t_l):
        return "v2"
    if re.fullmatch(r"v_?\(?1,?3\)?", t_l) or re.fullmatch(r"v_?13", t_l):
        return "v1,3"
    if re.fullmatch(r"v_?\(?3\)?", t_l):
        return "v3"
    # combination: v1,3 + vL  (CSV sometimes shows v_(t,3)+v_(L) due to OCR)
    if "v_(t,3)+v_(l)" in t.lower() or "v1,3+v(l" in t_l or "v13+vl" in t_l or "v1,3+vl" in t_l:
        return "v1,3+vL"
    if re.search(r"^2\(?v_?\(?1,?3\)?\)?", t_l):
        return "2(v1,3)"

    # Table 2 intermolecular phonons:
    # Detect primes (single vs double) and mode family (T or L or r for translational)
    # Examples: v^(')_(r), v^('')_(T), v^(')_(L), v^('')_(L)
    is_trans = "(t)" in t_l or "(_t)" in t_l or "(_r)" in t_l or "(r)" in t_l
    is_lib   = "(l)" in t_l or "(_l)" in t_l
    
    # Count prime symbols more carefully
    # Double prime: ('')  or  "  or ''
    has_double_prime = ("('')" in t or '""' in t or "''" in t)
    # Single prime: (')
    has_single_prime = ("(')" in t or "'" in t) and not has_double_prime

    if is_trans or "(_r)" in t_l:
        return "vT2" if has_double_prime else "vT1"
    if is_lib:
        return "vL2" if has_double_prime else "vL1"

    # Unknown -> ignore
    return None

# ---------- Physics helpers ----------

def interp_linear(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    xi = x[m]; yi = y[m]
    idx = np.argsort(xi)
    return float(np.interp(xq, xi[idx], yi[idx]))

def p_solid_angle(theta_rad: np.ndarray, gamma: float) -> np.ndarray:
    """
    Henyey-Greenstein phase function: normalized density over solid angle.
    Integrates to 1 over 4π steradians.
    
    Returns p(cos θ) in units of 1/sr.
    
    Parameters:
        theta_rad: scattering angle in radians
        gamma: anisotropy parameter (g in HG notation)
               g = 0  → isotropic
               g → 1  → forward scattering
               g → -1 → backward scattering
    """
    g = float(np.clip(gamma, -0.9999, 0.9999))
    
    if abs(g) < 1e-14:
        # Isotropic limit
        return np.full_like(theta_rad, 1.0 / (4.0 * np.pi))
    
    cos_theta = np.cos(theta_rad)
    
    # Henyey-Greenstein: p(cos θ) = (1 - g²) / [4π(1 + g² - 2g cos θ)^(3/2)]
    denominator = (1.0 + g*g - 2.0*g*cos_theta)**(1.5)
    return (1.0 - g*g) / (4.0 * np.pi * denominator)

# ---------- HG forward-fraction mapping (γ as hemisphere-based) ----------

def hg_forward_fraction(g: float) -> float:
    """
    Forward hemisphere probability under Henyey–Greenstein with parameter g.
    Y(g) = ∫_{0}^{π/2} 2π sinθ p_HG(θ|g) dθ.
    Closed form: Y = (1+g)/(2g) * [1 - (1-g)/sqrt(1+g^2)] for g!=0.
    Handle g≈0 by limit Y→1/2.
    """
    if not np.isfinite(g):
        return 0.5
    g = float(np.clip(g, -0.999999, 0.999999))
    if abs(g) < 1e-12:
        return 0.5
    A0 = 1.0 + g*g
    term = (1.0 - g) / np.sqrt(A0)
    return (1.0 + g) / (2.0 * g) * (1.0 - term)

def invert_forward_fraction_to_g(Y: float, tol: float = 1e-10, maxit: int = 100) -> float:
    """
    Given forward hemisphere fraction Y in [0,1], find g ∈ (-1,1) such that
    hg_forward_fraction(g) = Y via bisection. Monotone in g.
    """
    if not np.isfinite(Y):
        return 0.0
    Y = float(np.clip(Y, 0.0, 1.0))
    if Y <= 0.0:
        return -0.999999
    if Y >= 1.0:
        return 0.999999
    lo, hi = -0.999999, 0.999999
    f_lo = hg_forward_fraction(lo) - Y
    f_hi = hg_forward_fraction(hi) - Y
    # Ensure proper bracketing (should hold given monotonicity)
    if f_lo > 0 and f_hi > 0:
        return 0.0
    if f_lo < 0 and f_hi < 0:
        return 0.0
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid = hg_forward_fraction(mid) - Y
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return float(np.clip(mid, -0.999999, 0.999999))
        # Bisection step
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(np.clip(0.5 * (lo + hi), -0.999999, 0.999999))

# ---------- Writers ----------

def write_sigmadiff_combined(out_dir: Path, Egrid: np.ndarray, th_deg: np.ndarray, 
                             dsdO_dict: Dict[str, np.ndarray]):
    """
    Write single file with all 8 channels:
    E[eV]  θ[deg]  dσ/dΩ_ch0  dσ/dΩ_ch1  ...  dσ/dΩ_ch7
    """
    with open(out_dir / "sigmadiff_excitationvib_e_michaud.dat", "w") as f:
        for i, e in enumerate(Egrid):
            for j, th in enumerate(th_deg):
                line = f"{FMT_E % e} {FMT_T % th}"
                for key in TARGETS:
                    line += f" {FMT_XS % dsdO_dict[key][i, j]}"
                f.write(line + "\n")

def write_sigmadiff_cumulated_combined(out_dir: Path, Egrid: np.ndarray, th_deg: np.ndarray,
                                      cdf_dict: Dict[str, np.ndarray]):
    """
    Write cumulative file for all 8 channels (for sampling):
    E[eV]  channel  CDF  θ[deg]
    Each channel has its own CDF based on its own gamma parameter
    """
    with open(out_dir / "sigmadiff_cumulated_excitationvib_e_michaud_hp.dat", "w") as f:
        for i, e in enumerate(Egrid):
            for ch_idx, key in enumerate(TARGETS):
                cdf_channel = cdf_dict[key][i, :]
                for j, th in enumerate(th_deg):
                    f.write(f"{FMT_E % e} {ch_idx} {FMT_C % cdf_channel[j]} {FMT_T % th}\n")

# ---------- Main pipeline ----------

def plot_anisotropy_and_angles(out_dir: Path, Egrid: np.ndarray, th_deg: np.ndarray, 
                                ch: Dict, targets: list, dsdO_dict: Dict[str, np.ndarray]):
    """
    Create six-panel plot (3x2 grid):
    - Panel 1: Anisotropy index γ(E) for each channel
    - Panels 2-6: Angular distribution dσ/dΩ(cos θ) at 5, 10, 25, 50, 99 eV
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    ax1 = axes[0]
    angular_axes = axes[1:]  # axes 1-5 for angular distributions
    
    # Define colors for each channel
    colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
    
    # Panel 1: Anisotropy index γ vs Energy
    for i, key in enumerate(targets):
        E_src, a_src, y_src = ch[key]
        ax1.plot(E_src, y_src, 'o-', label=key, color=colors[i], markersize=4, linewidth=2)
    
    ax1.set_xlabel('Energy (eV)', fontsize=fontsize)
    ax1.set_ylabel('Anisotropy $\gamma$', fontsize=fontsize)
    ax1.set_title('Anisotropy Index $\gamma$(E)', fontsize=fontsize, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(loc='best', fontsize=fontsize)
    ax1.set_ylim(-0.1, 1.1)
    ax1.tick_params(labelsize=fontsize)
    
    # Panels 2-6: Angular distributions at selected energies
    selected_energies = [5.0, 10.0, 25.0, 50.0, 99.0]
    th_rad = np.deg2rad(th_deg)
    cos_theta = np.cos(th_rad)
    
    for panel_idx, E_sel in enumerate(selected_energies):
        ax = angular_axes[panel_idx]
        
        # Find closest energy index
        idx = np.argmin(np.abs(Egrid - E_sel))
        E_actual = Egrid[idx]
        
        # Track how many channels have data
        channels_plotted = 0
        
        for i, key in enumerate(targets):
            # Get dσ/dΩ for this channel at this energy
            dsdO_E = dsdO_dict[key][idx, :]
            
            # Skip if all values are zero or negligible
            if np.max(np.abs(dsdO_E)) < 1e-30:
                continue
                
            # Normalize to peak for better visualization
            if np.max(dsdO_E) > 0:
                dsdO_norm = dsdO_E / np.max(dsdO_E)
            else:
                dsdO_norm = dsdO_E
            
            ax.plot(cos_theta, dsdO_norm, '-', color=colors[i], 
                   linewidth=1.5, label=key, alpha=0.8)
            channels_plotted += 1
        
        ax.set_xlabel('cos($\\theta$)', fontsize=fontsize)
        ax.set_ylabel('Normalized d$\sigma$/d$\Omega$', fontsize=fontsize)
        ax.set_title(f'E = {E_actual:.1f} eV ({channels_plotted}/{len(targets)} channels)', fontsize=fontsize, fontweight='bold')
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=fontsize)
        
        # Only show legend on first angular distribution panel
        if panel_idx == 0:
            ax.legend(loc='best', fontsize=fontsize, ncol=2)
    
    plt.tight_layout()
    plot_path = out_dir / "vibexc_anisotropy_and_angular_dist.pdf"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")
    plt.close()


def build_channel_dict(df2: pd.DataFrame, df3: pd.DataFrame) -> Dict[str, Tuple[np.ndarray,np.ndarray,np.ndarray]]:
    """
    Return {canon_key: (E, sigma, gamma)} for all channels available across both tables.
    """
    out: Dict[str, Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}

    def harvest(df: pd.DataFrame):
        # any column ending with _a / _y pair defines a channel
        cols = list(df.columns)
        E = df["E0_eV"].values.astype(float)
        for c in cols:
            if c.endswith("_a") and c[:-2] + "_y" in cols:
                key = c[:-2]
                if key in ("Elastic","Others"):  # ignore
                    continue
                sig = df[c].values.astype(float)
                gam = df[key + "_y"].values.astype(float)
                out[key] = (E, sig, gam)

    harvest(df2)
    harvest(df3)
    return out

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df2 = read_table(TABLE2_PATH)
    df3 = read_table(TABLE3_PATH)

    # Build a dictionary of all parsed channels
    ch = build_channel_dict(df2, df3)

    # Sanity print
    print("Parsed channels:", ", ".join(sorted(ch.keys())))

    # Ensure we have the 8 targets
    missing = [k for k in TARGETS if k not in ch]
    if missing:
        raise RuntimeError(f"Missing channels after parsing: {missing}. "
                           f"Check the CSV headers/labels.")

    # Grids
    Egrid = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_E)
    th_deg = np.linspace(THETA_MIN, THETA_MAX, N_TH)
    th_rad = np.deg2rad(th_deg)
    sin_th = np.sin(th_rad)

    # Store all channels' differential cross sections
    dsdO_dict = {}
    cdf_dict = {}

    # Process each target channel
    for key in TARGETS:
        E_src, a_src, y_src = ch[key]
        sig = np.array([interp_linear(E_src, a_src, e) for e in Egrid], dtype=float)
        # Michaud γ is treated as hemisphere-based anisotropy: γ = 2Y - 1
        # Map to HG parameter g by inverting the HG forward fraction Y(g) = (1+γ)/2
        gam_raw = np.array([interp_linear(E_src, y_src, e) for e in Egrid], dtype=float)
        gam_raw = np.clip(gam_raw, -0.999999, 0.999999)
        Y_target = 0.5 * (1.0 + gam_raw)
        gam = np.array([invert_forward_fraction_to_g(float(Y)) for Y in Y_target], dtype=float)

        dsdO = np.empty((Egrid.size, th_deg.size), dtype=float)
        cdf  = np.empty_like(dsdO)

        for i, (e, s, g) in enumerate(zip(Egrid, sig, gam)):
            pth = p_solid_angle(th_rad, float(g))   # 1/sr using HG with g mapped from γ via hemisphere constraint
            dsdO[i, :] = s * pth                    # 1e-16 cm^2/sr

            # cumulative over θ with 2π sinθ weighting
            integrand = dsdO[i, :] * 2.0 * np.pi * sin_th
            F = np.cumsum(np.concatenate([[0.0],
                  0.5*(integrand[1:] + integrand[:-1]) * np.diff(th_rad)]))
            total = F[-1] if F[-1] > 0 else 1.0
            C = F / total
            if C.size == th_deg.size + 1:
                C = C[:-1]
            cdf[i, :] = np.maximum.accumulate(C)

        dsdO_dict[key] = dsdO
        cdf_dict[key] = cdf
        print(f"Processed channel: {key}")

    # Write combined files
    write_sigmadiff_combined(out_dir, Egrid, th_deg, dsdO_dict)
    write_sigmadiff_cumulated_combined(out_dir, Egrid, th_deg, cdf_dict)

    print("\nWrote combined files:")
    print("  sigmadiff_excitationvib_e_michaud.dat")
    print("  sigmadiff_cumulated_excitationvib_e_michaud_hp.dat")
    
    # Plot anisotropy and angular distributions
    plot_anisotropy_and_angles(out_dir, Egrid, th_deg, ch, TARGETS, dsdO_dict)
    
    print("\nDone. Place outputs under $G4LEDATA/dna/")

if __name__ == "__main__":
    main()
