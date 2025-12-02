#!/usr/bin/env python3
"""
Plot elastic scattering cross-sections for electrons in ice:
- G4DNAMichaudElasticModel: 2-100 eV
- G4DNAScreenedRutherfordElasticModel: 100 eV - 1 MeV
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Font configuration
font = 'Courier'
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'
FONTSIZE = 18

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
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.major.width': 1.5,
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


def screened_rutherford_cross_section(energy_eV):
    """
    Calculate Screened Rutherford elastic cross-section for electrons in water/ice.
    """
    # Physical constants (Geant4 system)
    e_squared = 1.4399764  # MeV * fm
    electron_mass_c2 = 0.510998928  # MeV
    z = 10.0  # Effective charge for water (H2O)
    
    # Convert eV to MeV
    k_MeV = energy_eV * 1e-6
    
    # Rutherford cross-section formula
    length_fm = (e_squared * (k_MeV + electron_mass_c2)) / \
                (k_MeV * (k_MeV + 2 * electron_mass_c2))
    sigma_ruth_fm2 = z * (z + 1) * length_fm**2
    
    # Screening factor formula
    alpha_1 = 1.64
    beta_1 = -0.0825
    constK = 1.7e-5
    
    numerator = (alpha_1 + beta_1 * np.log(energy_eV)) * constK * (z**(2./3.))
    k_ratio = k_MeV / electron_mass_c2
    denominator = k_ratio * (2 + k_ratio)
    
    n = np.where(denominator > 0, numerator / denominator, 0)
    sigma_fm2 = np.pi * sigma_ruth_fm2 / (n * (n + 1.0))
    
    # Convert from fm^2 to cm^2
    sigma_cm2 = sigma_fm2 * 1e-26
    
    return sigma_cm2


def plot_elastic_cross_sections():
    """
    Plot elastic cross-sections: Michaud from CSV and Screened Rutherford.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load Michaud data from CSV Table 2
    try:
        csv_path = Path(__file__).parent.parent / "tabular" / "michaud_table2.csv"
        df = pd.read_csv(csv_path, skiprows=3, header=None)
        E_michaud = pd.to_numeric(df[0], errors="coerce").to_numpy()
        sigma_raw = pd.to_numeric(df[1], errors="coerce").to_numpy()
        
        # Filter valid data
        mask = np.isfinite(E_michaud) & np.isfinite(sigma_raw)
        E_michaud = E_michaud[mask]
        sigma_raw = sigma_raw[mask]
        
        # Apply scale factor: 1e-16 cm^2
        sigma_michaud = sigma_raw * 1e-16
        
        print(f"Loaded Michaud CSV data: {len(E_michaud)} points")
        print(f"Energy range: {E_michaud[0]:.1f} - {E_michaud[-1]:.1f} eV")
        print(f"Michaud @ {E_michaud[-1]:.1f} eV: {sigma_michaud[-1]:.6e} cm²")
        
        # Plot Michaud
        ax.loglog(E_michaud, sigma_michaud, color='lightgray', linewidth=5,
                 label='Michaud+03', zorder=3)
        # Plot Michaud x2
        # ax.loglog(E_michaud, sigma_michaud * 2, color='darkgray', linewidth=2,
        #          label='Michaud × 2', zorder=3)
        
    except Exception as e:
        print(f"Error loading Michaud: {e}")
    
    # Load ELSEPA elastic cross-sections (muffin potential) to use for high-energy branch
    try:
        elsepa_path = Path(__file__).parent.parent / "cross_sections" / "sigma_elastic_e_elsepa_muffin.dat"
        data_elsepa = np.loadtxt(elsepa_path)
        E_elsepa = data_elsepa[:, 0]
        sigma_elsepa = data_elsepa[:, 1]
        ax.loglog(E_elsepa, sigma_elsepa, color='slategray', linewidth=5,
                  label='ELSEPA (muffin)', zorder=3)
        print(f"Loaded ELSEPA elastic data: {len(E_elsepa)} points, {E_elsepa[0]:.1f}-{E_elsepa[-1]:.1f} eV")
    except Exception as e:
        E_elsepa = sigma_elsepa = None
        print(f"Error loading ELSEPA elastic data: {e}")
    
    # Create blended cross-section using C1-smooth smoothstep kernel
    try:
        # Define transition parameters
        E0 = 100.0  # Start of transition
        t = 494   # End of transition
        
        # Create blended cross-section array spanning full range
        E_blend = np.logspace(np.log10(2), np.log10(1e7), 500)
        sigma_blend = np.zeros_like(E_blend)

        # Target at 100 eV: regular Michaud
        target_100 = sigma_michaud[-1]
        
        for i, E in enumerate(E_blend):
            if E < E0:
                # Below 100 eV: Use regular Michaud (interpolate from loaded data)
                sigma_m = np.interp(E, E_michaud, sigma_michaud)
                sigma_blend[i] = sigma_m
            elif E >= t and E_elsepa is not None:
                # Above transition: pure ELSEPA (interpolated/extrapolated)
                sigma_blend[i] = float(np.interp(E, E_elsepa, sigma_elsepa, left=sigma_elsepa[0], right=sigma_elsepa[-1]))
            else:
                # Between 100-t eV: smoothstep blend
                s = (E - E0) / (t - E0)
                w = s * s * (3 - 2 * s)  # C1-smooth weight function
                sigma_sr_E = float(np.interp(E, E_elsepa, sigma_elsepa, left=sigma_elsepa[0], right=sigma_elsepa[-1])) if E_elsepa is not None else 0.0
                
                # Blend: start at Michaud (at 100 eV), end at SR (at t eV)
                sigma_blend[i] = (1 - w) * target_100 + w * sigma_sr_E
        
        ax.loglog(E_blend, sigma_blend, 'k-.', linewidth=2,
                 label='Blended', zorder=3)
        
        # Print values at key points
        idx_2 = np.argmin(np.abs(E_blend - 2))
        print(f"Blended @ 2 eV: {sigma_blend[idx_2]:.6e} cm²")
        idx_100 = np.argmin(np.abs(E_blend - 100))
        print(f"Blended @ 100 eV: {sigma_blend[idx_100]:.6e} cm²")
        idx_490 = np.argmin(np.abs(E_blend - 490))
        print(f"Blended @ 490 eV: {sigma_blend[idx_490]:.6e} cm²")
        
    except Exception as e:
        print(f"Error creating blended cross-section: {e}")

    # Mark transition zone
    # ax.axvline(100, color='red', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
    # ax.axvline(t, color='red', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
    ax.axvspan(100, t, color='lightgray', alpha=0.3, zorder=0)
    
    # Formatting
    ax.set_xlabel('Electron Energy (eV)')
    ax.set_ylabel(r'Cross-Section (cm$^2$)')
    # ax.set_title('Electron Elastic Scattering Cross-Sections in Ice')
    ax.legend(loc='best')
    # ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_xlim(1, 1e7)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(__file__).parent / 'output/elastic_cross_sections.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.show()


def plot_vibrational_excitations():
    """
    Plot vibrational excitation cross-sections from Michaud et al. 2003.
    Two panels: intermolecular (phonons + librations) and intramolecular modes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Use plasma colormap (without bright yellow at the end)
    cmap = plt.cm.plasma
    
    # ============== PANEL 1: INTERMOLECULAR MODES ==============
    try:
        csv_path = Path(__file__).parent.parent / "tabular" / "michaud_table2.csv"
        df = pd.read_csv(csv_path, skiprows=3, header=None)
        
        E = pd.to_numeric(df[0], errors="coerce").to_numpy()
        
        # Intermolecular modes from Table 2
        # Each mode has (sigma, gamma) pairs
        # v'_T (phonon 1): columns 3-4
        # v''_T (phonon 2): columns 5-6  
        # v'_L (libration 1): columns 7-8
        # v''_L (libration 2): columns 9-10
        
        modes_inter = [
            ("$v'_{\\mathrm{T}}$", 3),
            ("$v''_{\\mathrm{T}}$", 5),
            ("$v'_{\\mathrm{L}}$", 7),
            ("$v''_{\\mathrm{L}}$", 9),
        ]
        
        n_modes = len(modes_inter)
        for i, (label, col_idx) in enumerate(modes_inter):
            # Use 0.0 to 0.85 range to avoid bright yellow
            color = cmap(i / (n_modes - 1) * 0.85)
            sigma_raw = pd.to_numeric(df[col_idx], errors="coerce").to_numpy()
            mask = np.isfinite(E) & np.isfinite(sigma_raw) & (sigma_raw > 0)
            E_valid = E[mask]
            sigma = sigma_raw[mask] * 1e-16  # Apply scale factor
            
            ax1.plot(E_valid, sigma, color=color, linewidth=3, label=label, zorder=3)
        
        # ax1.set_xlabel('Electron Energy (eV)')
        ax1.set_ylabel(r'Cross-Section (cm$^2$)')
        ax1.legend(loc='best')
        ax1.set_xlim(1, 100)
        
        print("Loaded intermolecular modes from Table 2")
        
    except Exception as e:
        print(f"Error loading intermolecular modes: {e}")
    
    # ============== PANEL 2: INTRAMOLECULAR MODES ==============
    try:
        csv_path = Path(__file__).parent.parent / "tabular" / "michaud_table3.csv"
        df = pd.read_csv(csv_path, skiprows=3, header=None)
        
        E = pd.to_numeric(df[0], errors="coerce").to_numpy()
        
        # Intramolecular modes from Table 3
        # v_2 (bending): columns 1-2
        # v_1,3 (stretching): columns 3-4
        # v_3 (asymmetric stretch): columns 5-6
        # v_1,3+v_L (combination): columns 7-8
        # 2v_1,3 (overtone): columns 9-10
        
        modes_intra = [
            ("$v_2$", 1),
            ("$v_{1,3}$", 3),
            ("$v_3$", 5),
            ("$v_{1,3}+v_{\\mathrm{L}}$", 7),
            ("$2v_{1,3}$", 9),
        ]
        
        n_modes = len(modes_intra)
        for i, (label, col_idx) in enumerate(modes_intra):
            # Use 0.0 to 0.85 range to avoid bright yellow
            color = cmap(i / (n_modes - 1) * 0.85)
            sigma_raw = pd.to_numeric(df[col_idx], errors="coerce").to_numpy()
            mask = np.isfinite(E) & np.isfinite(sigma_raw) & (sigma_raw > 0)
            E_valid = E[mask]
            sigma = sigma_raw[mask] * 1e-16  # Apply scale factor
            
            ax2.plot(E_valid, sigma, color=color, linewidth=3, label=label, zorder=3)
        
        ax2.set_xlabel('Electron Energy (eV)')
        ax2.set_ylabel(r'Cross-Section (cm$^2$)')
        ax2.legend(loc='best')
        ax2.set_xlim(1, 100)
        
        print("Loaded intramolecular modes from Table 3")
        
    except Exception as e:
        print(f"Error loading intramolecular modes: {e}")
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(__file__).parent / 'output/vibrational_excitations.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\nVibrational excitations plot saved to: {output_file}")
    
    plt.show()


def plot_vibrational_energy_distributions():
    """
    Plot Gaussian energy distributions for vibrational excitation modes.
    Two panels: intermolecular and intramolecular modes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Use plasma colormap (without bright yellow)
    cmap = plt.cm.plasma
    
    # Energy loss parameters from G4DNAMichaudExcitationModel
    # Centers (eV) and FWHM values (eV)
    modes_inter = [
        ("$v''_{\\mathrm{T}}$", 0.024, 0.025),  # phonon
        ("$v'_{\\mathrm{L}}$", 0.061, 0.030),   # libration 1
        ("$v''_{\\mathrm{L}}$", 0.092, 0.040),  # libration 2
    ]
    
    modes_intra = [
        ("$v_2$", 0.205, 0.016),                      # bending
        ("$v_{1,3}$", 0.417, 0.050),                  # stretching
        ("$v_3$", 0.460, 0.005),                      # asymmetric stretch
        ("$v_{1,3}+v_{\\mathrm{L}}$", 0.510, 0.040),  # combination
        ("$2v_{1,3}$", 0.834, 0.075),                 # overtone
    ]
    
    # Gaussian function: g(ω) = 1/(√π b) exp(-(ω-ω₀)²/b²)
    # where b = FWHM / (2√ln2) ≈ FWHM / 1.665
    def gaussian(omega, omega0, fwhm):
        b = fwhm / 1.665
        return (1 / (np.sqrt(np.pi) * b)) * np.exp(-((omega - omega0) / b)**2)
    
    # ============== PANEL 1: INTERMOLECULAR MODES ==============
    n_modes = len(modes_inter)
    for i, (label, omega0, fwhm) in enumerate(modes_inter):
        color = cmap(i / (n_modes - 1) * 0.85)
        
        # Energy range around the center
        omega_range = np.linspace(max(0, omega0 - 3*fwhm), omega0 + 3*fwhm, 500)
        g_omega = gaussian(omega_range, omega0, fwhm)
        
        ax1.plot(omega_range * 1000, g_omega, color=color, linewidth=3, label=label, zorder=3)
    
    ax1.set_ylabel('Probability Density (eV$^{-1}$)')
    ax1.legend(loc='best')
    ax1.set_xlim(0, 150)
    
    # ============== PANEL 2: INTRAMOLECULAR MODES ==============
    n_modes = len(modes_intra)
    for i, (label, omega0, fwhm) in enumerate(modes_intra):
        color = cmap(i / (n_modes - 1) * 0.85)
        
        # Energy range around the center
        omega_range = np.linspace(max(0, omega0 - 3*fwhm), omega0 + 3*fwhm, 500)
        g_omega = gaussian(omega_range, omega0, fwhm)
        
        ax2.plot(omega_range, g_omega, color=color, linewidth=3, label=label, zorder=3)
    
    ax2.set_xlabel('Energy Loss (eV)')
    ax2.set_ylabel('Probability Density (eV$^{-1}$)')
    ax2.legend(loc='best')
    ax2.set_xlim(0.15, 0.95)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(__file__).parent / 'output/vibrational_energy_distributions.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\nVibrational energy distributions plot saved to: {output_file}")
    
    plt.show()


if __name__ == "__main__":
    plot_elastic_cross_sections()
    plot_vibrational_excitations()
    plot_vibrational_energy_distributions()
