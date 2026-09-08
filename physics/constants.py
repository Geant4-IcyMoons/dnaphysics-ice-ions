"""Shared constants, ice phases, projectile registry, and output locations."""
from __future__ import annotations
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIELECTRIC_ROOT = PROJECT_ROOT / "physics" / "inelastic_dielectric"
OUTPUT_DIR = DIELECTRIC_ROOT / "output" / "caches"
CROSS_SECTIONS_DIR = DIELECTRIC_ROOT / "output" / "tables"
# Legacy DAT values times this area scale give m^2/eV (DCS) or m^2 (TCS).
EMFI_DCS_SCALE_M2 = 1.0e-22 / 3.343
CROSS_SECTION_PLOTS_DIR = DIELECTRIC_ROOT / "plots"
DIAGNOSTIC_PLOTS_DIR = DIELECTRIC_ROOT / "plots"
DIELECTRIC_PLOTS_DIR = DIELECTRIC_ROOT / "finite_q" / "benchmarking" / "plots"
ICE_DATA_XLSX_PATH = DIELECTRIC_ROOT / "finite_q" / "data" / "ice data.xlsx"
DEFAULT_WORKERS = 10
WATER_TOTAL_OSCILLATOR_STRENGTH = 10.0


# --- Plot styles ---
# Nimbus Mono PS is the installed, metrically compatible Courier family used
# for paper figures on the cluster.  Using its registered Matplotlib name
# avoids silently falling back to an unrelated proportional sans-serif font.
FONT_COURIER = "Nimbus Mono PS"

HFONT_COURIER = {"fontname": FONT_COURIER}

FONTSIZE_16 = 16

FONTSIZE_18 = 18

FONTSIZE_24 = 24


# Shared full-width, three-panel paper-figure geometry.
AASTEX_FULL_WIDTH_IN = 7.1

THREE_PANEL_ROW_HEIGHT_IN = 5.15 / 2.0

PAPER_FONTSIZE = 8.0


RC_BASE_STANDARD = {
    "axes.linewidth": 1.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markerfacecolor": "white",
    "lines.markeredgecolor": "k",
    "xtick.major.size": 0,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 0,
    "xtick.minor.width": 1.5,
    "xtick.direction": "in",
    "xtick.major.pad": 5,
    "ytick.major.size": 0,
    "ytick.major.width": 1.5,
    "ytick.minor.size": 0,
    "ytick.minor.width": 1.5,
    "ytick.direction": "in",
    "axes.titleweight": "normal",
    "axes.titlepad": 20,
}


RC_BASE_ELASTIC = {
    "axes.linewidth": 1.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markerfacecolor": "white",
    "lines.markeredgecolor": "k",
    "xtick.major.size": 8,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 4,
    "xtick.minor.width": 1.5,
    "xtick.direction": "in",
    "xtick.major.pad": 5,
    "ytick.major.size": 8,
    "ytick.minor.size": 4,
    "ytick.major.width": 1.5,
    "ytick.minor.width": 1.5,
    "ytick.direction": "in",
    "axes.titleweight": "normal",
    "axes.titlepad": 20,
}



def rcparams_with_fontsize(base, fontsize, overrides=None):
    rc = dict(base)
    rc.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
        }
    )
    if overrides:
        rc.update(overrides)
    return rc



# --- Physical constants and conversions ---
EH = 27.211386245988  # eV, Hartree

RY = 13.605693009  # eV, Rydberg

EV_TO_HA = 1.0 / EH


a0 = 5.291e-11


# Projectile masses in electron-mass atomic units and bare-ion charges.
# Bound-electron screening is supplied separately by projectile_form_factors.
# The charge below is always nuclear Z, not an ionic charge or fitted Zeff.
PROTON_MASS_AU = 1836.152673

ALPHA_MASS_AU = 7294.299536

CARBON_12_BARE_MASS_AU = 21868.6618

# Bare O-16 nuclear mass in electron-mass atomic units.  Derived from the
# NIST neutral-atom relative mass 15.99491461957 u by subtracting eight
# electron masses and restoring the 2043.8429988 eV total electronic binding
# energy (the sum of the NIST O0--O7+ successive ionization energies).
OXYGEN_16_BARE_MASS_AU = 29148.9497

# Bare S-32 nuclear mass in electron-mass atomic units.  Derived from the
# NIST neutral-atom relative mass 31.9720711744 u by subtracting sixteen
# electron masses and restoring the 10859.5983847 eV total electronic binding
# energy (the sum of the NIST S0--S15+ successive ionization energies).
SULFUR_32_BARE_MASS_AU = 58265.5417

PROJECTILE_LIBRARY = {
    "proton": {
        "aliases": ("p", "h+", "proton", "h", "hydrogen"),
        "element": "H",
        "mass_au": PROTON_MASS_AU,
        "mass_number": 1.0,
        "charge": 1.0,
        "file_token": "proton",
        "label": "Proton",
    },
    "alpha": {
        "aliases": ("alpha", "he2+", "helium", "he"),
        "element": "He",
        "mass_au": ALPHA_MASS_AU,
        "mass_number": 4.0,
        "charge": 2.0,
        "file_token": "alpha",
        "label": "Alpha particle He2+",
    },
    "carbon": {
        "element": "C",
        "aliases": ("carbon", "c6+", "carbon6+", "c"),
        "mass_au": CARBON_12_BARE_MASS_AU,
        "mass_number": 12.0,
        "charge": 6.0,
        "file_token": "carbon",
        "label": "Carbon ion C6+",
    },
    "oxygen": {
        "element": "O",
        "aliases": ("oxygen", "o", "o8+", "oxygen8+"),
        "mass_au": OXYGEN_16_BARE_MASS_AU,
        "mass_number": 16.0,
        "charge": 8.0,
        "file_token": "oxygen",
        "label": "Oxygen ion O8+ constant-charge approximation",
    },
    "sulfur": {
        "element": "S",
        "aliases": ("sulfur", "s", "s16+", "sulfur16+"),
        "mass_au": SULFUR_32_BARE_MASS_AU,
        "mass_number": 32.0,
        "charge": 16.0,
        "file_token": "sulfur",
        "label": "Sulfur ion S16+ constant-charge approximation",
    },
}


# Molecular-density / mass-density mapping for H2O
AVOGADRO = 6.02214076e23  # mol^-1

H2O_MOLAR_MASS_G_MOL = 18.01528


# Ice phase mass densities (g/cm^3). Ice Ih is the 100 K density from the
# corrected Rottger et al. experimental lattice fit. LDA is derived from the
# validated 3000-water, 45.796719 A EPSR model archived at 10.5286/edata/729.
ICE_HEXAGONAL_DENSITY_G_CM3 = 0.9335

ICE_AMORPHOUS_DENSITY_G_CM3 = 0.9343471678603292


C_AU = 137.035999084

MC2_eV = 510998.95

MC2_HA = MC2_eV * EV_TO_HA

REGIME_II_MAX_eV = 1.0e5


ELF_ROLLOFF_E0_eV = 5.0e4

ELF_ROLLOFF_COEF = 0.05
