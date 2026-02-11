from __future__ import annotations

from pathlib import Path

import numpy as np

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GEANT4_PROJECTS_ROOT = PROJECT_ROOT.parent
TOP_ROOT = PROJECT_ROOT.parent.parent

OUTPUT_DIR = SCRIPT_DIR / "output"
CROSS_SECTIONS_DIR = PROJECT_ROOT / "cross_sections"
TABULAR_DIR = PROJECT_ROOT / "tabular"
BACKUP_TABULAR_DIR = TOP_ROOT / "backup" / "geant4_icyMoons" / "tabular"

CUSTOM_DATA_ROOT_GEANT4 = (
    GEANT4_PROJECTS_ROOT / "g4_custom_ice" / "install" / "share" / "Geant4" / "data"
)
CUSTOM_DATA_ROOT_PROJECT = (
    PROJECT_ROOT / "g4_custom_ice" / "install" / "share" / "Geant4" / "data"
)

MICHAUD_TABLE2_PATH = TABULAR_DIR / "michaud_table2.csv"
MICHAUD_TABLE3_PATH = TABULAR_DIR / "michaud_table3.csv"
BACKUP_MICHAUD_TABLE2_PATH = BACKUP_TABULAR_DIR / "michaud_table2.csv"
BACKUP_MICHAUD_TABLE3_PATH = BACKUP_TABULAR_DIR / "michaud_table3.csv"
ICE_DATA_XLSX_PATH = TABULAR_DIR / "ice data.xlsx"

ELSEPA_MUFFIN_TOTAL = CROSS_SECTIONS_DIR / "sigma_elastic_e_elsepa_muffin.dat"
ELSEPA_MUFFIN_CDF = CROSS_SECTIONS_DIR / "sigmadiff_cumulated_elastic_e_elsepa_muffin.dat"

# --- Plot styles ---
FONT_COURIER = "Courier"
FONT_DEJAVU_SANS = "DejaVu Sans"
HFONT_COURIER = {"fontname": FONT_COURIER}
HFONT_DEJAVU_SANS = {"fontname": FONT_DEJAVU_SANS}

FONTSIZE_12 = 12
FONTSIZE_16 = 16
FONTSIZE_18 = 18
FONTSIZE_24 = 24

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

RC_BASE_MINIMAL = {}


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
N = 3.34e28
mass = 1.0

# Molecular-density / mass-density mapping for H2O
AVOGADRO = 6.02214076e23  # mol^-1
H2O_MOLAR_MASS_G_MOL = 18.01528

# Ice phase nominal mass densities (g/cm^3)
ICE_HEXAGONAL_DENSITY_G_CM3 = 0.917
ICE_AMORPHOUS_DENSITY_G_CM3 = 0.94

# Reference mass density implied by N above (used in XS normalization code)
N_REFERENCE_DENSITY_G_CM3 = (N / 1.0e6) * H2O_MOLAR_MASS_G_MOL / AVOGADRO

C_AU = 137.035999084
MC2_eV = 510998.95
MC2_HA = MC2_eV * EV_TO_HA

REGIME_I_MAX_eV = 1.0e3
REGIME_II_MAX_eV = 1.0e5
REGIME_III_MAX_eV = 1.0e6
REGIME_IV_MAX_eV = 1.0e7

ELF_ROLLOFF_E0_eV = 5.0e4
ELF_ROLLOFF_COEF = 0.05

EV_TO_MEV = 1e-6
FM2_TO_CM2 = 1e-26

SR_E_SQUARED_MEV_FM = 1.4399764
SR_ELECTRON_MASS_MEV = 0.510998928
SR_Z_WATER = 10.0
SR_ALPHA_1 = 1.64
SR_BETA_1 = -0.0825
SR_CONST_K = 1.7e-5

MICHAUD_SIGMA_SCALE_CM2 = 1e-16
MELTON_SIGMA_SCALE_CM2 = 1e-18
EMFIETZOGLOU_SCALE_1E16 = (1e-22 / 3.343) * 1e4 / 1e-16

# --- Emfietzoglou channel markers ---
EMFI_EXCITATION_EEV = np.array([8.22, 10.00, 11.24, 12.61, 13.77], dtype=float)
EMFI_ION_BINDING_EEV = np.array([10.0, 13.0, 17.0, 32.2, 539.7], dtype=float)
EMFI_EXCITATION_TOL_EEV = 2.0

# --- Elastic blending parameters ---
BLEND_E_MIN_LOW = 1.7
BLEND_E_SPLIT = 200.0
BLEND_E_MAX = 1.0e7 + 1.0

ELASTIC_BLEND_E0 = 100.0
ELASTIC_BLEND_T = 494.0
ELASTIC_BLEND_E_MIN = 2.0
ELASTIC_BLEND_E_MAX = 1.0e7
ELASTIC_BLEND_N_POINTS = 500

# --- Vibrational excitation tables ---
VIB_CHANNEL_MAPPING = [
    ("table2", 5, 6),
    ("table2", 7, 8),
    ("table2", 9, 10),
    ("table3", 1, 2),
    ("table3", 3, 4),
    ("table3", 5, 6),
    ("table3", 7, 8),
    ("table3", 9, 10),
]

VIB_TARGET_NAMES = [
    "vT2",
    "vL1",
    "vL2",
    "v2",
    "v1,3",
    "v3",
    "v1,3+vL",
    "2(v1,3)",
]
VIB_TARGETS = VIB_TARGET_NAMES

VIB_E_MIN = 1.7
VIB_E_MAX = 100.0
VIB_N_E = 201
VIB_THETA_MIN = 0.0
VIB_THETA_MAX = 180.0
VIB_N_TH = 181

FMT_E = "%.8f"
FMT_XS = "%.10e"
FMT_T = "%.8f"
FMT_C = "%.10f"
