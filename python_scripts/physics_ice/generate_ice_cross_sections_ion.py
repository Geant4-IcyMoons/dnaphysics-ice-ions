#!/usr/bin/env python3

"""Generate phase-specific ion excitation and ionisation tables for ice.

The optional relativistic projectile kernel implements the finite-Q RPWBA
DDCS of Dominguez-Munoz et al., Radiat. Phys. Chem. 199 (2022) 110363,
doi:10.1016/j.radphyschem.2022.110363, Eqs. (1)-(4), with the condensed-medium
Fermi density correction of Eqs. (7)-(9). Their liquid-water GOS is replaced
by this repository's phase-specific finite-q ice dielectric response. The
paper validates protons from 100 to 300 MeV; use for other bare ions is an
explicit first-Born extrapolation at the same projectile velocity.
"""

import os, sys
import json
import shutil
from functools import lru_cache
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.constants import elementary_charge, electron_mass, epsilon_0, hbar
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (
    C_AU,
    CROSS_SECTION_PLOTS_DIR,
    CUSTOM_DATA_ROOT_GEANT4,
    CUSTOM_DATA_ROOT_PROJECT,
    CROSS_SECTIONS_DIR,
    EH,
    EV_TO_HA,
    ELF_ROLLOFF_COEF,
    ELF_ROLLOFF_E0_eV,
    FONT_COURIER,
    FONTSIZE_24,
    MC2_HA,
    MC2_eV,
    AVOGADRO,
    H2O_MOLAR_MASS_G_MOL,
    ICE_AMORPHOUS_DENSITY_G_CM3,
    ICE_HEXAGONAL_DENSITY_G_CM3,
    OUTPUT_DIR,
    PROJECTILE_LIBRARY,
    PROTON_MASS_AU,
    RC_BASE_ELASTIC,
    REGIME_I_MAX_eV,
    REGIME_II_MAX_eV,
    REGIME_III_MAX_eV,
    REGIME_IV_MAX_eV,
    a0,
    rcparams_with_fontsize,
)
import emfietzoglou_model_finite_q as model
import barkas_dcs

PROJECTILE_KEY = "proton"
PROJECTILE_MASS_AU = PROTON_MASS_AU
PROJECTILE_CHARGE = 1.0
PROJECTILE_MASS_NUMBER = 1.0
PROJECTILE_FILE_TOKEN = "proton"
PROJECTILE_LABEL = "Proton"
CHARGE_MODE = "bare"
EXPLICIT_PROJECTILE_CHARGE = None
INCLUDE_BARKAS_DCS = False
INCLUDE_BLOCH_DCS = False
ENERGY_UNIT = "total"
BORN_REFERENCE_CHARGE = "bare_Z"
BORN_REFERENCE_EXPLICIT_CHARGE = None
PROJECTILE_RELATIVISTIC_DCS = False
INCLUDE_TRANSVERSE_DCS = False
RPWBA_DENSITY_EFFECT = False
RPWBA_MODEL_NAME = "dominguez-munoz-2022-finite-Q"
RPWBA_REFERENCE_DOI = "10.1016/j.radphyschem.2022.110363"
DEFAULT_ENERGY_MIN_EV = 1.0e6
DEFAULT_ENERGY_MAX_EV = 1.0e8
DEFAULT_ENERGY_POINTS = 1000
DEFAULT_ENERGY_GRID = "log"
DEFAULT_DE_POINTS = 1000
DEFAULT_DQ_POINTS = 1000

def _projectile_config(key):
    lookup = str(key).strip().lower()
    for name, cfg in PROJECTILE_LIBRARY.items():
        aliases = cfg.get("aliases", ())
        if lookup == name or lookup in aliases:
            return name, cfg
    supported = ", ".join(PROJECTILE_LIBRARY)
    raise ValueError(f"Unsupported projectile {key!r}; choose one of: {supported}")

def set_projectile(key):
    global PROJECTILE_KEY, PROJECTILE_MASS_AU, PROJECTILE_CHARGE
    global PROJECTILE_MASS_NUMBER, PROJECTILE_FILE_TOKEN, PROJECTILE_LABEL
    name, cfg = _projectile_config(key)
    PROJECTILE_KEY = name
    PROJECTILE_MASS_AU = float(cfg["mass_au"])
    PROJECTILE_MASS_NUMBER = float(cfg.get("mass_number", 1.0))
    PROJECTILE_CHARGE = float(cfg["charge"])
    PROJECTILE_FILE_TOKEN = str(cfg["file_token"])
    PROJECTILE_LABEL = str(cfg["label"])

def _projectile_from_argv(default="proton"):
    for idx, arg in enumerate(sys.argv[1:]):
        if arg == "--projectile" and idx + 2 <= len(sys.argv[1:]):
            return sys.argv[1:][idx + 1]
        if arg.startswith("--projectile="):
            return arg.split("=", 1)[1]
    return os.environ.get("ICE_PROJECTILE", default)

def _bool_from_text(value, default=True):
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default

def _argv_value(names, default=None):
    argv = sys.argv[1:]
    names = tuple(names)
    value = default
    for idx, arg in enumerate(argv):
        if arg in names and idx + 1 < len(argv):
            value = argv[idx + 1]
        else:
            for name in names:
                prefix = f"{name}="
                if arg.startswith(prefix):
                    value = arg.split("=", 1)[1]
    return value

def _float_cli_value(names, env_names=(), default=None, scale=1.0):
    value = default
    value_scale = 1.0
    for env_name, env_scale in env_names:
        if env_name in os.environ:
            value = os.environ[env_name]
            value_scale = env_scale
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        for name in names:
            if arg == name and idx + 1 < len(argv):
                value = argv[idx + 1]
                value_scale = scale
            elif arg.startswith(f"{name}="):
                value = arg.split("=", 1)[1]
                value_scale = scale
    return float(value) * value_scale

def _int_cli_value(names, env_name=None, default=None):
    value = os.environ.get(env_name, default) if env_name else default
    value = _argv_value(names, default=value)
    return int(value)

def _max_workers_from_environment(default=None):
    for env_name in ("ICE_MAX_WORKERS", "PBS_NCPUS", "PBS_NP", "NCPUS", "NSLOTS", "SLURM_CPUS_PER_TASK"):
        raw = os.environ.get(env_name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value

    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if pbs_nodefile and os.path.exists(pbs_nodefile):
        try:
            with open(pbs_nodefile, "r", encoding="utf-8") as handle:
                value = sum(1 for line in handle if line.strip())
            if value > 0:
                return value
        except OSError:
            pass

    if default is None:
        default = os.cpu_count() or 1
    return max(1, int(default))

def _energy_range_from_argv():
    emin = _float_cli_value(
        ("--energy-min-eV",),
        env_names=(("ICE_ENERGY_MIN_EV", 1.0), ("ICE_ENERGY_MIN_MEV", 1.0e6)),
        default=DEFAULT_ENERGY_MIN_EV,
        scale=1.0,
    )
    emax = _float_cli_value(
        ("--energy-max-eV",),
        env_names=(("ICE_ENERGY_MAX_EV", 1.0), ("ICE_ENERGY_MAX_MEV", 1.0e6)),
        default=DEFAULT_ENERGY_MAX_EV,
        scale=1.0,
    )
    # MeV flags are scanned after eV flags so explicit MeV arguments can override env/defaults.
    emin = _float_cli_value(("--energy-min-MeV",), default=emin, scale=1.0e6)
    emax = _float_cli_value(("--energy-max-MeV",), default=emax, scale=1.0e6)
    n_points = _int_cli_value(("--energy-points", "--n-energy"), env_name="ICE_ENERGY_POINTS", default=DEFAULT_ENERGY_POINTS)
    grid = str(_argv_value(("--energy-grid",), default=os.environ.get("ICE_ENERGY_GRID", DEFAULT_ENERGY_GRID))).strip().lower()
    if grid not in ("log", "linear"):
        raise ValueError("--energy-grid must be 'log' or 'linear'.")
    if not (np.isfinite(emin) and np.isfinite(emax) and emin > 0.0 and emax > emin):
        raise ValueError("--energy-min/--energy-max must be finite, positive, and increasing.")
    if n_points < 2:
        raise ValueError("--energy-points must be at least 2.")
    return emin, emax, n_points, grid

def _energy_unit_from_argv(default="total"):
    unit = str(_argv_value(("--energy-unit",), default=os.environ.get("ICE_ENERGY_UNIT", default))).strip().lower()
    if unit not in ("total", "per_u"):
        raise ValueError("--energy-unit must be 'total' or 'per_u'.")
    return unit

def _energy_range_to_total_eV(emin, emax, energy_unit):
    if str(energy_unit) == "total":
        return float(emin), float(emax)
    return float(emin) * PROJECTILE_MASS_NUMBER, float(emax) * PROJECTILE_MASS_NUMBER

def _charge_mode_from_argv(default="bare"):
    mode = str(_argv_value(("--charge-mode",), default=os.environ.get("ICE_CHARGE_MODE", default))).strip().lower()
    if mode not in ("bare", "zeff", "explicit"):
        raise ValueError("--charge-mode must be bare, zeff, or explicit.")
    return mode

def _explicit_charge_from_argv(default=None):
    raw = _argv_value(("--explicit-charge", "--charge-state", "--q-charge"), default=os.environ.get("ICE_EXPLICIT_CHARGE", default))
    if raw is None or str(raw).strip() == "":
        return None
    val = float(raw)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError("--explicit-charge must be finite and positive.")
    return val

def _include_barkas_dcs_from_argv(default=False):
    include = _bool_from_text(os.environ.get("ICE_INCLUDE_BARKAS_DCS"), default)
    for arg in sys.argv[1:]:
        if arg == "--include-barkas-dcs":
            include = True
        elif arg == "--no-include-barkas-dcs":
            include = False
        elif arg.startswith("--include-barkas-dcs="):
            include = _bool_from_text(arg.split("=", 1)[1], include)
    return bool(include)

def _include_bloch_dcs_from_argv(default=False):
    include = _bool_from_text(os.environ.get("ICE_INCLUDE_BLOCH_DCS"), default)
    for arg in sys.argv[1:]:
        if arg == "--include-bloch-dcs":
            include = True
        elif arg == "--no-include-bloch-dcs":
            include = False
        elif arg.startswith("--include-bloch-dcs="):
            include = _bool_from_text(arg.split("=", 1)[1], include)
    if include:
        raise ValueError("Bloch corrections are not implemented or allowed in the DCS generator.")
    return False

def _projectile_relativistic_dcs_from_argv(default=False):
    enabled = _bool_from_text(os.environ.get("ICE_PROJECTILE_RELATIVISTIC_DCS"), default)
    for arg in sys.argv[1:]:
        if arg == "--relativistic-projectile-dcs":
            enabled = True
        elif arg == "--no-relativistic-projectile-dcs":
            enabled = False
        elif arg.startswith("--relativistic-projectile-dcs="):
            enabled = _bool_from_text(arg.split("=", 1)[1], enabled)
    return bool(enabled)

def _include_transverse_dcs_from_argv(default=False):
    enabled = _bool_from_text(os.environ.get("ICE_INCLUDE_TRANSVERSE_DCS"), default)
    for arg in sys.argv[1:]:
        if arg == "--include-transverse-dcs":
            enabled = True
        elif arg == "--no-include-transverse-dcs":
            enabled = False
        elif arg.startswith("--include-transverse-dcs="):
            enabled = _bool_from_text(arg.split("=", 1)[1], enabled)
    return bool(enabled)

def _rpwba_density_effect_from_argv(default=True):
    enabled = _bool_from_text(os.environ.get("ICE_RPWBA_DENSITY_EFFECT"), default)
    for arg in sys.argv[1:]:
        if arg == "--rpwba-density-effect":
            enabled = True
        elif arg == "--no-rpwba-density-effect":
            enabled = False
        elif arg.startswith("--rpwba-density-effect="):
            enabled = _bool_from_text(arg.split("=", 1)[1], enabled)
    return bool(enabled)

def _set_projectile_relativistic_dcs(
    enabled=False,
    include_transverse=None,
    use_density_effect=True,
):
    global PROJECTILE_RELATIVISTIC_DCS, INCLUDE_TRANSVERSE_DCS
    global RPWBA_DENSITY_EFFECT
    PROJECTILE_RELATIVISTIC_DCS = bool(enabled)
    if include_transverse is None:
        include_transverse = PROJECTILE_RELATIVISTIC_DCS
    INCLUDE_TRANSVERSE_DCS = bool(include_transverse)
    RPWBA_DENSITY_EFFECT = bool(PROJECTILE_RELATIVISTIC_DCS and use_density_effect)
    if INCLUDE_TRANSVERSE_DCS and not PROJECTILE_RELATIVISTIC_DCS:
        raise ValueError("--include-transverse-dcs requires --relativistic-projectile-dcs.")
    if PROJECTILE_RELATIVISTIC_DCS and not INCLUDE_TRANSVERSE_DCS:
        raise ValueError(
            "Dominguez-Munoz RPWBA requires both finite-Q longitudinal and "
            "transverse terms; --no-include-transverse-dcs is not a physical mode."
        )

def _born_reference_charge_from_argv(default="bare_Z"):
    ref = str(
        _argv_value(
            ("--born-reference-charge",),
            default=os.environ.get("ICE_BORN_REFERENCE_CHARGE", default),
        )
    ).strip()
    if ref not in barkas_dcs.BORN_REFERENCE_CHOICES:
        choices = "|".join(barkas_dcs.BORN_REFERENCE_CHOICES)
        raise ValueError(f"--born-reference-charge must be {choices}.")
    return ref

def _born_reference_explicit_charge_from_argv(default=None):
    raw = _argv_value(
        ("--born-reference-explicit-charge", "--born-reference-q"),
        default=os.environ.get("ICE_BORN_REFERENCE_EXPLICIT_CHARGE", default),
    )
    if raw is None or str(raw).strip() == "":
        return None
    val = float(raw)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError("--born-reference-explicit-charge must be finite and positive.")
    return val

def _set_charge_options(
    charge_mode,
    explicit_charge,
    include_barkas_dcs,
    include_bloch_dcs,
    energy_unit,
    born_reference_charge,
    born_reference_explicit_charge,
):
    global CHARGE_MODE, EXPLICIT_PROJECTILE_CHARGE, INCLUDE_BARKAS_DCS
    global INCLUDE_BLOCH_DCS, ENERGY_UNIT
    global BORN_REFERENCE_CHARGE, BORN_REFERENCE_EXPLICIT_CHARGE
    CHARGE_MODE = str(charge_mode)
    EXPLICIT_PROJECTILE_CHARGE = None if explicit_charge is None else float(explicit_charge)
    INCLUDE_BARKAS_DCS = bool(include_barkas_dcs)
    INCLUDE_BLOCH_DCS = bool(include_bloch_dcs)
    ENERGY_UNIT = str(energy_unit)
    BORN_REFERENCE_CHARGE = str(born_reference_charge)
    BORN_REFERENCE_EXPLICIT_CHARGE = (
        None if born_reference_explicit_charge is None else float(born_reference_explicit_charge)
    )

def _integration_resolution_from_argv():
    dE = _int_cli_value(
        ("--dE", "--NE", "--nE", "--energy-integration-points"),
        env_name="ICE_DE",
        default=DEFAULT_DE_POINTS,
    )
    dq = _int_cli_value(
        ("--dq", "--Nq", "--nq", "--q-integration-points"),
        env_name="ICE_DQ",
        default=DEFAULT_DQ_POINTS,
    )
    if dE < 2:
        raise ValueError("--dE must be at least 2.")
    if dq < 2:
        raise ValueError("--dq must be at least 2.")
    return dE, dq

def _merge_energy_patches_from_argv(default=True):
    merge = _bool_from_text(os.environ.get("ICE_MERGE_ENERGY_PATCHES"), default)
    for arg in sys.argv[1:]:
        if arg == "--merge-energy-patches":
            merge = True
        elif arg == "--no-merge-energy-patches":
            merge = False
        elif arg.startswith("--merge-energy-patches="):
            merge = _bool_from_text(arg.split("=", 1)[1], merge)
    return merge

def _format_energy_tag_value(value):
    text = f"{float(value):.6g}"
    return (
        text.replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
    )

def _energy_range_tag(emin, emax, n_points, grid):
    default = (
        np.isclose(float(emin), DEFAULT_ENERGY_MIN_EV)
        and np.isclose(float(emax), DEFAULT_ENERGY_MAX_EV)
        and int(n_points) == DEFAULT_ENERGY_POINTS
        and str(grid) == DEFAULT_ENERGY_GRID
    )
    if default:
        return ""
    return (
        f"_range_{_format_energy_tag_value(emin)}_"
        f"{_format_energy_tag_value(emax)}eV_{grid}_n{int(n_points)}"
    )

def _charge_mode_tag(
    charge_mode,
    include_barkas_dcs,
    explicit_charge=None,
    born_reference_charge="bare_Z",
    born_reference_explicit_charge=None,
):
    tag = ""
    if str(charge_mode) != "bare":
        tag += f"_charge_{charge_mode}"
    if str(charge_mode) == "explicit":
        tag += f"_q{_format_energy_tag_value(explicit_charge)}"
    if str(born_reference_charge) != "bare_Z":
        tag += f"_bornref_{str(born_reference_charge).lower()}"
    if str(born_reference_charge) == "explicit_q":
        tag += f"_qref{_format_energy_tag_value(born_reference_explicit_charge)}"
    if include_barkas_dcs:
        tag += "_barkas_dcs"
    return tag

def _projectile_kernel_tag():
    if not PROJECTILE_RELATIVISTIC_DCS:
        return ""
    suffix = "_rpwba_dm2022"
    if not RPWBA_DENSITY_EFFECT:
        suffix += "_no_density"
    return suffix

def _print_cli_help_and_exit():
    supported = ", ".join(PROJECTILE_LIBRARY)
    print(
        "Usage: ICE_TYPE=hexagonal python -u python_scripts/physics_ice/generate_ice_cross_sections_ion.py [options]\n"
        "\n"
        "Options:\n"
        f"  --projectile NAME              projectile: {supported} (default: proton)\n"
        "  --include-kshell               include O K-shell (default)\n"
        "  --no-kshell                    omit O K-shell\n"
        "  --kshell-model MODEL           hydrogenic-gos, old-optical, or none\n"
        "  --energy-min-eV VALUE          incident-energy minimum in eV\n"
        "  --energy-max-eV VALUE          incident-energy maximum in eV\n"
        "  --energy-min-MeV VALUE         incident-energy minimum in MeV\n"
        "  --energy-max-MeV VALUE         incident-energy maximum in MeV\n"
        "  --energy-unit total|per_u      interpret energy inputs as total ion energy or energy/u (default: total)\n"
        "  --energy-points N              number of incident-energy grid points (default: 1000)\n"
        "  --energy-grid log|linear       incident-energy grid type (default: log)\n"
        "  --dE N                         energy-loss integration points (default: 1000)\n"
        "  --dq N                         q-integration points (default: 1000)\n"
        "  --charge-mode bare|zeff|explicit\n"
        "                                 interaction charge for Born/Barkas terms (default: bare)\n"
        "  --explicit-charge Q            charge state for --charge-mode explicit\n"
        "  --born-reference-charge bare_Z|unit_charge|explicit_q\n"
        "                                 charge convention already present in input Born DCS (default: bare_Z)\n"
        "  --born-reference-explicit-charge Q\n"
        "                                 reference charge for --born-reference-charge explicit_q\n"
        "  --include-barkas-dcs[=true|false]\n"
        "                                 add the OOS Barkas Z^3 DCS kernel (default: false)\n"
        "  --include-bloch-dcs=false      Bloch DCS is intentionally unsupported\n"
        "  --relativistic-projectile-dcs[=true|false]\n"
        "                                 use the finite-Q Dominguez-Munoz RPWBA kernel\n"
        "  --include-transverse-dcs[=true|false]\n"
        "                                 compatibility flag; full RPWBA requires true\n"
        "  --rpwba-density-effect[=true|false]\n"
        "                                 apply the finite-Q dielectric Fermi correction (default: true in RPWBA)\n"
        "  --merge-energy-patches         merge this energy patch into existing DAT tables (default)\n"
        "  --no-merge-energy-patches      write only this energy patch to DAT tables\n"
    )
    sys.exit(0)

if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    _print_cli_help_and_exit()

def _include_kshell_from_argv(default=True):
    include = _bool_from_text(os.environ.get("ICE_INCLUDE_KSHELL"), default)
    for arg in sys.argv[1:]:
        if arg == "--include-kshell":
            include = True
        elif arg == "--no-kshell":
            include = False
        elif arg.startswith("--include-kshell="):
            include = _bool_from_text(arg.split("=", 1)[1], include)
    return include

KSHELL_MODEL_CHOICES = ("none", "hydrogenic-gos", "old-optical")
KSHELL_MODEL = "hydrogenic-gos"

def _kshell_model_from_argv(include_kshell=True):
    default = "hydrogenic-gos" if include_kshell else "none"
    selected = os.environ.get("ICE_KSHELL_MODEL", default).strip().lower()
    for idx, arg in enumerate(sys.argv[1:]):
        if arg == "--kshell-model" and idx + 2 <= len(sys.argv[1:]):
            selected = sys.argv[1:][idx + 1].strip().lower()
        elif arg.startswith("--kshell-model="):
            selected = arg.split("=", 1)[1].strip().lower()
    if selected not in KSHELL_MODEL_CHOICES:
        choices = "|".join(KSHELL_MODEL_CHOICES)
        raise ValueError(f"Unsupported --kshell-model={selected!r}; choose {choices}.")
    return selected

def _set_kshell_model(kshell_model):
    global KSHELL_MODEL
    kshell_model = str(kshell_model).strip().lower()
    if kshell_model not in KSHELL_MODEL_CHOICES:
        choices = "|".join(KSHELL_MODEL_CHOICES)
        raise ValueError(f"Unsupported K-shell model {kshell_model!r}; choose {choices}.")
    KSHELL_MODEL = kshell_model

def _projectile_energy_label(math=False):
    if math:
        return f"{PROJECTILE_LABEL} kinetic energy $s$ (eV)"
    return f"{PROJECTILE_LABEL} kinetic energy s (eV)"

set_projectile(_projectile_from_argv())

# Select ice structure: "amorphous" or "hexagonal"
ICE_TYPE = os.environ.get("ICE_TYPE", "hexagonal").strip().lower()
ICE_LABEL = f"{ICE_TYPE}_ice"

# One material density per phase. Normalize the ELF-derived GOS to its
# physical electron density (10 electrons/H2O), then divide by N_H2O.
ION_NORMALIZATION_VERSION = "optical-fsum-per-H2O-v1"
OPTICAL_SUM_POINTS = 60001
OPTICAL_SUM_MAX_EV = 1.0e8
# integral W*ELF dW = OPTICAL_SUM_UNIT_EV2_M3 * electron density.
# SI plasma-frequency identity, with hbar converted to eV s.
OPTICAL_SUM_UNIT_EV2_M3 = (
    0.5 * np.pi * (hbar / elementary_charge)**2
    * elementary_charge**2 / (electron_mass * epsilon_0)
)

# Extend DCS grid beyond Born table using a linear T grid.
DCS_T_MAX_EEV = 1.0e7
DCS_T_STEP_EEV = 2.0e5

# Geant4 Emfietzoglou DCS table scale: file values * scale -> m^2
EMFI_DCS_SCALE_M2 = 1.0e-22 / 3.343

KSHELL_B_EV = model.OXYGEN_K_B_EV
KSHELL_ZEFF = model.OXYGEN_K_ZEFF
KSHELL_FSUM_TARGET = model.OXYGEN_K_FSUM_TARGET
HYDROGENIC_KSHELL_ROLLOFF_APPLIED = False

# ----------------------------------------------------------------------
# Constants for integration (from constants.py)
# ----------------------------------------------------------------------
def _normalize_ice_type(name):
    if not name:
        return None
    key = str(name).strip().lower()
    if "amorph" in key:
        return "amorphous"
    if "hex" in key:
        return "hexagonal"
    return None

def _material_density_g_cm3(ice_type):
    norm = _normalize_ice_type(ice_type)
    if norm == "amorphous":
        return float(ICE_AMORPHOUS_DENSITY_G_CM3)
    if norm == "hexagonal":
        return float(ICE_HEXAGONAL_DENSITY_G_CM3)
    raise ValueError(f"Unsupported ice phase: {ice_type}")

@lru_cache(maxsize=8)
def _optical_normalization(Ep, Bmin, excitations, ionizations, density_g_cm3):
    """Normalize the full optical ELF f-sum at the phase's physical density.

    The f-sum identity is integral W*Im[-1/epsilon(W,0)] dW
    = (pi/2)*(hbar*omega_p)^2, with omega_p^2 = n_e*e^2/(m_e*epsilon_0).
    Set N_H2O = rho*N_A/M_H2O, and scale the ELF-derived GOS by
    N_H2O*10*OPTICAL_SUM_UNIT_EV2_M3/integral before dividing by N_H2O.
    This gives microscopic cross sections per H2O with one material density.
    See Dominguez-Munoz et al. (2022), Eqs. (3),(9), for the same ELF/GOS
    conversion. No stopping-power reference enters this normalization.

    Always include the normalized optical K-shell moment, even for a
    no-K-shell diagnostic: omitting a process must not amplify valence.
    The existing 0.179 K fraction is not replaced by the separate Barkas
    8+2 OOS convention. This is an optical normalization, not a repair of
    the finite-q GOS or a refit of the complex dielectric function.
    """
    s = model.IceOpticalSet(Ep, Bmin, list(excitations), list(ionizations), None)
    energies = np.geomspace(Bmin, OPTICAL_SUM_MAX_EV, OPTICAL_SUM_POINTS)
    # At q=0 the dispersion coefficients drop out. Use the same partitioned
    # response as the ion kernel, not the unpartitioned optical diagnostic.
    C = model.DispersionCoeffs(0.0, 0.0, 0.0)
    elf = model.elf_Eq(energies, 0.0, s, C, include_kshell=False)[0]
    valence_moment = float(np.trapezoid(energies * elf, energies))
    core_moment = 0.5 * np.pi * Ep**2 * KSHELL_FSUM_TARGET
    full_moment = valence_moment + core_moment
    if (not np.isfinite(full_moment) or valence_moment <= 0.0
            or np.any(~np.isfinite(elf)) or np.any(elf < 0.0)
            or not np.isfinite(density_g_cm3) or density_g_cm3 <= 0.0):
        raise RuntimeError("Invalid ion optical ELF f-sum; cannot normalize per H2O.")
    density = density_g_cm3 * 1e6 * AVOGADRO / H2O_MOLAR_MASS_G_MOL
    scale = (
        density * OPTICAL_SUM_UNIT_EV2_M3 * model.WATER_TOTAL_OSCILLATOR_STRENGTH
        / full_moment
    )
    return density, scale, valence_moment, core_moment


def _ion_elf_per_molecule_factor(s):
    density, scale, _, _ = _optical_normalization(
        s.Ep, s.Bmin, tuple(s.excitations), tuple(s.ionizations),
        _material_density_g_cm3(s.material),
    )
    return scale / density


def _ion_normalization_metadata(ice_type, s=None):
    phase = _normalize_ice_type(ice_type)
    rho = _material_density_g_cm3(phase)
    if s is None:
        s = model.epsilon_optical(phase)
    if s.material != phase:
        raise ValueError("Optical model and requested ice phase differ.")
    density, scale, valence, core = _optical_normalization(
        s.Ep, s.Bmin, tuple(s.excitations), tuple(s.ionizations), rho,
    )
    return {
        "ion_normalization_version": ION_NORMALIZATION_VERSION,
        "ice_type": phase,
        "optical_fit_Ep_eV": float(s.Ep),
        "physical_plasma_energy_eV": float(np.sqrt(
            2.0 / np.pi * OPTICAL_SUM_UNIT_EV2_M3 * density
            * model.WATER_TOTAL_OSCILLATOR_STRENGTH
        )),
        "optical_elf_fsum_scale": scale,
        "optical_full_moment_eV2": valence + core,
        "optical_fsum_fraction": (valence + core) / (0.5 * np.pi * s.Ep**2),
        "optical_electrons_per_H2O": model.WATER_TOTAL_OSCILLATOR_STRENGTH,
        "optical_normalization_includes_full_kshell": True,
        "optical_valence_electrons_per_H2O": model.WATER_TOTAL_OSCILLATOR_STRENGTH * valence / (valence + core),
        "optical_kshell_electrons_per_H2O": model.WATER_TOTAL_OSCILLATOR_STRENGTH * core / (valence + core),
        "material_density_g_cm3": rho,
        "material_molecular_density_m3": density,
        "material_density_applied_to_cross_sections": False,
        "density_scale_factor": 1.0,
    }


def _require_current_normalization(data, ice_type=None):
    phase = ice_type or _npz_str_value(data, "ice_type")
    if phase is None:
        raise ValueError("Legacy ion normalization: regenerate the cross sections.")
    for key, expected in _ion_normalization_metadata(phase).items():
        if key not in data:
            raise ValueError(f"Missing {key}: regenerate the ion cross sections.")
        actual = np.asarray(data[key]).item()
        matches = (
            np.isclose(actual, expected, rtol=1e-10, atol=0.0)
            if isinstance(expected, float) else actual == expected
        )
        if not matches:
            raise ValueError(f"Incompatible {key}: regenerate the ion cross sections.")

def _infer_ice_type_from_path(path_like):
    text = str(path_like).lower()
    if "amorph" in text:
        return "amorphous"
    if "hex" in text:
        return "hexagonal"
    return None

# ----------------------------------------------------------------------
# Helpers: Relativistic corrections
# ----------------------------------------------------------------------
# Regime correction handles (toggle corrections in regimes II/III/IV).
# APPLY_CORRECTIONS_REGIME_II = True
# APPLY_CORRECTIONS_REGIME_III = True
# APPLY_CORRECTIONS_REGIME_IV = True
# APPLY_MOTT_COULOMB = True

APPLY_CORRECTIONS_REGIME_II = False
APPLY_CORRECTIONS_REGIME_III = False
APPLY_CORRECTIONS_REGIME_IV = False
APPLY_MOTT_COULOMB = False

def _set_regime_corrections(apply_regime_ii=None, apply_regime_iii=None, apply_regime_iv=None):
    global APPLY_CORRECTIONS_REGIME_II, APPLY_CORRECTIONS_REGIME_III, APPLY_CORRECTIONS_REGIME_IV
    if apply_regime_ii is not None:
        APPLY_CORRECTIONS_REGIME_II = bool(apply_regime_ii)
    if apply_regime_iii is not None:
        APPLY_CORRECTIONS_REGIME_III = bool(apply_regime_iii)
    if apply_regime_iv is not None:
        APPLY_CORRECTIONS_REGIME_IV = bool(apply_regime_iv)

def _set_mc_correction(apply_mc=None):
    global APPLY_MOTT_COULOMB
    APPLY_MOTT_COULOMB = False

def _simpson_integrate(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if n < 2:
        return 0.0
    if n != x.size:
        raise ValueError("x and y must have the same length")
    if n == 2:
        return 0.5 * (y[0] + y[1]) * (x[1] - x[0])

    dx = np.diff(x)
    if not np.allclose(dx, dx[0]):
        return float(np.sum(0.5 * dx * (y[:-1] + y[1:])))

    h = dx[0]
    if n % 2 == 1:
        return float(
            (h / 3.0)
            * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
        )

    n1 = n - 1
    simpson_part = (h / 3.0) * (
        y[0]
        + y[n1 - 1]
        + 4.0 * np.sum(y[1 : n1 - 1 : 2])
        + 2.0 * np.sum(y[2 : n1 - 2 : 2])
    )
    trap_part = 0.5 * (y[-2] + y[-1]) * (x[-1] - x[-2])
    return float(simpson_part + trap_part)

def _energy_grid(Emin, Emax, N, use_log=True):
    Emin = float(Emin)
    Emax = float(Emax)
    if (not use_log) or (Emin <= 0.0) or (Emax <= Emin):
        return np.linspace(Emin, Emax, N)
    log_min = np.log(Emin)
    log_max = np.log(Emax)
    if (not np.isfinite(log_min)) or (not np.isfinite(log_max)) or (log_max <= log_min):
        return np.linspace(Emin, Emax, N)
    return np.exp(np.linspace(log_min, log_max, N))

def _regime_flags(Tj):
    # Electron exchange/Mott corrections remain excluded. This optional path is
    # the complete finite-Q projectile RPWBA, not the electron correction.
    if PROJECTILE_RELATIVISTIC_DCS:
        return False, True, True, bool(RPWBA_DENSITY_EFFECT)
    return False, False, False, False

def _elf_rolloff_factor(Ei):
    Ei = np.asarray(Ei, dtype=float)
    factor = np.ones_like(Ei)
    mask = Ei > ELF_ROLLOFF_E0_eV
    if np.any(mask):
        factor[mask] = 1.0 - ELF_ROLLOFF_COEF * np.log10(Ei[mask] / ELF_ROLLOFF_E0_eV)
    return factor

def projectile_rest_energy_eV(projectile_mass_au=None):
    if projectile_mass_au is None:
        projectile_mass_au = PROJECTILE_MASS_AU
    return float(projectile_mass_au) * MC2_eV


def projectile_beta2(Tp_eV, projectile_mass_au=None):
    Tp_eV = float(Tp_eV)
    if Tp_eV <= 0.0:
        return 0.0
    gamma = 1.0 + Tp_eV / projectile_rest_energy_eV(projectile_mass_au)
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    return float(np.clip(beta2, 0.0, 1.0 - np.finfo(float).eps))


def heavy_projectile_Emax(Tp_eV, projectile_mass_au=None):
    beta2 = projectile_beta2(Tp_eV, projectile_mass_au)
    if beta2 <= 0.0:
        return 0.0
    return float(2.0 * MC2_eV * beta2 / max(1.0 - beta2, np.finfo(float).tiny))


def _projectile_energy_loss_upper_eV(Tp_eV):
    return float(min(float(Tp_eV), heavy_projectile_Emax(Tp_eV)))


def beta2_rel(Tj):
    return projectile_beta2(Tj)

def delta_fermi(T):
    """
    Sternheimer-Fano density effect for liquid water.
    """
    b2 = beta2_rel(T)
    b2 = min(max(float(b2), np.finfo(float).tiny), 1.0 - np.finfo(float).eps)
    beta = np.sqrt(b2)
    X = np.log10(np.sqrt(beta / (1.0 - beta * beta)))

    # liquid water parameters
    alpha = 0.09116
    X1 = 2.8004
    m = 3.4773
    C = -3.5017
    X0 = 0.24

    if X < X0:
        return 0.0
    if X < X1:
        return 4.6052 * X + alpha * (X1 - X) ** m + C
    return 4.6052 * X + C

def Q_q(q_au):
    """
    Q(q) in eV.
    """
    q_au = np.asarray(q_au, dtype=float)
    Q_Ha = np.sqrt((C_AU * q_au)**2 + (MC2_HA)**2) - MC2_HA
    return Q_Ha * EH

# ----------------------------------------------------------------------
# Helper: q-bounds for scalar Ei, Tj (eV)
# ----------------------------------------------------------------------
def _q_bounds_scalar(Ei, Tj, projectile_mass_au=None):
    """Return (q_lo, q_hi) for scalar Ei, Tj (both in eV)."""
    if projectile_mass_au is None:
        projectile_mass_au = PROJECTILE_MASS_AU
    Ei_H = Ei * EV_TO_HA
    T_H = Tj * EV_TO_HA
    if Ei_H >= T_H:
        return 0.0, 0.0

    d = T_H - Ei_H
    if d <= 0.0:
        return 0.0, 0.0

    sqrtT = np.sqrt(T_H)
    sqrt_d = np.sqrt(d)
    qlo = np.sqrt(2.0 * projectile_mass_au) * (sqrtT - sqrt_d)
    qhi = np.sqrt(2.0 * projectile_mass_au) * (sqrtT + sqrt_d)

    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        return 0.0, 0.0
    return float(qlo), float(qhi)

def _q_bounds_scalar_rel(Ei, Tj):
    """Return exact relativistic ion momentum-transfer bounds in a0^-1.

    For ion rest energy M c^2 and energy loss W,
        q_- = [p(T) - p(T-W)] / (hbar/a0),
        q_+ = [p(T) + p(T-W)] / (hbar/a0),
    where p c = sqrt[T(T + 2 M c^2)]. The selected projectile supplies M.
    """
    Ei = float(Ei)
    Tj = float(Tj)
    if Ei <= 0.0 or Ei >= Tj:
        return 0.0, 0.0
    rest_H = projectile_rest_energy_eV() * EV_TO_HA
    T_H = Tj * EV_TO_HA
    final_H = (Tj - Ei) * EV_TO_HA
    p_initial = np.sqrt(T_H * (T_H + 2.0 * rest_H)) / C_AU
    p_final = np.sqrt(final_H * (final_H + 2.0 * rest_H)) / C_AU
    qhi = p_initial + p_final
    loss_H = Ei * EV_TO_HA
    # Rationalized p(T)-p(T-W), avoiding cancellation for W << T.
    qlo = (
        loss_H * (2.0 * (T_H + rest_H) - loss_H)
        / (C_AU**2 * qhi)
    )
    if (not np.isfinite(qlo)) or (not np.isfinite(qhi)) or qlo <= 0.0 or qhi <= qlo:
        return 0.0, 0.0
    return float(qlo), float(qhi)

def _projectile_relativistic_longitudinal_prefactor(Tj, s):
    """Microscopic Dominguez-Munoz RPWBA longitudinal DCS prefactor.

    d sigma_L/dW = 2 z^2 / (pi a0 N m_e c^2 beta^2)
                   * integral[dq/q Im(-1/epsilon(W,q))].

    This is Eq. (3) of Dominguez-Munoz et al. (2022), after substituting
    their Eq. (9), changing variables from recoil energy Q to momentum q,
    and dividing the sum-rule-normalized DIMFP by the phase molecular
    density N. _ion_elf_per_molecule_factor supplies the ELF scale/N. The selected
    projectile supplies z and beta. The established high-mass energy-loss
    cutoff remains unchanged in heavy_projectile_Emax().
    """
    beta2 = projectile_beta2(Tj)
    if beta2 <= 0.0:
        return 0.0
    return float(
        2.0 * PROJECTILE_CHARGE**2 * _ion_elf_per_molecule_factor(s)
        / (np.pi * a0 * MC2_eV * beta2)
    )

def _rpwba_transverse_ratio(
    W_eV,
    q_au,
    beta2,
    epsilon1=None,
    epsilon2=None,
    use_density_effect=False,
):
    """Return the finite-Q transverse/longitudinal integrand ratio.

    With R = Q(Q + 2 m_e c^2) = (q c)^2 and rho = W^2/R, the second
    term in braces in Dominguez-Munoz et al. Eq. (3), divided by the first,
    is

        rho * (beta^2 - rho) / (1 - rho)^2.

    When requested, Eq. (8) replaces that vacuum transverse term with the
    condensed-medium result

        W/(2 m_e c^2) * |epsilon|^2 * (beta^2 - rho)
        / [(1 - rho epsilon_1)^2 + (rho epsilon_2)^2].

    The latter is the algebraic sum of the Eq. (3) transverse term and the
    Eq. (8) Fermi correction. No optical-q=0 or Sternheimer approximation is
    used here.
    """
    q_au = np.asarray(q_au, dtype=float)
    W_eV = float(W_eV)
    beta2 = float(beta2)
    qc_eV = C_AU * EH * q_au
    recoil_product = qc_eV * qc_eV
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rho = (W_eV * W_eV) / recoil_product

    tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(beta2))
    if np.any(rho > beta2 + tolerance):
        raise FloatingPointError("RPWBA q grid violates beta^2 - W^2/(qc)^2 >= 0.")
    beta_minus_rho = np.maximum(beta2 - rho, 0.0)

    if use_density_effect:
        if epsilon1 is None or epsilon2 is None:
            raise ValueError("Finite-Q epsilon1 and epsilon2 are required for the density effect.")
        epsilon1 = np.asarray(epsilon1, dtype=float)
        epsilon2 = np.asarray(epsilon2, dtype=float)
        denominator = (1.0 - rho * epsilon1) ** 2 + (rho * epsilon2) ** 2
        denominator = np.maximum(denominator, np.finfo(float).tiny)
        ratio = (
            (W_eV / (2.0 * MC2_eV))
            * (epsilon1 * epsilon1 + epsilon2 * epsilon2)
            * beta_minus_rho
            / denominator
        )
    else:
        denominator = np.maximum((1.0 - rho) ** 2, np.finfo(float).tiny)
        ratio = rho * beta_minus_rho / denominator

    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise FloatingPointError("Non-finite or negative RPWBA transverse kernel.")
    return ratio

# ----------------------------------------------------------------------
# Inner q-integral at fixed Ei for channels (excitation / ionization)
# ----------------------------------------------------------------------
def _integrate_channel_single_E(
    Ei, Tj, idx, channel_type, s, C, Nq=400, use_rel_bounds=False
):
    """
    Compute inner integral over q:

        ∫ dq [ ELF_channel(Ei, q) / q ]

    for one excitation or ionization channel, at fixed Ei, Tj.
    """
    if use_rel_bounds:
        raise RuntimeError("Relativistic electron q-bounds are disabled for heavy projectiles.")
    if Ei > _projectile_energy_loss_upper_eV(Tj):
        return 0.0
    qlo, qhi = _q_bounds_scalar(Ei, Tj, projectile_mass_au=PROJECTILE_MASS_AU)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0

    # q-grid
    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    qvals = np.exp(xi)
    E_arr = np.array([Ei], float)

    # Vectorized dielectric functions at (Ei, qvals)
    e1 = model.epsilon1_valence_Eq(E_arr, qvals, s, C)
    e2 = model.epsilon2_valence_Eq(E_arr, qvals, s, C)

    e1t = e1["total"][:, 0]   # shape (Nq,)
    e2t = e2["total"][:, 0]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

    if channel_type == "excitation":
        vals = e2["excitations"][idx][:, 0] / denom
    elif channel_type == "ionization":
        vals = e2["ionizations"][idx][:, 0] / denom
    else:
        raise ValueError("channel_type must be 'excitation' or 'ionization'")

    vals = vals * _elf_rolloff_factor(Ei)
    accum = float(_simpson_integrate(vals, xi))

    T_scaled = Tj / PROJECTILE_MASS_AU
    int_cons = PROJECTILE_CHARGE**2 * _ion_elf_per_molecule_factor(s) / (
        np.pi * a0 * T_scaled
    )

    return float(int_cons * accum)

def _integrate_channel_single_E_rpwba_components(
    Ei,
    Tj,
    idx,
    channel_type,
    s,
    C,
    Nq=400,
    use_density_effect=False,
):
    """Return finite-Q longitudinal and transverse RPWBA DCS components.

    The integration is Eq. (4) of Dominguez-Munoz et al. (2022), evaluated
    on logarithmic q after applying Eqs. (2), (3), and (9). Both terms use
    the same channel-resolved finite-q GOS/ELF. The density option applies
    their Eqs. (7)-(9) directly through the complex ice dielectric function.
    """
    if Ei > _projectile_energy_loss_upper_eV(Tj):
        return 0.0, 0.0
    qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0, 0.0

    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    qvals = np.exp(xi)
    E_arr = np.array([Ei], float)
    e1 = model.epsilon1_valence_Eq(E_arr, qvals, s, C)
    e2 = model.epsilon2_valence_Eq(E_arr, qvals, s, C)
    denominator = e1["total"][:, 0] ** 2 + e2["total"][:, 0] ** 2
    denominator = np.where(denominator == 0.0, np.finfo(float).tiny, denominator)
    if channel_type == "excitation":
        vals = e2["excitations"][idx][:, 0] / denominator
    elif channel_type == "ionization":
        vals = e2["ionizations"][idx][:, 0] / denominator
    else:
        raise ValueError("channel_type must be 'excitation' or 'ionization'")
    vals = vals * _elf_rolloff_factor(Ei)
    beta2 = projectile_beta2(Tj)
    transverse_ratio = _rpwba_transverse_ratio(
        Ei,
        qvals,
        beta2,
        epsilon1=e1["total"][:, 0],
        epsilon2=e2["total"][:, 0],
        use_density_effect=use_density_effect,
    )
    prefactor = _projectile_relativistic_longitudinal_prefactor(Tj, s)
    longitudinal = prefactor * _simpson_integrate(vals, xi)
    transverse = prefactor * _simpson_integrate(vals * transverse_ratio, xi)
    return float(longitudinal), float(transverse)

def _integrate_channel_single_E_rel(Ei, Tj, idx, channel_type, s, C, Nq=400):
    longitudinal, _ = _integrate_channel_single_E_rpwba_components(
        Ei, Tj, idx, channel_type, s, C, Nq=Nq, use_density_effect=False
    )
    return longitudinal

def _integrate_channel_single_E_trans(
    Ei, Tj, idx, channel_type, s, C, Nq=400, use_density_effect=False
):
    _, transverse = _integrate_channel_single_E_rpwba_components(
        Ei,
        Tj,
        idx,
        channel_type,
        s,
        C,
        Nq=Nq,
        use_density_effect=use_density_effect,
    )
    return transverse

# ----------------------------------------------------------------------
# Low-energy Mott–Coulomb (MC) corrections using PWBA kernel evaluations
# ----------------------------------------------------------------------
def _dsigma_pwba_dE(Ei, Tj, idx, channel_type, s, C, Nq=400, use_rel=False):
    """
    Return the differential cross section d sigma/dE at (Ei,Tj) for one channel.
    If use_rel=True, this uses the longitudinal relativistic kernel; otherwise it uses
    the nonrelativistic PWBA kernel.
    """
    if use_rel:
        return _integrate_channel_single_E_rel(Ei, Tj, idx, channel_type, s, C, Nq=Nq)
    return _integrate_channel_single_E(Ei, Tj, idx, channel_type, s, C, Nq=Nq)

def _dsigma_mc_ionization_dE(Ei, Tj, j, s, C, Nq=400, use_rel=False):
    raise RuntimeError("Mott-Coulomb/exchange correction is disabled for heavy projectiles.")

def _sigma_pwba_excitation_shifted_T(s, C, Tshift, Tj, k, NE=400, Nq=400, use_rel=False):
    """sigma_PWBA for excitation k, with shifted kernel but kinematic E-window from Tj."""
    Emin = float(s.Bmin)
    Emax = _projectile_energy_loss_upper_eV(Tj)
    if Emin >= Emax:
        return 0.0

    Egrid = _energy_grid(Emin, Emax, NE)
    vals = np.empty_like(Egrid)
    for i, Ei in enumerate(Egrid):
        vals[i] = _dsigma_pwba_dE(Ei, Tshift, k, "excitation", s, C, Nq=Nq, use_rel=use_rel)

    vals = np.where(vals < 0.0, 0.0, vals)
    return float(_simpson_integrate(vals, Egrid))

def _sigma_mc_ionization(s, C, Tj, j, NE=400, Nq=400, use_rel=False):
    raise RuntimeError("Mott-Coulomb/exchange correction is disabled for heavy projectiles.")

def _total_transverse_sigma(
    s,
    C,
    Tj,
    NE=400,
    Nq=400,
    use_density_effect=False,
    include_kshell=True,
    return_valence=False,
):
    """Integrate the finite-Q RPWBA transverse DCS over all energy losses."""
    E_upper = _projectile_energy_loss_upper_eV(Tj)
    total = 0.0
    for idx in range(len(s.excitations)):
        if s.Bmin >= E_upper:
            continue
        energies = _energy_grid(float(s.Bmin), E_upper, NE)
        values = np.array(
            [
                _integrate_channel_single_E_trans(
                    W,
                    Tj,
                    idx,
                    "excitation",
                    s,
                    C,
                    Nq=Nq,
                    use_density_effect=use_density_effect,
                )
                for W in energies
            ]
        )
        total += _simpson_integrate(values, energies)
    for idx, oscillator in enumerate(s.ionizations):
        if oscillator.Bth >= E_upper:
            continue
        energies = _energy_grid(float(oscillator.Bth), E_upper, NE)
        values = np.array(
            [
                _integrate_channel_single_E_trans(
                    W,
                    Tj,
                    idx,
                    "ionization",
                    s,
                    C,
                    Nq=Nq,
                    use_density_effect=use_density_effect,
                )
                for W in energies
            ]
        )
        total += _simpson_integrate(values, energies)
    valence_total = float(total)
    if include_kshell and s.kshell is not None and KSHELL_MODEL != "none":
        threshold = _kshell_threshold_eV(s)
        if threshold is not None and threshold < E_upper:
            energies = _energy_grid(float(threshold), E_upper, NE)
            values = np.array(
                [
                    _integrate_kshell_single_E_rpwba_components(
                        W,
                        Tj,
                        s,
                        C,
                        Nq=Nq,
                        include_kshell=True,
                        use_density_effect=use_density_effect,
                    )[1]
                    for W in energies
                ]
            )
            total += _simpson_integrate(values, energies)
    if return_valence:
        return valence_total, float(total)
    return float(total)

def _kshell_threshold_eV(s):
    if KSHELL_MODEL == "hydrogenic-gos":
        return float(KSHELL_B_EV)
    if KSHELL_MODEL == "old-optical" and s.kshell is not None:
        return float(s.kshell.Bth)
    return None

def _kshell_old_optical_elf(Ei, s):
    ks_arr = model.oxygen_K_electron_optical_elf(np.array([Ei], float), s, fsum_corrected=False)
    return float(ks_arr[0])

def _kshell_hydrogenic_gos_elf(Ei, qvals, s):
    return model.oxygen_K_ion_hydrogenic_gos_elf(
        Ei,
        qvals,
        B_K_eV=KSHELL_B_EV,
        Zeff=KSHELL_ZEFF,
        normalize_fsum=True,
        Ep_eV=float(s.Ep),
        fsum_target=KSHELL_FSUM_TARGET,
    )

# ----------------------------------------------------------------------
# Inner q-integral at fixed Ei for K-shell channel
# ----------------------------------------------------------------------
def _integrate_kshell_single_E(
    Ei, Tj, s, Nq=400, include_kshell=True, use_rel_bounds=False
):
    """
    Inner integral over q for K-shell:
    """
    if not include_kshell or (s.kshell is None) or KSHELL_MODEL == "none":
        return 0.0

    if use_rel_bounds:
        raise RuntimeError("Relativistic electron q-bounds are disabled for heavy projectiles.")
    kshell_B = _kshell_threshold_eV(s)
    if kshell_B is None or Ei <= kshell_B:
        return 0.0
    if Ei > _projectile_energy_loss_upper_eV(Tj):
        return 0.0
    qlo, qhi = _q_bounds_scalar(Ei, Tj, projectile_mass_au=PROJECTILE_MASS_AU)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0

    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    qvals = np.exp(xi)

    if KSHELL_MODEL == "old-optical":
        ks_val = _kshell_old_optical_elf(Ei, s)
        if ks_val == 0.0:
            return 0.0
        vals = np.full_like(xi, ks_val)
        vals = vals * _elf_rolloff_factor(Ei)
    elif KSHELL_MODEL == "hydrogenic-gos":
        vals = _kshell_hydrogenic_gos_elf(Ei, qvals, s)
    else:
        return 0.0

    accum = float(_simpson_integrate(vals, xi))

    T_scaled = Tj / PROJECTILE_MASS_AU
    int_cons = PROJECTILE_CHARGE**2 * _ion_elf_per_molecule_factor(s) / (
        np.pi * a0 * T_scaled
    )

    return float(int_cons * accum)

def _integrate_kshell_single_E_rpwba_components(
    Ei,
    Tj,
    s,
    C,
    Nq=400,
    include_kshell=True,
    use_density_effect=False,
):
    """Return longitudinal and transverse RPWBA O K-shell DCS components.

    The hydrogenic K-shell GOS is additive and has no corresponding complex
    K-shell epsilon in the current ice model. The Eq. (8) screening factor is
    therefore evaluated with the finite-q valence epsilon. This approximation
    is recorded in output metadata and is relevant only when the density
    correction is enabled.
    """
    if not include_kshell or (s.kshell is None) or KSHELL_MODEL == "none":
        return 0.0, 0.0
    threshold = _kshell_threshold_eV(s)
    if threshold is None or Ei <= threshold or Ei > _projectile_energy_loss_upper_eV(Tj):
        return 0.0, 0.0
    qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    if qhi <= qlo or qlo <= 0.0:
        return 0.0, 0.0
    xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
    qvals = np.exp(xi)
    if KSHELL_MODEL == "old-optical":
        vals = np.full_like(xi, _kshell_old_optical_elf(Ei, s))
        vals = vals * _elf_rolloff_factor(Ei)
    elif KSHELL_MODEL == "hydrogenic-gos":
        vals = _kshell_hydrogenic_gos_elf(Ei, qvals, s)
    else:
        return 0.0, 0.0

    E_arr = np.array([Ei], float)
    e1 = model.epsilon1_valence_Eq(E_arr, qvals, s, C)
    e2 = model.epsilon2_valence_Eq(E_arr, qvals, s, C)
    transverse_ratio = _rpwba_transverse_ratio(
        Ei,
        qvals,
        projectile_beta2(Tj),
        epsilon1=e1["total"][:, 0],
        epsilon2=e2["total"][:, 0],
        use_density_effect=use_density_effect,
    )
    prefactor = _projectile_relativistic_longitudinal_prefactor(Tj, s)
    longitudinal = prefactor * _simpson_integrate(vals, xi)
    transverse = prefactor * _simpson_integrate(vals * transverse_ratio, xi)
    return float(longitudinal), float(transverse)

def _integrate_kshell_single_E_rel(Ei, Tj, s, C, Nq=400, include_kshell=True):
    longitudinal, _ = _integrate_kshell_single_E_rpwba_components(
        Ei,
        Tj,
        s,
        C,
        Nq=Nq,
        include_kshell=include_kshell,
        use_density_effect=False,
    )
    return longitudinal

# ----------------------------------------------------------------------
# Q-integrated ELF per channel, on its own E-grid
# ----------------------------------------------------------------------
def integrate_elf_channels_per_channel_q(
    s,
    C,
    T,
    NE=400,
    Nq=400,
    include_kshell=True,
    use_rel_long=False,
    use_rel_trans=False,
    use_density_effect=False,
):
    """
    Integrate ELF(E,q)/q over q for each excitation, ionization, and K-shell,
    and ALSO compute a second "relativistic longitudinal-corrected" version
    of the same inner-q integrals (stored with *_rel keys).

    use_rel_long and use_rel_trans control whether the corresponding arrays
    are computed or filled with zeros.
    """

    # --- Channel energy windows ---
    E_upper = _projectile_energy_loss_upper_eV(T)
    exc_Emin = np.full(len(s.excitations), float(s.Bmin), dtype=float)
    exc_Emax = np.array([E_upper for _ in s.excitations], float)

    ion_Emin = np.array([osc.Bth for osc in s.ionizations], float)
    ion_Emax = np.array([E_upper for _ in s.ionizations], float)

    if include_kshell and (s.kshell is not None) and KSHELL_MODEL != "none":
        kshell_Emin = _kshell_threshold_eV(s)
        kshell_Emax = E_upper
    else:
        kshell_Emin = None
        kshell_Emax = None

    results = {
        "excitation_E": [],
        "excitation_int": [],
        "excitation_int_rel": [],
        "excitation_int_rel_trans": [],

        "ionization_E": [],
        "ionization_int": [],
        "ionization_int_rel": [],
        "ionization_int_rel_trans": [],

        "kshell_E": None,
        "kshell_int": None,
        "kshell_int_rel": None,
        "kshell_int_rel_trans": None,
    }

    # ------------------- Excitations -------------------
    for k in range(len(s.excitations)):
        Emin, Emax = exc_Emin[k], exc_Emax[k]
        if Emin >= Emax:
            results["excitation_E"].append(np.array([], float))
            results["excitation_int"].append(np.array([], float))
            results["excitation_int_rel"].append(np.array([], float))
            results["excitation_int_rel_trans"].append(np.array([], float))
            continue

        Egrid = _energy_grid(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.zeros_like(Egrid)
        vals_rel_trans = np.zeros_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_channel_single_E(
                Ei, T, k, "excitation", s, C, Nq=Nq, use_rel_bounds=False
            )

            # Full finite-Q RPWBA components.
            if use_rel_long:
                vals_rel[i], vals_rel_trans[i] = _integrate_channel_single_E_rpwba_components(
                    Ei,
                    T,
                    k,
                    "excitation",
                    s,
                    C,
                    Nq=Nq,
                    use_density_effect=bool(use_rel_trans and use_density_effect),
                )


        results["excitation_E"].append(Egrid)
        results["excitation_int"].append(vals)
        results["excitation_int_rel"].append(vals_rel)
        results["excitation_int_rel_trans"].append(vals_rel_trans)

    # ------------------- Ionizations -------------------
    for j in range(len(s.ionizations)):
        Emin, Emax = ion_Emin[j], ion_Emax[j]
        if Emin >= Emax:
            results["ionization_E"].append(np.array([], float))
            results["ionization_int"].append(np.array([], float))
            results["ionization_int_rel"].append(np.array([], float))
            results["ionization_int_rel_trans"].append(np.array([], float))
            continue

        Egrid = _energy_grid(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.zeros_like(Egrid)
        vals_rel_trans = np.zeros_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_channel_single_E(
                Ei, T, j, "ionization", s, C, Nq=Nq, use_rel_bounds=False
            )

            # Full finite-Q RPWBA components.
            if use_rel_long:
                vals_rel[i], vals_rel_trans[i] = _integrate_channel_single_E_rpwba_components(
                    Ei,
                    T,
                    j,
                    "ionization",
                    s,
                    C,
                    Nq=Nq,
                    use_density_effect=bool(use_rel_trans and use_density_effect),
                )


        results["ionization_E"].append(Egrid)
        results["ionization_int"].append(vals)
        results["ionization_int_rel"].append(vals_rel)
        results["ionization_int_rel_trans"].append(vals_rel_trans)

    # ------------------- K-shell -----------------------
    if (kshell_Emin is not None) and (kshell_Emin < kshell_Emax):
        Egrid = _energy_grid(kshell_Emin, kshell_Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.zeros_like(Egrid)
        vals_rel_trans = np.zeros_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_kshell_single_E(
                Ei, T, s, Nq=Nq, include_kshell=include_kshell, use_rel_bounds=False
            )

            # Full finite-Q RPWBA components.
            if use_rel_long:
                vals_rel[i], vals_rel_trans[i] = _integrate_kshell_single_E_rpwba_components(
                    Ei,
                    T,
                    s,
                    C,
                    Nq=Nq,
                    include_kshell=include_kshell,
                    use_density_effect=bool(use_rel_trans and use_density_effect),
                )

        results["kshell_E"] = Egrid
        results["kshell_int"] = vals
        results["kshell_int_rel"] = vals_rel if use_rel_long else None
        results["kshell_int_rel_trans"] = vals_rel_trans if use_rel_trans else None

    return results

# ----------------------------------------------------------------------
# Full double integral over E and q: sigma(T) per channel and totals
# ----------------------------------------------------------------------
def integrate_elf_double_integral(
    s,
    C,
    T,
    NE=200,
    Nq=200,
    include_kshell=True,
    use_mott_coulomb=False,
):
    """
    Compute full double integral sigma(T) per channel and totals.
    """

    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(T)
    if (not use_mott_coulomb) or (not APPLY_MOTT_COULOMB):
        use_mc = False

    # 1) Inner q-integrals as functions of E
    integ = integrate_elf_channels_per_channel_q(
        s,
        C,
        T=T,
        NE=NE,
        Nq=Nq,
        include_kshell=include_kshell,
        use_rel_long=use_rel_long,
        use_rel_trans=use_rel_trans,
        use_density_effect=use_density_effect,
    )

    # -------------------- PWBA baseline --------------------
    exc_sigma_pwba = []
    ion_sigma_pwba = []
    kshell_sigma_pwba = None

    for k in range(len(s.excitations)):
        E_k = integ["excitation_E"][k]
        y_k = integ["excitation_int"][k]
        exc_sigma_pwba.append(0.0 if E_k.size == 0 else float(_simpson_integrate(y_k, E_k)))

    for j in range(len(s.ionizations)):
        E_j = integ["ionization_E"][j]
        y_j = integ["ionization_int"][j]
        ion_sigma_pwba.append(0.0 if E_j.size == 0 else float(_simpson_integrate(y_j, E_j)))

    if integ["kshell_E"] is not None:
        E_K = integ["kshell_E"]
        y_K = integ["kshell_int"]
        kshell_sigma_pwba = 0.0 if E_K.size == 0 else float(_simpson_integrate(y_K, E_K))

    valence_sigma_pwba = float(np.sum(exc_sigma_pwba) + np.sum(ion_sigma_pwba))
    total_sigma_pwba = (
        valence_sigma_pwba + kshell_sigma_pwba if kshell_sigma_pwba is not None else valence_sigma_pwba
    )

    # -------------------- Relativistic components --------------------
    exc_sigma_rel = []
    exc_sigma_rel_trans = []
    ion_sigma_rel = []
    ion_sigma_rel_trans = []
    kshell_sigma_rel = None
    kshell_sigma_rel_trans = None

    for k in range(len(s.excitations)):
        E_k = integ["excitation_E"][k]
        y_k_rel = integ["excitation_int_rel"][k]
        y_k_trans = integ["excitation_int_rel_trans"][k]
        exc_sigma_rel.append(0.0 if E_k.size == 0 else float(_simpson_integrate(y_k_rel, E_k)))
        exc_sigma_rel_trans.append(0.0 if E_k.size == 0 else float(_simpson_integrate(y_k_trans, E_k)))

    for j in range(len(s.ionizations)):
        E_j = integ["ionization_E"][j]
        y_j_rel = integ["ionization_int_rel"][j]
        y_j_trans = integ["ionization_int_rel_trans"][j]
        ion_sigma_rel.append(0.0 if E_j.size == 0 else float(_simpson_integrate(y_j_rel, E_j)))
        ion_sigma_rel_trans.append(0.0 if E_j.size == 0 else float(_simpson_integrate(y_j_trans, E_j)))

    if integ["kshell_E"] is not None:
        E_K = integ["kshell_E"]
        y_K_rel = integ.get("kshell_int_rel", None)
        y_K_trans = integ.get("kshell_int_rel_trans", None)
        if (y_K_rel is None) or (E_K.size == 0):
            kshell_sigma_rel = 0.0
        else:
            kshell_sigma_rel = float(_simpson_integrate(y_K_rel, E_K))
        if (y_K_trans is None) or (E_K.size == 0):
            kshell_sigma_rel_trans = 0.0
        else:
            kshell_sigma_rel_trans = float(_simpson_integrate(y_K_trans, E_K))

    valence_sigma_rel = float(np.sum(exc_sigma_rel) + np.sum(ion_sigma_rel))
    valence_sigma_rel_trans = float(np.sum(exc_sigma_rel_trans) + np.sum(ion_sigma_rel_trans))
    if kshell_sigma_rel is not None:
        total_sigma_rel = valence_sigma_rel + kshell_sigma_rel
    else:
        total_sigma_rel = valence_sigma_rel
    total_sigma_rel_trans = valence_sigma_rel_trans + (
        kshell_sigma_rel_trans if kshell_sigma_rel_trans is not None else 0.0
    )
    total_sigma_rel_total = total_sigma_rel + total_sigma_rel_trans
    valence_sigma_rel_trans_no_density = None
    total_sigma_rel_trans_no_density = None
    if use_rel_trans and use_density_effect:
        (
            valence_sigma_rel_trans_no_density,
            total_sigma_rel_trans_no_density,
        ) = _total_transverse_sigma(
            s,
            C,
            T,
            NE=NE,
            Nq=Nq,
            use_density_effect=False,
            include_kshell=include_kshell,
            return_valence=True,
        )

    # -------------------- Mott-Coulomb --------------------
    exc_sigma_mc = None
    ion_sigma_mc = None
    valence_sigma_mc = None
    total_sigma_mc = None

    if use_mc:
        use_rel_in_mc = use_rel_long
        exc_sigma_mc = []
        for k in range(len(s.excitations)):
            Bk = float(s.Bmin)
            Tshift = float(T + 2.0 * Bk)
            exc_sigma_mc.append(
                _sigma_pwba_excitation_shifted_T(
                    s, C, Tshift, T, k, NE=NE, Nq=Nq, use_rel=use_rel_in_mc
                )
            )

        ion_sigma_mc = []
        for j in range(len(s.ionizations)):
            ion_sigma_mc.append(
                _sigma_mc_ionization(s, C, T, j, NE=NE, Nq=Nq, use_rel=use_rel_in_mc)
            )

        valence_sigma_mc = float(np.sum(exc_sigma_mc) + np.sum(ion_sigma_mc))
        kshell_for_mc = (
            kshell_sigma_rel if (use_rel_long and kshell_sigma_rel is not None) else kshell_sigma_pwba
        )
        total_sigma_mc = valence_sigma_mc + (kshell_for_mc if kshell_for_mc is not None else 0.0)

    # -------------------- Default output selection --------------------
    exc_sigma = list(exc_sigma_pwba)
    ion_sigma = list(ion_sigma_pwba)
    kshell_sigma = kshell_sigma_pwba
    valence_sigma = float(valence_sigma_pwba)
    total_sigma = float(total_sigma_pwba)

    if use_mc and (exc_sigma_mc is not None) and (ion_sigma_mc is not None):
        exc_sigma = list(exc_sigma_mc)
        ion_sigma = list(ion_sigma_mc)
        kshell_sigma = (
            kshell_sigma_rel if (use_rel_long and kshell_sigma_rel is not None) else kshell_sigma_pwba
        )
        valence_sigma = float(np.sum(exc_sigma) + np.sum(ion_sigma))
        total_sigma = valence_sigma + (kshell_sigma if kshell_sigma is not None else 0.0)
    elif use_rel_long:
        if use_rel_trans:
            exc_sigma = [a + b for a, b in zip(exc_sigma_rel, exc_sigma_rel_trans)]
            ion_sigma = [a + b for a, b in zip(ion_sigma_rel, ion_sigma_rel_trans)]
            valence_sigma = float(valence_sigma_rel + valence_sigma_rel_trans)
            kshell_sigma = (
                (kshell_sigma_rel or 0.0) + (kshell_sigma_rel_trans or 0.0)
                if kshell_sigma_rel is not None
                else kshell_sigma_pwba
            )
            total_sigma = float(total_sigma_rel_total)
        else:
            exc_sigma = list(exc_sigma_rel)
            ion_sigma = list(ion_sigma_rel)
            valence_sigma = float(valence_sigma_rel)
            kshell_sigma = kshell_sigma_rel if kshell_sigma_rel is not None else kshell_sigma_pwba
            total_sigma = valence_sigma + (kshell_sigma if kshell_sigma is not None else 0.0)

    # Optional convenience: combined
    total_sigma_plus_rel = total_sigma_pwba + total_sigma_rel
    total_sigma_plus_rel_total = total_sigma_pwba + total_sigma_rel_total

    return {
        # PWBA Baseline
        "excitation_sigma_pwba": exc_sigma_pwba,
        "ionization_sigma_pwba": ion_sigma_pwba,
        "valence_sigma_pwba": valence_sigma_pwba,
        "total_sigma_pwba": total_sigma_pwba,

        "excitation_sigma": exc_sigma,
        "ionization_sigma": ion_sigma,
        "kshell_sigma": kshell_sigma,
        "valence_sigma": valence_sigma,
        "total_sigma": total_sigma,

        # Low-energy Mott–Coulomb
        "excitation_sigma_mc": exc_sigma_mc,
        "ionization_sigma_mc": ion_sigma_mc,
        "valence_sigma_mc": valence_sigma_mc,
        "total_sigma_mc": total_sigma_mc,

        # Longitudinal relativistic outputs
        "excitation_sigma_rel": exc_sigma_rel,
        "excitation_sigma_rel_trans": exc_sigma_rel_trans,

        "ionization_sigma_rel": ion_sigma_rel,
        "ionization_sigma_rel_trans": ion_sigma_rel_trans,

        "kshell_sigma_rel": kshell_sigma_rel,
        "kshell_sigma_rel_trans": kshell_sigma_rel_trans,
        "valence_sigma_rel": valence_sigma_rel,
        "valence_sigma_rel_trans": valence_sigma_rel_trans,
        "valence_sigma_rel_trans_no_density": valence_sigma_rel_trans_no_density,

        "total_sigma_rel": total_sigma_rel,
        "total_sigma_rel_trans": total_sigma_rel_trans,
        "total_sigma_rel_trans_no_density": total_sigma_rel_trans_no_density,
        "total_sigma_rel_total": total_sigma_rel_total,

        "total_sigma_plus_rel": total_sigma_plus_rel,
        "total_sigma_plus_rel_total": total_sigma_plus_rel_total,
    }

# ----------------------------------------------------------------------
# Plotting: full cross sections per channel vs T
# ----------------------------------------------------------------------
def plot_full_cross_sections_per_channel(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """
    Plot a *comparison* of PWBA vs the default model (per channel)

    Baseline (PWBA):
        uses keys '*_sigma_pwba' computed by the integrator.

    Default model uses the regime flags from _regime_flags(T).
    """

    # Allow passing (ax_ion, ax_exc) or None
    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        ax_ion, ax_exc = ax

    T_arr = np.asarray(T_list, dtype=float)
    nT = len(T_arr)

    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

    def _get_list(i, key, default):
        return sigma_list[i].get(key, default)

    # ----------------- Ionizations: PWBA vs Default model -----------------
    n_ion = len(s.ionizations)
    for j in range(n_ion):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])

            pwba_list = _get_list(i, "ionization_sigma_pwba", [0.0] * n_ion)
            pwba = pwba_list[j] if j < len(pwba_list) else 0.0

            use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
            if use_mc:
                mc_list = _get_list(i, "ionization_sigma_mc", None)
                model = (mc_list[j] if (mc_list is not None and j < len(mc_list)) else pwba)
            elif use_rel_long:
                relL_list = _get_list(i, "ionization_sigma_rel", [0.0] * n_ion)
                relT_list = _get_list(i, "ionization_sigma_rel_trans", [0.0] * n_ion)
                relL = relL_list[j] if j < len(relL_list) else 0.0
                relT = relT_list[j] if j < len(relT_list) else 0.0
                model = relL + (relT if use_rel_trans else 0.0)
            else:
                model = pwba

            y_pwba.append(pwba)
            y_model.append(model)

        ax_ion.loglog(
            T_arr, y_pwba,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            ls="--",
            label=f"Ion. {j+1} PWBA",
        )
        ax_ion.loglog(
            T_arr, y_model,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Ion. {j+1} Default model",
        )

    ax_ion.set_xlabel(_projectile_energy_label())
    ax_ion.set_ylabel("Cross section sigma(T)")
    ax_ion.set_title("Ionizations: PWBA baseline vs Default model")
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)
    ax_ion.legend(loc="best", fontsize=8)

    # ----------------- Excitations: PWBA vs Default model -----------------
    n_exc = len(s.excitations)
    for k in range(n_exc):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])

            pwba_list = _get_list(i, "excitation_sigma_pwba", [0.0] * n_exc)
            pwba = pwba_list[k] if k < len(pwba_list) else 0.0

            use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
            if use_mc:
                mc_list = _get_list(i, "excitation_sigma_mc", None)
                model = (mc_list[k] if (mc_list is not None and k < len(mc_list)) else pwba)
            elif use_rel_long:
                relL_list = _get_list(i, "excitation_sigma_rel", [0.0] * n_exc)
                relT_list = _get_list(i, "excitation_sigma_rel_trans", [0.0] * n_exc)
                relL = relL_list[k] if k < len(relL_list) else 0.0
                relT = relT_list[k] if k < len(relT_list) else 0.0
                model = relL + (relT if use_rel_trans else 0.0)
            else:
                model = pwba

            y_pwba.append(pwba)
            y_model.append(model)

        ax_exc.loglog(
            T_arr, y_pwba,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            ls="--",
            label=f"Exc. {k+1} PWBA",
        )
        ax_exc.loglog(
            T_arr, y_model,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Exc. {k+1} Default model",
        )

    ax_exc.set_xlabel(_projectile_energy_label())
    ax_exc.set_ylabel("Cross section sigma(T)")
    ax_exc.set_title("Excitations: PWBA baseline vs Default model")
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)
    ax_exc.legend(loc="best", fontsize=8)

    return (ax_ion, ax_exc)


def plot_relativistic_component_per_channel(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8),
        include_kshell=False, include_totals=False):
    """ Make two plots:
      (1) Ionizations: longitudinal (solid) and transverse (dashed)
      (2) Excitations: longitudinal (solid) and transverse (dashed)

    Uses keys:
      - ionization_sigma_rel (longitudinal)
      - ionization_sigma_rel_trans (transverse)
      - excitation_sigma_rel (longitudinal)
      - excitation_sigma_rel_trans (transverse)

    Plots only for T >= REGIME_II_MAX_eV by default.
    """

    T_arr = np.asarray(T_list, dtype=float)
    mask = T_arr >= REGIME_II_MAX_eV
    if not np.any(mask):
        raise ValueError("No T values >= REGIME_II_MAX_eV found in T_list.")

    Tm = T_arr[mask]
    idxs = np.where(mask)[0]

    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

    n_exc = len(s.excitations)
    n_ion = len(s.ionizations)

    # Ax handling: allow ax=None or ax=(ax_ion, ax_exc)
    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        if not (isinstance(ax, (tuple, list)) and len(ax) == 2):
            raise ValueError("ax must be None or a (ax_ion, ax_exc) tuple/list")
        ax_ion, ax_exc = ax

    # ----------------- Ionizations -----------------
    print(f"Plot IONIZATIONS (REL longitudinal solid, transverse dashed; T>={REGIME_II_MAX_eV:.1e} eV)")
    for j in tqdm(range(n_ion)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("ionization_sigma_rel", [0.0] * n_ion)
            trans_list = sigma_list[i].get("ionization_sigma_rel_trans", [0.0] * n_ion)
            y_long.append(long_list[j] if j < len(long_list) else 0.0)
            y_trans.append(trans_list[j] if j < len(trans_list) else 0.0)

        ax_ion.loglog(
            Tm, y_long,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            label=f"Ion. {j+1} Long",
        )
        ax_ion.loglog(
            Tm, y_trans,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Ion. {j+1} Trans",
        )

    ax_ion.set_xlabel(_projectile_energy_label())
    ax_ion.set_ylabel("Relativistic component sigma_rel(T)")
    ax_ion.legend(loc="best", fontsize=8)
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)

    # ----------------- Excitations -----------------
    print(f"Plot EXCITATIONS (REL longitudinal solid, transverse dashed; T>={REGIME_II_MAX_eV:.1e} eV)")
    for k in tqdm(range(n_exc)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("excitation_sigma_rel", [0.0] * n_exc)
            trans_list = sigma_list[i].get("excitation_sigma_rel_trans", [0.0] * n_exc)
            y_long.append(long_list[k] if k < len(long_list) else 0.0)
            y_trans.append(trans_list[k] if k < len(trans_list) else 0.0)

        ax_exc.loglog(
            Tm, y_long,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            label=f"Exc. {k+1} Long",
        )
        ax_exc.loglog(
            Tm, y_trans,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Exc. {k+1} Trans",
        )

    ax_exc.set_xlabel(_projectile_energy_label())
    ax_exc.set_ylabel("Relativistic component sigma_rel(T)")
    ax_exc.legend(loc="best", fontsize=8)
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)

    return (ax_ion, ax_exc)

def plot_total_cross_section(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """
    Plot TOTAL cross section with *all* corrections (as available in sigma_list).

    The corrected total is chosen per T as:
      * if regime selects MC: use total_sigma_mc when present
      * elif regime selects rel: use total_sigma_rel_total (long+trans) or total_sigma_rel
      * else fall back to PWBA
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)

    y_pwba = []
    y_corr = []

    for i, Tj in enumerate(T_arr):
        pwba = float(sigma_list[i].get("total_sigma_pwba", sigma_list[i].get("total_sigma", 0.0)) or 0.0)

        use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
        if use_mc:
            corr = float(sigma_list[i].get("total_sigma_mc", pwba) or pwba)
        elif use_rel_long:
            if use_rel_trans:
                corr = float(sigma_list[i].get("total_sigma_rel_total", pwba) or pwba)
            else:
                corr = float(sigma_list[i].get("total_sigma_rel", pwba) or pwba)
        else:
            corr = pwba

        y_pwba.append(pwba)
        y_corr.append(corr)

    ax.loglog(T_arr, y_pwba, lw=linewidth, alpha=alpha, ls=":", label="Total PWBA")
    ax.loglog(T_arr, y_corr, lw=linewidth+1, alpha=alpha, ls="-", label="Total (all corrections)")

    ax.set_xlabel(_projectile_energy_label())
    ax.set_ylabel("Total cross section sigma(T)")
    ax.set_title("Total cross section: PWBA vs all corrections")
    ax.legend(loc="best", fontsize=9)
    return ax

def _load_total_sigma_npz(npz_path):
    with np.load(npz_path) as data:
        _require_current_normalization(data)
        T = np.asarray(data["T_eV"], float)
        pwba = np.asarray(data.get("total_sigma_pwba", []), float)
        corrected = np.asarray(
            data.get("total_sigma_corrected", data.get("total_sigma", [])), float
        )
    return T, pwba, corrected

def plot_total_cross_section_two_panel(
    amorphous_npz,
    hexagonal_npz,
    out_path=None,
):
    """
    Two-panel comparison (amorphous vs hexagonal) with a shared legend row below.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    T_a, pwba_a, corr_a = _load_total_sigma_npz(amorphous_npz)
    T_h, pwba_h, corr_h = _load_total_sigma_npz(hexagonal_npz)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, height_ratios=[3.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    ln1 = ax_a.loglog(T_a, pwba_a, "k:", linewidth=2, label="Total PWBA")[0]
    ln2 = ax_a.loglog(T_a, corr_a, "k-", linewidth=2.5, label="Total (all corrections)")[0]
    ax_a.set_xlabel(_projectile_energy_label(math=True))
    ax_a.set_ylabel("Total cross section sigma(T)")
    ax_a.set_title("Amorphous ice")

    ax_h.loglog(T_h, pwba_h, "k:", linewidth=2, label="Total PWBA")
    ax_h.loglog(T_h, corr_h, "k-", linewidth=2.5, label="Total (all corrections)")
    ax_h.set_xlabel(_projectile_energy_label(math=True))
    ax_h.set_ylabel("")
    ax_h.set_title("Hexagonal ice")

    # Enforce common y-range and y-ticks across both panels.
    y_lo = min(ax_a.get_ylim()[0], ax_h.get_ylim()[0])
    y_hi = max(ax_a.get_ylim()[1], ax_h.get_ylim()[1])
    if not np.isfinite(y_lo) or y_lo <= 0.0:
        y_lo = 1e-30
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = y_lo * 10.0
    ax_a.set_ylim(y_lo, y_hi)
    ax_h.set_ylim(y_lo, y_hi)
    for ax in (ax_a, ax_h):
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(NullLocator())

    legend_ax.legend(
        [ln1, ln2],
        ["Total PWBA", "Total (all corrections)"],
        loc="center",
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
        labelspacing=0.8,
    )

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved two-panel total cross section plot to {out_path}")
    plt.close(fig)

def plot_channel_cross_sections_two_panel(
    amorphous_npz,
    hexagonal_npz,
    out_path=None,
):
    """
    Two-panel comparison (amorphous vs hexagonal) of channel-resolved cross sections
    with a shared legend row below (excitation top, ionization bottom).
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    with np.load(amorphous_npz) as data_a:
        _require_current_normalization(data_a, "amorphous")
        T_a, sigma_a = _sigma_list_from_npz(data_a)

    with np.load(hexagonal_npz) as data_h:
        _require_current_normalization(data_h, "hexagonal")
        T_h, sigma_h = _sigma_list_from_npz(data_h)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, height_ratios=[3.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    plot_corrected_exc_ion_scaled(T_a, sigma_a, ax=ax_a)
    plot_corrected_exc_ion_scaled(T_h, sigma_h, ax=ax_h)
    ax_a.set_title("Amorphous ice")
    ax_h.set_title("Hexagonal ice")
    ax_a.set_xlabel(_projectile_energy_label(math=True))
    ax_h.set_xlabel(_projectile_energy_label(math=True))
    ax_a.set_ylabel(r"Total cross-section (cm$^2$)")
    ax_h.set_ylabel("")

    max_x = max(float(np.max(T_a)), float(np.max(T_h)))
    ax_a.set_xlim(1.0, max_x)
    ax_h.set_xlim(1.0, max_x)
    ax_a.set_ylim(bottom=1e-25)
    ax_h.set_ylim(bottom=1e-25)

    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    handles, labels = ax_a.get_legend_handles_labels()
    n_exc = len(sigma_a[0].get("excitation_sigma_pwba", [])) if sigma_a else 0
    n_ion = len(sigma_a[0].get("ionization_sigma_pwba", [])) if sigma_a else 0
    exc_handles = handles[:n_exc]
    ion_handles = handles[n_exc:n_exc + n_ion]
    kshell_handle = None
    if "K-shell" in labels:
        try:
            kshell_handle = handles[labels.index("K-shell")]
        except Exception:
            kshell_handle = None

    gs_leg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], height_ratios=[1.0, 1.0], hspace=0.30)
    ax_leg_exc = fig.add_subplot(gs_leg[0, 0])
    ax_leg_ion = fig.add_subplot(gs_leg[1, 0])
    ax_leg_exc.axis("off")
    ax_leg_ion.axis("off")

    if exc_handles:
        exc_labels = ["Excitation"] + [str(i + 1) for i in range(n_exc)]
        exc_handles = [Line2D([], [], color="none", linestyle="none")] + exc_handles
        exc_ncol = max(1, len(exc_labels))
        ax_leg_exc.legend(
            exc_handles,
            exc_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=exc_ncol,
            frameon=False,
            columnspacing=0.9,
            handlelength=2.2,
            handletextpad=0.6,
            borderaxespad=0.0,
        )
    if ion_handles:
        ion_labels = ["Ionization"] + [str(i + 1) for i in range(n_ion)]
        ion_handles = [Line2D([], [], color="none", linestyle="none")] + ion_handles
        if kshell_handle is not None:
            ion_labels.append("K-shell")
            ion_handles.append(kshell_handle)
        ion_ncol = max(1, len(ion_labels))
        ax_leg_ion.legend(
            ion_handles,
            ion_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=ion_ncol,
            frameon=False,
            columnspacing=0.9,
            handlelength=2.2,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

    for ax in (ax_a, ax_h):
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(NullLocator())

    # Enforce common y-range and y-ticks across both panels.
    y_lo = min(ax_a.get_ylim()[0], ax_h.get_ylim()[0])
    y_hi = max(ax_a.get_ylim()[1], ax_h.get_ylim()[1])
    if not np.isfinite(y_lo) or y_lo <= 0.0:
        y_lo = 1e-25
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = y_lo * 10.0
    ax_a.set_ylim(y_lo, y_hi)
    ax_h.set_ylim(y_lo, y_hi)
    for ax in (ax_a, ax_h):
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved two-panel channel cross section plot to {out_path}")
    plt.close(fig)

def plot_corrected_exc_ion_scaled(
        T_list, sigma_list,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """
    Plot corrected excitation and ionization channels (dashed),
    scaled by the provided factor.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)
    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

    n_exc = len(sigma_list[0].get("excitation_sigma_pwba", [])) if sigma_list else 0
    n_ion = len(sigma_list[0].get("ionization_sigma_pwba", [])) if sigma_list else 0

    exc_scaled = np.zeros((len(T_arr), n_exc), float)
    ion_scaled = np.zeros((len(T_arr), n_ion), float)
    kshell_scaled = np.full(len(T_arr), np.nan, float)

    for i, Tj in enumerate(T_arr):
        sigma = sigma_list[i]
        use_mc, use_rel_long, use_rel_trans, _ = _regime_flags(Tj)
        if sigma.get("total_sigma_mc", None) is None:
            use_mc = False

        if use_mc:
            exc_vals = sigma.get("excitation_sigma_mc", []) or []
            ion_vals = sigma.get("ionization_sigma_mc", []) or []
        elif use_rel_long:
            exc_vals = sigma.get("excitation_sigma_rel", []) or []
            ion_vals = sigma.get("ionization_sigma_rel", []) or []
            if use_rel_trans:
                exc_vals = [a + b for a, b in zip(exc_vals, (sigma.get("excitation_sigma_rel_trans", []) or []))]
                ion_vals = [a + b for a, b in zip(ion_vals, (sigma.get("ionization_sigma_rel_trans", []) or []))]
        else:
            exc_vals = sigma.get("excitation_sigma_pwba", []) or []
            ion_vals = sigma.get("ionization_sigma_pwba", []) or []

        for j in range(min(n_exc, len(exc_vals))):
            exc_scaled[i, j] = float(exc_vals[j])
        for j in range(min(n_ion, len(ion_vals))):
            ion_scaled[i, j] = float(ion_vals[j])
        kshell_val = None
        if use_rel_long and sigma.get("kshell_sigma_rel", None) is not None:
            kshell_val = sigma.get("kshell_sigma_rel", None)
        else:
            kshell_val = sigma.get("kshell_sigma", None)
        if kshell_val is not None:
            try:
                kshell_scaled[i] = float(kshell_val)
            except Exception:
                kshell_scaled[i] = np.nan

    for j in range(n_exc):
        ax.loglog(
            T_arr,
            exc_scaled[:, j],
            lw=linewidth,
            alpha=alpha,
            ls="-",
            color=exc_colors[j % len(exc_colors)],
            label=f"{j+1}",
        )
    for j in range(n_ion):
        ax.loglog(
            T_arr,
            ion_scaled[:, j],
            lw=linewidth,
            alpha=alpha,
            ls="-",
            color=ion_colors[j % len(ion_colors)],
            label=f"{j+1}",
        )
    if np.any(np.isfinite(kshell_scaled)):
        ax.loglog(
            T_arr,
            kshell_scaled,
            lw=linewidth,
            alpha=alpha,
            ls="-",
            color="purple",
            label="K-shell",
        )
    ax.set_xlabel(_projectile_energy_label(math=True), labelpad=1)
    ax.set_ylabel(r"Cross-Section (cm$^2$)")
    ax.set_xlim(1.0, np.max(T_arr))
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())
    return ax

# ----------------------------------------------------------------------
# Logging: per-energy corrections
# ----------------------------------------------------------------------
def _compute_correction_row(T, sigma):
    Tj = float(T)
    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(Tj)
    if sigma.get("total_sigma_mc", None) is None:
        use_mc = False

    pwba = float(sigma.get("total_sigma_pwba", 0.0) or 0.0)
    corrected = float(sigma.get("total_sigma", 0.0) or 0.0)

    total_mc = sigma.get("total_sigma_mc", None)
    corr_mc = float(total_mc) - pwba if total_mc is not None else 0.0
    corr_rel_long = float(sigma.get("total_sigma_rel", 0.0) or 0.0)

    corr_density = 0.0
    trans_no_density = sigma.get("total_sigma_rel_trans_no_density", None)
    if use_density_effect and trans_no_density is not None:
        corr_rel_trans = float(trans_no_density)
        corr_density = float(sigma.get("total_sigma_rel_trans", 0.0) or 0.0) - corr_rel_trans
    else:
        corr_rel_trans = float(sigma.get("total_sigma_rel_trans", 0.0) or 0.0)

    return {
        "T_eV": Tj,
        "sigma_pwba": pwba,
        "corr_stage1_mc": corr_mc,
        "corr_stage2_rel_long": corr_rel_long,
        "corr_stage3_rel_trans": corr_rel_trans,
        "corr_stage4_density": corr_density,
        "sigma_corrected": corrected,
        "use_mc": int(use_mc),
        "use_rel_long": int(use_rel_long),
        "use_rel_trans": int(use_rel_trans),
        "use_density_effect": int(use_density_effect),
    }

def save_cross_section_corrections_npz(
    T_list,
    sigma_list,
    out_path=None,
    NE=None,
    Nq=None,
    dcs_data=None,
    ice_label=None,
    include_kshell=True,
    energy_unit="total",
    charge_mode="bare",
    explicit_charge=None,
    include_barkas_dcs=False,
    include_bloch_dcs=False,
    born_reference_charge="bare_Z",
    born_reference_explicit_charge=None,
):
    """
    Save PWBA, per-stage correction terms, corrected totals, and per-channel
    cross sections for each energy to an NPZ file.
    """
    if out_path is None:
        if ice_label is None:
            ice_label = ICE_LABEL
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"cross_section_corrections_{ice_label}.npz"

    T_arr = np.asarray(T_list, dtype=float)
    nT = len(T_arr)
    n_exc = len(sigma_list[0].get("excitation_sigma_pwba", [])) if nT else 0
    n_ion = len(sigma_list[0].get("ionization_sigma_pwba", [])) if nT else 0

    def _stack_channel(key, n_chan):
        arr = np.zeros((nT, n_chan), float)
        for i, sigma in enumerate(sigma_list):
            vals = sigma.get(key, None)
            if vals is None:
                continue
            for j in range(min(n_chan, len(vals))):
                arr[i, j] = float(vals[j])
        return arr

    rows = [_compute_correction_row(T, sigma) for T, sigma in zip(T_list, sigma_list)]
    pwba_total = np.array([row["sigma_pwba"] for row in rows], float)
    corr_mc = np.array([row["corr_stage1_mc"] for row in rows], float)
    corr_rel_long = np.array([row["corr_stage2_rel_long"] for row in rows], float)
    corr_rel_trans = np.array([row["corr_stage3_rel_trans"] for row in rows], float)
    corr_density = np.array([row["corr_stage4_density"] for row in rows], float)
    corrected_total = np.array([row["sigma_corrected"] for row in rows], float)
    use_mc = np.array([row["use_mc"] for row in rows], int)
    use_rel_long = np.array([row["use_rel_long"] for row in rows], int)
    use_rel_trans = np.array([row["use_rel_trans"] for row in rows], int)
    use_density_effect = np.array([row["use_density_effect"] for row in rows], int)

    exc_pwba = _stack_channel("excitation_sigma_pwba", n_exc)
    ion_pwba = _stack_channel("ionization_sigma_pwba", n_ion)
    exc_rel_long = _stack_channel("excitation_sigma_rel", n_exc)
    ion_rel_long = _stack_channel("ionization_sigma_rel", n_ion)
    exc_rel_trans = _stack_channel("excitation_sigma_rel_trans", n_exc)
    ion_rel_trans = _stack_channel("ionization_sigma_rel_trans", n_ion)
    exc_mc = _stack_channel("excitation_sigma_mc", n_exc)
    ion_mc = _stack_channel("ionization_sigma_mc", n_ion)

    exc_selected = np.zeros((nT, n_exc), float)
    ion_selected = np.zeros((nT, n_ion), float)
    for i, (Tj, sigma) in enumerate(zip(T_arr, sigma_list)):
        use_mc_i, use_rel_long_i, use_rel_trans_i, _ = _regime_flags(Tj)
        if sigma.get("total_sigma_mc", None) is None:
            use_mc_i = False
        if use_mc_i:
            exc_selected[i, :] = exc_mc[i, :]
            ion_selected[i, :] = ion_mc[i, :]
        elif use_rel_long_i:
            if use_rel_trans_i:
                exc_selected[i, :] = exc_rel_long[i, :] + exc_rel_trans[i, :]
                ion_selected[i, :] = ion_rel_long[i, :] + ion_rel_trans[i, :]
            else:
                exc_selected[i, :] = exc_rel_long[i, :]
                ion_selected[i, :] = ion_rel_long[i, :]
        else:
            exc_selected[i, :] = exc_pwba[i, :]
            ion_selected[i, :] = ion_pwba[i, :]

    total_sigma = np.array([float(s.get("total_sigma", 0.0) or 0.0) for s in sigma_list], float)
    total_sigma_mc = np.array(
        [float(s.get("total_sigma_mc", np.nan)) if s.get("total_sigma_mc", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_rel = np.array(
        [float(s.get("total_sigma_rel", np.nan)) if s.get("total_sigma_rel", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_rel_trans = np.array(
        [float(s.get("total_sigma_rel_trans", np.nan)) if s.get("total_sigma_rel_trans", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_rel_trans_no_density = np.array(
        [
            float(s.get("total_sigma_rel_trans_no_density", np.nan))
            if s.get("total_sigma_rel_trans_no_density", None) is not None
            else np.nan
            for s in sigma_list
        ],
        float,
    )
    total_sigma_rel_total = np.array(
        [float(s.get("total_sigma_rel_total", np.nan)) if s.get("total_sigma_rel_total", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_plus_rel = np.array(
        [float(s.get("total_sigma_plus_rel", np.nan)) if s.get("total_sigma_plus_rel", None) is not None else np.nan for s in sigma_list],
        float,
    )
    total_sigma_plus_rel_total = np.array(
        [float(s.get("total_sigma_plus_rel_total", np.nan)) if s.get("total_sigma_plus_rel_total", None) is not None else np.nan for s in sigma_list],
        float,
    )
    kshell_sigma = np.array(
        [float(s.get("kshell_sigma", np.nan)) if s.get("kshell_sigma", None) is not None else np.nan for s in sigma_list],
        float,
    )
    kshell_sigma_rel = np.array(
        [float(s.get("kshell_sigma_rel", np.nan)) if s.get("kshell_sigma_rel", None) is not None else np.nan for s in sigma_list],
        float,
    )
    kshell_sigma_rel_trans = np.array(
        [
            float(s.get("kshell_sigma_rel_trans", np.nan))
            if s.get("kshell_sigma_rel_trans", None) is not None
            else np.nan
            for s in sigma_list
        ],
        float,
    )

    np_save_args = dict(
        T_eV=T_arr,
        projectile_key=PROJECTILE_KEY,
        projectile_mass_au=float(PROJECTILE_MASS_AU),
        projectile_charge=float(PROJECTILE_CHARGE),
        projectile_mass_number=float(PROJECTILE_MASS_NUMBER),
        projectile_file_token=PROJECTILE_FILE_TOKEN,
        projectile_label=PROJECTILE_LABEL,
        energy_unit=str(energy_unit),
        charge_mode=str(charge_mode),
        explicit_projectile_charge=float(explicit_charge) if explicit_charge is not None else np.nan,
        include_barkas_dcs=bool(include_barkas_dcs),
        include_bloch_dcs=bool(include_bloch_dcs),
        born_reference_charge=str(born_reference_charge),
        born_reference_explicit_charge=(
            float(born_reference_explicit_charge)
            if born_reference_explicit_charge is not None
            else np.nan
        ),
        barkas_arbi_source_sha256=barkas_dcs.ARBI_SOURCE_SHA256,
        heavy_projectile_emax_applied=True,
        dcs_table_variable="energy_loss_eV",
        electron_exchange_correction_applied=False,
        electron_relativistic_q_bounds_applied=False,
        projectile_relativistic_dcs=bool(PROJECTILE_RELATIVISTIC_DCS),
        projectile_relativistic_q_bounds_applied=bool(PROJECTILE_RELATIVISTIC_DCS),
        projectile_relativistic_beta_prefactor_applied=bool(PROJECTILE_RELATIVISTIC_DCS),
        projectile_transverse_dcs_applied=bool(INCLUDE_TRANSVERSE_DCS),
        projectile_density_effect_dcs_applied=bool(RPWBA_DENSITY_EFFECT),
        projectile_rpwba_model=(
            RPWBA_MODEL_NAME if PROJECTILE_RELATIVISTIC_DCS else "none"
        ),
        projectile_rpwba_reference_doi=(
            RPWBA_REFERENCE_DOI if PROJECTILE_RELATIVISTIC_DCS else ""
        ),
        projectile_rpwba_finite_q_transverse=bool(PROJECTILE_RELATIVISTIC_DCS),
        projectile_rpwba_optical_transverse_approximation=False,
        projectile_rpwba_validated_scope="proton 100-300 MeV",
        projectile_rpwba_heavy_ion_extrapolation=bool(
            PROJECTILE_RELATIVISTIC_DCS and PROJECTILE_KEY != "proton"
        ),
        projectile_rpwba_kshell_density_epsilon=(
            "valence-only"
            if PROJECTILE_RELATIVISTIC_DCS and RPWBA_DENSITY_EFFECT and include_kshell
            else "not-used"
        ),
        include_kshell=bool(include_kshell),
        kshell_model=KSHELL_MODEL,
        kshell_B_eV=float(KSHELL_B_EV if KSHELL_MODEL == "hydrogenic-gos" else np.nan),
        kshell_Zeff=float(KSHELL_ZEFF if KSHELL_MODEL == "hydrogenic-gos" else np.nan),
        kshell_fsum_target=float(KSHELL_FSUM_TARGET if KSHELL_MODEL == "hydrogenic-gos" else np.nan),
        kshell_q_dependent=bool(KSHELL_MODEL == "hydrogenic-gos"),
        old_optical_kshell_used=bool(KSHELL_MODEL == "old-optical"),
        hydrogenic_kshell_rolloff_applied=bool(HYDROGENIC_KSHELL_ROLLOFF_APPLIED),
        total_sigma_pwba=pwba_total,
        total_sigma_corrected=corrected_total,
        total_sigma=total_sigma,
        total_sigma_mc=total_sigma_mc,
        total_sigma_rel=total_sigma_rel,
        total_sigma_rel_trans=total_sigma_rel_trans,
        total_sigma_rel_trans_no_density=total_sigma_rel_trans_no_density,
        total_sigma_rel_total=total_sigma_rel_total,
        total_sigma_plus_rel=total_sigma_plus_rel,
        total_sigma_plus_rel_total=total_sigma_plus_rel_total,
        kshell_sigma=kshell_sigma,
        kshell_sigma_rel=kshell_sigma_rel,
        kshell_sigma_rel_trans=kshell_sigma_rel_trans,
        corr_stage1_mc=corr_mc,
        corr_stage2_rel_long=corr_rel_long,
        corr_stage3_rel_trans=corr_rel_trans,
        corr_stage4_density=corr_density,
        use_mc=use_mc,
        use_rel_long=use_rel_long,
        use_rel_trans=use_rel_trans,
        use_density_effect=use_density_effect,
        excitation_sigma_pwba=exc_pwba,
        ionization_sigma_pwba=ion_pwba,
        excitation_sigma_rel_long=exc_rel_long,
        ionization_sigma_rel_long=ion_rel_long,
        excitation_sigma_rel_trans=exc_rel_trans,
        ionization_sigma_rel_trans=ion_rel_trans,
        excitation_sigma_mc=exc_mc,
        ionization_sigma_mc=ion_mc,
        excitation_sigma_selected=exc_selected,
        ionization_sigma_selected=ion_selected,
    )

    phase = _infer_ice_type_from_path(ice_label) if ice_label else ICE_TYPE
    np_save_args.update(_ion_normalization_metadata(phase))

    if NE is not None:
        np_save_args["NE"] = int(NE)
    if Nq is not None:
        np_save_args["Nq"] = int(Nq)

    if dcs_data is not None:
        np_save_args["dcs_T_line"] = np.asarray(dcs_data.get("T_line", []), float)
        np_save_args["dcs_E_line"] = np.asarray(dcs_data.get("E_line", []), float)
        np_save_args["dcs_exc_vals"] = np.asarray(dcs_data.get("exc_vals", []), float)
        np_save_args["dcs_ion_vals"] = np.asarray(dcs_data.get("ion_vals", []), float)
        diag = dcs_data.get("barkas_diagnostics")
        if diag is not None:
            np_save_args["barkas_z_int"] = np.asarray(diag.z_int, float)
            np_save_args["TCS_T_eV"] = np.asarray(diag.TCS_T_eV, float)
            np_save_args["DCS_Born_m2_per_eV"] = np.asarray(diag.DCS_Born_m2_per_eV, float)
            np_save_args["DCS_Barkas_m2_per_eV"] = np.asarray(diag.DCS_Barkas_m2_per_eV, float)
            np_save_args["DCS_total_m2_per_eV"] = np.asarray(diag.DCS_total_m2_per_eV, float)
            np_save_args["TCS_Born_m2"] = np.asarray(diag.TCS_Born_m2, float)
            np_save_args["TCS_Barkas_m2"] = np.asarray(diag.TCS_Barkas_m2, float)
            np_save_args["TCS_total_m2"] = np.asarray(diag.TCS_total_m2, float)
            np_save_args["S_Barkas_check_eV_m2"] = np.asarray(diag.S_Barkas_check_eV_m2, float)
            np_save_args["DCS_total_min_m2_per_eV"] = float(diag.dcs_total_min_m2_per_eV)
            np_save_args["DCS_total_max_m2_per_eV"] = float(diag.dcs_total_max_m2_per_eV)
            np_save_args["barkas_negative_or_unstable_T_eV"] = np.asarray(diag.negative_or_unstable_T_eV, float)
            np_save_args["barkas_negative_channel_T_eV"] = np.asarray(diag.negative_channel_T_eV, float)
            np_save_args["barkas_nonfinite_T_eV"] = np.asarray(diag.nonfinite_T_eV, float)
            np_save_args["df_dW_total"] = np.asarray(diag.df_dW_total, float)
            np_save_args["df_dW_valence"] = np.asarray(diag.df_dW_valence, float)
            np_save_args["df_dW_OK"] = np.asarray(diag.df_dW_OK, float)
            np_save_args["df_dW_total_unique"] = np.asarray(diag.df_dW_total_unique, float)
            np_save_args["df_dW_valence_unique"] = np.asarray(diag.df_dW_valence_unique, float)
            np_save_args["df_dW_OK_unique"] = np.asarray(diag.df_dW_OK_unique, float)
            np_save_args["df_dW_integral"] = float(diag.df_dW_integral)
            np_save_args["df_dW_integral_norm_grid"] = float(diag.df_dW_integral_norm_grid)
            np_save_args["df_dW_integral_unique_grid"] = float(diag.df_dW_integral_unique_grid)
            np_save_args["df_dW_valence_raw_integral"] = float(diag.df_dW_valence_raw_integral)
            np_save_args["df_dW_OK_raw_integral"] = float(diag.df_dW_OK_raw_integral)
            np_save_args["df_dW_valence_norm"] = float(diag.df_dW_valence_norm)
            np_save_args["df_dW_OK_norm"] = float(diag.df_dW_OK_norm)
            np_save_args["barkas_born_reference_charge"] = str(diag.born_reference_charge)
            np_save_args["barkas_born_reference_q"] = float(diag.born_reference_q)
            np_save_args["barkas_channel_distribution"] = str(diag.barkas_channel_distribution)

    np.savez(
        out_path,
        **np_save_args,
    )

    print(f"Saved correction log to {out_path}")

def _npz_int_value(npz_data, key):
    if key not in npz_data:
        return None
    try:
        val = np.asarray(npz_data[key]).reshape(-1)[0]
    except Exception:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def _npz_float_value(npz_data, key):
    if key not in npz_data:
        return None
    try:
        val = np.asarray(npz_data[key]).reshape(-1)[0]
        return float(val)
    except Exception:
        return None

def _npz_str_value(npz_data, key):
    if key not in npz_data:
        return None
    try:
        val = np.asarray(npz_data[key]).reshape(-1)[0]
        return str(val)
    except Exception:
        return None

def _npz_matches_params(
    npz_data,
    NE,
    Nq,
    T_list=None,
    include_kshell=True,
    kshell_model=None,
    energy_unit="total",
    charge_mode="bare",
    explicit_charge=None,
    include_barkas_dcs=False,
    include_bloch_dcs=False,
    born_reference_charge="bare_Z",
    born_reference_explicit_charge=None,
):
    try:
        _require_current_normalization(npz_data, ICE_TYPE)
    except ValueError:
        return False
    ne = _npz_int_value(npz_data, "NE")
    nq = _npz_int_value(npz_data, "Nq")
    if ne is None or nq is None:
        return False
    if int(ne) != int(NE) or int(nq) != int(Nq):
        return False
    stored_projectile = _npz_str_value(npz_data, "projectile_key")
    stored_mass = _npz_float_value(npz_data, "projectile_mass_au")
    stored_charge = _npz_float_value(npz_data, "projectile_charge")
    if stored_projectile != PROJECTILE_KEY:
        return False
    if stored_mass is None or not np.isclose(stored_mass, PROJECTILE_MASS_AU):
        return False
    if stored_charge is None or not np.isclose(stored_charge, PROJECTILE_CHARGE):
        return False
    stored_energy_unit = _npz_str_value(npz_data, "energy_unit")
    if stored_energy_unit is None:
        if str(energy_unit) != "total":
            return False
    elif stored_energy_unit != str(energy_unit):
        return False
    stored_charge_mode = _npz_str_value(npz_data, "charge_mode")
    if stored_charge_mode is None:
        if str(charge_mode) != "bare":
            return False
    elif stored_charge_mode != str(charge_mode):
        return False
    stored_explicit_charge = _npz_float_value(npz_data, "explicit_projectile_charge")
    if explicit_charge is not None:
        if stored_explicit_charge is None or not np.isclose(stored_explicit_charge, explicit_charge):
            return False
    stored_barkas = bool(np.asarray(npz_data.get("include_barkas_dcs", np.array([False]))).reshape(-1)[0])
    if stored_barkas != bool(include_barkas_dcs):
        return False
    stored_bloch = bool(np.asarray(npz_data.get("include_bloch_dcs", np.array([False]))).reshape(-1)[0])
    if stored_bloch != bool(include_bloch_dcs):
        return False
    stored_born_ref = _npz_str_value(npz_data, "born_reference_charge")
    if stored_born_ref is None:
        if str(born_reference_charge) != "bare_Z":
            return False
    elif stored_born_ref != str(born_reference_charge):
        return False
    stored_born_q = _npz_float_value(npz_data, "born_reference_explicit_charge")
    if str(born_reference_charge) == "explicit_q":
        if stored_born_q is None or not np.isclose(stored_born_q, born_reference_explicit_charge):
            return False
    if "heavy_projectile_emax_applied" not in npz_data:
        return False
    if not bool(np.asarray(npz_data["heavy_projectile_emax_applied"]).reshape(-1)[0]):
        return False
    table_variable = _npz_str_value(npz_data, "dcs_table_variable")
    if table_variable != "energy_loss_eV":
        return False
    if "include_kshell" not in npz_data:
        return False
    stored_include_kshell = bool(np.asarray(npz_data["include_kshell"]).reshape(-1)[0])
    if stored_include_kshell != bool(include_kshell):
        return False
    if kshell_model is None:
        kshell_model = KSHELL_MODEL
    stored_kshell_model = _npz_str_value(npz_data, "kshell_model")
    if stored_kshell_model != str(kshell_model):
        return False
    if str(kshell_model) == "hydrogenic-gos":
        if "kshell_q_dependent" not in npz_data:
            return False
        if not bool(np.asarray(npz_data["kshell_q_dependent"]).reshape(-1)[0]):
            return False
        stored_B = _npz_float_value(npz_data, "kshell_B_eV")
        stored_Zeff = _npz_float_value(npz_data, "kshell_Zeff")
        stored_fsum = _npz_float_value(npz_data, "kshell_fsum_target")
        if stored_B is None or not np.isclose(stored_B, KSHELL_B_EV):
            return False
        if stored_Zeff is None or not np.isclose(stored_Zeff, KSHELL_ZEFF):
            return False
        if stored_fsum is None or not np.isclose(stored_fsum, KSHELL_FSUM_TARGET):
            return False
        if bool(np.asarray(npz_data.get("old_optical_kshell_used", np.array([True]))).reshape(-1)[0]):
            return False
        if "hydrogenic_kshell_rolloff_applied" not in npz_data:
            return False
        if bool(np.asarray(npz_data["hydrogenic_kshell_rolloff_applied"]).reshape(-1)[0]):
            return False
    if T_list is not None and "T_eV" in npz_data:
        T_arr = np.asarray(npz_data["T_eV"], float)
        if len(T_arr) != len(T_list):
            return False
        if not np.allclose(T_arr, np.asarray(T_list, float)):
            return False
    return True

def _dcs_data_from_npz(npz_data):
    keys = ("dcs_T_line", "dcs_E_line", "dcs_exc_vals", "dcs_ion_vals")
    if not all(key in npz_data for key in keys):
        return None
    T_line = np.asarray(npz_data["dcs_T_line"], float)
    E_line = np.asarray(npz_data["dcs_E_line"], float)
    if T_line.size == 0 or E_line.size == 0:
        return None
    if "T_eV" in npz_data:
        T_arr = np.asarray(npz_data["T_eV"], float)
        T_arr = T_arr[np.isfinite(T_arr) & (T_arr > 0.0)]
        if T_arr.size:
            target_min = float(np.min(T_arr))
            target_max = float(np.max(T_arr))
            dcs_min = float(np.min(T_line))
            dcs_max = float(np.max(T_line))
            if target_min > dcs_min and not np.isclose(target_min, dcs_min):
                return None
            if target_max > dcs_max and not np.isclose(target_max, dcs_max):
                return None
    exc_vals = np.asarray(npz_data["dcs_exc_vals"], float)
    ion_vals = np.asarray(npz_data["dcs_ion_vals"], float)
    E_upper = np.array([_projectile_energy_loss_upper_eV(T) for T in T_line], float)
    over_cutoff = E_line > (E_upper * (1.0 + 1.0e-12) + 1.0e-12)
    if np.any(over_cutoff):
        nonzero_over = (
            np.any(np.abs(exc_vals[over_cutoff]) > 0.0)
            or np.any(np.abs(ion_vals[over_cutoff]) > 0.0)
        )
        if nonzero_over:
            return None
    charge_mode = _npz_str_value(npz_data, "charge_mode") or "bare"
    include_barkas_dcs = bool(np.asarray(npz_data.get("include_barkas_dcs", np.array([False]))).reshape(-1)[0])
    born_reference_charge = _npz_str_value(npz_data, "born_reference_charge") or "bare_Z"
    return {
        "T_line": T_line,
        "E_line": E_line,
        "exc_vals": exc_vals,
        "ion_vals": ion_vals,
        "barkas_charge_applied": bool(str(charge_mode) != "bare" or include_barkas_dcs),
        "barkas_charge_mode": str(charge_mode),
        "born_reference_charge": str(born_reference_charge),
    }

def _sigma_list_from_npz(npz_data):
    if "T_eV" not in npz_data:
        raise KeyError("Missing T_eV in NPZ cache.")
    T_arr = np.asarray(npz_data["T_eV"], float)
    nT = len(T_arr)

    def _array(key):
        if key not in npz_data:
            return None
        return np.asarray(npz_data[key], float)

    def _row_vals(arr, i):
        if arr is None:
            return []
        row = np.asarray(arr[i], float)
        if row.ndim == 0:
            return [float(row)]
        return [float(val) for val in row]

    def _scalar(arr, i):
        if arr is None:
            return None
        val = float(arr[i])
        if not np.isfinite(val):
            return None
        return val

    exc_pwba = _array("excitation_sigma_pwba")
    ion_pwba = _array("ionization_sigma_pwba")
    exc_rel_long = _array("excitation_sigma_rel_long")
    ion_rel_long = _array("ionization_sigma_rel_long")
    exc_rel_trans = _array("excitation_sigma_rel_trans")
    ion_rel_trans = _array("ionization_sigma_rel_trans")
    exc_mc = _array("excitation_sigma_mc")
    ion_mc = _array("ionization_sigma_mc")

    total_sigma = _array("total_sigma")
    total_sigma_pwba = _array("total_sigma_pwba")
    total_sigma_mc = _array("total_sigma_mc")
    total_sigma_rel = _array("total_sigma_rel")
    total_sigma_rel_trans = _array("total_sigma_rel_trans")
    total_sigma_rel_trans_no_density = _array("total_sigma_rel_trans_no_density")
    total_sigma_rel_total = _array("total_sigma_rel_total")
    total_sigma_plus_rel = _array("total_sigma_plus_rel")
    total_sigma_plus_rel_total = _array("total_sigma_plus_rel_total")
    kshell_sigma = _array("kshell_sigma")
    kshell_sigma_rel = _array("kshell_sigma_rel")

    sigma_list = []
    for i in range(nT):
        sigma = {
            "excitation_sigma_pwba": _row_vals(exc_pwba, i),
            "ionization_sigma_pwba": _row_vals(ion_pwba, i),
            "excitation_sigma_rel": _row_vals(exc_rel_long, i),
            "ionization_sigma_rel": _row_vals(ion_rel_long, i),
            "excitation_sigma_rel_trans": _row_vals(exc_rel_trans, i),
            "ionization_sigma_rel_trans": _row_vals(ion_rel_trans, i),
            "excitation_sigma_mc": _row_vals(exc_mc, i),
            "ionization_sigma_mc": _row_vals(ion_mc, i),
            "total_sigma": _scalar(total_sigma, i),
            "total_sigma_pwba": _scalar(total_sigma_pwba, i),
            "total_sigma_mc": _scalar(total_sigma_mc, i),
            "total_sigma_rel": _scalar(total_sigma_rel, i),
            "total_sigma_rel_trans": _scalar(total_sigma_rel_trans, i),
            "total_sigma_rel_trans_no_density": _scalar(total_sigma_rel_trans_no_density, i),
            "total_sigma_rel_total": _scalar(total_sigma_rel_total, i),
            "total_sigma_plus_rel": _scalar(total_sigma_plus_rel, i),
            "total_sigma_plus_rel_total": _scalar(total_sigma_plus_rel_total, i),
            "kshell_sigma": _scalar(kshell_sigma, i),
            "kshell_sigma_rel": _scalar(kshell_sigma_rel, i),
        }
        sigma_list.append(sigma)
    return T_arr.tolist(), sigma_list

def load_cross_section_corrections_npz(
    npz_path,
    NE,
    Nq,
    T_list=None,
    require_dcs=False,
    include_kshell=True,
    kshell_model=None,
    energy_unit="total",
    charge_mode="bare",
    explicit_charge=None,
    include_barkas_dcs=False,
    include_bloch_dcs=False,
    born_reference_charge="bare_Z",
    born_reference_explicit_charge=None,
):
    if npz_path is None or not os.path.exists(npz_path):
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as npz_data:
            if not _npz_matches_params(
                npz_data,
                NE,
                Nq,
                T_list=T_list,
                include_kshell=include_kshell,
                kshell_model=kshell_model,
                energy_unit=energy_unit,
                charge_mode=charge_mode,
                explicit_charge=explicit_charge,
                include_barkas_dcs=include_barkas_dcs,
                include_bloch_dcs=include_bloch_dcs,
                born_reference_charge=born_reference_charge,
                born_reference_explicit_charge=born_reference_explicit_charge,
            ):
                return None
            dcs_data = _dcs_data_from_npz(npz_data)
            if require_dcs and dcs_data is None:
                return None
            T_loaded, sigma_list = _sigma_list_from_npz(npz_data)
    except Exception as exc:
        print(f"Failed to load cached NPZ {npz_path}: {exc}")
        return None
    return T_loaded, sigma_list, dcs_data

def _geant4_dna_dir():
    for root in (CUSTOM_DATA_ROOT_GEANT4, CUSTOM_DATA_ROOT_PROJECT):
        if root is None:
            continue
        if not root.exists():
            continue
        dna_dir = root / "G4EMLOW8.6.1" / "dna"
        if dna_dir.exists():
            return dna_dir
    return None

def _default_dcs_template_paths(ice_label):
    labels = []
    for label in (ice_label, ICE_LABEL):
        if label and label not in labels:
            labels.append(label)

    for label in labels:
        path = (
            CROSS_SECTIONS_DIR
            / f"sigmadiff_ionisation_e_{label}_emfietzoglou_kyriakou.dat"
        )
        if path.exists():
            return path, None

    dna_dir = _geant4_dna_dir()
    if dna_dir is None:
        return None, None
    return (
        dna_dir / "sigmadiff_ionisation_e_emfietzoglou.dat",
        dna_dir / "sigmadiff_ionisation_e_born.dat",
    )

def _load_dcs_template_grid(path, t_min=None, t_max=None, include_min=True, include_max=True):
    from collections import OrderedDict

    grid = OrderedDict()
    with open(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                T = float(parts[0])
                E = float(parts[1])
            except ValueError:
                continue
            if t_min is not None:
                if T < t_min or (not include_min and T == t_min):
                    continue
            if t_max is not None:
                if T > t_max or (not include_max and T == t_max):
                    continue
            grid.setdefault(T, []).append(E)
    return grid

def _merge_dcs_template_grids(*grids):
    from collections import OrderedDict

    merged = OrderedDict()
    for grid in grids:
        for T, E_list in grid.items():
            if T in merged:
                continue
            merged[T] = E_list
    return merged

def _extend_dcs_grid(grid, t_max, t_step):
    from collections import OrderedDict

    if not grid:
        return grid
    t_max = float(t_max)
    t_step = float(t_step)
    if t_step <= 0.0:
        return grid

    grid_sorted = OrderedDict(sorted(grid.items(), key=lambda kv: kv[0]))
    last_T = max(grid_sorted.keys())
    if t_max <= last_T:
        return grid_sorted

    template_E = list(grid_sorted[last_T])
    t = last_T + t_step
    while t <= t_max + 0.5 * t_step:
        grid_sorted[float(t)] = list(template_E)
        t += t_step
    return grid_sorted

def _format_dcs_row(T, E, vals):
    fields = [f"{T:.9E}", f"{E:.9E}"]
    fields.extend(f"{val:.9E}" for val in vals)
    return " ".join(fields) + "\n"

def _write_dcs_tables_from_data(
    dcs_data,
    exc_out,
    ion_out,
    exc_t_min=None,
    ion_t_min=None,
    t_max=None,
    table_metadata=None,
):
    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = np.asarray(dcs_data.get("exc_vals", []), float)
    ion_vals = np.asarray(dcs_data.get("ion_vals", []), float)

    if T_line.size == 0 or E_line.size == 0:
        raise ValueError("DCS data is empty.")
    if T_line.shape != E_line.shape:
        raise ValueError("DCS T/E arrays must have the same shape.")
    if exc_vals.ndim == 1:
        exc_vals = exc_vals.reshape(-1, 1)
    if ion_vals.ndim == 1:
        ion_vals = ion_vals.reshape(-1, 1)
    if exc_vals.shape[0] != T_line.size or ion_vals.shape[0] != T_line.size:
        raise ValueError("DCS value arrays must align with T/E lines.")

    exc_mask = np.ones(T_line.size, dtype=bool)
    ion_mask = np.ones(T_line.size, dtype=bool)
    if exc_t_min is not None:
        exc_mask &= T_line >= float(exc_t_min)
    if ion_t_min is not None:
        ion_mask &= T_line >= float(ion_t_min)
    if t_max is not None:
        tmax = float(t_max)
        exc_mask &= T_line <= tmax
        ion_mask &= T_line <= tmax

    with open(exc_out, "w") as exc_handle, open(ion_out, "w") as ion_handle:
        _write_table_metadata(exc_handle, table_metadata)
        _write_table_metadata(ion_handle, table_metadata)
        for i in range(T_line.size):
            if exc_mask[i]:
                exc_handle.write(_format_dcs_row(T_line[i], E_line[i], exc_vals[i]))
            if ion_mask[i]:
                ion_handle.write(_format_dcs_row(T_line[i], E_line[i], ion_vals[i]))

def _normalize_dcs_value_array(vals):
    vals = np.asarray(vals, float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    return vals

def _sort_dcs_data(dcs_data):
    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = _normalize_dcs_value_array(dcs_data.get("exc_vals", []))
    ion_vals = _normalize_dcs_value_array(dcs_data.get("ion_vals", []))
    if T_line.size == 0 or E_line.size != T_line.size:
        raise ValueError("DCS T/E arrays must be nonempty and aligned.")
    if exc_vals.shape[0] != T_line.size or ion_vals.shape[0] != T_line.size:
        raise ValueError("DCS value arrays must align with T/E lines.")
    order = np.lexsort((E_line, T_line))
    return {
        "T_line": T_line[order],
        "E_line": E_line[order],
        "exc_vals": exc_vals[order],
        "ion_vals": ion_vals[order],
    }

def _slice_dcs_data(dcs_data, t_min=None, t_max=None):
    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = _normalize_dcs_value_array(dcs_data.get("exc_vals", []))
    ion_vals = _normalize_dcs_value_array(dcs_data.get("ion_vals", []))
    if T_line.size == 0 or E_line.size != T_line.size:
        raise ValueError("DCS T/E arrays must be nonempty and aligned.")
    mask = np.isfinite(T_line) & np.isfinite(E_line)
    if t_min is not None:
        mask &= T_line >= float(t_min)
    if t_max is not None:
        mask &= T_line <= float(t_max)
    return _sort_dcs_data(
        {
            "T_line": T_line[mask],
            "E_line": E_line[mask],
            "exc_vals": exc_vals[mask],
            "ion_vals": ion_vals[mask],
        }
    )

def _merge_dcs_energy_patch(existing_data, patch_data):
    existing_data = _sort_dcs_data(existing_data)
    patch_data = _sort_dcs_data(patch_data)
    T_patch = np.asarray(patch_data["T_line"], float)
    if T_patch.size == 0:
        return existing_data
    if existing_data["exc_vals"].shape[1] != patch_data["exc_vals"].shape[1]:
        raise ValueError("Existing and current excitation DCS tables have different channel counts.")
    if existing_data["ion_vals"].shape[1] != patch_data["ion_vals"].shape[1]:
        raise ValueError("Existing and current ionisation DCS tables have different channel counts.")

    replace_min = float(np.min(T_patch))
    replace_max = float(np.max(T_patch))
    T_existing = np.asarray(existing_data["T_line"], float)
    keep_existing = (T_existing < replace_min) | (T_existing > replace_max)

    return _sort_dcs_data(
        {
            "T_line": np.concatenate([T_existing[keep_existing], patch_data["T_line"]]),
            "E_line": np.concatenate([existing_data["E_line"][keep_existing], patch_data["E_line"]]),
            "exc_vals": np.vstack([existing_data["exc_vals"][keep_existing], patch_data["exc_vals"]]),
            "ion_vals": np.vstack([existing_data["ion_vals"][keep_existing], patch_data["ion_vals"]]),
        }
    )

def _write_table_metadata(handle, metadata):
    if metadata is not None:
        handle.write("# ion_table_metadata: " + json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")


def _check_energy_patch_normalization(exc_out, ion_out, metadata):
    existing = [path for path in (exc_out, ion_out) if path.exists()]
    if not existing:
        return
    if len(existing) != 2:
        raise ValueError("Incomplete existing DCS pair; regenerate both tables with --no-merge-energy-patches.")
    for path in existing:
        with open(path) as handle:
            first_line = handle.readline()
        prefix = "# ion_table_metadata: "
        if not first_line.startswith(prefix):
            raise ValueError(
                f"Legacy/unversioned DCS table {path}; cannot merge different normalizations. "
                "Regenerate the full energy range with --no-merge-energy-patches."
            )
        stored = json.loads(first_line[len(prefix):])
        _require_current_normalization(stored, metadata["ice_type"])
        for key, value in metadata.items():
            actual = stored.get(key)
            matches = key in stored and (
                actual is not None and np.isclose(actual, value, rtol=1e-10, atol=0.0)
                if isinstance(value, float) else actual == value
            )
            if not matches:
                raise ValueError(
                    f"Incompatible DCS metadata {key} in {path}; use --no-merge-energy-patches "
                    "to regenerate the full energy range."
                )


def _prepare_dcs_data_for_output(dcs_data, exc_out, ion_out, t_min=None, t_max=None, merge_energy_patches=True, table_metadata=None):
    patch_data = _slice_dcs_data(dcs_data, t_min=t_min, t_max=t_max)
    if not merge_energy_patches:
        return patch_data
    if table_metadata is None:
        raise ValueError("DCS energy-patch merge requires normalization metadata.")
    _check_energy_patch_normalization(exc_out, ion_out, table_metadata)
    if not exc_out.exists():
        return patch_data
    existing_data = _load_dcs_pair(exc_out, ion_out)
    merged = _merge_dcs_energy_patch(existing_data, patch_data)
    print(
        "Merged DCS energy patches: "
        f"existing rows={len(existing_data['T_line'])}, "
        f"current rows={len(patch_data['T_line'])}, "
        f"merged rows={len(merged['T_line'])}."
    )
    return merged

def _load_dcs_table(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"DCS table {path} has fewer than 3 columns.")
    T_line = np.asarray(data[:, 0], float)
    E_line = np.asarray(data[:, 1], float)
    vals = np.asarray(data[:, 2:], float)
    return T_line, E_line, vals

def _load_dcs_pair(exc_path, ion_path):
    T_exc, E_exc, exc_vals = _load_dcs_table(exc_path)
    T_ion, E_ion, ion_vals = _load_dcs_table(ion_path)
    if (T_exc.shape != T_ion.shape) or (E_exc.shape != E_ion.shape):
        raise ValueError("Excitation and ionization DCS grids have different shapes.")
    if not np.allclose(T_exc, T_ion) or not np.allclose(E_exc, E_ion):
        raise ValueError("Excitation and ionization DCS grids do not match.")
    return {
        "T_line": T_exc,
        "E_line": E_exc,
        "exc_vals": exc_vals,
        "ion_vals": ion_vals,
    }

def _integrate_dcs_to_totals(T_line, E_line, vals):
    T_line = np.asarray(T_line, float)
    E_line = np.asarray(E_line, float)
    vals = np.asarray(vals, float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if (T_line.size != E_line.size) or (T_line.size != vals.shape[0]):
        raise ValueError("DCS arrays must have matching lengths.")

    unique_T = []
    totals = []
    n = T_line.size
    start = 0
    for i in range(1, n + 1):
        if i == n or T_line[i] != T_line[start]:
            Tval = float(T_line[start])
            E_seg = np.asarray(E_line[start:i], float)
            V_seg = np.asarray(vals[start:i], float)
            E_upper = _projectile_energy_loss_upper_eV(Tval)
            physical = np.isfinite(E_seg) & (E_seg <= E_upper)
            E_seg = E_seg[physical]
            V_seg = V_seg[physical]
            if E_seg.size == 0:
                start = i
                continue
            order = np.argsort(E_seg)
            E_sorted = E_seg[order]
            V_sorted = V_seg[order]
            row = [_simpson_integrate(V_sorted[:, j], E_sorted) for j in range(V_sorted.shape[1])]
            unique_T.append(Tval)
            totals.append(row)
            start = i
    return np.asarray(unique_T, float), np.asarray(totals, float)

def _run_projectile_pwba_sanity_checks(s, C, dcs_data=None, Nq=80, include_kshell=True):
    Ei = 100.0
    Tj = 1.0e5
    qp = _q_bounds_scalar(Ei, Tj, projectile_mass_au=PROJECTILE_MASS_AU)
    qe = _q_bounds_scalar(Ei, Tj, projectile_mass_au=1.0)
    ratio = qp[1] / qe[1] if qe[1] > 0.0 else np.nan
    expected = np.sqrt(PROJECTILE_MASS_AU)
    print(
        "PWBA q-limit sanity: "
        f"projectile={PROJECTILE_KEY}, projectile_mass_au={PROJECTILE_MASS_AU:.6f}, "
        f"charge={PROJECTILE_CHARGE:.6g}, "
        f"qhi(projectile)/qhi(electron)={ratio:.6g}, sqrt(M)={expected:.6g}"
    )
    print(
        "Heavy-projectile Emax sanity: "
        f"Emax(1 MeV {PROJECTILE_KEY})={heavy_projectile_Emax(1.0e6):.6g} eV"
    )

    if dcs_data is None:
        return

    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = np.asarray(dcs_data.get("exc_vals", []), float)
    ion_vals = np.asarray(dcs_data.get("ion_vals", []), float)
    if T_line.size == 0 or E_line.size != T_line.size:
        return

    E_upper = np.array([_projectile_energy_loss_upper_eV(T) for T in T_line], float)
    over_cutoff = E_line > (E_upper * (1.0 + 1.0e-12) + 1.0e-12)
    if np.any(over_cutoff):
        if np.any(np.abs(exc_vals[over_cutoff]) > 0.0) or np.any(np.abs(ion_vals[over_cutoff]) > 0.0):
            raise RuntimeError("DCS has nonzero values above heavy-projectile Emax.")
    valid_q = np.isfinite(E_line) & np.isfinite(T_line) & (E_line > 0.0) & (E_line < T_line) & (E_line <= E_upper)
    if np.any(valid_q):
        for Ei_check, T_check in zip(E_line[valid_q], T_line[valid_q]):
            qlo, qhi = _q_bounds_scalar(Ei_check, T_check, projectile_mass_au=PROJECTILE_MASS_AU)
            if not (np.isfinite(qlo) and np.isfinite(qhi) and 0.0 < qlo < qhi):
                raise RuntimeError(
                    f"Invalid projectile q bounds at T={T_check:.6g} eV, E={Ei_check:.6g} eV."
                )
    print(
        "DCS kinematic sanity: "
        f"{int(np.count_nonzero(over_cutoff))} template rows above Emax forced to zero; "
        f"{int(np.count_nonzero(valid_q))} rows have ordered projectile q-bounds."
    )

    for Tval in np.unique(T_line):
        idx = np.flatnonzero(T_line == Tval)
        physical_idx = idx[E_line[idx] <= _projectile_energy_loss_upper_eV(Tval)]
        if physical_idx.size >= 3:
            break
    else:
        return

    E_seg = E_line[physical_idx]
    order = np.argsort(E_seg)
    E_sorted = E_seg[order]
    idx_sorted = physical_idx[order]

    exc_direct = np.zeros((E_sorted.size, len(s.excitations)), float)
    kshell_B = (
        _kshell_threshold_eV(s)
        if include_kshell and s.kshell is not None and KSHELL_MODEL != "none"
        else None
    )
    ion_direct = np.zeros((E_sorted.size, len(s.ionizations) + (1 if kshell_B is not None else 0)), float)
    exc_B = [float(s.Bmin) for _ in s.excitations]
    ion_B = [float(osc.Bth) for osc in s.ionizations]
    for j in range(len(s.excitations)):
        exc_direct[:, j] = _compute_dcs_channel_values(
            "excitation", j, s, C, Nq, np.full_like(E_sorted, Tval), E_sorted, exc_B, ion_B, kshell_B
        )
    for j in range(len(s.ionizations)):
        ion_direct[:, j] = _compute_dcs_channel_values(
            "ionization", j, s, C, Nq, np.full_like(E_sorted, Tval), E_sorted, exc_B, ion_B, kshell_B
        )
    if kshell_B is not None:
        ion_direct[:, -1] = _compute_dcs_channel_values(
            "kshell", 0, s, C, Nq, np.full_like(E_sorted, Tval), E_sorted, exc_B, ion_B, kshell_B
        )

    exc_from_dcs = np.asarray(exc_vals[idx_sorted], float)
    ion_from_dcs = np.asarray(ion_vals[idx_sorted], float)
    exc_total_dcs = float(np.sum([_simpson_integrate(exc_from_dcs[:, j], E_sorted) for j in range(exc_from_dcs.shape[1])]))
    ion_total_dcs = float(np.sum([_simpson_integrate(ion_from_dcs[:, j], E_sorted) for j in range(ion_from_dcs.shape[1])]))
    exc_total_direct = float(np.sum([_simpson_integrate(exc_direct[:, j], E_sorted) for j in range(exc_direct.shape[1])]))
    ion_total_direct = float(np.sum([_simpson_integrate(ion_direct[:, j], E_sorted) for j in range(ion_direct.shape[1])]))
    exc_ok = np.isclose(exc_total_dcs, exc_total_direct, rtol=1.0e-10, atol=1.0e-12)
    ion_ok = np.isclose(ion_total_dcs, ion_total_direct, rtol=1.0e-10, atol=1.0e-12)
    print(
        "DCS/direct same-grid sanity: "
        f"T={Tval:.6g} eV, excitation relerr={abs(exc_total_dcs - exc_total_direct) / max(abs(exc_total_direct), 1.0e-300):.3e}, "
        f"ionization relerr={abs(ion_total_dcs - ion_total_direct) / max(abs(ion_total_direct), 1.0e-300):.3e}"
    )
    if not (exc_ok and ion_ok):
        raise RuntimeError("DCS-integrated totals differ from direct same-grid totals.")

def _write_total_table(T_vals, totals, out_path, table_metadata=None):
    with open(out_path, "w") as handle:
        _write_table_metadata(handle, table_metadata)
        for Tval, row in zip(T_vals, totals):
            fields = [f"{Tval:.9E}"]
            fields.extend(f"{val:.9E}" for val in row)
            handle.write(" ".join(fields) + "\n")

def _filter_dcs_lines(T_line, E_line, vals, t_min=None, t_max=None):
    mask = np.ones(np.asarray(T_line).size, dtype=bool)
    if t_min is not None:
        mask &= np.asarray(T_line, float) >= float(t_min)
    if t_max is not None:
        mask &= np.asarray(T_line, float) <= float(t_max)
    return (
        np.asarray(T_line, float)[mask],
        np.asarray(E_line, float)[mask],
        np.asarray(vals, float)[mask],
    )


def _write_total_tables_from_dcs(
    dcs_data,
    exc_out,
    ion_out,
    exc_t_min=None,
    ion_t_min=None,
    t_max=None,
    table_metadata=None,
):
    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = np.asarray(dcs_data.get("exc_vals", []), float)
    ion_vals = np.asarray(dcs_data.get("ion_vals", []), float)
    if T_line.size == 0 or E_line.size == 0:
        raise ValueError("DCS data is empty; cannot compute totals.")
    T_exc_line, E_exc_line, exc_vals = _filter_dcs_lines(
        T_line, E_line, exc_vals, t_min=exc_t_min, t_max=t_max
    )
    T_ion_line, E_ion_line, ion_vals = _filter_dcs_lines(
        T_line, E_line, ion_vals, t_min=ion_t_min, t_max=t_max
    )
    T_exc, exc_totals = _integrate_dcs_to_totals(T_exc_line, E_exc_line, exc_vals)
    T_ion, ion_totals = _integrate_dcs_to_totals(T_ion_line, E_ion_line, ion_vals)
    _write_total_table(T_exc, exc_totals, exc_out, table_metadata)
    _write_total_table(T_ion, ion_totals, ion_out, table_metadata)

def _replace_sigma_list_with_dcs_totals(T_list, sigma_list, dcs_data):
    if sigma_list is None or dcs_data is None:
        return sigma_list
    T_line = np.asarray(dcs_data.get("T_line", []), float)
    E_line = np.asarray(dcs_data.get("E_line", []), float)
    exc_vals = np.asarray(dcs_data.get("exc_vals", []), float)
    ion_vals = np.asarray(dcs_data.get("ion_vals", []), float)
    if T_line.size == 0 or E_line.size == 0:
        return sigma_list
    T_exc, exc_totals = _integrate_dcs_to_totals(T_line, E_line, exc_vals * EMFI_DCS_SCALE_M2)
    T_ion, ion_totals = _integrate_dcs_to_totals(T_line, E_line, ion_vals * EMFI_DCS_SCALE_M2)

    out = []
    for T, sigma in zip(T_list, sigma_list):
        sigma_new = dict(sigma)
        i_exc = int(np.argmin(np.abs(T_exc - float(T)))) if T_exc.size else None
        i_ion = int(np.argmin(np.abs(T_ion - float(T)))) if T_ion.size else None
        if i_exc is not None and np.isclose(T_exc[i_exc], float(T)):
            exc = [float(v) for v in np.asarray(exc_totals[i_exc], float)]
            sigma_new["excitation_sigma_pwba"] = exc
            sigma_new["excitation_sigma"] = exc
        else:
            exc = sigma_new.get("excitation_sigma", sigma_new.get("excitation_sigma_pwba", [])) or []
        if i_ion is not None and np.isclose(T_ion[i_ion], float(T)):
            ion = [float(v) for v in np.asarray(ion_totals[i_ion], float)]
            sigma_new["ionization_sigma_pwba"] = ion
            sigma_new["ionization_sigma"] = ion
        else:
            ion = sigma_new.get("ionization_sigma", sigma_new.get("ionization_sigma_pwba", [])) or []
        total = float(np.sum(exc) + np.sum(ion))
        sigma_new["valence_sigma_pwba"] = total
        sigma_new["total_sigma_pwba"] = total
        sigma_new["valence_sigma"] = total
        sigma_new["total_sigma"] = total
        out.append(sigma_new)
    return out

def _export_to_custom_geant4(paths):
    roots = []
    for root in (CUSTOM_DATA_ROOT_GEANT4, CUSTOM_DATA_ROOT_PROJECT):
        if root is None:
            continue
        dna_dir = root / "G4EMLOW8.6.1" / "dna"
        if dna_dir.exists():
            roots.append(dna_dir)
    if not roots:
        print("No custom Geant4 DNA data dir found; skipping export.")
        return
    for dna_dir in roots:
        for path in paths:
            if path is None:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Missing file for export: {path}")
            dest = dna_dir / path.name
            shutil.copy2(path, dest)
        print(f"Exported {len(paths)} files to {dna_dir}")

_DCS_WORKER_S = None
_DCS_WORKER_C = None
_DCS_WORKER_T_LINE = None
_DCS_WORKER_E_LINE = None
_DCS_WORKER_NQ = None
_DCS_WORKER_EXC_B = None
_DCS_WORKER_ION_B = None
_DCS_WORKER_KSHELL_B = None

def _compute_dcs_channel_values(
    channel_type,
    idx,
    s,
    C,
    Nq,
    T_line,
    E_line,
    exc_B,
    ion_B,
    kshell_B,
):
    T_line = np.asarray(T_line, float)
    E_line = np.asarray(E_line, float)
    n_lines = T_line.size
    vals = np.zeros(n_lines, float)

    for i in range(n_lines):
        Tj = float(T_line[i])
        Ei = float(E_line[i])
        E_upper = _projectile_energy_loss_upper_eV(Tj)

        if channel_type == "excitation":
            Bk = exc_B[idx]
            if Ei < Bk or Ei > E_upper:
                val = 0.0
            else:
                val = _selected_dsigma_excitation(Ei, Tj, idx, s, C, Nq)
        elif channel_type == "ionization":
            Bj = ion_B[idx]
            if Ei < Bj or Ei > E_upper:
                val = 0.0
            else:
                val = _selected_dsigma_ionization(Ei, Tj, idx, s, C, Nq)
        elif channel_type == "kshell":
            if kshell_B is None:
                val = 0.0
            elif Ei < kshell_B or Ei > E_upper:
                val = 0.0
            else:
                val = _selected_dsigma_kshell(Ei, Tj, s, C, Nq)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")

        if (not np.isfinite(val)) or (val < 0.0):
            val = 0.0
        vals[i] = val / EMFI_DCS_SCALE_M2

    return vals

def _init_dcs_worker(
    a_vec,
    b_vec,
    c_vec,
    T_line,
    E_line,
    Nq,
    apply_mc,
    apply_regime_ii,
    apply_regime_iii,
    apply_regime_iv,
    ice_type,
    projectile_key,
    kshell_model,
    projectile_relativistic_dcs,
    include_transverse_dcs,
    rpwba_density_effect,
):
    global _DCS_WORKER_S, _DCS_WORKER_C
    global _DCS_WORKER_T_LINE, _DCS_WORKER_E_LINE, _DCS_WORKER_NQ
    global _DCS_WORKER_EXC_B, _DCS_WORKER_ION_B, _DCS_WORKER_KSHELL_B

    _set_regime_corrections(
        apply_regime_ii=apply_regime_ii,
        apply_regime_iii=apply_regime_iii,
        apply_regime_iv=apply_regime_iv,
    )
    _set_mc_correction(apply_mc=apply_mc)
    set_projectile(projectile_key)
    _set_kshell_model(kshell_model)
    _set_projectile_relativistic_dcs(
        projectile_relativistic_dcs,
        include_transverse_dcs,
        use_density_effect=rpwba_density_effect,
    )

    _DCS_WORKER_S = model.epsilon_optical(ice_type)
    _DCS_WORKER_C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)
    _DCS_WORKER_T_LINE = np.asarray(T_line, float)
    _DCS_WORKER_E_LINE = np.asarray(E_line, float)
    _DCS_WORKER_NQ = int(Nq)
    _DCS_WORKER_EXC_B = [float(_DCS_WORKER_S.Bmin) for _ in _DCS_WORKER_S.excitations]
    _DCS_WORKER_ION_B = [float(osc.Bth) for osc in _DCS_WORKER_S.ionizations]
    _DCS_WORKER_KSHELL_B = (
        _kshell_threshold_eV(_DCS_WORKER_S)
        if _DCS_WORKER_S.kshell is not None and KSHELL_MODEL != "none"
        else None
    )

def _compute_dcs_channel_worker(args):
    channel_type, idx = args
    vals = _compute_dcs_channel_values(
        channel_type,
        idx,
        _DCS_WORKER_S,
        _DCS_WORKER_C,
        _DCS_WORKER_NQ,
        _DCS_WORKER_T_LINE,
        _DCS_WORKER_E_LINE,
        _DCS_WORKER_EXC_B,
        _DCS_WORKER_ION_B,
        _DCS_WORKER_KSHELL_B,
    )
    return channel_type, idx, vals

def _selected_dsigma_excitation(Ei, Tj, k, s, C, Nq):
    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(Tj)
    if use_mc:
        Bk = float(s.Bmin)
        Tshift = float(Tj + 2.0 * Bk)
        return _dsigma_pwba_dE(Ei, Tshift, k, "excitation", s, C, Nq=Nq, use_rel=use_rel_long)
    if use_rel_long:
        longitudinal, transverse = _integrate_channel_single_E_rpwba_components(
            Ei,
            Tj,
            k,
            "excitation",
            s,
            C,
            Nq=Nq,
            use_density_effect=use_density_effect,
        )
        return longitudinal + (transverse if use_rel_trans else 0.0)
    return _integrate_channel_single_E(Ei, Tj, k, "excitation", s, C, Nq=Nq, use_rel_bounds=False)

def _selected_dsigma_ionization(Ei, Tj, j, s, C, Nq):
    use_mc, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(Tj)
    if use_mc:
        return _dsigma_mc_ionization_dE(Ei, Tj, j, s, C, Nq=Nq, use_rel=use_rel_long)
    if use_rel_long:
        longitudinal, transverse = _integrate_channel_single_E_rpwba_components(
            Ei,
            Tj,
            j,
            "ionization",
            s,
            C,
            Nq=Nq,
            use_density_effect=use_density_effect,
        )
        return longitudinal + (transverse if use_rel_trans else 0.0)
    return _integrate_channel_single_E(Ei, Tj, j, "ionization", s, C, Nq=Nq, use_rel_bounds=False)

def _selected_dsigma_kshell(Ei, Tj, s, C, Nq):
    if s.kshell is None or KSHELL_MODEL == "none":
        return 0.0
    _, use_rel_long, use_rel_trans, use_density_effect = _regime_flags(Tj)
    if use_rel_long:
        longitudinal, transverse = _integrate_kshell_single_E_rpwba_components(
            Ei,
            Tj,
            s,
            C,
            Nq=Nq,
            include_kshell=True,
            use_density_effect=use_density_effect,
        )
        return longitudinal + (transverse if use_rel_trans else 0.0)
    return _integrate_kshell_single_E(Ei, Tj, s, Nq=Nq, include_kshell=True, use_rel_bounds=False)

def write_emfietzoglou_dcs_tables(
    s,
    C,
    Nq=200,
    out_dir=None,
    template_path=None,
    dcs_data=None,
    return_data=False,
    parallel_channels=True,
    max_workers=None,
    a_vec=None,
    b_vec=None,
    c_vec=None,
    ice_label=None,
    ice_type=None,
    apply_mc=None,
    apply_regime_ii=None,
    apply_regime_iii=None,
    apply_regime_iv=None,
    reuse_existing_tables=False,
    T_list=None,
    include_kshell=True,
    merge_energy_patches=True,
    charge_mode=None,
    include_barkas_dcs=None,
    explicit_charge=None,
    born_reference_charge=None,
    born_reference_explicit_charge=None,
):
    if out_dir is None:
        out_dir = CROSS_SECTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if ice_label is None:
        ice_label = ICE_LABEL
    if charge_mode is None:
        charge_mode = CHARGE_MODE
    if include_barkas_dcs is None:
        include_barkas_dcs = INCLUDE_BARKAS_DCS
    if explicit_charge is None:
        explicit_charge = EXPLICIT_PROJECTILE_CHARGE
    if born_reference_charge is None:
        born_reference_charge = BORN_REFERENCE_CHARGE
    if born_reference_explicit_charge is None:
        born_reference_explicit_charge = BORN_REFERENCE_EXPLICIT_CHARGE
    mode_suffix = _projectile_kernel_tag() + _charge_mode_tag(
        charge_mode,
        include_barkas_dcs,
        explicit_charge=explicit_charge,
        born_reference_charge=born_reference_charge,
        born_reference_explicit_charge=born_reference_explicit_charge,
    )
    exc_out = out_dir / f"sigmadiff_excitation_{PROJECTILE_FILE_TOKEN}_{ice_label}{mode_suffix}_emfietzoglou_kyriakou.dat"
    ion_out = out_dir / f"sigmadiff_ionisation_{PROJECTILE_FILE_TOKEN}_{ice_label}{mode_suffix}_emfietzoglou_kyriakou.dat"
    exc_total_out = out_dir / f"sigma_excitation_{PROJECTILE_FILE_TOKEN}_{ice_label}{mode_suffix}_emfietzoglou_kyriakou.dat"
    ion_total_out = out_dir / f"sigma_ionisation_{PROJECTILE_FILE_TOKEN}_{ice_label}{mode_suffix}_emfietzoglou_kyriakou.dat"

    table_metadata = _ion_normalization_metadata(ice_type or s.material, s)
    table_metadata.update({
        "projectile": PROJECTILE_KEY,
        "projectile_mass_au": float(PROJECTILE_MASS_AU),
        "projectile_charge": float(PROJECTILE_CHARGE),
        "charge_mode": charge_mode,
        "explicit_charge": explicit_charge,
        "include_barkas_dcs": bool(include_barkas_dcs),
        "born_reference_charge": born_reference_charge,
        "born_reference_explicit_charge": born_reference_explicit_charge,
        "projectile_kernel": RPWBA_MODEL_NAME if PROJECTILE_RELATIVISTIC_DCS else "pwba",
        "rpwba_density_effect": bool(RPWBA_DENSITY_EFFECT),
        "include_kshell": bool(include_kshell),
        "kshell_model": KSHELL_MODEL,
    })
    if merge_energy_patches:
        _check_energy_patch_normalization(exc_out, ion_out, table_metadata)

    if reuse_existing_tables and dcs_data is None and exc_out.exists() and ion_out.exists():
        print("Ignoring existing DCS tables; heavy-projectile Emax metadata is not available in DAT files.")

    exc_B = [float(s.Bmin) for _ in s.excitations]
    ion_B = [float(osc.Bth) for osc in s.ionizations]
    t_list_min = None
    t_list_max = None
    if T_list is not None:
        T_arr_for_dcs = np.asarray(T_list, float)
        T_arr_for_dcs = T_arr_for_dcs[np.isfinite(T_arr_for_dcs) & (T_arr_for_dcs > 0.0)]
        if T_arr_for_dcs.size:
            t_list_min = float(np.min(T_arr_for_dcs))
            t_list_max = float(np.max(T_arr_for_dcs))
    exc_write_t_min = float(min(exc_B)) if exc_B else None
    ion_write_t_min = float(min(ion_B)) if ion_B else None
    exc_t_min = exc_write_t_min
    ion_t_min = ion_write_t_min
    if t_list_min is not None:
        if exc_t_min is not None:
            exc_t_min = max(exc_t_min, t_list_min)
        if ion_t_min is not None:
            ion_t_min = max(ion_t_min, t_list_min)
    grid_t_min = min(v for v in (exc_t_min, ion_t_min) if v is not None)
    grid_t_max = t_list_max if t_list_max is not None else DCS_T_MAX_EEV

    if dcs_data is None:
        if template_path is None:
            emfi_path, born_path = _default_dcs_template_paths(ice_label)
            if emfi_path is None:
                raise FileNotFoundError(
                    "Could not locate a DCS template. Expected a local file such as "
                    f"{CROSS_SECTIONS_DIR / f'sigmadiff_ionisation_e_{ice_label}_emfietzoglou_kyriakou.dat'} "
                    "or a Geant4 DNA data directory."
                )
            if not os.path.exists(emfi_path):
                raise FileNotFoundError(f"Missing DCS template file: {emfi_path}")
            print(f"Using DCS template grid: {emfi_path}")
            grid_low = _load_dcs_template_grid(
                emfi_path, t_min=grid_t_min, t_max=grid_t_max
            )

            grid = grid_low
            if born_path is not None and os.path.exists(born_path):
                t_switch = max(grid_low.keys()) if grid_low else grid_t_min
                grid_high = _load_dcs_template_grid(
                    born_path,
                    t_min=t_switch,
                    t_max=grid_t_max,
                    include_min=not bool(grid_low),
                )
                if grid_high:
                    grid = _merge_dcs_template_grids(grid_low, grid_high)
            if not grid:
                raise ValueError(
                    f"No template DCS incident-energy grid at or above {grid_t_min:.6g} eV."
                )
            if grid_t_max > max(grid.keys()):
                grid = _extend_dcs_grid(grid, grid_t_max, DCS_T_STEP_EEV)
        else:
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Missing DCS template file: {template_path}")
            grid = _load_dcs_template_grid(
                template_path, t_min=grid_t_min, t_max=grid_t_max
            )
            if not grid:
                raise ValueError(
                    f"No template DCS incident-energy grid at or above {grid_t_min:.6g} eV."
                )

        T_line = []
        E_line = []
        for T, E_list in grid.items():
            Tj = float(T)
            for Ei in E_list:
                T_line.append(Tj)
                E_line.append(float(Ei))

        T_line = np.asarray(T_line, float)
        E_line = np.asarray(E_line, float)

        kshell_B = (
            _kshell_threshold_eV(s)
            if include_kshell and s.kshell is not None and KSHELL_MODEL != "none"
            else None
        )

        n_lines = T_line.size
        n_exc = len(exc_B)
        n_ion = len(ion_B)

        exc_vals = np.zeros((n_lines, n_exc), float)
        ion_vals = np.zeros((n_lines, n_ion + (1 if kshell_B is not None else 0)), float)

        if apply_mc is None:
            apply_mc = APPLY_MOTT_COULOMB
        if apply_regime_ii is None:
            apply_regime_ii = APPLY_CORRECTIONS_REGIME_II
        if apply_regime_iii is None:
            apply_regime_iii = APPLY_CORRECTIONS_REGIME_III
        if apply_regime_iv is None:
            apply_regime_iv = APPLY_CORRECTIONS_REGIME_IV
        if ice_type is None:
            ice_type = ICE_TYPE

        tasks = [("excitation", k) for k in range(n_exc)]
        tasks.extend(("ionization", j) for j in range(n_ion))
        if kshell_B is not None:
            tasks.append(("kshell", 0))

        do_parallel = (
            parallel_channels
            and len(tasks) > 1
            and a_vec is not None
            and b_vec is not None
            and c_vec is not None
        )

        if do_parallel:
            if max_workers is None:
                max_workers = _max_workers_from_environment()
            max_workers = max(1, min(int(max_workers), len(tasks)))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_dcs_worker,
                initargs=(
                    a_vec,
                    b_vec,
                    c_vec,
                    T_line,
                    E_line,
                    Nq,
                    apply_mc,
                    apply_regime_ii,
                    apply_regime_iii,
                    apply_regime_iv,
                    ice_type,
                    PROJECTILE_KEY,
                    KSHELL_MODEL,
                    PROJECTILE_RELATIVISTIC_DCS,
                    INCLUDE_TRANSVERSE_DCS,
                    RPWBA_DENSITY_EFFECT,
                ),
            ) as ex:
                futures = [ex.submit(_compute_dcs_channel_worker, task) for task in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="DCS channels"):
                    channel_type, idx, vals = fut.result()
                    if channel_type == "excitation":
                        exc_vals[:, idx] = vals
                    elif channel_type == "ionization":
                        ion_vals[:, idx] = vals
                    elif channel_type == "kshell":
                        ion_vals[:, -1] = vals
        else:
            for k in tqdm(range(n_exc), desc="DCS excitation"):
                exc_vals[:, k] = _compute_dcs_channel_values(
                    "excitation", k, s, C, Nq, T_line, E_line, exc_B, ion_B, kshell_B
                )
            for j in tqdm(range(n_ion), desc="DCS ionization"):
                ion_vals[:, j] = _compute_dcs_channel_values(
                    "ionization", j, s, C, Nq, T_line, E_line, exc_B, ion_B, kshell_B
                )
            if kshell_B is not None:
                ion_vals[:, -1] = _compute_dcs_channel_values(
                    "kshell", 0, s, C, Nq, T_line, E_line, exc_B, ion_B, kshell_B
                )

        dcs_data = {
            "T_line": T_line,
            "E_line": E_line,
            "exc_vals": exc_vals,
            "ion_vals": ion_vals,
        }

    needs_charge_kernel = (str(charge_mode) != "bare") or bool(include_barkas_dcs)
    if needs_charge_kernel and not bool(dcs_data.get("barkas_charge_applied", False)):
        dcs_data, barkas_diag = barkas_dcs.apply_barkas_correction_to_dcs_data(
            dcs_data,
            s,
            material=ice_type if ice_type is not None else ICE_TYPE,
            projectile_mass_me=PROJECTILE_MASS_AU,
            nuclear_charge=PROJECTILE_CHARGE,
            charge_mode=charge_mode,
            include_barkas_dcs=include_barkas_dcs,
            explicit_charge=explicit_charge,
            include_kshell=include_kshell,
            dcs_scale_m2=EMFI_DCS_SCALE_M2,
            born_reference_charge=born_reference_charge,
            born_reference_q=born_reference_explicit_charge,
        )
        dcs_data["barkas_charge_applied"] = True
        dcs_data["barkas_charge_mode"] = str(charge_mode)
        if barkas_diag.negative_or_unstable_T_eV.size:
            bad = ", ".join(f"{v:.6g}" for v in barkas_diag.negative_or_unstable_T_eV[:10])
            raise RuntimeError(f"DCS_total is negative or unstable at projectile energies: {bad}")
        print(
            "Barkas/charge DCS mode: "
            f"charge_mode={charge_mode}, include_barkas_dcs={bool(include_barkas_dcs)}, "
            f"df/dW integral={barkas_diag.df_dW_integral:.6g}, "
            f"born_reference_charge={born_reference_charge}, "
            f"DCS_total min/max={barkas_diag.dcs_total_min_m2_per_eV:.6e}/"
            f"{barkas_diag.dcs_total_max_m2_per_eV:.6e} m^2/eV"
        )

    dcs_output_data = _prepare_dcs_data_for_output(
        dcs_data,
        exc_out,
        ion_out,
        t_min=grid_t_min,
        t_max=grid_t_max,
        merge_energy_patches=merge_energy_patches,
        table_metadata=table_metadata,
    )
    _write_dcs_tables_from_data(
        dcs_output_data,
        exc_out,
        ion_out,
        exc_t_min=exc_write_t_min,
        ion_t_min=ion_write_t_min,
        t_max=None,
        table_metadata=table_metadata,
    )
    _write_total_tables_from_dcs(
        dcs_output_data,
        exc_total_out,
        ion_total_out,
        exc_t_min=exc_write_t_min,
        ion_t_min=ion_write_t_min,
        t_max=None,
        table_metadata=table_metadata,
    )
    _export_to_custom_geant4([exc_out, ion_out, exc_total_out, ion_total_out])

    print(
        "DCS/table energy bounds: "
        f"excitation >= {exc_t_min:.6g} eV, "
        f"ionisation >= {ion_t_min:.6g} eV, "
        f"max <= {grid_t_max:.6g} eV."
    )
    print(f"Saved excitation DCS to {exc_out}")
    print(f"Saved ionization DCS to {ion_out}")
    print(f"Saved excitation total to {exc_total_out}")
    print(f"Saved ionization total to {ion_total_out}")
    if return_data:
        return dcs_data
    return None

def plot_total_cross_section_corrections(T_list, sigma_list, out_path=None, ice_label=None):
    """
    Plot PWBA vs corrected totals and per-stage correction terms vs energy.
    """
    if out_path is None:
        if ice_label is None:
            ice_label = ICE_LABEL
        CROSS_SECTION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CROSS_SECTION_PLOTS_DIR / f"cross_section_corrections_{ice_label}.png"

    rows = [_compute_correction_row(T, sigma) for T, sigma in zip(T_list, sigma_list)]
    T_arr = np.array([row["T_eV"] for row in rows], float)
    pwba = np.array([row["sigma_pwba"] for row in rows], float)
    corrected = np.array([row["sigma_corrected"] for row in rows], float)
    corr_mc = np.array([row["corr_stage1_mc"] for row in rows], float)
    corr_rel_long = np.array([row["corr_stage2_rel_long"] for row in rows], float)
    corr_rel_trans = np.array([row["corr_stage3_rel_trans"] for row in rows], float)
    corr_density = np.array([row["corr_stage4_density"] for row in rows], float)

    abs_vals = np.abs(
        np.concatenate([corr_mc, corr_rel_long, corr_rel_trans, corr_density])
    )
    abs_vals = abs_vals[abs_vals > 0.0]
    linthresh = float(np.min(abs_vals)) if abs_vals.size else 1e-40

    fig, (ax_tot, ax_corr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax_tot.loglog(T_arr, pwba, lw=2, ls=":", label="PWBA total")
    ax_tot.loglog(T_arr, corrected, lw=2, ls="-", label="Corrected total")
    ax_tot.set_ylabel("Total cross section sigma(T)")
    ax_tot.grid(True, which="both", ls="--", alpha=0.3)
    ax_tot.legend(loc="best", fontsize=9)

    ax_corr.set_xscale("log")
    ax_corr.set_yscale("symlog", linthresh=linthresh)
    ax_corr.plot(T_arr, corr_mc, lw=1.8, label="Stage 1: MC")
    ax_corr.plot(T_arr, corr_rel_long, lw=1.8, label="Stage 2: Rel long")
    ax_corr.plot(T_arr, corr_rel_trans, lw=1.8, label="Stage 3: Rel trans")
    ax_corr.plot(T_arr, corr_density, lw=1.8, label="Stage 4: Density effect")
    ax_corr.set_xlabel(_projectile_energy_label())
    ax_corr.set_ylabel("Correction term")
    ax_corr.grid(True, which="both", ls="--", alpha=0.3)
    ax_corr.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved correction diagnostics to {out_path}")

# ----------------------------------------------------------------------
# Workers for parallel computing
# ----------------------------------------------------------------------
# Globals set once per worker process
_WORKER_S = None
_WORKER_C = None
_WORKER_KW = None

def _init_worker(
    a_vec,
    b_vec,
    c_vec,
    NE,
    Nq,
    include_kshell,
    use_mott_coulomb,
    apply_regime_ii,
    apply_regime_iii,
    apply_regime_iv,
    apply_mc,
    ice_type,
    projectile_key,
    kshell_model,
    projectile_relativistic_dcs,
    include_transverse_dcs,
    rpwba_density_effect,
):
    """
    Runs once inside each worker process. Builds s and C once to avoid repeated pickling.
    """
    global _WORKER_S, _WORKER_C, _WORKER_KW

    import emfietzoglou_model_finite_q as model
    _set_regime_corrections(
        apply_regime_ii=apply_regime_ii,
        apply_regime_iii=apply_regime_iii,
        apply_regime_iv=apply_regime_iv,
    )
    _set_mc_correction(apply_mc=apply_mc)
    set_projectile(projectile_key)
    _set_kshell_model(kshell_model)
    _set_projectile_relativistic_dcs(
        projectile_relativistic_dcs,
        include_transverse_dcs,
        use_density_effect=rpwba_density_effect,
    )
    _WORKER_S = model.epsilon_optical(ice_type)
    _WORKER_C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)
    _WORKER_KW = dict(
        NE=NE,
        Nq=Nq,
        include_kshell=include_kshell,
        use_mott_coulomb=use_mott_coulomb,
    )

def _compute_for_T(T):
    """
    Compute sigma dict for one T.
    Returns (T, sigma_dict)
    """
    sigma = integrate_elf_double_integral(_WORKER_S, _WORKER_C, float(T), **_WORKER_KW)
    return float(T), sigma

def main():
    if ICE_TYPE not in ("amorphous", "hexagonal"):
        raise ValueError(f"Unsupported ICE_TYPE: {ICE_TYPE}")
    include_kshell = _include_kshell_from_argv(default=True)
    kshell_model = _kshell_model_from_argv(include_kshell=include_kshell)
    if not include_kshell:
        kshell_model = "none"
    if kshell_model == "none":
        include_kshell = False
    _set_kshell_model(kshell_model)
    input_energy_min_eV, input_energy_max_eV, energy_points, energy_grid = _energy_range_from_argv()
    energy_unit = _energy_unit_from_argv(default="total")
    energy_min_eV, energy_max_eV = _energy_range_to_total_eV(
        input_energy_min_eV,
        input_energy_max_eV,
        energy_unit,
    )
    NE, Nq = _integration_resolution_from_argv()
    merge_energy_patches = _merge_energy_patches_from_argv(default=True)
    charge_mode = _charge_mode_from_argv(default="bare")
    explicit_charge = _explicit_charge_from_argv(default=None)
    include_barkas_dcs = _include_barkas_dcs_from_argv(default=False)
    include_bloch_dcs = _include_bloch_dcs_from_argv(default=False)
    projectile_relativistic_dcs = _projectile_relativistic_dcs_from_argv(default=False)
    include_transverse_dcs = _include_transverse_dcs_from_argv(
        default=projectile_relativistic_dcs
    )
    rpwba_density_effect = _rpwba_density_effect_from_argv(default=True)
    _set_projectile_relativistic_dcs(
        projectile_relativistic_dcs,
        include_transverse_dcs,
        use_density_effect=rpwba_density_effect,
    )
    born_reference_charge = _born_reference_charge_from_argv(default="bare_Z")
    born_reference_explicit_charge = _born_reference_explicit_charge_from_argv(default=None)
    if charge_mode == "explicit" and explicit_charge is None:
        raise ValueError("--charge-mode explicit requires --explicit-charge.")
    if born_reference_charge == "explicit_q" and born_reference_explicit_charge is None:
        raise ValueError("--born-reference-charge explicit_q requires --born-reference-explicit-charge.")
    _set_charge_options(
        charge_mode=charge_mode,
        explicit_charge=explicit_charge,
        include_barkas_dcs=include_barkas_dcs,
        include_bloch_dcs=include_bloch_dcs,
        energy_unit=energy_unit,
        born_reference_charge=born_reference_charge,
        born_reference_explicit_charge=born_reference_explicit_charge,
    )
    run_label = f"{PROJECTILE_FILE_TOKEN}_{ICE_LABEL}"
    if kshell_model == "none":
        run_label = f"{run_label}_no_kshell"
    else:
        run_label = f"{run_label}_kshell_{kshell_model.replace('-', '_')}"
    if energy_unit == "per_u":
        run_label = f"{run_label}_per_u"
    run_label = (
        f"{run_label}"
        f"{_projectile_kernel_tag()}"
        f"{_charge_mode_tag(charge_mode, include_barkas_dcs, explicit_charge=explicit_charge, born_reference_charge=born_reference_charge, born_reference_explicit_charge=born_reference_explicit_charge)}"
        f"{_energy_range_tag(energy_min_eV, energy_max_eV, energy_points, energy_grid)}"
    )
    print(
        f"Projectile: {PROJECTILE_LABEL} "
        f"(key={PROJECTILE_KEY}, mass={PROJECTILE_MASS_AU:.6g} m_e, charge={PROJECTILE_CHARGE:.6g} e)"
    )
    print(f"Include K-shell: {include_kshell}")
    print(f"K-shell model: {KSHELL_MODEL}")
    print(
        "Incident-energy grid: "
        f"{energy_min_eV:.9g} to {energy_max_eV:.9g} eV, "
        f"N={energy_points}, grid={energy_grid}"
    )
    if energy_unit == "per_u":
        print(
            "Energy input convention: per_u; "
            f"input range {input_energy_min_eV:.9g} to {input_energy_max_eV:.9g} eV/u "
            f"converted using A={PROJECTILE_MASS_NUMBER:.6g}."
        )
    else:
        print("Energy input convention: total projectile kinetic energy.")
    print(
        "Charge/Barkas mode: "
        f"charge_mode={charge_mode}, explicit_charge={explicit_charge}, "
        f"include_barkas_dcs={include_barkas_dcs}, include_bloch_dcs={include_bloch_dcs}, "
        f"born_reference_charge={born_reference_charge}"
    )
    print(
        "Ion dielectric kernel: "
        f"relativistic={PROJECTILE_RELATIVISTIC_DCS}, "
        f"finite_q_transverse={INCLUDE_TRANSVERSE_DCS}, "
        f"density_effect={RPWBA_DENSITY_EFFECT}"
    )
    if PROJECTILE_RELATIVISTIC_DCS:
        print(
            "RPWBA reference: Dominguez-Munoz et al., Radiat. Phys. Chem. "
            f"199 (2022) 110363, doi:{RPWBA_REFERENCE_DOI}; finite-Q Eqs. "
            "(1)-(4) and dielectric density correction Eqs. (7)-(9)."
        )
        if PROJECTILE_KEY != "proton":
            print(
                "WARNING: Dominguez-Munoz et al. validated protons at "
                "100-300 MeV; this projectile is a bare-ion first-Born "
                "extrapolation at the calculated beta."
            )
    print(f"Integration resolution: dE={NE}, dq={Nq}")
    print(f"Merge energy patches into DAT tables: {merge_energy_patches}")
    if KSHELL_MODEL == "old-optical":
        print("WARNING: old-optical K-shell is q-independent and invalid for production proton/ion tables.")
    # Optical model / dispersion coefficients
    s = model.epsilon_optical(ICE_TYPE)
    if KSHELL_MODEL == "hydrogenic-gos":
        k_fsum = model.oxygen_K_hydrogenic_gos_fsum(
            B_K_eV=KSHELL_B_EV,
            Zeff=KSHELL_ZEFF,
            Ep_eV=float(s.Ep),
            normalize_fsum=True,
            fsum_target=KSHELL_FSUM_TARGET,
        )
        print(
            "Hydrogenic O K-shell: "
            f"B={KSHELL_B_EV:.6g} eV, Zeff={KSHELL_ZEFF:.6g}, "
            f"optical f-sum={k_fsum:.6g} (target {KSHELL_FSUM_TARGET:.6g}), "
            f"hydrogenic rolloff applied={HYDROGENIC_KSHELL_ROLLOFF_APPLIED}"
        )
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])

    # RR2017 defaults for c_disp, d_disp, b1, b2
    C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)

    # Incident projectile energy grid (eV)
    T_list = _energy_grid(
        energy_min_eV,
        energy_max_eV,
        energy_points,
        use_log=(energy_grid == "log"),
    )

    # Computing Choices
    # use_mott_coulomb = True
    # apply_mc = True
    # apply_regime_ii = True
    # apply_regime_iii = True
    # apply_regime_iv = True
    use_mott_coulomb = False     # Changed to False
    apply_mc = False             # Changed to False
    apply_regime_ii = False      # Changed to False
    apply_regime_iii = False     # Changed to False
    apply_regime_iv = False      # Changed to False

    _set_regime_corrections(
        apply_regime_ii=apply_regime_ii,
        apply_regime_iii=apply_regime_iii,
        apply_regime_iv=apply_regime_iv,
    )
    _set_mc_correction(apply_mc=apply_mc)

    normalization = _ion_normalization_metadata(ICE_TYPE, s)
    print(
        f"Ion normalization: {ION_NORMALIZATION_VERSION}; "
        f"optical sum={normalization['optical_electrons_per_H2O']:.6g} electrons/H2O, "
        f"ELF f-sum scale={normalization['optical_elf_fsum_scale']:.9g}."
    )
    print(
        f"Material density ({ICE_TYPE})={normalization['material_density_g_cm3']:.9g} g/cm^3; "
        "applied only to macroscopic rates in transport, not to per-molecule tables."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / f"cross_section_corrections_pwba_{run_label}.npz"
    cached = load_cross_section_corrections_npz(
        cache_path,
        NE=NE,
        Nq=Nq,
        T_list=T_list,
        include_kshell=include_kshell,
        kshell_model=KSHELL_MODEL,
        energy_unit=energy_unit,
        charge_mode=charge_mode,
        explicit_charge=explicit_charge,
        include_barkas_dcs=include_barkas_dcs,
        include_bloch_dcs=include_bloch_dcs,
        born_reference_charge=born_reference_charge,
        born_reference_explicit_charge=born_reference_explicit_charge,
    )

    sigma_list = None
    dcs_data = None
    dcs_written = False

    if cached is not None:
        T_list, sigma_list, dcs_data = cached
        print(f"Loaded cached cross sections from {cache_path}")
    else:
        print("Computing double-integrated cross sections (parallel over T)...")
        max_workers = max(1, min(_max_workers_from_environment(), len(T_list)))
        print("Using %i workers" % (max_workers))

        results_by_T = {}

        with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(
                    a_vec,
                    b_vec,
                    c_vec,
                    NE,
                    Nq,
                    include_kshell,
                    use_mott_coulomb,
                    apply_regime_ii,
                    apply_regime_iii,
                    apply_regime_iv,
                    apply_mc,
                    ICE_TYPE,
                    PROJECTILE_KEY,
                    KSHELL_MODEL,
                    PROJECTILE_RELATIVISTIC_DCS,
                    INCLUDE_TRANSVERSE_DCS,
                    RPWBA_DENSITY_EFFECT,
                ),
        ) as ex:
            futures = [ex.submit(_compute_for_T, T) for T in T_list]

            for fut in tqdm(as_completed(futures), total=len(futures)):
                T_val, sigma = fut.result()
                results_by_T[T_val] = sigma

        # Restore original T order
        sigma_list = [results_by_T[float(T)] for T in T_list]

        dcs_data = write_emfietzoglou_dcs_tables(
            s,
            C,
            Nq=Nq,
            return_data=True,
            ice_label=ICE_LABEL,
            ice_type=ICE_TYPE,
            a_vec=a_vec,
            b_vec=b_vec,
            c_vec=c_vec,
            apply_mc=apply_mc,
            apply_regime_ii=apply_regime_ii,
            apply_regime_iii=apply_regime_iii,
            apply_regime_iv=apply_regime_iv,
            T_list=T_list,
            include_kshell=include_kshell,
            merge_energy_patches=merge_energy_patches,
            charge_mode=charge_mode,
            include_barkas_dcs=include_barkas_dcs,
            explicit_charge=explicit_charge,
            born_reference_charge=born_reference_charge,
            born_reference_explicit_charge=born_reference_explicit_charge,
        )
        if charge_mode != "bare" or include_barkas_dcs:
            sigma_list = _replace_sigma_list_with_dcs_totals(T_list, sigma_list, dcs_data)
        dcs_written = True
        save_cross_section_corrections_npz(
            T_list,
            sigma_list,
            out_path=cache_path,
            NE=NE,
            Nq=Nq,
            dcs_data=dcs_data,
            ice_label=run_label,
            include_kshell=include_kshell,
            energy_unit=energy_unit,
            charge_mode=charge_mode,
            explicit_charge=explicit_charge,
            include_barkas_dcs=include_barkas_dcs,
            include_bloch_dcs=include_bloch_dcs,
            born_reference_charge=born_reference_charge,
            born_reference_explicit_charge=born_reference_explicit_charge,
        )

    if not dcs_written:
        if dcs_data is None:
            dcs_data = write_emfietzoglou_dcs_tables(
                s,
                C,
                Nq=Nq,
                return_data=True,
                ice_label=ICE_LABEL,
                ice_type=ICE_TYPE,
                a_vec=a_vec,
                b_vec=b_vec,
                c_vec=c_vec,
                apply_mc=apply_mc,
                apply_regime_ii=apply_regime_ii,
                apply_regime_iii=apply_regime_iii,
                apply_regime_iv=apply_regime_iv,
                T_list=T_list,
                include_kshell=include_kshell,
                merge_energy_patches=merge_energy_patches,
                charge_mode=charge_mode,
                include_barkas_dcs=include_barkas_dcs,
                explicit_charge=explicit_charge,
                born_reference_charge=born_reference_charge,
                born_reference_explicit_charge=born_reference_explicit_charge,
            )
            if charge_mode != "bare" or include_barkas_dcs:
                sigma_list = _replace_sigma_list_with_dcs_totals(T_list, sigma_list, dcs_data)
            dcs_written = True
            save_cross_section_corrections_npz(
                T_list,
                sigma_list,
                out_path=cache_path,
                NE=NE,
                Nq=Nq,
                dcs_data=dcs_data,
                ice_label=run_label,
                include_kshell=include_kshell,
                energy_unit=energy_unit,
                charge_mode=charge_mode,
                explicit_charge=explicit_charge,
                include_barkas_dcs=include_barkas_dcs,
                include_bloch_dcs=include_bloch_dcs,
                born_reference_charge=born_reference_charge,
                born_reference_explicit_charge=born_reference_explicit_charge,
            )
        else:
            write_emfietzoglou_dcs_tables(
                s,
                C,
                Nq=Nq,
                dcs_data=dcs_data,
                ice_label=ICE_LABEL,
                ice_type=ICE_TYPE,
                T_list=T_list,
                include_kshell=include_kshell,
                merge_energy_patches=merge_energy_patches,
                charge_mode=charge_mode,
                include_barkas_dcs=include_barkas_dcs,
                explicit_charge=explicit_charge,
                born_reference_charge=born_reference_charge,
                born_reference_explicit_charge=born_reference_explicit_charge,
            )
            if charge_mode != "bare" or include_barkas_dcs:
                sigma_list = _replace_sigma_list_with_dcs_totals(T_list, sigma_list, dcs_data)
            dcs_written = True

    if charge_mode == "bare" and not include_barkas_dcs:
        _run_projectile_pwba_sanity_checks(s, C, dcs_data=dcs_data, Nq=Nq, include_kshell=include_kshell)
    else:
        _run_projectile_pwba_sanity_checks(s, C, dcs_data=None, Nq=Nq, include_kshell=include_kshell)
        diag = dcs_data.get("barkas_diagnostics") if isinstance(dcs_data, dict) else None
        if diag is not None:
            print(
                "Barkas DCS diagnostics: "
                f"TCS_total rows={diag.TCS_total_m2.size}, "
                f"S_Barkas_check rows={diag.S_Barkas_check_eV_m2.size}, "
                f"unstable energies={diag.negative_or_unstable_T_eV.size}"
            )

    # ----------------- Log corrections per energy -----------------
    plot_total_cross_section_corrections(T_list, sigma_list, ice_label=run_label)

    # ----------------- Plot 0: corrected excitation/ionization (scaled) -----------------
    style = rcparams_with_fontsize(
        RC_BASE_ELASTIC,
        FONTSIZE_24,
        overrides={
            "font.family": FONT_COURIER,
            "mathtext.rm": FONT_COURIER,
            "mathtext.fontset": "custom",
        },
    )
    with plt.rc_context(style):
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        fig = plt.figure(figsize=(9, 7.8))
        gs = GridSpec(2, 1, height_ratios=[4.7, 1.3], hspace=0.30)
        ax = fig.add_subplot(gs[0, 0])
        plot_corrected_exc_ion_scaled(T_list, sigma_list, ax=ax)

        gs_leg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[1.0, 1.0], hspace=0.30)
        ax_leg_exc = fig.add_subplot(gs_leg[0, 0])
        ax_leg_ion = fig.add_subplot(gs_leg[1, 0])
        ax_leg_exc.axis("off")
        ax_leg_ion.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            n_exc = len(s.excitations)
            n_ion = len(s.ionizations)
            exc_handles = handles[:n_exc]
            ion_handles = handles[n_exc:n_exc + n_ion]
            if exc_handles:
                exc_labels = [str(i + 1) for i in range(n_exc)]
                ax_leg_exc.legend(
                    exc_handles,
                    exc_labels,
                    loc="center left",
                    bbox_to_anchor=(0.0, 0.5),
                    ncol=max(1, n_exc),
                    frameon=False,
                    columnspacing=0.9,
                    handlelength=2.2,
                    handletextpad=0.6,
                    borderaxespad=0.0,
                )
            if ion_handles:
                ion_labels = [str(i + 1) for i in range(n_ion)]
                ax_leg_ion.legend(
                    ion_handles,
                    ion_labels,
                    loc="center left",
                    bbox_to_anchor=(0.0, 0.5),
                    ncol=max(1, n_ion),
                    frameon=False,
                    columnspacing=0.9,
                    handlelength=2.2,
                    handletextpad=0.6,
                    borderaxespad=0.0,
                )

        CROSS_SECTION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CROSS_SECTION_PLOTS_DIR / f"corrected_excitation_ionization_scaled_{run_label}.png"
        fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.06)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved corrected excitation/ionization plot to {out_path}")

    # ----------------- Plot 1: comparison per channel (PWBA vs selected corrections) -----------------
    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    ax_ion, ax_exc = plot_full_cross_sections_per_channel(
        T_list, sigma_list, s
    )

    CROSS_SECTION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ion_compare_path = CROSS_SECTION_PLOTS_DIR / f"comparison_ionizations_pwba_vs_model_{run_label}.png"
    exc_compare_path = CROSS_SECTION_PLOTS_DIR / f"comparison_excitations_pwba_vs_model_{run_label}.png"
    ax_ion.figure.tight_layout()
    ax_ion.figure.savefig(ion_compare_path, dpi=300)
    plt.close(ax_ion.figure)

    ax_exc.figure.tight_layout()
    ax_exc.figure.savefig(exc_compare_path, dpi=300)
    plt.close(ax_exc.figure)
    print(
        f"Saved comparison plots to {ion_compare_path} "
        f"and {exc_compare_path}"
    )


    # ----------------- Plot 2: REL components per channel (L solid, T dashed) -----------------
    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    plot_relativistic_component_per_channel(T_list, sigma_list, s, ax=(ax_ion, ax_exc))
    fig_ion.tight_layout()
    fig_exc.tight_layout()
    rel_ion_path = CROSS_SECTION_PLOTS_DIR / f"rel_cross_sections_ionizations_LT_{run_label}.png"
    rel_exc_path = CROSS_SECTION_PLOTS_DIR / f"rel_cross_sections_excitations_LT_{run_label}.png"
    fig_ion.savefig(rel_ion_path, dpi=300)
    fig_exc.savefig(rel_exc_path, dpi=300)
    plt.close(fig_ion)
    plt.close(fig_exc)
    print(
        f"Saved REL component plots to {rel_ion_path} "
        f"and {rel_exc_path}"
    )

    # ----------------- Plot 3: TOTAL cross section with all corrections -----------------
    fig, ax = plt.subplots()
    plot_total_cross_section(T_list, sigma_list, s, ax=ax)
    fig.tight_layout()
    total_plot_path = CROSS_SECTION_PLOTS_DIR / f"total_cross_section_all_corrections_{run_label}.png"
    fig.savefig(total_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved total plot to {total_plot_path}")

    # ----------------- Plot 4: two-panel amorphous vs hexagonal totals -----------------
    amorphous_npz = OUTPUT_DIR / f"cross_section_corrections_pwba_{PROJECTILE_FILE_TOKEN}_amorphous_ice.npz"
    hexagonal_npz = OUTPUT_DIR / f"cross_section_corrections_pwba_{PROJECTILE_FILE_TOKEN}_hexagonal_ice.npz"
    if amorphous_npz.exists() and hexagonal_npz.exists():
        out_path = CROSS_SECTION_PLOTS_DIR / f"channel_cross_section_two_panel_{PROJECTILE_FILE_TOKEN}_amorphous_hexagonal.png"
        plot_channel_cross_sections_two_panel(amorphous_npz, hexagonal_npz, out_path=out_path)
    else:
        missing = []
        if not amorphous_npz.exists():
            missing.append(amorphous_npz.name)
        if not hexagonal_npz.exists():
            missing.append(hexagonal_npz.name)
        if missing:
            print(f"Skipping two-panel total plot (missing {', '.join(missing)}).")

if __name__ == "__main__":
    main()
