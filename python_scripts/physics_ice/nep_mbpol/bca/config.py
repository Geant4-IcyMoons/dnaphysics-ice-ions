"""Shared numerical and isotope constants for the NLH collision pipeline."""

from __future__ import annotations


ATOMIC_MASS_UNIT_C2_EV = 931.494_103_72e6
AVOGADRO_MOL_MINUS_ONE = 6.022_140_76e23
WATER_MOLAR_MASS_G_MOL = 18.015_28

# Dominant isotopes used for the projectile and target kinematics.  Electron
# binding changes these atomic masses by far less than the uncertainty of the
# present classical pair-potential model.
ISOTOPE_MASS_U = {
    "H": 1.007_825_032_23,
    "He": 4.002_603_254_13,
    "C": 12.0,
    "O": 15.994_914_619_57,
    "S": 31.972_071_174_4,
}

ELEMENT_ALIASES = {
    "h": "H",
    "hydrogen": "H",
    "proton": "H",
    "he": "He",
    "helium": "He",
    "alpha": "He",
    "c": "C",
    "carbon": "C",
    "o": "O",
    "oxygen": "O",
    "s": "S",
    "sulfur": "S",
    "sulphur": "S",
}

DEFAULT_PROJECTILES = ("C", "O", "S")
ICE_TARGETS = ("H", "O")

# These are numerical defaults, not additional physical fit parameters.  The
# convergence report produced with a publication table must justify them.
DEFAULT_ENERGY_MIN_EV = 1.0e3
DEFAULT_ENERGY_MAX_EV = 1.0e8
DEFAULT_ENERGY_POINTS = 121
DEFAULT_IMPACT_POINTS = 129
DEFAULT_QUADRATURE_ORDER = 96
DEFAULT_WORKERS = 10

# NLH reports its strongest accuracy above 30 eV.  Evaluation down to 10 eV is
# supported, but must be requested explicitly by changing this threshold.
PUBLISHED_MINIMUM_TURNING_POTENTIAL_EV = 10.0
DEFAULT_MINIMUM_TURNING_POTENTIAL_EV = 30.0


def canonical_element(value: str) -> str:
    """Return the canonical symbol for one supported element name."""

    try:
        return ELEMENT_ALIASES[value.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unsupported element {value!r}.") from exc
