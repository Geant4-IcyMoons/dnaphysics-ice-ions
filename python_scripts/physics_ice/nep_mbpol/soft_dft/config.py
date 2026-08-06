"""Shared projectile definitions and numerical settings for fixed-charge CDFT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ion_ice import (
    IonChargeState,
    ProjectileDefinition,
    get_projectile,
    load_projectile_definition as _load_projectile_definition,
    projectile_registry,
)


HARTREE_TO_EV = 27.211_386_245_981


def load_projectile_definition(path: Path) -> ProjectileDefinition:
    """Load a unified species definition that contains a complete DFT layer."""

    definition = _load_projectile_definition(path)
    if definition.component_status("soft_dft") != "implemented":
        raise ValueError(
            f"The {definition.symbol} registry entry has no soft-DFT definition."
        )
    return definition


def load_builtin_projectile(symbol: str = "C") -> ProjectileDefinition:
    """Load one reviewed registry species with a complete soft-DFT layer."""

    definition = get_projectile(symbol)
    if definition.component_status("soft_dft") != "implemented":
        available = ", ".join(
            definition.symbol
            for definition in projectile_registry().values()
            if definition.component_status("soft_dft") == "implemented"
        )
        raise ValueError(
            f"No reviewed soft-DFT definition for {definition.symbol}. "
            f"Available: {available or 'none'}. Supply a complete external "
            "ion--ice species definition."
        )
    return definition


@dataclass(frozen=True)
class CP2KSettings:
    """Explicit numerical settings; every item remains subject to convergence."""

    cp2k_series: str = "2025.2"
    method: str = "GAPW"
    basis_file: str = "BASIS_MOLOPT_UZH"
    projectile_basis_set: str = "TZVPP-MOLOPT-GGA-ae"
    water_basis_set: str = "TZVPP-MOLOPT-GGA-ae"
    potential: str = "ALL"
    xc_functional: str = "PBE"
    cell_angstrom: float = 30.0
    mgrid_cutoff_ry: float = 800.0
    mgrid_rel_cutoff_ry: float = 80.0
    scf_eps: float = 1.0e-7
    scf_max: int = 100
    cdft_eps: float = 1.0e-5
    cdft_max: int = 30

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


DEFAULT_CP2K_SETTINGS = CP2KSettings()
DEFAULT_PROJECTILE = load_builtin_projectile("C")


CP2K_PROVENANCE = {
    "software": "CP2K",
    "documentation": "https://manual.cp2k.org/cp2k-2025_2-branch/",
    "cdft_tutorial": (
        "https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/constrained.html"
    ),
    "gapw_documentation": "https://manual.cp2k.org/trunk/methods/dft/gapw.html",
    "basis_documentation": (
        "https://manual.cp2k.org/trunk/methods/dft/basis_sets.html"
    ),
    "constraint_definition": (
        "Becke charge-density constraint without empirical atomic-radius "
        "adjustment; fixed nuclei; all-electron GAPW."
    ),
}
