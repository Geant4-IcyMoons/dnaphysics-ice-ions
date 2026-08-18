"""Shared projectile definitions and numerical settings for fixed-charge CDFT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from ion_ice import (
    IonChargeState,
    ProjectileDefinition,
    get_projectile,
    load_projectile_definition as _load_projectile_definition,
    projectile_registry,
)


HARTREE_TO_EV = 27.211_386_245_981
OT_MINIMIZERS = ("CG", "DIIS")
OT_LINESEARCHES = ("2PNT", "3PNT", "GOLD", "ADAPT")
OT_ALGORITHMS = ("STRICT", "IRAC")
OT_PRECONDITIONERS = ("FULL_ALL", "FULL_KINETIC")
SCF_SOLVERS = ("OT", "DIAGONALIZATION")
CDFT_OPTIMIZERS = ("NEWTON_LS", "BISECT")
CDFT_CONSTRAINT_TYPES = ("HIRSHFELD", "BECKE")
MIXING_METHODS = (
    "DIRECT_P_MIXING",
    "PULAY_MIXING",
    "BROYDEN_MIXING",
)


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
    complex_ot_inner_scf_max: int = 24
    ot_minimizer: str = "CG"
    ot_linesearch: str = "3PNT"
    ot_algorithm: str = "STRICT"
    ot_preconditioner: str = "FULL_ALL"
    ot_energy_gap_hartree: float = 0.001
    counterpoise_ot_minimizer: str = "CG"
    counterpoise_ot_linesearch: str = "2PNT"
    counterpoise_ot_algorithm: str = "IRAC"
    water_counterpoise_scf_solver: str = "OT"
    projectile_counterpoise_scf_solver: str = "DIAGONALIZATION"
    complex_scf_solver: str = "DIAGONALIZATION"
    counterpoise_mixing_alpha: float = 0.5
    counterpoise_mixing_npulay: int = 5
    counterpoise_mixing_method: str = "PULAY_MIXING"
    complex_mixing_alpha: float = 0.5
    complex_mixing_npulay: int = 5
    complex_mixing_method: str = "PULAY_MIXING"
    cdft_eps: float = 1.0e-5
    cdft_max: int = 60
    cdft_optimizer: str = "BISECT"
    cdft_constraint_type: str = "HIRSHFELD"

    def __post_init__(self) -> None:
        if self.scf_max < 1:
            raise ValueError("SCF maximum iteration count must be positive.")
        if self.complex_ot_inner_scf_max < 1:
            raise ValueError(
                "Complex OT inner-SCF maximum iteration count must be positive."
            )
        if self.ot_minimizer not in OT_MINIMIZERS:
            raise ValueError(f"Unsupported CP2K OT minimizer: {self.ot_minimizer}")
        if self.ot_linesearch not in OT_LINESEARCHES:
            raise ValueError(f"Unsupported CP2K OT line search: {self.ot_linesearch}")
        if self.ot_algorithm not in OT_ALGORITHMS:
            raise ValueError(f"Unsupported CP2K OT algorithm: {self.ot_algorithm}")
        if self.ot_preconditioner not in OT_PRECONDITIONERS:
            raise ValueError(
                f"Unsupported complex OT preconditioner: {self.ot_preconditioner}"
            )
        if (
            not math.isfinite(self.ot_energy_gap_hartree)
            or self.ot_energy_gap_hartree <= 0.0
        ):
            raise ValueError("The complex OT ENERGY_GAP must be finite and positive.")
        if self.counterpoise_ot_minimizer not in OT_MINIMIZERS:
            raise ValueError(
                "Unsupported counterpoise OT minimizer: "
                f"{self.counterpoise_ot_minimizer}"
            )
        if self.counterpoise_ot_linesearch not in OT_LINESEARCHES:
            raise ValueError(
                "Unsupported counterpoise OT line search: "
                f"{self.counterpoise_ot_linesearch}"
            )
        if self.counterpoise_ot_algorithm not in OT_ALGORITHMS:
            raise ValueError(
                "Unsupported counterpoise OT algorithm: "
                f"{self.counterpoise_ot_algorithm}"
            )
        for role, solver in (
            ("water", self.water_counterpoise_scf_solver),
            ("projectile", self.projectile_counterpoise_scf_solver),
            ("complex", self.complex_scf_solver),
        ):
            if solver not in SCF_SOLVERS:
                raise ValueError(
                    f"Unsupported {role} counterpoise SCF solver: {solver}"
                )
        if not math.isfinite(self.counterpoise_mixing_alpha) or not (
            0.0 < self.counterpoise_mixing_alpha <= 1.0
        ):
            raise ValueError("Counterpoise mixing alpha must lie in (0, 1].")
        if self.counterpoise_mixing_npulay < 1:
            raise ValueError("Counterpoise NPULAY must be positive.")
        if self.counterpoise_mixing_method not in MIXING_METHODS:
            raise ValueError(
                "Unsupported counterpoise mixing method: "
                f"{self.counterpoise_mixing_method}"
            )
        if not math.isfinite(self.complex_mixing_alpha) or not (
            0.0 < self.complex_mixing_alpha <= 1.0
        ):
            raise ValueError("Complex mixing alpha must lie in (0, 1].")
        if self.complex_mixing_npulay < 1:
            raise ValueError("Complex NPULAY must be positive.")
        if self.complex_mixing_method not in MIXING_METHODS:
            raise ValueError(
                f"Unsupported complex mixing method: {self.complex_mixing_method}"
            )
        if self.cdft_optimizer not in CDFT_OPTIMIZERS:
            raise ValueError(
                f"Unsupported CDFT optimizer: {self.cdft_optimizer}"
            )
        if self.cdft_constraint_type not in CDFT_CONSTRAINT_TYPES:
            raise ValueError(
                "Unsupported CDFT constraint type: "
                f"{self.cdft_constraint_type}"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_CP2K_SETTINGS = CP2KSettings()
DEFAULT_PROJECTILE = load_builtin_projectile("C")


CP2K_PROVENANCE = {
    "software": "CP2K",
    "validated_series": "2025.2",
    "validated_source_revision": "c3a8adfec5",
    "documentation": "https://manual.cp2k.org/cp2k-2025_2-branch/",
    "cdft_tutorial": (
        "https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/constrained.html"
    ),
    "gapw_documentation": "https://manual.cp2k.org/trunk/methods/dft/gapw.html",
    "basis_documentation": (
        "https://manual.cp2k.org/trunk/methods/dft/basis_sets.html"
    ),
    "constraint_definition": (
        "Density-Hirshfeld charge-density constraint; fixed nuclei; "
        "all-electron GAPW. Becke is retained only as an explicit "
        "population-definition sensitivity calculation."
    ),
    "cdft_implementation": (
        "Ahart, Rosso, and Blumberger, Journal of Chemical Theory and "
        "Computation 18, 4438-4446 (2022)."
    ),
    "cdft_implementation_doi": "10.1021/acs.jctc.2c00284",
    "bisect_source": (
        "https://github.com/cp2k/cp2k/blob/v2025.2/src/qs_outer_scf.F"
    ),
    "atomic_guess_documentation": (
        "https://manual.cp2k.org/cp2k-2025_2-branch/CP2K_INPUT/"
        "FORCE_EVAL/SUBSYS/KIND/BS.html"
    ),
}
