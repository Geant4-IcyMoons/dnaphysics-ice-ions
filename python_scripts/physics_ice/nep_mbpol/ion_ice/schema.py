"""Strict schemas for the shared ion--ice model registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


SPECIES_SCHEMA_VERSION = 1
PHASE_SCHEMA_VERSION = 1

ELEMENT_SYMBOLS = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_ELEMENT_SYMBOL = re.compile(r"[A-Z][a-z]?")
_COMPONENT_STATUSES = frozenset(("implemented", "missing", "blocked"))


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _positive_float(value: object, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return converted


def _positive_sequence(values: object, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    converted = tuple(_positive_float(value, name) for value in values)
    if len(set(converted)) != len(converted):
        raise ValueError(f"{name} cannot contain duplicates.")
    return converted


def _provenance(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not str(value.get("source", "")).strip():
        raise ValueError(f"{name} requires a source.")
    if not any(
        str(value.get(key, "")).strip()
        for key in ("doi", "data_doi", "url")
    ):
        raise ValueError(f"{name} requires a DOI, data DOI, or URL.")
    return dict(value)


@dataclass(frozen=True)
class IonChargeState:
    charge: int
    electrons_on_projectile: int
    multiplicity: int
    configuration: str
    term: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class IsotopeDefinition:
    mass_number: int
    neutral_atomic_mass_u: float
    provenance: Mapping[str, Any]

@dataclass(frozen=True)
class ProjectileDefinition:
    """One atomic projectile and the declared availability of each component."""

    symbol: str
    name: str
    atomic_number: int
    aliases: tuple[str, ...]
    default_isotope: IsotopeDefinition
    components: Mapping[str, Mapping[str, Any]]
    source_path: Path

    def validate(self) -> None:
        if not _ELEMENT_SYMBOL.fullmatch(self.symbol):
            raise ValueError(f"Invalid element symbol {self.symbol!r}.")
        if not 1 <= self.atomic_number <= 118:
            raise ValueError("Atomic number must lie in 1..118.")
        if ELEMENT_SYMBOLS[self.atomic_number] != self.symbol:
            raise ValueError(
                f"{self.symbol} does not match atomic number {self.atomic_number}."
            )
        if not self.name.strip():
            raise ValueError("Projectile name cannot be empty.")
        normalized_aliases = tuple(alias.strip().lower() for alias in self.aliases)
        if any(not alias for alias in normalized_aliases) or len(
            set(normalized_aliases)
        ) != len(normalized_aliases):
            raise ValueError("Projectile aliases must be non-empty and unique.")
        if self.symbol.lower() not in normalized_aliases:
            raise ValueError("Projectile aliases must include its lower-case symbol.")
        if self.default_isotope.mass_number < 1:
            raise ValueError("Default isotope mass number must be positive.")
        _positive_float(
            self.default_isotope.neutral_atomic_mass_u,
            "Default-isotope neutral atomic mass",
        )
        _provenance(self.default_isotope.provenance, "Isotope provenance")

        required = {
            "nlh_pair_potential",
            "hard_transport",
            "soft_dft",
            "geant4_hard_runtime",
            "geant4_soft_runtime",
        }
        if set(self.components) != required:
            raise ValueError(
                "Components must be exactly: " + ", ".join(sorted(required))
            )
        for name, component in self.components.items():
            status = component.get("status")
            if status not in _COMPONENT_STATUSES:
                raise ValueError(f"Invalid {name} component status {status!r}.")
            if status != "implemented" and not str(
                component.get("reason", "")
            ).strip():
                raise ValueError(f"Non-implemented {name} requires a reason.")

        nlh = self.components["nlh_pair_potential"]
        if nlh["status"] == "implemented":
            targets = nlh.get("targets")
            if targets != ["H", "O"]:
                raise ValueError("Implemented ice NLH requires H and O targets.")
            if not str(nlh.get("coefficient_source", "")).strip():
                raise ValueError("Implemented NLH requires a coefficient source.")
            _provenance(nlh.get("provenance"), "NLH provenance")

        soft = self.components["soft_dft"]
        if soft["status"] == "implemented":
            self._validate_soft_dft(soft)

    def _validate_soft_dft(self, soft: Mapping[str, Any]) -> None:
        if soft.get("complete_charge_ladder") is not True:
            raise ValueError("Soft DFT requires a complete q=0..Z charge ladder.")
        states = self.states
        charges = [state.charge for state in states]
        if charges != list(range(self.atomic_number + 1)):
            raise ValueError(
                f"{self.symbol} soft DFT requires the complete ordered charge "
                f"ladder q=0..{self.atomic_number}; got {charges}."
            )
        for state in states:
            expected = self.atomic_number - state.charge
            if state.electrons_on_projectile != expected:
                raise ValueError(
                    f"{self.symbol}{state.charge:+d} must contain Z-q={expected} "
                    "projectile electrons."
                )
            if state.multiplicity < 1:
                raise ValueError("Every soft-DFT state needs positive multiplicity.")
            if not state.configuration.strip() or not state.term.strip():
                raise ValueError("Every soft-DFT state needs configuration and term.")
        _provenance(soft.get("state_provenance"), "Electronic-state provenance")
        cp2k = soft.get("cp2k", {})
        basis = cp2k.get("projectile_basis_set")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("Soft DFT requires a projectile all-electron basis.")
        basis_file = cp2k.get("basis_file")
        if not isinstance(basis_file, str) or not basis_file.strip():
            raise ValueError("Soft DFT requires the source CP2K basis filename.")
        _provenance(
            cp2k.get("basis_provenance"), "All-electron basis provenance"
        )
        scan = soft.get("scan_grid")
        if not isinstance(scan, dict):
            raise ValueError("Soft DFT requires a scan_grid object.")
        if scan.get("kind") == "nlh_overlap":
            if self.components["nlh_pair_potential"]["status"] != "implemented":
                raise ValueError("An NLH overlap grid requires implemented NLH pairs.")
            _positive_sequence(scan.get("thresholds_ev"), "NLH thresholds")
            _positive_sequence(scan.get("soft_probe_angstrom"), "Soft probes")
        elif scan.get("kind") == "explicit":
            distances = scan.get("distances_angstrom")
            if not isinstance(distances, dict) or set(distances) != {"H", "O"}:
                raise ValueError("Explicit scans require H and O distance lists.")
            _positive_sequence(distances["H"], "H scan distances")
            _positive_sequence(distances["O"], "O scan distances")
        else:
            raise ValueError("Soft scan kind must be nlh_overlap or explicit.")
        _provenance(scan.get("provenance"), "Soft scan-grid provenance")

    @property
    def states(self) -> tuple[IonChargeState, ...]:
        soft = self.components["soft_dft"]
        if soft["status"] != "implemented":
            raise ValueError(f"No soft-DFT definition is available for {self.symbol}.")
        return tuple(
            IonChargeState(
                charge=_integer(item["charge"], "Charge"),
                electrons_on_projectile=_integer(
                    item["electrons_on_projectile"], "Projectile electron count"
                ),
                multiplicity=_integer(item["multiplicity"], "Multiplicity"),
                configuration=str(item["configuration"]),
                term=str(item["term"]),
            )
            for item in soft["states"]
        )

    def state(self, charge: int) -> IonChargeState:
        if not 0 <= charge <= self.atomic_number:
            raise ValueError(
                f"Unsupported {self.symbol} q={charge}; expected 0.."
                f"{self.atomic_number}."
            )
        return self.states[charge]

    @property
    def state_provenance(self) -> Mapping[str, Any]:
        return self.components["soft_dft"]["state_provenance"]

    @property
    def scan_grid(self) -> Mapping[str, Any]:
        return self.components["soft_dft"]["scan_grid"]

    @property
    def projectile_basis_set(self) -> str:
        return str(
            self.components["soft_dft"]["cp2k"]["projectile_basis_set"]
        )

    @property
    def basis_file(self) -> str:
        return str(self.components["soft_dft"]["cp2k"]["basis_file"])

    def component_status(self, component: str) -> str:
        try:
            return str(self.components[component]["status"])
        except KeyError as exc:
            raise ValueError(f"Unknown model component {component!r}.") from exc

    def soft_dft_dict(self) -> dict[str, Any]:
        self.validate()
        soft = dict(self.components["soft_dft"])
        if soft.pop("status") != "implemented":
            raise ValueError(f"No soft-DFT definition is available for {self.symbol}.")
        return {
            "schema_version": SPECIES_SCHEMA_VERSION,
            "symbol": self.symbol,
            "name": self.name,
            "atomic_number": self.atomic_number,
            **soft,
        }


@dataclass(frozen=True)
class PhaseDefinition:
    phase_id: str
    name: str
    target_temperature_k: float
    structure_registry: str | None
    preparation: Mapping[str, Any]
    validation: Mapping[str, Any]
    density: Mapping[str, Any]
    source_path: Path

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", self.phase_id):
            raise ValueError(f"Invalid phase ID {self.phase_id!r}.")
        if not self.name.strip():
            raise ValueError("Phase name cannot be empty.")
        _positive_float(self.target_temperature_k, "Phase temperature")
        if not str(self.preparation.get("pbs_script", "")).strip():
            raise ValueError("Phase preparation requires a PBS script.")
        if not str(self.preparation.get("pbs_job_name", "")).strip():
            raise ValueError("Phase preparation requires an exact PBS job name.")
        qsub_arguments = self.preparation.get("qsub_arguments", [])
        if not isinstance(qsub_arguments, list) or any(
            not isinstance(value, str) for value in qsub_arguments
        ):
            raise ValueError("Phase qsub_arguments must be a list of strings.")
        for index, value in enumerate(qsub_arguments):
            resource_value = (
                qsub_arguments[index + 1]
                if value == "-l" and index + 1 < len(qsub_arguments)
                else value
            )
            if any(
                token in resource_value.lower()
                for token in ("select=", "ncpus=", "mem=", "ngpus=")
            ):
                raise ValueError(
                    "Phase qsub_arguments cannot override bounded CPU, memory, "
                    "or GPU profiles."
                )
        if not str(self.validation.get("protocol", "")).strip():
            raise ValueError("Phase validation requires a protocol.")
        if self.density.get("value_g_cm3") is not None:
            _positive_float(self.density["value_g_cm3"], "Phase density")
            _provenance(self.density.get("provenance"), "Density provenance")

    def resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.source_path.parent / path
        return path.resolve()

def load_projectile_definition(path: str | Path) -> ProjectileDefinition:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SPECIES_SCHEMA_VERSION:
        raise ValueError(f"Unsupported species schema in {source}.")
    isotope = payload["default_isotope"]
    definition = ProjectileDefinition(
        symbol=str(payload["symbol"]),
        name=str(payload["name"]),
        atomic_number=_integer(payload["atomic_number"], "Atomic number"),
        aliases=tuple(str(value) for value in payload["aliases"]),
        default_isotope=IsotopeDefinition(
            mass_number=_integer(isotope["mass_number"], "Mass number"),
            neutral_atomic_mass_u=_positive_float(
                isotope["neutral_atomic_mass_u"], "Neutral atomic mass"
            ),
            provenance=_provenance(isotope["provenance"], "Isotope provenance"),
        ),
        components={
            str(name): dict(component)
            for name, component in payload["components"].items()
        },
        source_path=source,
    )
    definition.validate()
    return definition


def load_phase_definition(path: str | Path) -> PhaseDefinition:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PHASE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported phase schema in {source}.")
    definition = PhaseDefinition(
        phase_id=str(payload["phase_id"]),
        name=str(payload["name"]),
        target_temperature_k=_positive_float(
            payload["target_temperature_k"], "Phase temperature"
        ),
        structure_registry=(
            None
            if payload.get("structure_registry") is None
            else str(payload["structure_registry"])
        ),
        preparation=dict(payload["preparation"]),
        validation=dict(payload["validation"]),
        density=dict(payload["density"]),
        source_path=source,
    )
    definition.validate()
    return definition
