"""Validated ingestion of periodic GPUMD/extended-XYZ ice snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import shlex
from typing import Iterator, TextIO

import numpy as np
from numpy.typing import NDArray

from .config import AVOGADRO_MOL_MINUS_ONE, WATER_MOLAR_MASS_G_MOL


class StructureValidationError(ValueError):
    """Raised when a structure is not suitable for collision calculations."""


@dataclass(frozen=True)
class XYZFrame:
    """One parsed frame from a GPUMD or extended-XYZ trajectory."""

    species: NDArray[np.str_]
    positions_angstrom: NDArray[np.float64]
    lattice_angstrom: NDArray[np.float64]
    pbc: tuple[bool, bool, bool]
    frame_index: int


@dataclass(frozen=True)
class IceStructure:
    """One periodic H2O snapshot and its traceable source metadata."""

    species: NDArray[np.str_]
    positions_angstrom: NDArray[np.float64]
    lattice_angstrom: NDArray[np.float64]
    pbc: tuple[bool, bool, bool]
    source_path: Path
    source_sha256: str
    frame_index: int
    metadata: dict[str, object]
    collision_ready: bool

    @property
    def atom_count(self) -> int:
        return int(self.species.size)

    @property
    def water_molecule_count(self) -> int:
        return int(np.count_nonzero(self.species == "O"))

    @property
    def volume_angstrom3(self) -> float:
        return abs(float(np.linalg.det(self.lattice_angstrom)))

    @property
    def density_g_cm3(self) -> float:
        mass_g = (
            self.water_molecule_count
            * WATER_MOLAR_MASS_G_MOL
            / AVOGADRO_MOL_MINUS_ONE
        )
        return mass_g / (self.volume_angstrom3 * 1.0e-24)

    @property
    def phase(self) -> str:
        return str(self.metadata.get("phase", "unspecified"))

    @property
    def use_class(self) -> str:
        return "production-input" if self.collision_ready else "diagnostic-only"

    def manifest_record(self) -> dict[str, object]:
        return {
            "path": str(self.source_path),
            "sha256": self.source_sha256,
            "frame_index": self.frame_index,
            "phase": self.phase,
            "state": self.metadata.get("state", "unspecified"),
            "collision_ready": self.collision_ready,
            "use_class": self.use_class,
            "atoms": self.atom_count,
            "water_molecules": self.water_molecule_count,
            "density_g_cm3": self.density_g_cm3,
            "volume_angstrom3": self.volume_angstrom3,
            "lattice_angstrom": self.lattice_angstrom.tolist(),
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header_attributes(header: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for token in shlex.split(header, posix=True):
        if "=" in token:
            name, value = token.split("=", 1)
            attributes[name.lower()] = value
    return attributes


def _property_slices(specification: str) -> dict[str, slice]:
    fields = specification.split(":")
    if len(fields) % 3:
        raise StructureValidationError(
            f"Malformed extended-XYZ Properties field {specification!r}."
        )
    result: dict[str, slice] = {}
    offset = 0
    for index in range(0, len(fields), 3):
        name, _, count_text = fields[index : index + 3]
        try:
            count = int(count_text)
        except ValueError as exc:
            raise StructureValidationError(
                f"Invalid property width {count_text!r}."
            ) from exc
        if count <= 0:
            raise StructureValidationError("Property widths must be positive.")
        result[name.lower()] = slice(offset, offset + count)
        offset += count
    return result


def _next_nonempty_line(handle: TextIO) -> str | None:
    for line in handle:
        if line.strip():
            return line
    return None


def _read_frame(
    handle: TextIO, source: Path, frame_index: int
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float64], tuple[bool, bool, bool]] | None:
    count_line = _next_nonempty_line(handle)
    if count_line is None:
        return None
    try:
        atom_count = int(count_line.strip())
    except ValueError as exc:
        raise StructureValidationError(
            f"{source}: frame {frame_index} does not begin with an atom count."
        ) from exc
    if atom_count <= 0:
        raise StructureValidationError("Atom count must be positive.")

    header = handle.readline()
    if not header:
        raise StructureValidationError(f"{source}: frame {frame_index} has no header.")
    attributes = _header_attributes(header)
    try:
        lattice_values = [float(value) for value in attributes["lattice"].split()]
        properties = _property_slices(attributes["properties"])
    except KeyError as exc:
        raise StructureValidationError(
            f"{source}: frame {frame_index} requires Lattice and Properties."
        ) from exc
    if len(lattice_values) != 9:
        raise StructureValidationError("Lattice must contain nine numbers.")
    lattice = np.asarray(lattice_values, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(lattice)) or abs(np.linalg.det(lattice)) <= 0.0:
        raise StructureValidationError("The periodic lattice is singular or non-finite.")

    try:
        species_slice = properties["species"]
        position_slice = properties["pos"]
    except KeyError as exc:
        raise StructureValidationError(
            "Properties must contain species:S:1 and pos:R:3."
        ) from exc
    if species_slice.stop - species_slice.start != 1:
        raise StructureValidationError("The species property must have width one.")
    if position_slice.stop - position_slice.start != 3:
        raise StructureValidationError("The pos property must have width three.")
    required_fields = max(item.stop for item in properties.values())

    species = np.empty(atom_count, dtype="U2")
    positions = np.empty((atom_count, 3), dtype=np.float64)
    for atom_index in range(atom_count):
        atom_line = handle.readline()
        if not atom_line:
            raise StructureValidationError(
                f"{source}: frame {frame_index} ended at atom {atom_index}."
            )
        values = atom_line.split()
        if len(values) < required_fields:
            raise StructureValidationError(
                f"{source}: incomplete atom record at frame {frame_index}, "
                f"atom {atom_index}."
            )
        species[atom_index] = values[species_slice.start]
        try:
            positions[atom_index] = [float(value) for value in values[position_slice]]
        except ValueError as exc:
            raise StructureValidationError("Non-numeric atomic position.") from exc

    pbc_fields = attributes.get("pbc", "F F F").split()
    if len(pbc_fields) != 3:
        raise StructureValidationError("pbc must contain three flags.")
    pbc = tuple(value.lower() in {"t", "true", "1"} for value in pbc_fields)
    return species, positions, lattice, pbc  # type: ignore[return-value]


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_xyz_frames(path: str | Path) -> Iterator[XYZFrame]:
    """Yield every frame from a plain or gzip-compressed XYZ trajectory."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with _open_text(source) as handle:
        frame_index = 0
        while True:
            parsed = _read_frame(handle, source, frame_index)
            if parsed is None:
                break
            species, positions, lattice, pbc = parsed
            yield XYZFrame(
                species=species,
                positions_angstrom=positions,
                lattice_angstrom=lattice,
                pbc=pbc,
                frame_index=frame_index,
            )
            frame_index += 1


def _load_metadata(path: Path, metadata_path: Path | None) -> dict[str, object]:
    candidate = metadata_path or path.with_suffix(".json")
    if not candidate.is_file():
        return {}
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructureValidationError(f"Could not parse {candidate}.") from exc
    if not isinstance(value, dict):
        raise StructureValidationError("Structure metadata must be a JSON object.")
    return value


def load_ice_structure(
    path: str | Path,
    *,
    frame_index: int = -1,
    metadata_path: str | Path | None = None,
    allow_unvalidated: bool = False,
) -> IceStructure:
    """Load one frame and require an explicit collision-ready attestation.

    A production snapshot needs a sidecar JSON file whose ``sha256`` matches
    the extended-XYZ file and whose ``collision_ready`` field is true.  This
    makes structural acceptance an auditable step rather than an inference
    from a filename.  ``allow_unvalidated`` is only for diagnostic runs.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if frame_index < -1:
        raise ValueError("frame_index must be -1 (last) or a non-negative integer.")

    selected: XYZFrame | None = None
    for frame in iter_xyz_frames(source):
        if frame_index == -1 or frame.frame_index == frame_index:
            selected = frame
        if frame.frame_index == frame_index:
            break
    if selected is None:
        raise StructureValidationError(
            f"{source}: requested frame {frame_index} was not found."
        )

    species = selected.species
    positions = selected.positions_angstrom
    lattice = selected.lattice_angstrom
    pbc = selected.pbc
    selected_index = selected.frame_index
    if set(species.tolist()) != {"H", "O"}:
        raise StructureValidationError("An ice target must contain only H and O atoms.")
    oxygen_count = int(np.count_nonzero(species == "O"))
    hydrogen_count = int(np.count_nonzero(species == "H"))
    if hydrogen_count != 2 * oxygen_count:
        raise StructureValidationError(
            f"Invalid H2O stoichiometry: H={hydrogen_count}, O={oxygen_count}."
        )
    if not all(pbc):
        raise StructureValidationError("Collision structures must be periodic in 3D.")
    if not np.all(np.isfinite(positions)):
        raise StructureValidationError("Atomic positions contain non-finite values.")

    # Normalize coordinates to the primary triclinic cell without changing the
    # physical configuration.
    inverse_lattice = np.linalg.inv(lattice)
    fractional = positions @ inverse_lattice
    fractional -= np.floor(fractional)
    positions = fractional @ lattice

    digest = file_sha256(source)
    metadata_file = (
        Path(metadata_path).expanduser().resolve() if metadata_path else None
    )
    metadata = _load_metadata(source, metadata_file)
    resolved_metadata_file = metadata_file or source.with_suffix(".json")
    recorded_digest = metadata.get("sha256")
    digest_matches = isinstance(recorded_digest, str) and recorded_digest == digest
    phase_is_recorded = isinstance(metadata.get("phase"), str) and bool(
        str(metadata["phase"]).strip()
    )
    reports = metadata.get("validation_reports")
    evidence_is_recorded = isinstance(reports, list) and len(reports) > 0
    if evidence_is_recorded:
        for report in reports:
            if not isinstance(report, dict):
                evidence_is_recorded = False
                break
            report_path_value = report.get("path")
            report_digest = report.get("sha256")
            if not isinstance(report_path_value, str) or not isinstance(
                report_digest, str
            ):
                evidence_is_recorded = False
                break
            report_path = Path(report_path_value).expanduser()
            if not report_path.is_absolute():
                report_path = resolved_metadata_file.parent / report_path
            if not report_path.is_file() or file_sha256(report_path) != report_digest:
                evidence_is_recorded = False
                break
    collision_ready = (
        metadata.get("collision_ready") is True
        and digest_matches
        and phase_is_recorded
        and evidence_is_recorded
    )
    if not collision_ready and not allow_unvalidated:
        raise StructureValidationError(
            "The snapshot is not an accepted collision input. Its sidecar must "
            "contain collision_ready=true, the phase, validation-report records, "
            "their matching files, and the exact file sha256 after "
            "equilibration and structural "
            "validation. Use allow_unvalidated=True "
            "only for diagnostic infrastructure tests."
        )

    return IceStructure(
        species=species,
        positions_angstrom=positions,
        lattice_angstrom=lattice,
        pbc=pbc,
        source_path=source,
        source_sha256=digest,
        frame_index=selected_index,
        metadata=metadata,
        collision_ready=collision_ready,
    )
