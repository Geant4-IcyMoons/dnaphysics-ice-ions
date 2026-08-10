"""Lossless parser and converter for the archived EPSR ``.ato`` model."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray

from constants import AVOGADRO, H2O_MOLAR_MASS_G_MOL


class EpsrFormatError(ValueError):
    """Raised when an EPSR coordinate file violates its recorded layout."""


@dataclass(frozen=True)
class EpsrAtoStructure:
    """One cubic, molecular EPSR structure in the archive coordinate basis."""

    molecule_count: int
    box_length_angstrom: float
    temperature_k: float
    species: NDArray[np.str_]
    positions_angstrom: NDArray[np.float64]
    molecule_indices: NDArray[np.int64]
    source_sha256: str

    @property
    def lattice_angstrom(self) -> NDArray[np.float64]:
        return np.eye(3, dtype=np.float64) * self.box_length_angstrom

    @property
    def volume_angstrom3(self) -> float:
        return self.box_length_angstrom**3

    @property
    def atomic_number_density_angstrom3(self) -> float:
        return float(self.species.size / self.volume_angstrom3)

    @property
    def molecular_number_density_angstrom3(self) -> float:
        return float(self.molecule_count / self.volume_angstrom3)

    @property
    def density_g_cm3(self) -> float:
        mass_g = (
            self.molecule_count
            * H2O_MOLAR_MASS_G_MOL
            / AVOGADRO
        )
        return mass_g / (self.volume_angstrom3 * 1.0e-24)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(lines: list[str], index: int, context: str) -> list[str]:
    if index >= len(lines):
        raise EpsrFormatError(f"Unexpected end of file while reading {context}.")
    values = lines[index].split()
    if not values:
        raise EpsrFormatError(f"Empty record while reading {context}.")
    return values


def parse_ato(path: str | Path) -> EpsrAtoStructure:
    """Parse the EPSR v26 molecular ``.ato`` format without changing geometry.

    The molecule header supplies a molecular origin and each atom supplies a
    local Cartesian offset.  Their sum is the archived atomic coordinate.
    Connectivity records are checked but are not reinterpreted or fitted.
    """

    source = Path(path).expanduser().resolve()
    lines = source.read_text(encoding="utf-8").splitlines()
    header = _tokens(lines, 0, "global header")
    if len(header) < 3:
        raise EpsrFormatError("The global header requires molecule count, box, and T.")
    try:
        molecule_count = int(header[0])
        box_length = float(header[1])
        temperature = float(header[2])
    except ValueError as exc:
        raise EpsrFormatError("The global header contains a non-numeric value.") from exc
    if molecule_count <= 0 or box_length <= 0.0 or temperature <= 0.0:
        raise EpsrFormatError("Molecule count, box length, and T must be positive.")

    # EPSR global move-control record.  Retain it in the source archive; its
    # values do not enter coordinate conversion.
    _tokens(lines, 1, "global move-control record")
    species = np.empty(3 * molecule_count, dtype="U1")
    positions = np.empty((3 * molecule_count, 3), dtype=np.float64)
    molecule_indices = np.repeat(np.arange(molecule_count, dtype=np.int64), 3)
    line_index = 2

    for molecule_index in range(molecule_count):
        molecule = _tokens(lines, line_index, f"molecule {molecule_index + 1} header")
        line_index += 1
        if len(molecule) < 4 or molecule[0] != "3":
            raise EpsrFormatError(
                f"Molecule {molecule_index + 1} is not a three-site water record."
            )
        try:
            origin = np.asarray([float(value) for value in molecule[1:4]])
        except ValueError as exc:
            raise EpsrFormatError("A molecular origin is non-numeric.") from exc

        expected = ("OW", "HW", "HW")
        expected_connectivity = (
            ((2, 0.976), (3, 0.976)),
            ((1, 0.976), (3, 1.55)),
            ((1, 0.976), (2, 1.55)),
        )
        for site_index, expected_label in enumerate(expected):
            site = _tokens(lines, line_index, "site label")
            line_index += 1
            if site[0] != expected_label:
                raise EpsrFormatError(
                    f"Molecule {molecule_index + 1}, site {site_index + 1}: "
                    f"expected {expected_label}, found {site[0]}."
                )
            local = _tokens(lines, line_index, "site offset")
            line_index += 1
            if len(local) != 3:
                raise EpsrFormatError("Each EPSR site offset must have three values.")
            try:
                offset = np.asarray([float(value) for value in local])
            except ValueError as exc:
                raise EpsrFormatError("A site offset is non-numeric.") from exc
            connectivity = _tokens(lines, line_index, "connectivity record")
            line_index += 1
            try:
                parsed_connectivity = (
                    (int(connectivity[1]), float(connectivity[2])),
                    (int(connectivity[3]), float(connectivity[4])),
                )
            except (IndexError, ValueError) as exc:
                raise EpsrFormatError("Malformed water connectivity record.") from exc
            if connectivity[0] != "2" or parsed_connectivity != expected_connectivity[site_index]:
                raise EpsrFormatError(
                    "The archived water restraint topology is not OW-HW=0.976 A "
                    "and HW-HW=1.55 A."
                )
            atom_index = 3 * molecule_index + site_index
            species[atom_index] = "O" if expected_label == "OW" else "H"
            positions[atom_index] = origin + offset

        terminator = _tokens(lines, line_index, "molecule terminator")
        line_index += 1
        if terminator != ["0"]:
            raise EpsrFormatError(
                f"Molecule {molecule_index + 1} has no zero terminator."
            )

    # EPSR restart files may append ten records containing the site definitions,
    # random-number state, run label, and coarse/actual density.  They are not
    # coordinates, but recognize their fixed layout rather than silently
    # accepting arbitrary trailing content.
    trailing = [line.split() for line in lines[line_index:] if line.strip()]
    if trailing:
        valid_restart_tail = (
            len(trailing) == 10
            and trailing[0][:3] == ["OW", "O", "0"]
            and trailing[2][:3] == ["HW", "H", "1"]
            and trailing[7][:4] == ["Atomic", "number", "density", "(coarse"]
            and trailing[8][:2] == ["O", "0"]
            and trailing[9][:2] == ["H", "0"]
        )
        if not valid_restart_tail:
            raise EpsrFormatError("Unrecognized records follow the last molecule.")
    if not np.all(np.isfinite(positions)):
        raise EpsrFormatError("The parsed structure contains non-finite positions.")

    return EpsrAtoStructure(
        molecule_count=molecule_count,
        box_length_angstrom=box_length,
        temperature_k=temperature,
        species=species,
        positions_angstrom=positions,
        molecule_indices=molecule_indices,
        source_sha256=file_sha256(source),
    )


def _extended_xyz_text(structure: EpsrAtoStructure) -> str:
    length = structure.box_length_angstrom
    header = (
        f'Lattice="{length:.8f} 0 0 0 {length:.8f} 0 0 0 {length:.8f}" '
        'Properties=species:S:1:pos:R:3:molecule:I:1 pbc="T T T" '
        f'temperature_k={structure.temperature_k:.8f} '
        f'epsr_ato_sha256={structure.source_sha256}'
    )
    output = io.StringIO()
    output.write(f"{structure.species.size}\n{header}\n")
    for atom, position, molecule in zip(
        structure.species,
        structure.positions_angstrom,
        structure.molecule_indices,
        strict=True,
    ):
        output.write(
            f"{atom} {position[0]:.10f} {position[1]:.10f} "
            f"{position[2]:.10f} {molecule + 1}\n"
        )
    return output.getvalue()


def write_extended_xyz(structure: EpsrAtoStructure, path: str | Path) -> None:
    """Atomically write a deterministic gzip-compressed extended-XYZ file."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _extended_xyz_text(structure).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        if output.suffix == ".gz":
            with gzip.GzipFile(
                filename="", fileobj=handle, mode="wb", mtime=0
            ) as compressed:
                compressed.write(payload)
        else:
            handle.write(payload)
    temporary.replace(output)
    output.chmod(0o644)
