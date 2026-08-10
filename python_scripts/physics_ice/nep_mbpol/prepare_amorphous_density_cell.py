#!/usr/bin/env python3
"""Map a periodic amorphous H2O cell to an explicitly requested mass density."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


AVOGADRO_MOL_MINUS_ONE = 6.02214076e23
WATER_MOLAR_MASS_G_MOL = 18.01528


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o664)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_xyz(path: Path) -> tuple[list[str], list[list[float]], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    if len(lines) < atom_count + 2:
        raise ValueError(f"Truncated XYZ file: {path}")
    match = re.search(r'Lattice="([^"]+)"', lines[1])
    if match is None:
        raise ValueError(f"Missing Lattice field: {path}")
    lattice = [float(value) for value in match.group(1).split()]
    if len(lattice) != 9:
        raise ValueError("Lattice must contain nine components.")
    if any(abs(lattice[index]) > 1.0e-10 for index in (1, 2, 3, 5, 6, 7)):
        raise ValueError("The density scan currently requires an orthorhombic cell.")
    lengths = [lattice[0], lattice[4], lattice[8]]
    species: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2 : atom_count + 2]:
        fields = line.split()
        if len(fields) < 4 or fields[0] not in {"H", "O"}:
            raise ValueError(f"Invalid atom record: {line!r}")
        species.append(fields[0])
        positions.append([float(value) for value in fields[1:4]])
    if species.count("H") != 2 * species.count("O"):
        raise ValueError("The structure is not stoichiometric H2O.")
    return species, positions, lengths


def _density(water_molecules: int, lengths_a: list[float]) -> float:
    mass_g = (
        water_molecules
        * WATER_MOLAR_MASS_G_MOL
        / AVOGADRO_MOL_MINUS_ONE
    )
    return mass_g / (math.prod(lengths_a) * 1.0e-24)


def prepare(source: Path, output: Path, metadata: Path, target_density: float) -> None:
    if not math.isfinite(target_density) or target_density <= 0.0:
        raise ValueError("Target density must be finite and positive.")
    species, positions, source_lengths = _read_xyz(source)
    water_molecules = species.count("O")
    source_density = _density(water_molecules, source_lengths)
    length_scale = (source_density / target_density) ** (1.0 / 3.0)
    target_lengths = [length * length_scale for length in source_lengths]
    mapped = [
        [
            ((coordinate / source_length) % 1.0) * target_length
            for coordinate, source_length, target_length in zip(
                position, source_lengths, target_lengths
            )
        ]
        for position in positions
    ]
    lattice = (
        target_lengths[0], 0.0, 0.0,
        0.0, target_lengths[1], 0.0,
        0.0, 0.0, target_lengths[2],
    )
    lattice_text = " ".join(f"{value:.12f}" for value in lattice)
    lines = [
        str(len(species)),
        f'Lattice="{lattice_text}" '
        'Properties=species:S:1:pos:R:3:group:I:1 pbc="T T T"',
    ]
    lines.extend(
        f"{element:<2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f} 0"
        for element, xyz in zip(species, mapped)
    )
    _atomic_write(output, "\n".join(lines) + "\n")
    measured_density = _density(water_molecules, target_lengths)
    record = {
        "schema_version": 1,
        "operation": "isotropic_fixed_density_mapping_for_nvt_compatibility_scan",
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "atoms": len(species),
        "water_molecules": water_molecules,
        "source_lengths_a": source_lengths,
        "target_lengths_a": target_lengths,
        "source_density_g_cm3": source_density,
        "target_density_g_cm3": target_density,
        "measured_density_g_cm3": measured_density,
        "isotropic_length_scale": length_scale,
        "interpretation": (
            "This mapping creates an NVT test input. It is not evidence that "
            "the target density is mechanically or structurally stable."
        ),
    }
    _atomic_write(metadata, json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--target-density-g-cm3", type=float, required=True)
    args = parser.parse_args()
    prepare(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.metadata.expanduser().resolve(),
        args.target_density_g_cm3,
    )


if __name__ == "__main__":
    main()
