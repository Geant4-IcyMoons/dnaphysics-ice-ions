#!/usr/bin/env python3
"""Map periodic ice-Ih coordinates into the experimental H2O unit cell.

The target cell is derived from the corrected polynomial coefficients of
Rottger et al. (2012), DOI 10.1107/S0108768111046908.  Their independently
fitted unit-cell volume fixes the density, while the ratio of their fitted
``c`` and ``a`` lattice constants fixes the anisotropic cell shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


REFERENCE_DOI = "10.1107/S0108768111046908"
AVOGADRO_MOL_INV = 6.02214076e23
H2O_MOLAR_MASS_G_MOL = 18.01528

# A_i for sum_i A_i T**i.  These are the corrected H2O ice-Ih entries in
# Table 1 of Rottger et al. (2012); lengths are A and volume is A^3.
VOLUME_COEFFICIENTS = (
    128.2147,
    0.0,
    0.0,
    -1.3152e-6,
    2.4837e-8,
    -1.6064e-10,
    4.6097e-13,
    -4.9661e-16,
    0.0,
)
A_COEFFICIENTS = (
    4.496915,
    0.0,
    0.0,
    -1.9790e-8,
    3.8958e-10,
    -2.6930e-12,
    8.2861e-15,
    -9.5759e-18,
    0.0,
)
C_COEFFICIENTS = (
    7.321125,
    0.0,
    0.0,
    -2.4944e-8,
    4.6735e-10,
    -2.9799e-12,
    8.3902e-15,
    -8.8400e-18,
    0.0,
)

# The GenIce2 --rep 8 8 8 cell used by this project is an orthorhombic
# representation containing 2048 conventional four-molecule hexagonal cells.
EXPECTED_ATOMS = 24_576
EXPECTED_MOLECULES = 8_192
HEXAGONAL_CELLS = EXPECTED_MOLECULES // 4


def polynomial(coefficients: tuple[float, ...], temperature_k: float) -> float:
    """Evaluate a polynomial by Horner's method."""
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * temperature_k + coefficient
    return value


def experimental_cell(temperature_k: float) -> dict[str, float]:
    """Return the paper-derived orthorhombic supercell at ``temperature_k``."""
    if not 0.0 <= temperature_k <= 265.0:
        raise ValueError("Rottger et al. validate these fits only from 0 to 265 K.")

    volume = polynomial(VOLUME_COEFFICIENTS, temperature_k)
    a_fit = polynomial(A_COEFFICIENTS, temperature_k)
    c_fit = polynomial(C_COEFFICIENTS, temperature_k)
    c_over_a = c_fit / a_fit

    # Enforce the exact hexagonal-cell identity V = sqrt(3) a^2 c / 2 while
    # retaining the anisotropy c/a of the independent a and c fits.
    a = (2.0 * volume / (math.sqrt(3.0) * c_over_a)) ** (1.0 / 3.0)
    c = c_over_a * a
    length_x = 16.0 * a
    length_y = 8.0 * math.sqrt(3.0) * a
    length_z = 8.0 * c
    density = (
        4.0
        * H2O_MOLAR_MASS_G_MOL
        / (AVOGADRO_MOL_INV * volume * 1.0e-24)
    )
    return {
        "temperature_k": temperature_k,
        "unit_cell_volume_a3": volume,
        "a_fit_a": a_fit,
        "c_fit_a": c_fit,
        "c_over_a": c_over_a,
        "volume_constrained_a_a": a,
        "volume_constrained_c_a": c,
        "length_x_a": length_x,
        "length_y_a": length_y,
        "length_z_a": length_z,
        "density_g_cm3": density,
    }


def read_gpumd_xyz(path: Path) -> tuple[list[str], list[list[float]], list[float]]:
    """Read species, positions, and diagonal cell lengths from GPUMD XYZ."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Incomplete XYZ file: {path}")
    atom_count = int(lines[0])
    if atom_count != EXPECTED_ATOMS:
        raise ValueError(
            f"Expected {EXPECTED_ATOMS} atoms in the 8x8x8 cell; got {atom_count}."
        )
    if len(lines) < atom_count + 2:
        raise ValueError(f"XYZ atom block is truncated: {path}")

    match = re.search(r'Lattice="([^"]+)"', lines[1])
    if match is None:
        raise ValueError(f"Missing Lattice field: {path}")
    lattice = [float(value) for value in match.group(1).split()]
    if len(lattice) != 9:
        raise ValueError("Lattice must contain nine components.")
    off_diagonal = (lattice[1], lattice[2], lattice[3], lattice[5], lattice[6], lattice[7])
    if any(abs(value) > 1.0e-8 for value in off_diagonal):
        raise ValueError("This preparation requires the orthorhombic project cell.")
    lengths = [lattice[0], lattice[4], lattice[8]]
    if any(value <= 0.0 for value in lengths):
        raise ValueError("Periodic-cell lengths must be positive.")

    species: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2 : atom_count + 2]:
        fields = line.split()
        if len(fields) < 4 or fields[0] not in {"O", "H"}:
            raise ValueError(f"Invalid atom line: {line!r}")
        species.append(fields[0])
        positions.append([float(value) for value in fields[1:4]])
    if any(species[index : index + 3] != ["O", "H", "H"] for index in range(0, atom_count, 3)):
        raise ValueError("Expected all water molecules in O-H-H order.")
    return species, positions, lengths


def map_positions(
    positions: list[list[float]],
    source_lengths: list[float],
    target_lengths: list[float],
) -> list[list[float]]:
    """Preserve fractional coordinates while changing the periodic cell."""
    return [
        [
            ((coordinate / source_length) % 1.0) * target_length
            for coordinate, source_length, target_length in zip(
                position, source_lengths, target_lengths
            )
        ]
        for position in positions
    ]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_gpumd_xyz(
    path: Path,
    species: list[str],
    positions: list[list[float]],
    lengths: list[float],
) -> None:
    lattice = (
        lengths[0], 0.0, 0.0,
        0.0, lengths[1], 0.0,
        0.0, 0.0, lengths[2],
    )
    lattice_text = " ".join(f"{value:.12f}" for value in lattice)
    lines = [
        str(len(species)),
        f'Lattice="{lattice_text}" '
        'Properties=species:S:1:pos:R:3:group:I:1 pbc="T T T"',
    ]
    lines.extend(
        f"{element:<2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f} 0"
        for element, xyz in zip(species, positions)
    )
    atomic_write(path, "\n".join(lines) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--temperature-k", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else output.with_suffix(".json")
    )

    species, positions, source_lengths = read_gpumd_xyz(source)
    target = experimental_cell(args.temperature_k)
    target_lengths = [
        target["length_x_a"],
        target["length_y_a"],
        target["length_z_a"],
    ]
    mapped = map_positions(positions, source_lengths, target_lengths)
    write_gpumd_xyz(output, species, mapped, target_lengths)

    metadata = {
        "method": "fractional_coordinate_map_to_experimental_ice_ih_cell",
        "reference": {
            "authors": "Rottger et al.",
            "year": 2012,
            "doi": REFERENCE_DOI,
            "table": 1,
        },
        "polynomial_coefficients": {
            "unit_cell_volume_a3": VOLUME_COEFFICIENTS,
            "a_a": A_COEFFICIENTS,
            "c_a": C_COEFFICIENTS,
        },
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "lengths_a": source_lengths,
        },
        "target": target,
        "supercell": {
            "atoms": len(species),
            "molecules": EXPECTED_MOLECULES,
            "hexagonal_unit_cells": HEXAGONAL_CELLS,
            "periodic": [True, True, True],
            "vacuum": False,
        },
        "output": {"path": str(output), "sha256": sha256(output)},
    }
    atomic_write(metadata_path, json.dumps(metadata, indent=2) + "\n")
    print(
        f"Wrote {output} at {target['temperature_k']:.6g} K: "
        f"{target['density_g_cm3']:.9f} g/cm^3, "
        f"box {target_lengths[0]:.8f} x {target_lengths[1]:.8f} x "
        f"{target_lengths[2]:.8f} A"
    )


if __name__ == "__main__":
    main()
