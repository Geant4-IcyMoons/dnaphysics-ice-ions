#!/usr/bin/env python3
"""Generate a periodic, hydrogen-disordered ice-Ih cell for GPUMD.

GenIce2 constructs the initial crystal topology and proton arrangement.  This
script converts its output to the extended-XYZ dialect read by GPUMD.  The
result is an *initial*, unequilibrated structure; NEP-MB-pol must subsequently
relax/equilibrate it on a GPU node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parent
sys.path.insert(0, str(PHYSICS_ICE))

from constants import (  # noqa: E402
    AVOGADRO,
    H2O_MOLAR_MASS_G_MOL,
    ICE_HEXAGONAL_DENSITY_G_CM3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rep",
        nargs=3,
        type=int,
        default=(8, 8, 8),
        metavar=("NX", "NY", "NZ"),
        help="GenIce2 ice-Ih cell repetitions (default: 8 8 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Proton-disorder random seed (default: 1000).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=ICE_HEXAGONAL_DENSITY_G_CM3,
        help="Initial mass density in g/cm^3 (default: project constant).",
    )
    parser.add_argument(
        "--genice",
        type=Path,
        help="Path to genice2; defaults to .venv/bin/genice2 or PATH.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output GPUMD extended-XYZ file.",
    )
    return parser.parse_args()


def find_genice(requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested.expanduser())
    candidates.append(HERE / ".venv" / "bin" / "genice2")
    path_entry = shutil.which("genice2")
    if path_entry:
        candidates.append(Path(path_entry))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "genice2 was not found. Install requirements.txt in .venv or pass "
        "--genice /path/to/genice2."
    )


def parse_genice_exyz(text: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    lines = text.splitlines()
    atom_header = None
    atom_count = None
    for index, line in enumerate(lines[:-1]):
        try:
            candidate_count = int(line.strip())
        except ValueError:
            continue
        if lines[index + 1].strip() == "%PBC":
            atom_header = index
            atom_count = candidate_count
            break

    if atom_header is None or atom_count is None:
        raise ValueError("Could not locate the GenIce2 extended-XYZ atom block.")

    atom_lines = lines[atom_header + 2 : atom_header + 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError("GenIce2 output ended before all atoms were read.")

    species: list[str] = []
    positions = np.empty((atom_count, 3), dtype=float)
    for index, line in enumerate(atom_lines):
        fields = line.split()
        if len(fields) != 4 or fields[0] not in {"O", "H"}:
            raise ValueError(f"Unexpected GenIce2 atom line: {line!r}")
        species.append(fields[0])
        positions[index] = [float(value) for value in fields[1:4]]

    vectors: dict[str, list[float]] = {}
    for line in lines[atom_header + 2 + atom_count :]:
        fields = line.split()
        if len(fields) == 4 and fields[0] in {"Vector1", "Vector2", "Vector3"}:
            vectors[fields[0]] = [float(value) for value in fields[1:4]]
    if len(vectors) != 3:
        raise ValueError("Could not read all three periodic-cell vectors.")

    lattice = np.asarray(
        [vectors["Vector1"], vectors["Vector2"], vectors["Vector3"]],
        dtype=float,
    )

    # Put every site into the primary periodic cell. Cell vectors are rows.
    fractional = positions @ np.linalg.inv(lattice)
    fractional -= np.floor(fractional)
    positions = fractional @ lattice
    return species, positions, lattice


def write_gpumd_xyz(
    output: Path, species: list[str], positions: np.ndarray, lattice: np.ndarray
) -> None:
    lattice_values = " ".join(f"{value:.10f}" for value in lattice.ravel())
    header = (
        f'Lattice="{lattice_values}" '
        'Properties=species:S:1:pos:R:3:group:I:1 pbc="T T T"'
    )
    lines = [str(len(species)), header]
    lines.extend(
        f"{element:<2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f} 0"
        for element, xyz in zip(species, positions, strict=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if any(value <= 0 for value in args.rep):
        raise ValueError("All --rep values must be positive integers.")
    if args.density <= 0.0:
        raise ValueError("--density must be positive.")

    genice = find_genice(args.genice)
    rep_label = "x".join(str(value) for value in args.rep)
    output = args.output or (
        HERE
        / "structures"
        / f"ice_ih_{rep_label}_seed{args.seed}_initial.xyz"
    )
    output = output.expanduser().resolve()

    command = [
        str(genice),
        "--rep",
        *(str(value) for value in args.rep),
        "--dens",
        f"{args.density:.12g}",
        "--seed",
        str(args.seed),
        "--depol",
        "strict",
        "--water",
        "physical_water",
        "--format",
        "exyz",
        "--quiet",
        "one[hh]",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    species, positions, lattice = parse_genice_exyz(result.stdout)

    oxygen_count = species.count("O")
    hydrogen_count = species.count("H")
    if hydrogen_count != 2 * oxygen_count:
        raise ValueError(
            f"Invalid H2O stoichiometry: O={oxygen_count}, H={hydrogen_count}."
        )
    if not all(
        species[index : index + 3] == ["O", "H", "H"]
        for index in range(0, len(species), 3)
    ):
        raise ValueError("Expected each GenIce2 water molecule to be ordered O-H-H.")

    raw_volume_angstrom3 = abs(float(np.linalg.det(lattice)))
    raw_density = (
        oxygen_count * H2O_MOLAR_MASS_G_MOL / AVOGADRO
    ) / (raw_volume_angstrom3 * 1.0e-24)

    # GenIce2's structure scaling uses 18.000 g/mol for water, whereas the
    # shared project constants use 18.01528 g/mol. Isotropically correct the
    # cell volume so that the generated structure matches the project density
    # exactly without changing the Ih topology or c/a ratio.
    density_scale = (raw_density / args.density) ** (1.0 / 3.0)
    lattice *= density_scale
    positions *= density_scale

    volume_angstrom3 = abs(float(np.linalg.det(lattice)))
    density = (
        oxygen_count * H2O_MOLAR_MASS_G_MOL / AVOGADRO
    ) / (volume_angstrom3 * 1.0e-24)
    if not np.isclose(density, args.density, rtol=1.0e-10):
        raise ValueError(
            f"Generated density {density:.9f} does not match {args.density:.9f}."
        )

    inverse_lattice = np.linalg.inv(lattice)
    oxygen_positions = positions[0::3]
    oh_distances = []
    for hydrogen_positions in (positions[1::3], positions[2::3]):
        displacement_fractional = (
            hydrogen_positions - oxygen_positions
        ) @ inverse_lattice
        displacement_fractional -= np.rint(displacement_fractional)
        displacement = displacement_fractional @ lattice
        oh_distances.append(np.linalg.norm(displacement, axis=1))
    oh_distances_array = np.concatenate(oh_distances)
    if not np.all((oh_distances_array > 0.94) & (oh_distances_array < 0.98)):
        raise ValueError("Initial water geometry has an unexpected O-H distance.")

    write_gpumd_xyz(output, species, positions, lattice)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()

    metadata = {
        "phase": "hexagonal ice Ih",
        "state": "initial, proton-disordered, not yet NEP-MB-pol equilibrated",
        "generator": "GenIce2 2.2.13.3",
        "genice_lattice": "one[hh]",
        "orientation": "hexagonal c-axis (basal-plane normal) along z",
        "water_geometry": "physical_water",
        "depolarization": "strict",
        "replication": list(args.rep),
        "seed": args.seed,
        "atoms": len(species),
        "water_molecules": oxygen_count,
        "target_density_g_cm3": args.density,
        "genice_raw_density_g_cm3_using_project_molar_mass": raw_density,
        "isotropic_length_rescale_factor": density_scale,
        "measured_density_g_cm3": density,
        "volume_angstrom3": volume_angstrom3,
        "lattice_angstrom": lattice.tolist(),
        "oh_distance_angstrom": {
            "minimum": float(oh_distances_array.min()),
            "mean": float(oh_distances_array.mean()),
            "maximum": float(oh_distances_array.max()),
        },
        "gpumd_model": "../model/nep-mbpol.nep.txt",
        "sha256": digest,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {output}")
    print(f"Wrote {metadata_path}")
    print(
        f"Cell: {oxygen_count:,} H2O, {len(species):,} atoms, "
        f"{density:.6f} g/cm^3"
    )
    print("Status: initial Ih structure; NEP-MB-pol equilibration remains.")


if __name__ == "__main__":
    main()
