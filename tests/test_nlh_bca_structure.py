from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from bca.structure import (  # noqa: E402
    StructureValidationError,
    load_ice_structure,
)


INITIAL_STRUCTURE = (
    NEP_MBPOL.parent
    / "ice_structures"
    / "preparation"
    / "hexagonal_ih_genice2"
    / "ice_ih_8x8x8_seed1000_melt_start.xyz"
)


def _small_trajectory(path: Path) -> None:
    header = (
        'Lattice="3 0 0 0 3 0 0 0 3" '
        'Properties=species:S:1:pos:R:3:velocity:R:3 pbc="T T T"'
    )
    frame_0 = [
        "3",
        header,
        "O 0 0 0 0 0 0",
        "H 0.95 0 0 0 0 0",
        "H 0 0.95 0 0 0 0",
    ]
    frame_1 = [
        "3",
        header,
        "O 3.1 0 0 0 0 0",
        "H 1.05 0 0 0 0 0",
        "H 0.1 0.95 0 0 0 0",
    ]
    path.write_text("\n".join(frame_0 + frame_1) + "\n", encoding="utf-8")


def test_initial_unequilibrated_structure_is_diagnostic_only():
    with pytest.raises(StructureValidationError, match="not an accepted"):
        load_ice_structure(INITIAL_STRUCTURE)
    structure = load_ice_structure(INITIAL_STRUCTURE, allow_unvalidated=True)
    assert structure.atom_count == 24_576
    assert structure.water_molecule_count == 8_192
    # This committed GenIce input is a historical 273 K-density structure;
    # Geant4 transport and newly generated 100 K structures use 0.9335 g/cm3.
    assert structure.density_g_cm3 == pytest.approx(0.917, rel=2.0e-12)
    assert structure.use_class == "diagnostic-only"


def test_last_frame_arbitrary_properties_wrapping_and_attestation(tmp_path):
    trajectory = tmp_path / "accepted.xyz"
    _small_trajectory(trajectory)
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    report_digest = hashlib.sha256(report.read_bytes()).hexdigest()
    trajectory.with_suffix(".json").write_text(
        json.dumps(
            {
                "phase": "test ice",
                "state": "equilibrated and structurally validated",
                "collision_ready": True,
                "sha256": digest,
                "validation_reports": [
                    {"path": "report.json", "sha256": report_digest}
                ],
            }
        ),
        encoding="utf-8",
    )
    structure = load_ice_structure(trajectory)
    assert structure.frame_index == 1
    assert structure.positions_angstrom[0, 0] == pytest.approx(0.1)
    assert structure.collision_ready
    assert structure.use_class == "production-input"


def test_attestation_must_match_exact_file(tmp_path):
    trajectory = tmp_path / "changed.xyz"
    _small_trajectory(trajectory)
    trajectory.with_suffix(".json").write_text(
        json.dumps({"collision_ready": True, "sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(StructureValidationError, match="exact file sha256"):
        load_ice_structure(trajectory)


def test_gzip_xyz_is_parsed_and_attested(tmp_path):
    trajectory = tmp_path / "accepted.xyz"
    _small_trajectory(trajectory)
    compressed = tmp_path / "accepted.xyz.gz"
    with trajectory.open("rb") as source, gzip.open(compressed, "wb") as target:
        target.write(source.read())
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    compressed.with_suffix(".json").write_text(
        json.dumps(
            {
                "phase": "test ice",
                "collision_ready": True,
                "sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
                "validation_reports": [
                    {
                        "path": report.name,
                        "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    structure = load_ice_structure(compressed)
    assert structure.frame_index == 1
    assert structure.collision_ready
