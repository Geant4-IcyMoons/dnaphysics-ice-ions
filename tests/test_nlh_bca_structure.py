from __future__ import annotations

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
    NEP_MBPOL / "structures" / "ice_ih_8x8x8_seed1000_initial.xyz"
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
