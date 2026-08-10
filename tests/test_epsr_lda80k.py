from __future__ import annotations

import gzip
from pathlib import Path
import sys

import numpy as np
import pytest


PHYSICS_ICE = (
    Path(__file__).resolve().parents[1] / "python_scripts" / "physics_ice"
)
if str(PHYSICS_ICE) not in sys.path:
    sys.path.insert(0, str(PHYSICS_ICE))
NEP_MBPOL = PHYSICS_ICE / "nep_mbpol"
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from ice_structures.epsr_lda80k.model import (  # noqa: E402
    EpsrFormatError,
    file_sha256,
    parse_ato,
    write_extended_xyz,
)
from nep_mbpol.bca.structure import iter_xyz_frames  # noqa: E402


def _water(origin: tuple[float, float, float], index: int) -> str:
    return (
        f" 3 {origin[0]} {origin[1]} {origin[2]} 0 0 0 F 0 1.0 {index}\n"
        " OW 1 0\n"
        " 0.0 0.0 0.0\n"
        " 2 2 0.976 3 0.976\n"
        " HW 2 0\n"
        " 0.976 0.0 0.0\n"
        " 2 1 0.976 3 1.55\n"
        " HW 3 0\n"
        " -0.25483399 0.94213800 0.0\n"
        " 2 1 0.976 2 1.55\n"
        " 0\n"
    )


def _fixture(path: Path) -> None:
    path.write_text(
        " 2 10.0 80.0\n"
        " 1 2.2 0.3 0.3 1 65 1 10 1 10\n"
        + _water((1.0, 2.0, 3.0), 1)
        + _water((-4.5, 4.5, -4.5), 2),
        encoding="utf-8",
    )


def test_parse_ato_adds_molecular_origins_and_local_offsets(tmp_path: Path) -> None:
    source = tmp_path / "model.ato"
    _fixture(source)
    structure = parse_ato(source)
    assert structure.molecule_count == 2
    assert structure.temperature_k == 80.0
    assert structure.species.tolist() == ["O", "H", "H", "O", "H", "H"]
    np.testing.assert_allclose(structure.positions_angstrom[1], (1.976, 2.0, 3.0))
    np.testing.assert_array_equal(structure.molecule_indices, (0, 0, 0, 1, 1, 1))


def test_deterministic_conversion_is_lossless_modulo_periodicity(tmp_path: Path) -> None:
    source = tmp_path / "model.ato"
    _fixture(source)
    structure = parse_ato(source)
    first = tmp_path / "first.xyz.gz"
    second = tmp_path / "second.xyz.gz"
    write_extended_xyz(structure, first)
    write_extended_xyz(structure, second)
    assert file_sha256(first) == file_sha256(second)
    frame = next(iter_xyz_frames(first))
    assert frame.pbc == (True, True, True)
    assert frame.species.tolist() == structure.species.tolist()
    np.testing.assert_allclose(frame.positions_angstrom, structure.positions_angstrom)
    with gzip.open(first, mode="rt", encoding="utf-8") as handle:
        assert "epsr_ato_sha256=" in handle.readline() + handle.readline()


def test_parser_rejects_a_non_water_site_order(tmp_path: Path) -> None:
    source = tmp_path / "bad.ato"
    _fixture(source)
    source.write_text(
        source.read_text(encoding="utf-8").replace(" OW 1 0", " HW 1 0", 1),
        encoding="utf-8",
    )
    with pytest.raises(EpsrFormatError, match="expected OW"):
        parse_ato(source)
