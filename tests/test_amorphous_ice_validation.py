from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from validate_amorphous_ice import (  # noqa: E402
    AVOGADRO_MOL_MINUS_ONE,
    WATER_MOLAR_MASS_G_MOL,
    _density_from_thermo,
    _first_peak,
    _target_temperature,
)
from prepare_amorphous_density_cell import prepare  # noqa: E402


def test_density_uses_periodic_cell_volume_and_water_count() -> None:
    water_molecules = 100
    target_density = 0.94
    mass_g = water_molecules * WATER_MOLAR_MASS_G_MOL / AVOGADRO_MOL_MINUS_ONE
    length_a = (mass_g / target_density / 1.0e-24) ** (1.0 / 3.0)
    thermo = np.zeros((1, 12), dtype=np.float64)
    thermo[0, 9:12] = length_a
    np.testing.assert_allclose(
        _density_from_thermo(thermo, water_molecules), [target_density], rtol=1e-12
    )


def test_target_temperature_reproduces_all_five_protocol_stages() -> None:
    time_ns = np.asarray((0.5, 1.5, 2.5, 11.0, 19.5))
    np.testing.assert_allclose(
        _target_temperature(time_ns), (350.0, 295.0, 240.0, 160.0, 80.0)
    )


def test_first_peak_is_restricted_to_the_declared_search_window() -> None:
    q = np.asarray((0.9, 1.3, 1.9, 2.4, 3.0))
    values = np.asarray((100.0, 1.0, 5.0, 2.0, 200.0))
    assert _first_peak(q, values) == {"q_a_inverse": 1.9, "value": 5.0}


def test_fixed_density_mapping_preserves_fractional_coordinates(tmp_path: Path) -> None:
    source = tmp_path / "source.xyz"
    source.write_text(
        "3\n"
        'Lattice="10 0 0 0 20 0 0 0 30" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "O 1 4 9\n"
        "H 2 6 12\n"
        "H 3 8 15\n",
        encoding="utf-8",
    )
    output = tmp_path / "mapped.xyz"
    metadata = tmp_path / "mapped.json"
    prepare(source, output, metadata, 0.94)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    assert record["measured_density_g_cm3"] == pytest.approx(0.94)
    scale = record["isotropic_length_scale"]
    mapped_lines = output.read_text(encoding="utf-8").splitlines()
    mapped_oxygen = np.asarray([float(value) for value in mapped_lines[2].split()[1:4]])
    np.testing.assert_allclose(mapped_oxygen, np.asarray((1.0, 4.0, 9.0)) * scale)
