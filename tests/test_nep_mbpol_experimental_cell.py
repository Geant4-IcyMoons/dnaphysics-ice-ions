"""Regression tests for the experimental-density ice-Ih preparation."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
    / "prepare_hexagonal_ice_experimental_cell.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_ice_ih_cell", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_rottger_100k_polynomials_and_density() -> None:
    cell = PREPARE.experimental_cell(100.0)

    assert cell["unit_cell_volume_a3"] == pytest.approx(128.188109, abs=1.0e-9)
    assert cell["a_fit_a"] == pytest.approx(4.49648151, abs=1.0e-11)
    assert cell["c_fit_a"] == pytest.approx(7.32062320, abs=1.0e-11)
    assert cell["density_g_cm3"] == pytest.approx(
        0.9334742974038461, abs=1.0e-12
    )


def test_target_supercell_has_paper_volume_and_anisotropy() -> None:
    cell = PREPARE.experimental_cell(100.0)
    supercell_volume = (
        cell["length_x_a"] * cell["length_y_a"] * cell["length_z_a"]
    )

    assert supercell_volume == pytest.approx(
        PREPARE.HEXAGONAL_CELLS * cell["unit_cell_volume_a3"], rel=1.0e-14
    )
    assert cell["volume_constrained_c_a"] / cell["volume_constrained_a_a"] == (
        pytest.approx(cell["c_fit_a"] / cell["a_fit_a"], rel=1.0e-14)
    )
    assert cell["length_y_a"] / cell["length_x_a"] == pytest.approx(
        math.sqrt(3.0) / 2.0, rel=1.0e-14
    )


def test_fractional_mapping_wraps_into_target_cell() -> None:
    mapped = PREPARE.map_positions(
        [[5.0, -1.0, 20.0]],
        [10.0, 10.0, 10.0],
        [20.0, 30.0, 40.0],
    )
    assert mapped[0] == pytest.approx([10.0, 27.0, 0.0])


def test_fit_rejects_temperatures_outside_published_range() -> None:
    with pytest.raises(ValueError, match="0 to 265 K"):
        PREPARE.experimental_cell(266.0)
