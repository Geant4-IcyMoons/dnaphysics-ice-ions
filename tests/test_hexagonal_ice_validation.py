from __future__ import annotations

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

from bca.structure import iter_xyz_frames  # noqa: E402
from validate_hexagonal_ice import (  # noqa: E402
    bernal_fowler_topology,
    chill_plus,
    hac_linear_trend,
    holm_adjust,
)


INITIAL_STRUCTURE = (
    NEP_MBPOL.parent
    / "ice_structures"
    / "preparation"
    / "hexagonal_ih_genice2"
    / "ice_ih_8x8x8_seed1000_melt_start.xyz"
)


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_hac_trend_distinguishes_constant_and_exact_ramp():
    constant = hac_linear_trend(np.ones(500), 0.002)
    ramp = hac_linear_trend(np.arange(500, dtype=float), 0.002)
    assert constant["slope_per_ns"] == pytest.approx(0.0)
    assert constant["raw_two_sided_p_value"] == pytest.approx(1.0)
    assert ramp["slope_per_ns"] == pytest.approx(500.0)
    assert ramp["raw_two_sided_p_value"] == pytest.approx(0.0)


def test_committed_genice_ih_satisfies_chill_plus_and_ice_rules():
    frame = next(iter_xyz_frames(INITIAL_STRUCTURE))
    lengths = np.diag(frame.lattice_angstrom)
    positions = frame.positions_angstrom % lengths
    oxygen = positions[frame.species == "O"]
    hydrogen = positions[frame.species == "H"]
    chill, neighbors, _, tree = chill_plus(oxygen, lengths)
    topology, orientation = bernal_fowler_topology(
        oxygen, hydrogen, neighbors, tree
    )
    assert chill["hexagonal_fraction"] == pytest.approx(1.0)
    assert chill["passes_published_bulk_ih_benchmark"]
    assert topology["passes_bernal_fowler_rules"]
    assert orientation is not None
