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

import adaptive_nlh_particle_transport as adaptive  # noqa: E402


@pytest.mark.parametrize("projectile", ("H", "He", "C", "O", "S"))
def test_seed_and_case_layout_are_generic_for_every_projectile(projectile):
    case = adaptive.Case(1000, "c_axis", 1.0e5)
    seed = adaptive._stable_seed("production", projectile, case)
    assert 0 <= seed < 2**32
    assert seed == adaptive._stable_seed("production", projectile, case)
    assert adaptive._case_key(case) == "seed1000/c_axis/100000eV"


def test_cdf_midpoint_interpolation_is_exact_for_identical_samples():
    sample = np.asarray((0.0, 0.1, 0.4, 1.0), dtype=float)
    assert adaptive._cdf_interpolation_error(sample, sample, sample) == 0.0


def test_cdf_midpoint_interpolation_detects_a_shift():
    lower = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=float)
    upper = lower.copy()
    midpoint = np.asarray((0.0, 1.0, 1.0, 1.0), dtype=float)
    assert adaptive._cdf_interpolation_error(lower, upper, midpoint) == pytest.approx(
        0.25
    )
