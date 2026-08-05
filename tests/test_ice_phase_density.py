from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PHYSICS_ICE = ROOT / "python_scripts" / "physics_ice"
sys.path.insert(0, str(PHYSICS_ICE))

from constants import ICE_HEXAGONAL_DENSITY_G_CM3  # noqa: E402


def test_geant4_and_python_use_the_same_100k_ice_ih_density():
    header = (ROOT / "proton-pipeline/include/IcePhaseProperties.hh").read_text()
    match = re.search(
        r"kHexagonalIceDensityGPerCm3\s*=\s*([0-9.]+)", header
    )
    assert match is not None
    assert ICE_HEXAGONAL_DENSITY_G_CM3 == pytest.approx(0.9335)
    assert float(match.group(1)) == pytest.approx(
        ICE_HEXAGONAL_DENSITY_G_CM3
    )
