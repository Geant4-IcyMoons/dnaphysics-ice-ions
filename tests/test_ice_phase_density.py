from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PHYSICS_ICE = ROOT / "python_scripts" / "physics_ice"
sys.path.insert(0, str(PHYSICS_ICE))

from constants import ICE_AMORPHOUS_DENSITY_G_CM3, ICE_HEXAGONAL_DENSITY_G_CM3  # noqa: E402


@pytest.mark.parametrize("name,density,expected", [
    ("Hexagonal", ICE_HEXAGONAL_DENSITY_G_CM3, 0.9335),
    ("Amorphous", ICE_AMORPHOUS_DENSITY_G_CM3, 0.9343471678603292),
])
def test_geant4_and_python_use_the_same_ice_density(name, density, expected):
    header = (ROOT / "proton-pipeline/include/IcePhaseProperties.hh").read_text()
    match = re.search(
        rf"k{name}IceDensityGPerCm3\s*=\s*([0-9.]+)", header
    )
    assert match is not None
    assert density == pytest.approx(expected, rel=1e-14)
    assert float(match.group(1)) == pytest.approx(density, rel=1e-14)
