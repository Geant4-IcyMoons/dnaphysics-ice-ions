"""Small alpha exports check execution, not production-grid convergence."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("relativistic", [False, True])
def test_alpha_export(tmp_path, phase, relativistic):
    _check_export(tmp_path, phase, relativistic, False, "alpha", 2)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("projectile,charge", [("alpha", 0), ("alpha", 1), ("proton", 0)])
def test_screened_born_export(tmp_path, phase, relativistic, projectile, charge):
    _check_export(tmp_path, phase, relativistic, False, projectile, charge)


def _check_export(tmp_path, phase, relativistic, polarization, projectile, charge):
    env = dict(os.environ, ICE_TYPE=phase, ICE_MAX_WORKERS="2",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [sys.executable, "-m", "physics.inelastic_dielectric.generate_cross_sections",
               "--projectile", projectile, "--charge-state", str(charge),
               f"--relativistic-projectile-dcs={str(relativistic).lower()}",
               f"--include-barkas-dcs={str(polarization).lower()}",
               "--energy-min-MeV", "0.1", "--energy-max-MeV", "100",
               "--energy-unit", "total", "--energy-points", "2", "--dE", "12", "--dq", "24",
               "--output-dir", str(tmp_path / "tables"), "--cache-dir", str(tmp_path / "caches"),
               "--no-merge-energy-patches"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    tables = list((tmp_path / "tables").glob("*.dat"))
    assert len(tables) == 4
    assert len(list((tmp_path / "caches").glob("*.npz"))) == 1
    assert len(list((tmp_path / "caches").rglob("energy_*.npz"))) == 2
    for channel in ("excitation", "ionisation"):
        path = next(p for p in tables if p.name.startswith(f"sigmadiff_{channel}_"))
        assert projectile in path.name and f"_q{charge}_" in path.name and phase in path.name
        assert ("rpwba" in path.name) == relativistic
        assert ("barkas_dcs" in path.name) == polarization
        dcs = np.loadtxt(path)
        tcs = np.loadtxt(next(p for p in tables if p.name.startswith(f"sigma_{channel}_")))
        np.testing.assert_array_equal(np.unique(dcs[:, 0]), [1e5, 1e8])
        assert np.all(np.isfinite(dcs)) and np.all(dcs[:, 2:] >= 0)
        for i, energy in enumerate(tcs[:, 0]):
            rows = dcs[dcs[:, 0] == energy]
            np.testing.assert_allclose(np.trapezoid(rows[:, 2:], rows[:, 1], axis=0),
                                       tcs[i, 1:], rtol=1e-12, atol=0)
