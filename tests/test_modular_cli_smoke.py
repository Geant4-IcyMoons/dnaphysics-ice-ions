"""Small real CLI exports; these are not production convergence tests."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("charge", [0, 1])
def test_hydrogen_cli_dcs_tables(tmp_path, phase, relativistic, charge):
    _generate_and_check(tmp_path, phase, relativistic, charge, False)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("relativistic", [False, True])
def test_bare_proton_polarization_cli(tmp_path, phase, relativistic):
    _generate_and_check(tmp_path, phase, relativistic, 1, True)


def _generate_and_check(output, phase, relativistic, charge, polarization):
    environment = dict(os.environ, ICE_TYPE=phase, ICE_MAX_WORKERS="2",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MPLCONFIGDIR=str(output / "matplotlib"))
    command = [sys.executable, "-m", "physics.inelastic_dielectric.generate_cross_sections",
               "--projectile", "H", "--charge-state", str(charge),
               f"--relativistic-projectile-dcs={str(relativistic).lower()}",
               f"--include-barkas-dcs={str(polarization).lower()}",
               "--charge-mode=bare", "--born-reference-charge=bare_Z",
               "--energy-min-MeV", "10", "--energy-max-MeV", "20",
               "--energy-unit", "total", "--energy-points", "2", "--dE", "12", "--dq", "24",
               "--output-dir", str(output / "tables"), "--cache-dir", str(output / "caches"),
               "--no-merge-energy-patches"]
    result = subprocess.run(command, cwd=ROOT, env=environment,
                            text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    tables = sorted((output / "tables").glob("*.dat"))
    assert len(tables) == 4
    assert len(list((output / "caches").glob("*.npz"))) == 1
    for channel in ("excitation", "ionisation"):
        differential = next(path for path in tables if path.name.startswith(f"sigmadiff_{channel}_"))
        total = next(path for path in tables if path.name.startswith(f"sigma_{channel}_"))
        assert f"_q{charge}_" in differential.name
        assert phase in differential.name
        assert ("rpwba" in differential.name) == relativistic
        assert ("barkas_dcs" in differential.name) == polarization
        dcs, tcs = np.loadtxt(differential), np.loadtxt(total)
        np.testing.assert_array_equal(np.unique(dcs[:, 0]), [1e7, 2e7])
        assert np.all(np.isfinite(dcs)) and np.all(dcs[:, 2:] >= 0)
        for index, energy in enumerate(tcs[:, 0]):
            rows = dcs[dcs[:, 0] == energy]
            assert len(rows) >= 12
            np.testing.assert_allclose(np.trapezoid(rows[:, 2:], rows[:, 1], axis=0),
                                       tcs[index, 1:], rtol=1e-12, atol=0)
    # A second invocation must reuse the complete cache, not apply polarization twice.
    before = {path.name: path.read_bytes() for path in tables}
    # Keep partial products but remove one task and the complete cache. Resume
    # with a different worker count and require identical exported bytes.
    next((output / "caches").glob("*.npz")).unlink()
    task_root = next((output / "caches" / "checkpoints").iterdir())
    next(task_root.glob("energy_*.npz")).unlink()
    next((task_root / "dcs").glob("*.npz")).unlink()
    saved = {p: p.stat().st_mtime_ns for p in task_root.rglob("*.npz")}
    resumed = subprocess.run(command, cwd=ROOT, env=dict(environment, ICE_MAX_WORKERS="1"),
                             text=True, capture_output=True, timeout=180)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Integrated energies: 1/2 resumed" in resumed.stdout
    assert "DCS export: 19/20 tasks resumed" in resumed.stdout
    assert saved == {p: p.stat().st_mtime_ns for p in saved}
    assert before == {path.name: path.read_bytes() for path in tables}
    repeat = subprocess.run(command, cwd=ROOT, env=environment,
                            text=True, capture_output=True, timeout=180)
    assert repeat.returncode == 0, repeat.stdout + repeat.stderr
    assert "Loaded cached cross sections" in repeat.stdout
    assert before == {path.name: path.read_bytes() for path in tables}
