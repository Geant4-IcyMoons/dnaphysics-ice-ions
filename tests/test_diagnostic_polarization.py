"""Diagnostic output retains failures without changing strict acceptance."""
from types import SimpleNamespace
from pathlib import Path
import os
import subprocess
import sys
import json
import numpy as np
import pytest
from physics.inelastic_dielectric.polarization import oscillator_quadrature as oq
from physics.inelastic_dielectric.polarization import screened_barkas as sb
from physics.inelastic_dielectric.polarization import diagnostic_tables as dt
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import load_density


def test_unconverged_estimate_is_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(oq, "quad_vec", lambda *a, **k: (
        np.array([2., .2, .3]), .4, SimpleNamespace(neval=1, success=False)))
    with pytest.raises(RuntimeError, match="did not converge"):
        oq.integrate_kernel(.1, 1., 1., load_density("H", 0))
    value, error, report = oq.integrate_kernel(.1, 1., 1., load_density("H", 0), allow_unconverged=True)
    assert value == 2 and error == pytest.approx(.9)
    assert report["converged"] is False


def test_nonfinite_and_domain_failures_remain_missing(monkeypatch):
    def failure(*a, **k):
        raise FloatingPointError("nonfinite integrand")
    monkeypatch.setattr(oq, "integrate_kernel", failure)
    task = (1e7, np.array([15.]), 1836.152673, "H", 0)
    with pytest.raises(FloatingPointError):
        sb._energy_row(task)
    value, error, flags = sb._energy_row(task, diagnostic_only=True)
    assert np.isnan(value[0]) and np.isnan(error[0]) and flags[0] & 2
    value, error, flags = sb._energy_row((1., *task[1:]), diagnostic_only=True)
    assert np.isnan(value[0]) and flags[0] == 4


def test_rejected_rows_export_only_diagnostic_files(tmp_path, monkeypatch):
    monkeypatch.setattr(dt.bd, "oos_density", lambda *a, **k: SimpleNamespace(df_dW_total=np.ones(3)))
    monkeypatch.setattr(dt, "_diagnostic_row", lambda task: (
        np.array([2., -2., np.nan]), np.array([.1, .1, np.nan]), np.array([1, 0, 2], np.uint8)))
    data = dict(T_line=np.full(3, 1e7), E_line=np.array([15., 20., 30.]),
                exc_vals=np.ones((3, 1)), ion_vals=np.zeros((3, 1)))
    names = ("exc.dat", "ion.dat", "exc_total.dat", "ion_total.dat")
    dt.write_diagnostic_tables(data, SimpleNamespace(material="amorphous"), load_density("H", 0),
        1836.152673, tmp_path / "tables", tmp_path / "tasks", 1, {"include_kshell": True}, 1., dat_names=names)
    assert len(list(tmp_path.rglob("*.dat"))) == 4
    exported = np.loadtxt(tmp_path / "tables" / "exc.dat")
    np.testing.assert_array_equal(exported[:, 2], [3., -1., np.nan])
    assert np.isnan(np.loadtxt(tmp_path / "tables" / "exc_total.dat")[1])
    report = json.loads((tmp_path / "tables" / "DIAGNOSTIC_ONLY.json").read_text())
    assert report["transport_ready"] is False
    with np.load(tmp_path / "tables" / "DIAGNOSTIC_ONLY.npz") as result:
        np.testing.assert_array_equal(result["rejection_flags"], [9, 24, 2])
        np.testing.assert_array_equal(result["total_m2_eV"], [3., -1., np.nan])


def test_diagnostic_rejected_losses_resume_without_recomputation(tmp_path, monkeypatch):
    task = (1e7, np.array([15., 20.]), 1836.152673, "H", 0, tmp_path / "row.npz")
    calls = []
    def calculate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated cancellation")
        return np.array([np.nan]), np.array([np.nan]), np.array([2], np.uint8)
    monkeypatch.setattr(sb, "_energy_row", calculate)
    with pytest.raises(RuntimeError, match="cancellation"):
        dt._diagnostic_row(task)
    resumed = dt._diagnostic_row(task)
    assert len(calls) == 3
    assert np.all(np.isnan(resumed[0]))
    np.testing.assert_array_equal(resumed[2], [2, 2])


def test_real_he0_dat_export_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, ICE_TYPE="amorphous", ICE_MAX_WORKERS="2",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [sys.executable, "-m", "physics.inelastic_dielectric.generate_cross_sections",
           "--projectile", "He", "--charge-state", "0", "--include-barkas-dcs=true",
           "--diagnostic-only", "--energy-min-MeV", "40", "--energy-max-MeV", "41",
           "--energy-points", "2", "--dE", "2", "--dq", "24",
           "--output-dir", str(tmp_path / "tables"), "--cache-dir", str(tmp_path / "caches")]
    result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    tables = list((tmp_path / "tables").rglob("*.dat"))
    assert len(tables) == 4
    report = json.loads(next((tmp_path / "tables").rglob("DIAGNOSTIC_ONLY.json")).read_text())
    assert report["failure_counts"]["abs_correction_at_least_born"] > 0
    for kind in ("excitation", "ionisation"):
        dcs = np.loadtxt(next(p for p in tables if p.name.startswith(f"sigmadiff_{kind}_")))
        tcs = np.loadtxt(next(p for p in tables if p.name.startswith(f"sigma_{kind}_")))
        for i, energy in enumerate(tcs[:, 0]):
            rows = dcs[dcs[:, 0] == energy]
            np.testing.assert_allclose(np.trapezoid(rows[:, 2:], rows[:, 1], axis=0), tcs[i, 1:], rtol=1e-12)
    before = {p: p.read_bytes() for p in tables}
    result = subprocess.run(cmd, cwd=root, env=dict(env, ICE_MAX_WORKERS="1"),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    assert before == {p: p.read_bytes() for p in tables}
