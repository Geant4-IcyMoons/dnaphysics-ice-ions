"""Loss-level resume preserves results without relaxing physical guards."""
import numpy as np
import pytest
from physics.inelastic_dielectric.polarization import nonlinear_polarization as sb
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import load_density


def test_interrupted_loss_resume(tmp_path, monkeypatch):
    density = load_density("He", 0)
    path = tmp_path / "row.npz"
    calls = []
    def kernel(xi, scale, gamma, density, **kwargs):
        calls.append(xi)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return xi, abs(xi)*1e-6, {}
    monkeypatch.setattr(sb.nonlinear_oscillator, "integrate_kernel", kernel)
    w = np.array([15., 50., 100.])
    beta, gamma = .1, 1/np.sqrt(.99)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sb.converged_kernel(w, beta, gamma, density, checkpoint_path=path)
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["done"], [True, False, False])
    resumed = sb.converged_kernel(w, beta, gamma, density, checkpoint_path=path)
    assert len(calls) == 4
    fresh = sb.converged_kernel(w, beta, gamma, density)
    np.testing.assert_array_equal(resumed, fresh)
    with pytest.raises(ValueError, match="Incompatible"):
        sb.converged_kernel(w+1, beta, gamma, density, checkpoint_path=path)


def test_real_kernel_checkpoint_reuse(tmp_path, monkeypatch):
    task = (1e7, np.array([15.]), 1836.152673, "H", 0, tmp_path / "real.npz")
    value, error = sb._energy_row(task)
    assert np.all(np.isfinite(value)) and np.all(error >= 0)
    def unexpected(*args, **kwargs):
        raise AssertionError("A completed loss must not be recomputed")
    monkeypatch.setattr(sb.nonlinear_oscillator, "integrate_kernel", unexpected)
    np.testing.assert_array_equal(sb._energy_row(task), (value, error))


def test_velocity_guard_retained(tmp_path):
    with pytest.raises(ValueError, match="Bohr velocity"):
        sb._energy_row((1., np.array([15.]), 1836.152673, "H", 0, tmp_path / "bad.npz"))
    assert not (tmp_path / "bad.npz").exists()


def test_spawn_and_changed_worker_count_resume(tmp_path):
    density = load_density("H", 0)
    kwargs = dict(T_eV=np.array([1e7, 2e7]), W_eV=np.array([15., 15.]),
                  mass_me=1836.152673, df_dW=np.ones(2), density=density,
                  checkpoint_dir=tmp_path)
    parallel = sb.dcs_m2_per_eV(**kwargs, workers=2)
    saved = {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.npz")}
    serial = sb.dcs_m2_per_eV(**kwargs, workers=1)
    np.testing.assert_array_equal(parallel, serial)
    assert saved == {p: p.stat().st_mtime_ns for p in saved}
