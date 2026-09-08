"""Explicit interface fixtures; numerical tests use the unmodified solver."""
import pytest


@pytest.fixture
def synthetic_nonlinear_kernel(monkeypatch):
    """Cheap, signed test spectrum for assembly/cache tests, not physics validation."""
    from physics.inelastic_dielectric.polarization import nonlinear_polarization as npol
    calls = []

    def kernel(xi, scale, gamma, field, **kwargs):
        calls.append((xi, scale, gamma, field.z))
        return field.z / (1 + xi), 0., {"converged": True}

    original = npol.dcs_m2_per_eV

    def serial(*args, **kwargs):
        kwargs["workers"] = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(npol.nonlinear_oscillator, "integrate_kernel", kernel)
    monkeypatch.setattr(npol, "dcs_m2_per_eV", serial)
    monkeypatch.setenv("ICE_MAX_WORKERS", "1")
    return calls
