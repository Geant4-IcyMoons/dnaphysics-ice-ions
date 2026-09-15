"""Recovery mechanics; synthetic tests do not validate the response model."""
from types import SimpleNamespace
import numpy as np
import pytest
from physics.inelastic_dielectric.polarization import nonlinear_oscillator as no
from physics.inelastic_dielectric.polarization import nonlinear_polarization as pol


def test_only_dominant_error_is_refined(monkeypatch, tmp_path):
    import json
    monkeypatch.setenv('ICE_POLARIZATION_PROGRESS_DIR', str(tmp_path))
    calls = []
    def encounter(x, b, eta, field, **kw):
        calls.append(kw)
        return dict(difference=0.)
    results = iter(([1., 0., .002], [1., 0., .0001]))
    def quad(f, *args, **kw):
        f(0.)
        return np.array(next(results)), 0., SimpleNamespace(success=True)
    monkeypatch.setattr(no, 'encounter', encounter)
    monkeypatch.setattr(no, 'quad_vec', quad)
    _, _, report = no.integrate_kernel(.01, 3., 1., no.PointField(1.))
    assert report['converged']
    assert calls[3]['tail'] == 2*calls[0]['tail']
    assert calls[3]['rtol'] == calls[0]['rtol']
    progress = json.loads(next(tmp_path.glob('worker_*.json')).read_text())
    assert progress['stage'] == 'converged'
    assert progress['refinement'] == 1
    assert progress['completed_impact_evaluations'] == 2
    assert len(progress['history']) == 2


def test_strict_failure_preserves_later_losses_and_resume(monkeypatch, tmp_path):
    calls = []
    def kernel(xi, *args, **kw):
        calls.append(xi)
        if len(calls) == 1:
            raise RuntimeError('unresolved tail')
        return 1., 0., dict(converged=True)
    monkeypatch.setattr(no, 'integrate_kernel', kernel)
    path = tmp_path/'row.npz'
    beta = .1
    args = ([7., 8., 9.], beta, 1/np.sqrt(1-beta**2), None)
    with pytest.raises(RuntimeError, match='unresolved polarization losses'):
        pol.converged_kernel(*args, interaction_charge=1., checkpoint_path=path)
    with np.load(path) as saved:
        assert saved['done'].tolist() == [False, True, True]
    pol.converged_kernel(*args, interaction_charge=1., checkpoint_path=path)
    assert len(calls) == 4
    assert not path.with_suffix('.failures.json').exists()


def test_plot_failure_is_nonfatal(monkeypatch, tmp_path):
    from physics.inelastic_dielectric.polarization import plot_correction as plot
    def fail(*args):
        raise RuntimeError('missing font')
    monkeypatch.setattr(plot, 'plot_correction', fail)
    assert plot.plot_optional(tmp_path/'in.npz', tmp_path/'plot.pdf') is False
    assert 'missing font' in (tmp_path/'plot.status.json').read_text()


def test_successful_but_inconsistent_direct_solve_is_rechecked(monkeypatch):
    from physics.inelastic_dielectric.polarization import regularized_encounter as lc
    direct = iter((12.189578836,12.589507151,12.589507151))
    monkeypatch.setattr(no, 'encounter', lambda *a, **k:
                        dict(difference=next(direct),full=13.6,leading=1.))
    calls = []
    def regularized(*args, **kwargs):
        calls.append(kwargs)
        return dict(difference=12.5895071,full=13.6,leading=1.)
    monkeypatch.setattr(lc, 'encounter', regularized)
    def quad(f, *args, **kwargs):
        values = f(0.)
        assert values[1] == 0.
        return values, 0., SimpleNamespace(success=True)
    monkeypatch.setattr(no, 'quad_vec', quad)
    _, _, report = no.integrate_kernel(.01,3.,1.,no.PointField(1.))
    assert report['converged']
    assert report['regularized_encounters'] == 3
    assert len(calls) == 3
