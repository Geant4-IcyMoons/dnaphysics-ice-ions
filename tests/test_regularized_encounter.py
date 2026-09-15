import numpy as np
import pytest
from physics.inelastic_dielectric.polarization import nonlinear_oscillator as no
from physics.inelastic_dielectric.polarization import regularized_encounter as lc


def test_initial_transform():
    r,p = np.array([.2,-3.]),np.array([.3,.5])
    u,up,h = lc.initial_lc(r,p,2.)
    np.testing.assert_allclose(lc.lc_position(u),r,rtol=1e-14)
    np.testing.assert_allclose(2*lc.lc_matrix(u)@up/np.dot(u,u),p,rtol=1e-14)
    assert h == pytest.approx(.5*np.dot(p,p)-2/np.linalg.norm(r))


def test_regularized_vs_original_weak_encounter():
    field = no.PointField(1.)
    reference = no.encounter(.3,1.,.01,field,tail=4.,rtol=2e-10)
    actual = lc.encounter(.3,1.,.01,field,tail=4.,rtol=2e-10)
    assert actual['full'] == pytest.approx(reference['full'],rel=2e-6)
    assert actual['difference'] == pytest.approx(reference['difference'],rel=2e-5)
    assert abs(actual['kepler_energy_residual']) < 1e-7


def test_transformed_acceleration_preserves_coulomb_force():
    r,p,F = np.array([.2,-.3]),np.array([.4,.1]),np.array([.1,.6])
    mu = .7
    u,up,h = lc.initial_lc(r,p,mu)
    rho = np.dot(u,u)
    upp = .5*h*u+.5*rho*lc.lc_matrix(u).T@F
    acceleration = ((2*lc.lc_matrix(up)@up+2*lc.lc_matrix(u)@upp)/rho
                    -(2*lc.lc_matrix(u)@up)*(2*np.dot(u,up))/rho**2)/rho
    np.testing.assert_allclose(acceleration,-mu*r/rho**3+F,rtol=1e-13)


def test_screened_weak_encounter():
    field = no.frozen_field('H',0)
    reference = no.encounter(.3,.5,.05,field,tail=4.,rtol=2e-10)
    actual = lc.encounter(.3,.5,.05,field,tail=4.,rtol=2e-10)
    assert actual['difference'] == pytest.approx(reference['difference'],rel=2e-5)


def test_witnessed_h0_close_encounter_recovers_without_softening():
    result = no.encounter(.3720893743894298,.2986193522410363,.8370879566801157,
        no.frozen_field('H',0),gamma=1.0001065208808195,tail=10.750105419064154,rtol=2e-8)
    assert result['solver'] == 'levi-civita-fallback'
    assert result['difference'] == pytest.approx(10.8762022,rel=2e-6)
def test_comparison_checkpoint_and_direct_selection(monkeypatch, tmp_path):
    from physics.inelastic_dielectric.jobs import regularized_probe as probe
    calls = []
    def direct(*args, **kwargs):
        calls.append(kwargs)
        return dict(difference=1.)
    monkeypatch.setattr(probe.no, 'encounter', direct)
    point = dict(loss_eV=1., x=1., b_bohr=1., eta=.1, gamma=1., tail=4.)
    task = (point, 'direct', 2e-9, tmp_path, 'test-source')
    assert probe.compare(task)['result']['finished']
    assert probe.compare(task)['result']['difference'] == 1.
    assert len(calls) == 1
    assert calls[0]['regularize'] is False
