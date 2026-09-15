"""Failure witnesses distinguish accepted trajectories from rejected RK stages."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from physics.inelastic_dielectric.polarization import nonlinear_oscillator as no


def test_failed_solver_records_accepted_separation(monkeypatch, tmp_path):
    def solve(rhs, interval, initial, **kwargs):
        rhs(0., np.zeros(8))
        return SimpleNamespace(success=False, message='step too small',
                               t=np.array([-1.,0.]), y=np.zeros((8,2)))
    monkeypatch.setattr(no, 'solve_ivp', solve)
    monkeypatch.setenv('ICE_ENCOUNTER_FAILURE_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='step too small'):
        no.encounter(.3, .5, .1, no.PointField(1.), regularize=False)
    report = json.loads(next(tmp_path.glob('encounter_*.json')).read_text())
    assert report['minimum_accepted_separation_bohr'] == .5
    assert report['minimum_trial_separation_bohr'] == .5
    assert report['last_accepted_tau'] == 0.
    assert report['accepted_steps'] == 2
