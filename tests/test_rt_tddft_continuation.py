"""Continuation preserves physical input and updates only restart controls."""
import pytest
from physics.low_energy.RT_TDDFT.continue_clearing import restart_input


def test_restart_input_preserves_physics():
    text = 'FromScratch = yes\nTDMaxSteps = 8000\nTDTimeStep = 0.025\nOutputInterval = 2000\n'
    assert restart_input(text, 16000) == 'FromScratch = no\nTDMaxSteps = 16000\nTDTimeStep = 0.025\nOutputInterval = 2000\n'
    with pytest.raises(ValueError):
        restart_input('TDMaxSteps = 8000\n', 16000)
