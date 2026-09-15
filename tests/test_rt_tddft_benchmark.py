"""Frame-transform mathematics and the independently specified collision inputs."""
from pathlib import Path
import os
import shutil

import numpy as np
import pytest

from physics.low_energy.RT_TDDFT.benchmark import Settings, ground_input, td_input, water_geometry, run
from physics.low_energy.RT_TDDFT.capture import read_obf
from physics.low_energy.RT_TDDFT.frame import translate_orbital, write_obf


def test_complex_native_roundtrip(tmp_path):
    values = np.array([1+2j, 0.01-0.3j, -2j])
    path = tmp_path/'orbital.obf'
    write_obf(path, values)
    np.testing.assert_array_equal(read_obf(path), values)
    with pytest.raises(FileExistsError): write_obf(path, values)


def test_translation_and_boost_of_moving_gaussian():
    # A wavepacket moving at v becomes real and stationary in its own frame.
    h = 0.2
    points = np.stack(np.meshgrid(*([np.arange(-5, 5.01, h)]*3), indexing='ij'), axis=-1).reshape(-1, 3)
    position = np.array([0.37, -0.18, 0.11])
    velocity = np.array([0.2, 0.1, -0.15])
    source = 0.4*np.exp(-np.sum((points-position)**2, axis=1)/2)*np.exp(1j*((points-position)@velocity))
    result = translate_orbital(source, points, points, h, position, velocity)
    expected = 0.4*np.exp(-np.sum(points**2, axis=1)/2)
    assert np.max(np.abs(result-expected)) < 1e-5
    # A fractional occupation is retained; no normalization is hidden in the transform.
    assert abs(np.vdot(result, result).real*h**3 - np.vdot(expected, expected).real*h**3) < 2e-5


def test_integer_translation_is_exact_and_crop_loses_norm():
    points = np.indices((5,5,5)).reshape(3,-1).T.astype(float)
    values = np.arange(125).astype(complex) * (1+1j)
    result = translate_orbital(values, points, points, 1., [1,0,0], [0,0,0])
    expected = np.zeros((5,5,5), complex)
    expected[:-1] = values.reshape(5,5,5)[1:]
    np.testing.assert_array_equal(result, expected.ravel())
    assert np.vdot(result, result).real < np.vdot(values, values).real


def test_explicit_water_and_collision_inputs():
    xyz = water_geometry()
    np.testing.assert_allclose(np.linalg.norm(xyz[1:], axis=1), 1.809934, atol=1e-6)
    assert np.all(xyz[:,2] == 0) and np.all(xyz[1:,0] > 0)
    settings = Settings()
    settings.validate()
    gs = ground_input(settings, Path('/pseudo'))
    assert 'lda_x + lda_c_pz' in gs and 'SpinComponents = polarized' in gs
    assert gs.count(' | yes') == 3 and 'ExcessCharge = 0' in gs
    positions = np.vstack((xyz, [-15,2,0]))
    td = td_input(settings, Path('/pseudo'), positions, np.zeros((4,3)), 2, 1)
    assert td.count(' | yes') == 4
    assert 'ExcessCharge = 1' in td and 'TDPropagator = etrs' in td
    assert 'IonsConstantVelocity' not in td
    files = {s: [Path(f'/{s}{i}.obf') for i in range(4)] for s in ('up','down')}
    handoff = td_input(settings, Path('/pseudo'), positions, np.zeros((4,3)), 2, 1, sphere=True, files=files)
    assert handoff.count('normalize_no') == 8
    assert 'OnlyUserDefinedInitialStates = yes' in handoff
    with pytest.raises(ValueError): Settings(minimum_separation=20).validate()


@pytest.mark.skipif(not os.environ.get('OCTOPUS_TEST_EXECUTABLE'), reason='Set OCTOPUS_TEST_EXECUTABLE for the real solver integration test')
def test_octopus_collision_to_capture(tmp_path):
    """Reduced resolution verifies execution only, not numerical convergence."""
    executable = Path(shutil.which(os.environ['OCTOPUS_TEST_EXECUTABLE'])).resolve()
    settings = Settings(spacing=.8, dt=.1, initial=8., flight_end=16., minimum_separation=12.,
                        radius=8., cap_width=2., cap_height=.2, clearing_time=10., threads=2)
    result = run(settings, tmp_path/'case', str(executable), executable.parents[1]/'share/octopus/pseudopotentials/PSF')
    assert result['status'] == 'executed_not_numerically_validated'
    assert result['initial_orthogonality_error'] < 1e-6
    assert result['neutral_target_restart_max_amplitude_error'] < 1e-10
    assert result['handoff']['import_overlap_error'] < 1e-10
    assert result['handoff']['retained_electron_norm'] < result['handoff']['source_electron_norm']
    import json
    capture = json.loads((tmp_path/'case/capture.json').read_text())
    for frame in capture['frames']:
        probabilities = frame['probabilities_by_electron_count']
        assert len(probabilities) == 9 and min(probabilities) >= 0
        assert abs(sum(probabilities)-1) < 1e-12
