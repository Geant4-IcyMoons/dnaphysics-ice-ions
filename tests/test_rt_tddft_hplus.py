from types import SimpleNamespace
import numpy as np
import pytest
from physics.low_energy.RT_TDDFT.hplus import inputs, read_cube, density_populations, BOHR_ANGSTROM


def test_cube_integral_and_sphere(tmp_path):
    cube = tmp_path / 'density.cube'
    cube.write_text('test\ntest\n0 0 0 0\n2 1 0 0\n1 0 1 0\n1 0 0 1\n2 3\n')
    points, values, volume = read_cube(cube)
    assert volume == 1
    assert points[1, 0] == pytest.approx(BOHR_ANGSTROM)
    population = density_populations(cube, np.zeros(3), (.1, 1.))
    assert population['electron_count_on_grid'] == 5
    assert population['projectile_sphere_electrons'] == {'0.1': 2., '1.0': 5.}


def test_incoming_state_and_flight():
    args = SimpleNamespace(energy_ev=1000., separation=8., transverse_half=6.,
        clearance=4., cap_width=2., spacing=.4, impact=1., cap_height=.2)
    gs, td, flight = inputs(args, np.zeros((3,3)), .02)
    assert 'ExcessCharge = 0' in gs
    assert 'ExcessCharge = 1' in td
    assert gs.count('| no') == td.count('| no') == 3
    assert '| yes' not in gs and td.count('| yes') == 1
    assert 'RecalculateGSDuringEvolution = no' in td
    assert 8 <= flight['end_z_angstrom'] < 8.01
    assert flight['velocity_au'] == pytest.approx(.2000715, rel=1e-6)


def test_cluster_atom_and_velocity_counts():
    args = SimpleNamespace(energy_ev=1000., separation=8., transverse_half=8.,
        clearance=4.1, cap_width=2., spacing=.4, impact=1., cap_height=.2)
    gs, td, flight = inputs(args, np.zeros((15,3)), .04)
    assert gs.count('| no') == td.count('| no') == 15
    velocities = td.split('%Velocities\n')[1].split('%')[0].strip().splitlines()
    assert len(velocities) == 16
    assert flight['steps'] % 8 == 0
