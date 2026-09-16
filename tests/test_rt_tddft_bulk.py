"""Input and optional solver checks; these do not validate stopping physics."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from physics.low_energy.RT_TDDFT.bulk import prepare, trajectory, analyze, BOHR_ANGSTROM, EH


def settings():
    return dict(initialization='relaxed_charged_cell', energy_ev=20000,
                start_fractional=[.15, .37, .43], direction_cartesian=[1, 0, 0],
                length_angstrom=2., spacing_angstrom=.5, dt_au=.02, output_every=2)


def fixture_files(tmp_path, cell=None):
    cell = np.diag([8., 8., 8.]) if cell is None else cell
    source = tmp_path / 'water.xyz'
    # Deliberately not OHH order; this is a diagnostic cell, not validated ice.
    source.write_text('3\nLattice="' + ' '.join(map(str, np.asarray(cell).ravel())) +
                      '" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
                      'H 4.7 4.5 4\nO 4 4 4\nH 3.3 4.5 4\n')
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps(settings()))
    pseudos = tmp_path / 'pseudo'
    pseudos.mkdir()
    for element in ('H', 'O'):
        (pseudos / f'{element}.upf').write_text(
            f'<UPF><PP_HEADER element="{element}" functional="PBE" pseudo_type="NC" '
            f'has_so="F" z_valence="{1 if element == "H" else 6}"/></UPF>')
    return source, config, pseudos


def test_periodic_bundle_preserves_all_atoms_and_cell(tmp_path):
    cell = np.array([[8., 0, 0], [1, 8, 0], [0, 0, 8]])
    source, config, pseudos = fixture_files(tmp_path, cell)
    out = tmp_path / 'case'
    manifest = prepare(source, config, pseudos, out, diagnostic=True)
    assert manifest['source']['water_molecules'] == 1
    assert manifest['source']['collision_ready'] is False
    assert manifest['valence_electrons'] == 8
    np.testing.assert_allclose(manifest['source']['lattice_angstrom'], cell)
    td = (out / 'td.inp').read_text()
    assert 'PeriodicDimensions = 3' in td
    assert 'AbsorbingBoundaries' not in td
    assert td.count('| no') == 3
    assert td.count('| yes') == 1
    assert manifest['files_sha256']['pseudos/O.upf']
    with pytest.raises(FileExistsError):
        prepare(source, config, pseudos, out, diagnostic=True)


def test_unaccepted_structure_rejected_by_default(tmp_path):
    source, config, pseudos = fixture_files(tmp_path)
    with pytest.raises(ValueError):
        prepare(source, config, pseudos, tmp_path / 'case')
    assert not (tmp_path / 'case').exists()


def test_trajectory_units_and_containment():
    config = settings()
    cell = np.diag([8., 8., 8.])
    path = trajectory(config, cell)
    np.testing.assert_allclose((np.array(path['end_fractional']) - path['start_fractional']) @ cell,
                               [path['length_angstrom'], 0, 0])
    assert 0 < path['length_angstrom'] <= config['length_angstrom']
    assert path['steps'] % config['output_every'] == 0
    config['length_angstrom'] = 20
    with pytest.raises(ValueError, match='leaves the cell'):
        trajectory(config, cell)


@pytest.mark.parametrize('key,value', [('dt_au', 0), ('energy_ev', float('nan')),
                                     ('output_every', 2.5), ('direction_cartesian', [0, 0, 0]),
                                     ('initialization', 'bare_proton'), ('unknown', 1)])
def test_invalid_settings(key, value):
    config = settings()
    config[key] = value
    with pytest.raises(ValueError):
        trajectory(config, np.eye(3) * 8)


@pytest.mark.parametrize('skew', [0., 1.])
def test_octopus_periodic_execution(tmp_path, skew):
    executable = os.environ.get('OCTOPUS_TEST_EXECUTABLE')
    if not executable:
        pytest.skip('Set OCTOPUS_TEST_EXECUTABLE for a short solver check')
    source, config, _ = fixture_files(tmp_path, np.array([[8., 0, 0], [skew, 8., 0], [0, 0, 8.]]))
    values = settings()
    values.update(length_angstrom=.001, output_every=1, dt_au=.001)
    config.write_text(json.dumps(values))
    pseudo = Path(executable).resolve().parent.parent / 'share/octopus/pseudopotentials/pseudo-dojo.org/nc-sr-05_pbe_standard'
    case = tmp_path / 'case'
    prepare(source, config, pseudo, case, diagnostic=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for stage in ('gs', 'td'):
        shutil.copyfile(case / f'{stage}.inp', case / 'inp')
        result = subprocess.run([executable], cwd=case, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, result.stdout[-8000:]
    energy = np.loadtxt(case / 'td.general/energy')
    assert np.isfinite(energy).all()
    assert np.atleast_2d(energy)[-1, 0] >= 1
    coordinates = np.loadtxt(case / 'td.general/coordinates')
    np.testing.assert_allclose(coordinates[:, 2:11], np.tile(coordinates[0, 2:11], (len(coordinates), 1)))
    manifest = json.loads((case / 'manifest.json').read_text())
    expected = coordinates[:, 1, None] * np.asarray(manifest['trajectory']['velocity_au'])
    np.testing.assert_allclose(coordinates[:, 11:14] - coordinates[0, 11:14], expected, atol=1e-12)


def test_energy_slope_and_incomplete_run(tmp_path):
    source, config, pseudo = fixture_files(tmp_path)
    case = tmp_path / 'case'
    manifest = prepare(source, config, pseudo, case, diagnostic=True)
    n = manifest['trajectory']['steps']
    steps = np.arange(0, n + 1, 2)
    times = steps * settings()['dt_au']
    distances = times * manifest['trajectory']['speed_au'] * BOHR_ANGSTROM
    energies = -10 + 3 * distances / EH
    (case / 'td.general').mkdir()
    energy = case / 'td.general/energy'
    np.savetxt(energy, np.column_stack([steps, times, energies]))
    result = analyze(case, distances[1], distances[-1])
    assert result['finite_path_energy_slope_ev_per_angstrom'] == pytest.approx(3)
    np.savetxt(energy, np.column_stack([steps, times, energies])[:-1])
    with pytest.raises(ValueError, match='Incomplete'):
        analyze(case, 0, distances[-2])
