"""Test physical geometry preservation and input provenance without Octopus."""
from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from physics.low_energy.RT_TDDFT.prepare import extract_cluster, prepare, registered_structure
from physics.low_energy.RT_TDDFT.run import run
from physics.constants import EH, PROTON_MASS_AU


def args(output, phase='amorphous_lda_80k'):
    return Namespace(phase=phase, replica=0, center=0, molecules=5,
                     spacing=0.3, padding=6., dt_au=0.02, steps=10,
                     energy_ev=1000., impact=1., separation=15., output=output)


@pytest.mark.parametrize('phase', ['amorphous_lda_80k','hexagonal_ih_100k'])
def test_registered_cluster_preserves_bonds_and_is_nested(phase):
    structure, _ = registered_structure(phase, 0)
    ids, xyz, _ = extract_cluster(structure, 0, 5)
    larger, _, _ = extract_cluster(structure, 0, 10)
    assert np.array_equal(ids, larger[:5])
    assert ids[0] == 0
    assert np.allclose(xyz[0], 0)
    raw = structure.positions_angstrom.reshape(-1,3,3)[ids]
    bonds = raw[:,1:] - raw[:,:1]
    lengths = np.diag(structure.lattice_angstrom)
    bonds -= lengths * np.floor(bonds / lengths + .5)
    assert np.allclose(xyz.reshape(-1,3,3)[:,1:] - xyz.reshape(-1,3,3)[:,:1], bonds, atol=1e-12)


def test_boundary_crossing_water():
    structure, _ = registered_structure('amorphous_lda_80k', 0)
    fake = replace(structure, species=np.array(['O','H','H']),
                   positions_angstrom=np.array([[9.8,5,5],[.7,5,5],[9.8,5.9,5]]),
                   lattice_angstrom=np.eye(3)*10)
    _, xyz, _ = extract_cluster(fake,0,1)
    assert np.allclose(xyz, [[0,0,0],[.9,0,0],[0,.9,0]])
    with pytest.raises(ValueError, match='orthorhombic'):
        extract_cluster(replace(fake,lattice_angstrom=np.array([[10,1,0],[0,10,0],[0,0,10]])),0,1)


@pytest.mark.parametrize('phase', ['amorphous_lda_80k','hexagonal_ih_100k'])
def test_prepare_and_provenance(tmp_path, phase):
    case = prepare(args(tmp_path/'case',phase))
    manifest = json.loads((case/'manifest.json').read_text())
    velocity = manifest['collision']['velocity_au'][2]
    assert .5*PROTON_MASS_AU*velocity**2*EH == pytest.approx(1000.)
    assert manifest['collision']['runnable'] is False
    assert not Path(manifest['source']['path']).is_absolute()
    assert not Path(manifest['registry']).is_absolute()
    assert manifest['source']['path'].startswith('models/ice/')
    assert int((case/'cluster.xyz').read_text().splitlines()[0]) == 15
    assert 'CalculationMode = gs' in (case/'gs.inp').read_text()
    assert 'MoveIons = no' in (case/'control_td.inp').read_text()
    with pytest.raises(FileExistsError):
        prepare(args(case,phase))


@pytest.mark.parametrize('field,value', [('spacing',float('nan')),('dt_au',0),('energy_ev',-1),('center',-1),('molecules',0),('separation',.1)])
def test_invalid_request_no_output(tmp_path,field,value):
    request = args(tmp_path/'case')
    setattr(request,field,value)
    with pytest.raises(ValueError):
        prepare(request)
    assert not request.output.exists()


def test_missing_solver_no_output(tmp_path):
    with pytest.raises(FileNotFoundError,match='unavailable'):
        run(tmp_path/'case',tmp_path/'run','octopus_nonexistent_test_binary')
    assert not (tmp_path/'run').exists()


def test_runner_receipt_and_input_tamper(tmp_path):
    # This checks process orchestration only, not Octopus syntax or physics.
    binary = tmp_path/'fake_octopus'
    binary.write_text('#!/bin/sh\necho fake-test-executable\nexit 0\n')
    binary.chmod(0o755)
    case = prepare(args(tmp_path/'case'))
    run(case,tmp_path/'run',str(binary))
    receipt = json.loads((tmp_path/'run'/'receipt.json').read_text())
    assert receipt['status'] == 'executed_requires_numerical_review'
    assert receipt['physical_validation'] is False
    with (case/'gs.inp').open('a') as f:
        f.write('\n# changed\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        run(case,tmp_path/'other',str(binary))
    assert not (tmp_path/'other').exists()
