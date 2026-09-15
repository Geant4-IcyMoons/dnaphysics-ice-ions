"""Analytical/synthetic tests, not numerical collision validation."""
import itertools
import json
import struct

import numpy as np
import pytest

from physics.low_energy.RT_TDDFT.capture import (
    analyze_manifest, number_distribution, orbital_overlap, read_obf,
)


def write_obf(path, values, endian='<'):
    values = np.asarray(values, dtype=endian+'c16')
    header = struct.pack(endian+'7sBIfQdQI5I', b'pulpo\0\0', 0, 1, 1., 1, 1., len(values), 3, *([0]*5))
    path.write_bytes(header + values.tobytes())


def test_exact_channels_and_mean_are_different():
    p = number_distribution([np.diag([.5]), np.diag([.5])])
    np.testing.assert_allclose(p, [.25,.5,.25])
    assert np.dot(np.arange(3),p) == 1
    np.testing.assert_allclose(number_distribution([np.diag([1,0,0,0])]), [0,1,0,0,0])


def test_full_determinants_and_unitary_invariance():
    rng=np.random.default_rng(315)
    q,_=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))
    overlap=q @ np.diag([.1,.3,.6,.9]) @ q.conj().T
    reference=np.zeros(5,complex)
    # Direct row-permutation determinant expansion of det(I-S+zS).
    for selected in itertools.product([False,True],repeat=4):
        matrix=np.array([overlap[i] if selected[i] else (np.eye(4)-overlap)[i] for i in range(4)])
        reference[sum(selected)]+=np.linalg.det(matrix)
    p=number_distribution([overlap])
    np.testing.assert_allclose(p,reference.real,atol=1e-14)
    np.testing.assert_allclose(p,number_distribution([np.diag([.1,.3,.6,.9])]),atol=1e-14)
    assert not np.allclose(p,number_distribution([np.diag(np.diag(overlap))]))
    assert p.sum()==pytest.approx(1)
    assert np.dot(np.arange(5),p)==pytest.approx(np.trace(overlap).real)


@pytest.mark.parametrize('matrix',[np.array([[1.1]]),np.array([[-.1]]),np.array([[np.nan]]),
    np.array([[.5,.1],[0,.5]]),np.array([1,2])])
def test_bad_overlap_is_rejected(matrix):
    with pytest.raises(ValueError): number_distribution([matrix])


def test_cap_loss_is_not_renormalized():
    np.testing.assert_allclose(number_distribution([np.eye(2)*.2]), [.64,.32,.04])


@pytest.mark.parametrize('endian',['<','>'])
def test_native_complex_binary_and_overlap(tmp_path,endian):
    a=np.array([1,1j,0,0])/np.sqrt(2)
    b=np.array([1,-1j,0,0])/np.sqrt(2)
    files=[tmp_path/'a.obf',tmp_path/'b.obf']
    for f,v in zip(files,[a,b]):write_obf(f,v,endian)
    np.testing.assert_allclose(read_obf(files[0]),a)
    points=np.zeros((4,3))
    np.testing.assert_allclose(orbital_overlap(files,points,1,chunk_size=1),np.eye(2),atol=1e-15)
    region=np.array([True,False,False,False])
    np.testing.assert_allclose(orbital_overlap(files,points,1,region),np.ones((2,2))*.5)
    files[0].write_bytes(files[0].read_bytes()[:-1])
    with pytest.raises(ValueError,match='payload'):read_obf(files[0])


def synthetic_export(tmp_path):
    points=np.array([p for p in itertools.product(range(-2,3),repeat=3) if np.linalg.norm(p)<=2])
    np.savetxt(tmp_path/'mesh',np.column_stack([np.arange(1,len(points)+1),points]))
    frames=[]
    for label, scale in [('initial',1),('late1',.5),('late2',.5)]:
        names={'up':[],'down':[]}
        for spin in names:
            for i in range(4):
                values=np.zeros(len(points),complex);values[i]=scale
                name=f'{label}_{spin}_{i}.obf';write_obf(tmp_path/name,values);names[spin].append(name)
        frames.append({'orbitals':names,'time_au':len(frames)*10,
            'projectile_position_bohr':[0,0,0],'target_positions_bohr':[[40,0,0],[40,1,0],[40,0,1]]})
    config={'schema_version':1,'units':'atomic','projectile':'H+','active_electrons':8,
        'collision':{'energy_ev_total':1000,'orientation':'a','impact_parameter_bohr':1.},
        'mesh':'mesh','spacing_bohr':[1,1,1],'initial_orthogonality_tolerance':1e-8,
        'region':'projectile_centered_spherical_box','region_radius_bohr':2,
        'minimum_separation_bohr':40,'stationarity_tolerance':1e-4,'initial':frames[0],'frames':frames[1:]}
    path=tmp_path/'analysis.json';path.write_text(json.dumps(config));return path,config


def test_export_to_counting_end_to_end(tmp_path):
    path,_=synthetic_export(tmp_path)
    result=analyze_manifest(path)
    assert result['stationarity_passed']
    assert result['frames'][-1]['p_one_electron']==pytest.approx(8*.25*.75**7)
    assert result['frames'][-1]['mean_electrons']==pytest.approx(2)
    assert result['status']=='analyzed_not_literature_validated'


def test_reject_incoming_target_in_capture_box(tmp_path):
    path,config=synthetic_export(tmp_path)
    config['frames'][0]['target_positions_bohr'][0]=[0,0,1]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='separated'): analyze_manifest(path)


def test_curve_uses_probability_not_mean_and_requires_stationarity(tmp_path):
    from physics.low_energy.RT_TDDFT.capture_curve import collect,compare
    path,_=synthetic_export(tmp_path)
    result=analyze_manifest(path)
    output=tmp_path/'result.json';output.write_text(json.dumps(result))
    rows=collect([output])
    assert rows[0]['two_pi_b_p1_bohr']==pytest.approx(2*np.pi*8*.25*.75**7)
    reference=tmp_path/'reference.csv'
    reference.write_text('orientation,impact_parameter_bohr,two_pi_b_p1_bohr,provenance\na,1,0.5,synthetic unit test\n')
    assert compare(rows,reference)[0]['residual_bohr']==pytest.approx(rows[0]['two_pi_b_p1_bohr']-.5)
    with pytest.raises(ValueError,match='Duplicate'):collect([output,output])
    result['stationarity_passed']=False;output.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='stationarity'):collect([output])
