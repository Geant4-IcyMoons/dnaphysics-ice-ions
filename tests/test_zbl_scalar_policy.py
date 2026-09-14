"""Statistical contract tests; no physical collision integration."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics.elastic.bca.convergence import RatioStatistics, OBSERVABLES, relative_standard_error_report, progressive_schedule
from physics.elastic.zbl.run_campaign import check_scalar_report


def sample(values, denominator):
    s=RatioStatistics()
    for a,b in zip(values,denominator): s.add(float(a),float(b))
    return {name:s for name in OBSERVABLES}


def test_ratio_se_and_not_confidence_interval():
    x=np.tile([.8,1.2],100); y=np.ones(len(x))
    r=relative_standard_error_report(sample(x,y),.05,1,3)
    assert r['converged'] and not r['distribution_qualified']
    v=r['observables'][OBSERVABLES[0]]
    assert v['standard_error']==pytest.approx(x.std(ddof=1)/np.sqrt(len(x)))
    assert v['confidence_half_width'] is None
    check_scalar_report(r,{'statistical_relative_tolerance':.05})
    v['standard_error']=.2
    with pytest.raises(RuntimeError): check_scalar_report(r,{'statistical_relative_tolerance':.05})


@pytest.mark.parametrize('values',[[0]*10,[1]*10])
def test_zero_or_zero_variance_not_qualified(values):
    assert not relative_standard_error_report(sample(values,[1]*10),.05,1,1)['converged']


def test_schedule_includes_final_partial_budget():
    assert progressive_schedule(128,250,8)==(128,160,192,232,250)


def test_cli_accepts_scalar_and_rejects_mixed_gate(monkeypatch):
    from physics.elastic import simulate as sim
    argv=['sim','dummy.xyz','--projectile','C','--interaction-model','zbl_soft','--minimum-transfer-ev','1e-5',
          '--statistical-relative-tolerance','.05','--energy-ev','10000']
    monkeypatch.setattr(sys,'argv',argv)
    args=sim.parse_args(); sim._validate_args(args)
    args.trajectory_cdf_tolerance=.005
    with pytest.raises(ValueError,match='Choose scalar'):sim._validate_args(args)


def test_cutoff_comparison_reports_no_handoff_certificate():
    from physics.elastic.zbl.run_campaign import cutoff_comparison
    x=np.tile([.8,1.2],100)
    r=relative_standard_error_report(sample(x,np.ones(len(x))),.05,1,1)
    result=cutoff_comparison(('ice',0,'isotropic',10000),[(1e-5,r),(1e-4,r)])
    assert all(v['passes'] for v in result['comparisons'][0]['observables'].values())
    assert 'hard_collision_rate_per_angstrom' not in result['comparisons'][0]['observables']


def test_reducer_exports_se_without_cdf_gate(tmp_path, monkeypatch):
    import json
    from physics.elastic.zbl import run_campaign as rc
    x=np.tile([.8,1.2],100)
    r=relative_standard_error_report(sample(x,np.ones(len(x))),.05,1,1)
    cases=[]
    for i,cut in enumerate((1e-4,1e-5)):
        output=tmp_path/str(i); output.mkdir()
        manifest={'statistical_convergence':r,'structure':{'water_molecules':1,'volume_angstrom3':1},
                  'configuration':{'completed_trajectories':len(x)}}
        (output/'zbl_soft_collision_run.manifest.json').write_text(json.dumps(manifest))
        cases.append({'output_directory':str(output),'phase_id':'ice','structure_index':0,
                      'orientation':'isotropic','energy_ev':10000.,'minimum_transfer_ev':cut})
    campaign={'configuration_signature':'synthetic','configuration':{'interaction_model':'zbl_soft',
              'projectile':'C','statistical_relative_tolerance':.05},'cases':cases}
    monkeypatch.setattr(rc,'_campaign',lambda path:campaign)
    monkeypatch.setattr(rc,'_distribution_samples',lambda *a:np.array([0.,.1,.2]))
    cross, angular, final=rc.reduce_campaign(tmp_path/'campaign.json')
    assert 'sigma_standard_error_angstrom2' in cross.read_text()
    assert 'sigma_half_width_angstrom2' not in cross.read_text()
    result=json.loads(final.read_text())
    assert not result['distribution_qualified'] and not result['handoff_qualified'] and not result['soft_cutoff_qualified']
    assert len(result['cutoff_comparisons'])==1
