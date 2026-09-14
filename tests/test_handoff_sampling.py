"""Estimator and terminal-track tests without physical trajectory integration."""
from types import SimpleNamespace
import numpy as np
import pytest
from physics.elastic.handoff.run import statistics, summarize_history
from physics.elastic.handoff.kernel import HandoffKernel


class Kernel:
    def energy_bounds_ev(self,*args): return (1., 1e8)
    def branch(self,*args): return 'nlh'


def result(termination='energy_below_kernel_table', final=.6):
    return SimpleNamespace(termination=termination, final_energy_ev=final,
        traveled_path_length_angstrom=7., recoil_energy_ev=999.4,
        events=(), ambiguous_event_count=0)


def test_terminal_track_is_retained_without_inventing_deposition():
    row=summarize_history(result(),2.,Kernel())
    assert row[0]==7 and row[1]==999.4
    assert row[6:]==[2.,.6,1]
    with pytest.raises(RuntimeError,match='Incomplete'):
        summarize_history(result('maximum_collisions'),1.,Kernel())
    with pytest.raises(RuntimeError,match='contradicts'):
        summarize_history(result(final=2.),1.,Kernel())


def test_weighted_track_ratio_and_centered_se():
    rows=[[2,3,.1,2,1,0,.5,0,0],[4,9,.4,4,3,0,2.,.6,1],
          [5,4,.2,1,1,0,1.3,0,0]]
    report=statistics([{'rows':rows}],.05)
    a=np.asarray(rows);w=a[:,6];x=w*a[:,1];y=w*a[:,0]
    mu=x.sum()/y.sum();se=np.std(x-mu*y,ddof=1)/np.sqrt(len(a))/y.mean()
    assert report['observables']['stopping']['mean']==pytest.approx(mu)
    assert report['observables']['stopping']['standard_error']==pytest.approx(se)
    assert report['residual_energy_per_tracked_length_ev_angstrom']==pytest.approx(1.2/y.sum())
    assert report['terminated_histories']==1
    with pytest.raises(RuntimeError,match='schema'):
        statistics([{'rows':[r[:6] for r in rows]}],.05)


def test_geometric_sampling_intervals_preserve_entire_disk(monkeypatch):
    k=HandoffKernel(minimum_transfer_ev=1e-5)
    monkeypatch.setattr(k,'maximum_impact_parameter_angstrom',lambda *args:2.)
    monkeypatch.setattr(k,'boundary',lambda *args:1.)
    edges=k.area_quantile_breakpoints('C','H',1e8)
    assert edges[0]==0 and edges[-1]==1 and .25 in edges
    assert np.all(np.diff(edges)>0)
    assert np.sum(np.diff(edges)/(len(edges)-1)/np.diff(edges))==pytest.approx(1.)
    with pytest.raises(ValueError,match='Terminal'):
        HandoffKernel(minimum_transfer_ev=1e-5,terminal_energy_ev=.1)


def test_successful_blocks_survive_failure_and_resume_fills_holes(tmp_path,monkeypatch):
    import json
    from concurrent.futures import Future
    from physics.elastic.handoff import run
    case={'id':'case','seed':7}
    manifest={'configuration':{'relative_standard_error_target':.05,'path_length_angstrom':10},
              'signature':'test','cohorts':[case]}
    study=tmp_path/'study.json';study.write_text(json.dumps(manifest))
    monkeypatch.setattr(run,'prepare',lambda config:{'signature':'test'})
    submitted=[];fail=[True]
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,task):
            future=Future();start=task[1];submitted.append(start)
            if start==0 and fail[0]:future.set_exception(RuntimeError('test failure'))
            else:future.set_result({'start':start,'rows':[[10,1,.2,1,1,0,1,0,0],[10,2,.3,1,1,0,1,0,0]]})
            return future
    monkeypatch.setattr(run,'ProcessPoolExecutor',Pool)
    args=SimpleNamespace(study=study,case_index=0,cutoff_ev=1e-5,order=64,
                         terminal_energy_ev=1.,tube_fraction=.8,block_size=2,
                         output=tmp_path/'results',max_histories=4,min_histories=4,workers=2)
    with pytest.raises(RuntimeError,match='successful blocks were preserved'):run.execute(args)
    saved=list(args.output.glob('*/*/block*'))
    assert len(saved)==1 and json.loads(saved[0].read_text())['start']==2
    before=saved[0].read_bytes();fail[0]=False;submitted.clear();args.workers=1
    run.execute(args)
    assert submitted==[0] and saved[0].read_bytes()==before
    assert not list(args.output.glob('*/*/errors.json'))
