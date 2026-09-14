"""Compute-node integration smoke test, not physical qualification."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from physics.elastic.handoff.study import prepare,DEFAULT_CONFIG
from physics.elastic.handoff.kernel import HandoffKernel
from physics.elastic.handoff.binary import moments
from physics.elastic.bca.scattering import NLHCollisionKernel


def main():
    for target in ('H','O'):
        k=HandoffKernel(minimum_transfer_ev=1e-4)
        b=k.boundary(target,1e4)
        for impact in (b*.5,b*1.01):
            result=k.collide('C',target,1e4,impact)
            assert abs(result.energy_conservation_error_ev)<1e-8
            assert 0<=result.recoil_energy_ev<=1e4
            if impact<b:
                ref=NLHCollisionKernel('C',target,1e4).solve(impact)
                assert abs(result.theta_cm_rad-ref.theta_cm_rad)<1e-5
        fine=HandoffKernel(minimum_transfer_ev=1e-4,order=128)
        assert abs(k.zbl_angle(target,1e4,b*1.01)-fine.zbl_angle(target,1e4,b*1.01))<1e-5
    coarse=moments(HandoffKernel(minimum_transfer_ev=1e-4), 'O', 1e4, 4)
    fine=moments(HandoffKernel(minimum_transfer_ev=1e-4,order=128), 'O', 1e4, 8)
    assert all(a>0 and b>0 and abs(a-b)/b<.01 for a,b in zip(coarse,fine))
    config=json.loads(DEFAULT_CONFIG.read_text());config['energies_ev']=[10000];config['path_length_angstrom']=2.
    manifest=prepare(config)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);study=root/'study.json';study.write_text(json.dumps(manifest))
        for phase in config['phases']:
            index=next(i for i,c in enumerate(manifest['cohorts']) if c['phase']==phase)
            command=[sys.executable,'-B','-m','physics.elastic.handoff.run','run',str(study),'--case-index',str(index),'--output',str(root/'results'),'--cutoff-ev','1e-4','--workers','2','--block-size','2','--min-histories','2','--max-histories','2']
            subprocess.run(command,check=True)
            before={str(p):p.read_bytes() for p in root.glob('results/*/*/block*')}
            subprocess.run(command,check=True)
            assert before=={str(p):p.read_bytes() for p in root.glob('results/*/*/block*')}
        subprocess.run([sys.executable,'-B','-m','physics.elastic.handoff.run','reduce','--output',str(root/'results')],check=True)
    print('PASS: both phases, orbit refinement, branch selection, conservation and restart.')

if __name__=='__main__':main()
