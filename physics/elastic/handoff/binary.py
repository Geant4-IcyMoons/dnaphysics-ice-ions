"""Deterministic pair-moment checks for boundary and recoil-cutoff selection."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
from physics.elastic.handoff.kernel import HandoffKernel
from physics.elastic.handoff.run import atomic
from physics.elastic.handoff.study import prepare,digest


def moments(kernel,target,energy,points):
    outer=kernel.maximum_impact_parameter_angstrom('C',target,energy)
    if outer==0:return [0.,0.]
    qh=(kernel.boundary(target,energy)/outer)**2
    edges=sorted(set([0.,1.,qh,*np.logspace(-14,-1,14)]))
    nodes,weights=np.polynomial.legendre.leggauss(points)
    value=np.zeros(2)
    for left,right in zip(edges[:-1],edges[1:]):
        for q,w in zip(left+(nodes+1)*(right-left)/2,weights*(right-left)/2):
            event=kernel.collide('C',target,energy,outer*math.sqrt(q))
            value+=w*np.array([event.recoil_energy_ev,2*math.sin(event.theta_projectile_lab_rad/2)**2])
    return (math.pi*outer**2*value).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--task-index',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--points',type=int,default=16);a=p.parse_args()
    if a.points<4:p.error('Use at least four points per impact subinterval.')
    manifest=json.loads(a.study.read_text())
    if prepare(manifest['configuration'])['signature']!=manifest['signature']:raise RuntimeError('Changed study sources or inputs.')
    task=manifest['binary_tasks'][a.task_index]
    target,energy=task['target'],task['energy_ev']
    for boundary in tqdm([*task['boundary_potential_ev'],None],unit='boundaries'):
        for cutoff in task['soft_recoil_cutoffs_ev']:
            identity={'study':manifest['signature'],'task':task,'boundary_ev':boundary,'cutoff_ev':cutoff,'points':a.points}
            path=a.output/(digest(identity)+'.json')
            if path.exists():continue
            # Refine orbit and impact quadrature separately.
            low=moments(HandoffKernel(minimum_transfer_ev=cutoff,boundary_ev=boundary,order=64),target,energy,a.points)
            impact=moments(HandoffKernel(minimum_transfer_ev=cutoff,boundary_ev=boundary,order=64),target,energy,2*a.points)
            orbit=moments(HandoffKernel(minimum_transfer_ev=cutoff,boundary_ev=boundary,order=128),target,energy,2*a.points)
            differences=[max(abs(x-y),abs(y-z))/abs(z) if z else None for x,y,z in zip(low,impact,orbit)]
            atomic(path,{'identity':identity,'moments':dict(zip(('stopping_ev_angstrom2','transport_angstrom2'),orbit)),
                         'relative_refinement_changes':differences,'numerical_screen_pass':all(v is not None and v<=.005 for v in differences),
                         'handoff_qualified':False})
if __name__=='__main__':main()
