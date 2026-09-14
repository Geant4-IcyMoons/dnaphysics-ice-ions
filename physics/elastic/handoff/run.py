"""Run and reduce restartable combined handoff cohorts on a PBS allocation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from physics.elastic.handoff.study import prepare, DEFAULT_CONFIG, digest, compare_independent
from physics.elastic.handoff.kernel import HandoffKernel
from physics.elastic.bca.structure import load_ice_structure
from physics.elastic.bca.trajectory import PeriodicHardCollisionTransport
from physics.elastic.bca.convergence import RatioStatistics
from physics.elastic.zbl.generate_backend import REPOSITORY_ROOT

_TRANSPORT=None

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def initialize(case,cutoff,order):
    global _TRANSPORT
    structure=load_ice_structure(REPOSITORY_ROOT/case['structure'])
    _TRANSPORT=PeriodicHardCollisionTransport(structure,HandoffKernel(minimum_transfer_ev=cutoff,boundary_ev=case['boundary_potential_ev'],order=order))


def block(task):
    case,start,count,length=task
    rows=[]
    for i in range(start,start+count):
        rng=np.random.default_rng(np.random.SeedSequence([case['seed'],i]))
        position=rng.random(3)@_TRANSPORT.structure.lattice_angstrom
        direction={'c_axis':[0,0,1],'basal_a_axis':[1,0,0]}.get(case['direction'])
        if direction is None:
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
        result=_TRANSPORT.trace('C',case['energy_ev'],position,direction,length,rng=rng)
        if result.termination not in ('path_complete',):
            raise RuntimeError(f'Incomplete history {i}: {result.termination}; no silent acceptance.')
        angle=sum(2*math.sin(e.theta_projectile_lab_rad/2)**2 for e in result.events)
        branches=sum(_TRANSPORT.kernels.branch(e.target,e.projectile_energy_in_ev,e.impact_parameter_angstrom)=='nlh' for e in result.events)
        rows.append([result.traveled_path_length_angstrom,result.recoil_energy_ev,angle,len(result.events),branches,result.ambiguous_event_count])
    return {'start':start,'rows':rows}


def statistics(products,target):
    stats=[RatioStatistics(),RatioStatistics()];n=0;counts=[0,0,0]
    for product in products:
        for row in product['rows']:
            for s,x in zip(stats,row[1:3]):s.add(x,row[0])
            n+=1
            for k in range(3):counts[k]+=row[k+3]
    obs={}
    for name,s in zip(('stopping','angular_transport'),stats):
        v=s.interval(1.)
        # RatioStatistics uses estimate rather than mean.
        mean=v['estimate'];se=v['standard_error']
        obs[name]={'mean':mean,'standard_error':se,'passes':bool(mean is not None and se is not None and mean>0 and se>0 and se/mean<=target)}
    return {'histories':n,'observables':obs,'scalar_pass':all(v['passes'] for v in obs.values()),'collisions':counts[0],'nlh_collisions':counts[1],'ambiguous_collisions':counts[2]}


def execute(args):
    manifest=json.loads(args.study.read_text())
    if prepare(manifest['configuration'])['signature']!=manifest['signature']:
        raise RuntimeError('Study source/input signature changed; prepare a fresh study.')
    case=manifest['cohorts'][args.case_index]
    identity={'study_signature':manifest['signature'],'case':case,'cutoff_ev':args.cutoff_ev,'order':args.order,'block_size':args.block_size}
    root=args.output/case['id']/digest(identity)[:16];root.mkdir(parents=True,exist_ok=True)
    with (root/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(root/'identity.json',identity)
        paths=sorted(root.glob('block_*.json'));products=[json.loads(p.read_text()) for p in paths]
        n=sum(len(p['rows']) for p in products)
        if [p['start'] for p in products]!=list(range(0,n,args.block_size)):
            raise RuntimeError('Checkpoint block sequence is incomplete.')
        target=manifest['configuration']['relative_standard_error_target']
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(case,args.cutoff_ev,args.order)) as pool:
            with tqdm(total=args.max_histories,initial=n,unit='histories') as progress:
                while n<args.max_histories:
                    if n>=args.min_histories and statistics(products,target)['scalar_pass']:break
                    tasks=[(case,start,args.block_size,manifest['configuration']['path_length_angstrom']) for start in range(n,min(n+args.workers*args.block_size,args.max_histories),args.block_size)]
                    for product in pool.map(block,tasks):
                        atomic(root/f"block_{product['start']:012d}.json",product)
                        products.append(product);n+=len(product['rows']);progress.update(len(product['rows']))
        report=statistics(products,target)
        report.update(identity=identity,handoff_qualified=False,numerical_convergence_checked=False,status='scalar_target_met' if report['scalar_pass'] else 'sampling_limit')
        atomic(root/'result.json',report)
        print(root/'result.json')


def reduce(args):
    reports=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/*/result.json'))]
    groups={}
    for r in reports:
        i=r['identity'];c=i['case'];key=(i['study_signature'],c['phase'],c['structure'],c['direction'],c['energy_ev'],i['cutoff_ev'],i['order'])
        groups.setdefault(key,[]).append(r)
    comparisons=[]
    for key,group in groups.items():
        for a,b in itertools.combinations(group,2):
            ca=a['identity']['case'];cb=b['identity']['case']
            if ca['boundary_potential_ev'] is None or cb['boundary_potential_ev'] is None:continue
            comparisons.append({'condition':key,'boundaries':[ca['boundary_potential_ev'],cb['boundary_potential_ev']],
                                'observables':{name:compare_independent(a['observables'][name],b['observables'][name]) for name in a['observables']}})
    atomic(args.output/'comparison.json',{'cohorts_with_results':len(reports),'comparisons':comparisons,'handoff_qualified':False,'note':'Boundary screen only. Check completeness, orbit-order and soft-cutoff convergence before accepting a rule.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('study',type=Path);r.add_argument('--case-index',type=int,required=True)
    r.add_argument('--output',type=Path,required=True);r.add_argument('--cutoff-ev',type=float,required=True)
    r.add_argument('--order',type=int,default=64);r.add_argument('--workers',type=int,default=1)
    r.add_argument('--block-size',type=int,default=16);r.add_argument('--min-histories',type=int,default=256);r.add_argument('--max-histories',type=int,default=100000)
    s=sub.add_parser('reduce');s.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='run':
        if args.case_index<0 or args.workers<1 or args.block_size<1 or args.min_histories<2 or args.max_histories<args.min_histories or args.max_histories%args.block_size or not math.isfinite(args.cutoff_ev) or args.cutoff_ev<=0:p.error('Invalid case, resource, history or cutoff settings; maximum histories must be a block multiple.')
        execute(args)
    else:reduce(args)

if __name__=='__main__':main()
