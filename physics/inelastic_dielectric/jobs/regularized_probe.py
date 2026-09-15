"""Independent tolerance/window checks of the witnessed close encounter."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from time import monotonic
import json
import argparse
import math
from tqdm import tqdm
from physics.inelastic_dielectric.checkpoints import atomic_text, source_digest
from physics.inelastic_dielectric.polarization import nonlinear_oscillator as no
from physics.inelastic_dielectric.polarization import regularized_encounter as lc


def compare(task):
    point, method, tolerance, output, source = task
    path = output/f"loss_{point['loss_eV']}_{method}_{tolerance}.json"
    settings = dict(point=point, method=method, rtol=tolerance, source=source)
    if path.exists():
        old = json.loads(path.read_text())
        if old['settings'] != settings:
            raise RuntimeError('Incompatible comparison checkpoint')
        return old
    start = monotonic()
    try:
        args = (point['x'], point['b_bohr'], point['eta'], no.frozen_field('H',0))
        kw = dict(gamma=point['gamma'], tail=point['tail'], rtol=tolerance)
        result = (lc.encounter(*args, **kw) if method == 'lc'
                  else no.encounter(*args, regularize=False, **kw))
        result['finished'] = True
    except Exception as exc:
        result = dict(finished=False, error=str(exc))
    record = dict(settings=settings, result=result, seconds=monotonic()-start)
    with atomic_text(path) as stream:
        json.dump(record, stream, indent=2)
    return record


def row(task):
    report, factor, tolerance, output = task
    path = output/f'window_{factor}_rtol_{tolerance}.json'
    settings = dict(witness=report,window_factor=factor,rtol=tolerance,source=source_digest())
    if path.exists():
        old = json.loads(path.read_text())
        if old['settings'] == settings:
            return old
        raise RuntimeError('Incompatible regularized probe checkpoint')
    started = monotonic()
    try:
        result = lc.encounter(report['x'], report['b_bohr'], report['eta'], no.frozen_field('H',0),
            gamma=report['gamma'],tail=report['tail']*factor,rtol=tolerance)
        result['finished'] = True
    except Exception as exc:
        result = dict(finished=False,error=str(exc))
    record = dict(settings=settings,result=result,seconds=monotonic()-started)
    with atomic_text(path) as stream:
        json.dump(record,stream,indent=2)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-solvers', action='store_true')
    args = parser.parse_args()
    root = Path('physics/inelastic_dielectric/output')
    if args.compare_solvers:
        points = json.loads((root/'live_probe_inspection.json').read_text())['observations']
        betas = [0.017589887285868945, 0.014594784218305139]
        etas = [0.5847011912511584, 0.8371673667323619]
        for point,beta,eta in zip(points,betas,etas):
            point['gamma'] = 1/math.sqrt(1-beta*beta)
            point['eta'] = eta  # Exact locals recorded by live py-spy inspection.
            point['tail'] = 16*max(1.,1/point['x'])/max(4.,1/point['x'])
        output = root/'solver_comparison'; output.mkdir(exist_ok=True)
        source = source_digest()
        tasks = [(p,m,t,output,source) for p in points for m in ('direct','lc')
                 for t in (2e-8,2e-9,2e-10,2e-11)]
        with ProcessPoolExecutor(max_workers=8,mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(compare,t) for t in tasks]
            for future in tqdm(as_completed(futures),total=len(tasks),desc='Solver comparison',unit='encounter'):
                print(json.dumps(future.result()),flush=True)
        return
    witness = json.loads(next((root/'encounter_probe').glob('encounter_*.json')).read_text())
    output = root/'regularized_probe'; output.mkdir(parents=True,exist_ok=True)
    tasks = [(witness,factor,rtol,output) for factor in (1,2,4) for rtol in (2e-8,2e-9)]
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(row,task) for task in tasks]
        for future in tqdm(as_completed(futures),total=len(tasks),desc='Regularized encounter checks',unit='encounter'):
            print(json.dumps(future.result()),flush=True)


if __name__ == '__main__':
    main()
