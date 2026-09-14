"""Run and reduce restartable combined handoff cohorts on a PBS allocation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
from physics.elastic.handoff.study import prepare, digest, compare_independent
from physics.elastic.handoff.kernel import HandoffKernel
from physics.elastic.bca.structure import load_ice_structure
from physics.elastic.bca.trajectory import PeriodicHardCollisionTransport
from physics.elastic.zbl.generate_backend import REPOSITORY_ROOT

_TRANSPORT = None
_TUBE_FRACTION = 0.
ROW_COLUMNS = ('path_angstrom', 'recoil_ev', 'angular_transport', 'collisions',
               'nlh_collisions', 'ambiguous_collisions', 'weight',
               'residual_ev', 'terminated_at_floor')


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def initialize(case, cutoff, order, terminal_energy_ev=1., tube_fraction=0.):
    global _TRANSPORT, _TUBE_FRACTION
    structure = load_ice_structure(REPOSITORY_ROOT / case['structure'])
    kernel = HandoffKernel(minimum_transfer_ev=cutoff,
                           boundary_ev=case['boundary_potential_ev'], order=order,
                           terminal_energy_ev=terminal_energy_ev)
    _TRANSPORT = PeriodicHardCollisionTransport(structure, kernel)
    _TUBE_FRACTION = tube_fraction


def summarize_history(result, weight, kernel):
    """Retain valid tracks through the declared terminal energy; no deposition closure."""
    stopped = result.termination == 'energy_below_kernel_table'
    if result.termination not in ('path_complete', 'energy_below_kernel_table'):
        raise RuntimeError(f'Incomplete history: {result.termination}; no silent acceptance.')
    if not math.isfinite(weight) or weight <= 0 or result.traveled_path_length_angstrom <= 0:
        raise RuntimeError('Invalid weight or empty track.')
    floor = kernel.energy_bounds_ev('C', 'H')[0]
    if stopped and not 0 <= result.final_energy_ev < floor:
        raise RuntimeError('Terminal energy contradicts termination record.')
    angle = sum(2 * math.sin(e.theta_projectile_lab_rad / 2)**2 for e in result.events)
    branches = sum(kernel.branch(e.target, e.projectile_energy_in_ev,
                                 e.impact_parameter_angstrom) == 'nlh'
                   for e in result.events)
    residual = result.final_energy_ev if stopped else 0.
    return [result.traveled_path_length_angstrom, result.recoil_energy_ev, angle,
            len(result.events), branches, result.ambiguous_event_count, weight,
            residual, int(stopped)]


def block(task):
    case, start, count, length = task
    begun, cpu_begun = time.perf_counter(), time.process_time()
    rows = []
    for i in range(start, start + count):
        rng = np.random.default_rng(np.random.SeedSequence([case['seed'], i]))
        direction = {'c_axis': [0, 0, 1], 'basal_a_axis': [1, 0, 0]}.get(case['direction'])
        if direction is None:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
        if _TUBE_FRACTION:
            position, proposal = _TRANSPORT.sample_collision_tube_mixture(
                'C', case['energy_ev'], direction, length, _TUBE_FRACTION, rng)
            weight = proposal.target_over_proposal_weight
        else:
            position = rng.random(3) @ _TRANSPORT.structure.lattice_angstrom
            weight = 1.
        result = _TRANSPORT.trace('C', case['energy_ev'], position, direction,
                                  length, rng=rng)
        rows.append(summarize_history(result, weight, _TRANSPORT.kernels))
    return {'start': start, 'rows': rows,
            'worker_seconds': time.perf_counter() - begun,
            'cpu_seconds': time.process_time() - cpu_begun}


def ratio_estimate(numerator, denominator):
    """Centered whole-history ratio SE, including weight/track-length covariance."""
    n = len(numerator)
    mean = float(np.sum(numerator) / np.sum(denominator))
    residual = numerator - mean * denominator
    se = float(np.std(residual, ddof=1) / np.sqrt(n) / np.mean(denominator)) if n > 1 else None
    return mean, se


def statistics(products, target):
    rows = np.asarray([row for product in products for row in product['rows']], dtype=float)
    if rows.ndim != 2 or rows.shape[1] != len(ROW_COLUMNS):
        raise RuntimeError('Incompatible history schema; do not mix old and weighted blocks.')
    if not np.all(np.isfinite(rows)) or np.any(rows[:, 0] <= 0) or np.any(rows[:, 6] <= 0):
        raise RuntimeError('Nonfinite or invalid history observations.')
    weight = rows[:, 6]
    denominator = weight * rows[:, 0]
    obs = {}
    for name, column in (('stopping', 1), ('angular_transport', 2)):
        mean, se = ratio_estimate(weight * rows[:, column], denominator)
        residual_squared = (weight * rows[:, column] - mean * denominator)**2
        variance_sum = float(np.sum(residual_squared))
        obs[name] = {'mean': mean, 'standard_error': se,
                     'largest_history_variance_fraction': float(np.max(residual_squared) / variance_sum) if variance_sum > 0 else None,
                     'relative_variance_cpu_seconds': (se / mean)**2 * sum(p.get('cpu_seconds', 0.) for p in products) if mean > 0 and se is not None else None,
                     'passes': bool(mean > 0 and se is not None and se > 0 and se / mean <= target)}
    residual_rate = float(np.sum(weight * rows[:, 7]) / np.sum(denominator))
    return {'histories': len(rows), 'observables': obs,
            'scalar_pass': all(v['passes'] for v in obs.values()),
            'collisions': int(np.sum(rows[:, 3])), 'nlh_collisions': int(np.sum(rows[:, 4])),
            'ambiguous_collisions': int(np.sum(rows[:, 5])),
            'terminated_histories': int(np.sum(rows[:, 8])),
            'weighted_terminated_fraction': float(np.sum(weight * rows[:, 8]) / np.sum(weight)),
            'residual_energy_per_tracked_length_ev_angstrom': residual_rate,
            'mean_weight': float(np.mean(weight)),
            'weight_standard_error': float(np.std(weight, ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
            'weight_effective_sample_size': float(np.sum(weight)**2 / np.sum(weight**2)),
            'max_weight': float(np.max(weight)),
            'cpu_seconds': sum(p.get('cpu_seconds', 0.) for p in products),
            'worker_seconds': sum(p.get('worker_seconds', 0.) for p in products),
            'observable_scope': 'track-length averaged response through the declared terminal energy',
            'residual_note': 'Residual energy is retained, not deposited. No bound on omitted angular scattering is asserted.'}


def execute(args):
    manifest = json.loads(args.study.read_text())
    if prepare(manifest['configuration'])['signature'] != manifest['signature']:
        raise RuntimeError('Study source/input signature changed; prepare a fresh study.')
    case = dict(manifest['cohorts'][args.case_index])
    # Distinct settings receive independent streams; worker and sample counts do not.
    case['seed'] = int(digest([case['seed'], args.cutoff_ev, args.order,
                               args.terminal_energy_ev, args.tube_fraction])[:16], 16)
    identity = {'history_schema': 2, 'study_signature': manifest['signature'], 'case': case,
                'cutoff_ev': args.cutoff_ev, 'order': args.order, 'block_size': args.block_size,
                'terminal_energy_ev': args.terminal_energy_ev, 'tube_fraction': args.tube_fraction}
    root = args.output / case['id'] / digest(identity)[:16]
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(root / 'identity.json', identity)
        (root / 'result.json').unlink(missing_ok=True)
        products = {}
        for path in sorted(root.glob('block_*.json')):
            product = json.loads(path.read_text())
            start = product['start']
            if start < 0 or start % args.block_size or len(product['rows']) != args.block_size or start in products:
                raise RuntimeError('Invalid checkpoint block.')
            products[start] = product
        n = sum(len(p['rows']) for p in products.values())
        if args.max_histories and products and max(products) + args.block_size > args.max_histories:
            raise RuntimeError('Maximum histories is smaller than retained work.')
        target = manifest['configuration']['relative_standard_error_target']
        initargs = (case, args.cutoff_ev, args.order, args.terminal_energy_ev, args.tube_fraction)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=initargs) as pool:
            with tqdm(total=args.max_histories or None, initial=n, unit='histories') as progress:
                while not args.max_histories or n < args.max_histories:
                    ordered = [products[k] for k in sorted(products)]
                    # Fill any checkpoint holes before assessing, so completion order
                    # cannot select only fast histories after a failed/cancelled wave.
                    contiguous = sorted(products) == list(range(0, n, args.block_size))
                    if contiguous and n >= args.min_histories and statistics(ordered, target)['scalar_pass']:
                        break
                    end = args.max_histories or (max(products, default=-args.block_size) + args.block_size * (args.workers + 1))
                    starts = list(itertools.islice((k for k in range(0, end, args.block_size) if k not in products), args.workers))
                    tasks = [(case, k, args.block_size, manifest['configuration']['path_length_angstrom']) for k in starts]
                    futures = {pool.submit(block, task): task[1] for task in tasks}
                    errors = []
                    for future in as_completed(futures):
                        try:
                            product = future.result()
                        except Exception as exc:
                            errors.append({'start': futures[future], 'error': str(exc)})
                            continue
                        atomic(root / f"block_{product['start']:012d}.json", product)
                        products[product['start']] = product
                        n += len(product['rows'])
                        progress.update(len(product['rows']))
                    if errors:
                        atomic(root / 'errors.json', errors)
                        raise RuntimeError(f'{len(errors)} blocks failed; successful blocks were preserved: {errors[0]}')
                    atomic(root / 'progress.json', statistics([products[k] for k in sorted(products)], target))
        report = statistics([products[k] for k in sorted(products)], target)
        report.update(identity=identity, handoff_qualified=False, numerical_convergence_checked=False,
                      terminal_energy_sensitivity_checked=False,
                      status='scalar_target_met' if report['scalar_pass'] else 'sampling_limit')
        atomic(root / 'result.json', report)
        (root / 'errors.json').unlink(missing_ok=True)
        print(root / 'result.json')


def reduce(args):
    reports=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/*/result.json'))]
    groups={}
    for r in reports:
        i=r['identity'];c=i['case'];key=(i['study_signature'],c['phase'],c['structure'],c['direction'],c['energy_ev'],i['cutoff_ev'],i['order'],i.get('terminal_energy_ev'),i.get('tube_fraction'))
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
    r.add_argument('--terminal-energy-ev',type=float,default=1.)
    r.add_argument('--tube-fraction',type=float,default=0.)
    r.add_argument('--order',type=int,default=64);r.add_argument('--workers',type=int,default=1)
    r.add_argument('--block-size',type=int,default=16);r.add_argument('--min-histories',type=int,default=256);r.add_argument('--max-histories',type=int,default=0,help='0 means no history limit')
    s=sub.add_parser('reduce');s.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='run':
        if not math.isfinite(args.terminal_energy_ev) or not 1 <= args.terminal_energy_ev < 1e8 or not math.isfinite(args.tube_fraction) or not 0 <= args.tube_fraction < 1 or args.case_index<0 or args.workers<1 or args.block_size<1 or args.min_histories<2 or args.max_histories<0 or (args.max_histories and (args.max_histories<args.min_histories or args.max_histories%args.block_size)) or not math.isfinite(args.cutoff_ev) or args.cutoff_ev<=0:p.error('Invalid case, resource, history or cutoff settings; maximum histories must be a block multiple.')
        execute(args)
    else:reduce(args)

if __name__=='__main__':main()
