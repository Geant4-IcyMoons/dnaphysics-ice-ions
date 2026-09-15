"""Frozen-water H+ feasibility calculation; reports populations, not event probabilities."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.constants import physical_constants

from physics.constants import EH, PROTON_MASS_AU
from physics.low_energy.RT_TDDFT.prepare import registered_structure, extract_cluster, file_sha256

BOHR_ANGSTROM = physical_constants['Bohr radius'][0] * 1e10


def read_cube(path):
    """Read one real, scalar Gaussian cube, with the standard Bohr grid convention."""
    with Path(path).open() as f:
        next(f); next(f)
        header = next(f).split()
        atoms = int(header[0])
        if atoms < 0:
            raise ValueError('Orbital cubes are not scalar density cubes')
        origin = np.array(header[1:4], float)
        rows = [next(f).split() for _ in range(3)]
        shape = tuple(int(row[0]) for row in rows)
        if any(n <= 0 for n in shape):
            raise ValueError('Expected positive cube grid dimensions in Bohr')
        axes = np.array([row[1:4] for row in rows], float)
        for _ in range(atoms): next(f)
        values = np.fromstring(f.read(), sep=' ')
    if values.size != math.prod(shape):
        raise ValueError('Cube grid size mismatch')
    points = np.indices(shape).reshape(3, -1).T @ axes + origin
    return points * BOHR_ANGSTROM, values, abs(float(np.linalg.det(axes)))


def density_populations(path, projectile_position, radii=(1.0, 1.5, 2.0)):
    points, rho, dv = read_cube(path)
    distance = np.linalg.norm(points - projectile_position, axis=1)
    return {'electron_count_on_grid': float(rho.sum() * dv),
            'projectile_sphere_electrons': {str(r): float(rho[distance < r].sum() * dv) for r in radii}}


def inputs(args, positions, dt):
    v = math.sqrt(2 * args.energy_ev / EH / PROTON_MASS_AU)
    flight = 2 * args.separation / BOHR_ANGSTROM / v
    steps = 8 * math.ceil(flight / dt / 8)
    mass_u = PROTON_MASS_AU * physical_constants['electron mass in u'][0]
    half = (args.transverse_half, args.transverse_half, args.separation + args.clearance)
    if min(half) <= args.cap_width or args.clearance - args.cap_width < 2:
        raise ValueError('Keep the 2 A projectile analysis sphere outside the absorber')
    common = f'''# Octopus 16.4: atomic input units, with explicit length factors.
UnitsOutput = atomic
PeriodicDimensions = 0
BoxShape = parallelepiped
%Lsize
 {half[0]}*angstrom | {half[1]}*angstrom | {half[2]}*angstrom
%
Spacing = {args.spacing}*angstrom
PseudopotentialSet = standard
SpinComponents = unpolarized
%Species
 "H" | species_pseudo | set | standard | mass | {mass_u:.14g}
%
'''
    coords = '%Coordinates\n'
    for element, point in zip(['O','H','H'] * (len(positions)//3), positions):
        coords += f' "{element}" | ' + ' | '.join(f'{x:.14g}*angstrom' for x in point) + ' | no\n'
    output = '%Output\n density\n%\nOutputFormat = cube\n'
    gs = common + coords + '%\n' + '''CalculationMode = gs
FromScratch = yes
ExcessCharge = 0
MoveIons = no
ConvRelDens = 1e-8
ConvEigenError = yes
EigensolverTolerance = 1e-9
EigensolverMaxIter = 100
MaximumIter = 300
''' + output
    velocities = ''.join(f' \"{element}\" | 0 | 0 | 0\n' for element in ['O','H','H'] * (len(positions)//3))
    td = common + coords + f''' "H" | {args.impact}*angstrom | 0 | {-args.separation}*angstrom | yes
%
CalculationMode = td
FromScratch = yes
ExcessCharge = 1
ExperimentalFeatures = yes
MoveIons = yes
IonsConstantVelocity = yes
RecalculateGSDuringEvolution = no
%Velocities
{velocities} "H" | 0 | 0 | {v:.14g}
%
TDPropagator = aetrs
TDTimeStep = {dt}
TDMaxSteps = {steps}
AbsorbingBoundaries = cap
ABWidth = {args.cap_width}*angstrom
ABCapHeight = {-args.cap_height}
TDEnergyUpdateIter = 10
TDOutputComputeInterval = 10
%TDOutput
 energy
 multipoles
 geometry
%
OutputInterval = {max(1, steps // 8)}
RestartWriteInterval = {max(1, steps // 4)}
''' + output
    return gs, td, {'velocity_au': v, 'steps': steps, 'dt_au': dt,
                    'end_z_angstrom': -args.separation + steps*dt*v*BOHR_ANGSTROM,
                    'box_half_lengths_angstrom': half}


def execute_case(args, dt, positions, source):
    case = args.output / f'dt_{dt:g}'
    case.mkdir(parents=True, exist_ok=False)
    gs, td, trajectory = inputs(args, positions, dt)
    (case/'gs.inp').write_text(gs)
    (case/'td.inp').write_text(td)
    manifest = {'source': source, 'settings': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'trajectory': trajectory, 'model': 'standard LDA pseudopotentials; prescribed H+; frozen target',
                'code_sha256': file_sha256(Path(__file__)),
                'initialization': 'neutral-water GS orbitals read unchanged on identical mesh; H added only for TD; total electron count unchanged',
                'not_validated': ['finite initial separation', 'grid and box', 'absorber', 'projectile potential',
                                  'excitation and exclusive event probabilities'],
                'input_sha256': {name:file_sha256(case/name) for name in ('gs.inp','td.inp')},
                'executable_sha256': file_sha256(Path(args.executable)), 'stages': []}
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    try:
        for stage in ('gs', 'td'):
            shutil.copyfile(case/f'{stage}.inp', case/'inp')
            started = time.monotonic()
            with (case/f'{stage}.log').open('x') as log:
                result = subprocess.run([args.executable], cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
            log_text = (case/f'{stage}.log').read_text()
            manifest['stages'].append({'stage':stage,'returncode':result.returncode,'seconds':time.monotonic()-started})
            if result.returncode:
                raise RuntimeError(f'{case}: {stage} failed; inspect log')
            if stage == 'gs' and ('SCF converged' not in log_text or 'not fully converged' in log_text):
                raise RuntimeError(f'{case}: ground-state convergence not established')
        summary = analyze_case(case, args, trajectory)
        manifest.update(summary)
        manifest['status'] = 'feasibility_run_completed_not_physically_validated'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        (case/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def analyze_case(case, args, trajectory):
    cubes = sorted((case/'output_iter').glob('td.*/density.cube'), key=lambda p: int(p.parent.name.split('.')[-1]))
    if not cubes:
        raise RuntimeError('No TD density cubes found')
    samples = []
    for cube in cubes:
        step = int(cube.parent.name.split('.')[-1])
        z = -args.separation + step*trajectory['dt_au']*trajectory['velocity_au']*BOHR_ANGSTROM
        samples.append({'step':step,'projectile_z_angstrom':z,
                        **density_populations(cube, np.array([args.impact,0,z]))})
    initial = density_populations(case/'static/density.cube', np.array([args.impact,0,-args.separation]))
    if samples[0]['step'] != 0 or trajectory['steps'] - samples[-1]['step'] >= max(1, trajectory['steps']//8):
        raise RuntimeError('Missing initial or final density')
    _, gs_rho, dv = read_cube(case/'static/density.cube')
    _, td_rho, _ = read_cube(cubes[0])
    initial_difference = float(np.abs(gs_rho - td_rho).sum() * dv)
    if initial_difference > 1e-5:
        raise RuntimeError('Initial TD density differs from neutral-water density')
    coordinates = np.loadtxt(case/'td.general/coordinates', comments='#', ndmin=2)
    natoms = 3 * getattr(args, 'molecules', 1) + 1
    positions = coordinates[:,2:2+3*natoms].reshape(-1,natoms,3) * BOHR_ANGSTROM
    fixed_drift = float(np.max(np.abs(positions[:,:-1] - positions[0,:-1])))
    expected_z = -args.separation + coordinates[:,1]*trajectory['velocity_au']*BOHR_ANGSTROM
    path_error = float(np.max(np.abs(positions[:,-1,2] - expected_z)))
    if int(coordinates[-1,0]) != trajectory['steps']:
        raise RuntimeError('Incomplete nuclear trajectory')
    if fixed_drift > 1e-6 or path_error > 1e-5:
        raise RuntimeError('Nuclear trajectory check failed')
    return {'initial_target':initial, 'density_samples':samples,
            'initial_density_l1_difference_electrons':initial_difference,
            'fixed_target_drift_angstrom':fixed_drift,
            'projectile_path_error_angstrom':path_error,
            'largest_sphere_cap_overlap_angstrom': max(0., samples[-1]['projectile_z_angstrom'] + 2. - (args.separation + args.clearance - args.cap_width)),
            'absorbed_electrons_from_grid_norm':initial['electron_count_on_grid']-samples[-1]['electron_count_on_grid'],
            'interpretation': 'Sphere populations are capture diagnostics, not exclusive capture probabilities; absorbed norm is an emission proxy, not a validated ionization probability.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',default='amorphous_lda_80k',choices=['amorphous_lda_80k','hexagonal_ih_100k'])
    p.add_argument('--replica',type=int,default=0)
    p.add_argument('--molecules',type=int,default=1)
    p.add_argument('--center',type=int,default=0)
    p.add_argument('--energy-ev',type=float,default=1000.)
    p.add_argument('--impact',type=float,default=1.)
    p.add_argument('--separation',type=float,default=8.)
    p.add_argument('--spacing',type=float,default=.4)
    p.add_argument('--transverse-half',type=float,default=6.)
    p.add_argument('--clearance',type=float,default=4.1)
    p.add_argument('--cap-width',type=float,default=2.)
    p.add_argument('--cap-height',type=float,default=.2,help='Positive magnitude in Hartree of negative imaginary CAP')
    p.add_argument('--dt',type=float,nargs='+',default=[.04,.02])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--executable',default=str(Path.home()/'.local/opt/octopus-16.4/bin/octopus'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for name in ('energy_ev','separation','spacing','transverse_half','clearance','cap_width','cap_height'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0: p.error(f'{name} must be positive and finite')
    if args.molecules < 1: p.error('molecules must be positive')
    if not math.isfinite(args.impact) or args.threads<1 or args.workers<1 or any(not math.isfinite(d) or d<=0 for d in args.dt):
        p.error('Invalid numerical parameters')
    if len({f'{d:g}' for d in args.dt})!=len(args.dt): p.error('Time steps must have distinct names')
    binary=shutil.which(args.executable)
    if not binary: p.error('Octopus executable not found')
    args.executable=str(Path(binary).resolve())
    args.output=args.output.resolve()
    structure,registry=registered_structure(args.phase,args.replica)
    ids,positions,origin=extract_cluster(structure,args.center,args.molecules)
    if abs(args.impact) + 2 >= args.transverse_half - args.cap_width:
        p.error('Projectile analysis spheres must remain inside transverse absorber boundary')
    if args.separation <= np.max(np.abs(positions[:,2])): p.error('Projectile must start outside target')
    args.output.mkdir(parents=True,exist_ok=False)
    source={'sha256':structure.source_sha256,'phase':args.phase,'molecule_ids':ids.tolist(),
            'registry_sha256':file_sha256(registry),'wrapped_origin_angstrom':origin.tolist(),
            'positions_angstrom':positions.tolist()}
    version=subprocess.check_output([args.executable,'--version'],text=True)
    (args.output/'version.txt').write_text(version)
    with ThreadPoolExecutor(max_workers=min(args.workers,len(args.dt))) as pool:
        results=list(pool.map(lambda dt:execute_case(args,dt,positions,source),args.dt))
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(args.output/'summary.json')


if __name__=='__main__': main()
