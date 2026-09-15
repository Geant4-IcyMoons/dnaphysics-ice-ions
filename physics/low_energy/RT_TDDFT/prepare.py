"""Prepare finite ice targets for Octopus; collision propagation remains a template."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants

from physics.elastic.bca.structure import file_sha256, load_ice_structure
from physics.constants import EH, PROTON_MASS_AU, PROJECT_ROOT

STRUCTURES_ROOT = PROJECT_ROOT / "models" / "ice"


def registered_structure(phase, replica):
    index = json.loads((STRUCTURES_ROOT / 'registry.json').read_text())
    model = index['models'][phase]
    if model['status'] != 'collision_ready':
        raise ValueError('Model is not collision ready')
    registry = STRUCTURES_ROOT / model['registry']
    records = json.loads(registry.read_text())['structures']
    if not 0 <= replica < len(records):
        raise ValueError('Replica index outside registry')
    record = records[replica]
    if record['collision_ready'] is not True:
        raise ValueError('Snapshot is not collision ready')
    structure = load_ice_structure(registry.parent / record['path'], frame_index=record['frame_index'])
    if structure.source_sha256 != record['sha256']:
        raise ValueError('Registry checksum mismatch')
    if not np.allclose(structure.lattice_angstrom, record['lattice_angstrom'], rtol=0, atol=1e-8):
        raise ValueError('Registry lattice mismatch')
    return structure, registry


def extract_cluster(structure, center, count):
    """Nearest oxygen selection with whole OHH molecules; orthorhombic cells only."""
    lattice = structure.lattice_angstrom
    lengths = np.diag(lattice)
    if not np.allclose(lattice, np.diag(lengths), atol=1e-10) or np.any(lengths <= 0):
        raise ValueError('Only positive orthorhombic cells are supported')
    n = structure.water_molecule_count
    if not 0 <= center < n or not 1 <= count <= n:
        raise ValueError('Invalid center or molecule count')
    if not np.array_equal(structure.species, np.tile(['O', 'H', 'H'], n)):
        raise ValueError('Expected registered OHH molecule ordering')
    xyz = structure.positions_angstrom.reshape(n, 3, 3)
    def mic(delta):
        return delta - lengths * np.floor(delta / lengths + 0.5)
    oxygens = mic(xyz[:, 0] - xyz[center, 0])
    distances = np.linalg.norm(oxygens, axis=1)
    ids = np.lexsort((np.arange(n), distances))[:count]
    if distances[ids[-1]] >= min(lengths) / 2:
        raise ValueError('Cluster exceeds unique minimum-image sphere')
    bonds = mic(xyz[ids, 1:] - xyz[ids, :1])
    # Broad topology guard, not an equilibrium bond-length acceptance criterion.
    if np.any(np.linalg.norm(bonds, axis=2) > 1.5):
        raise ValueError('OHH records do not form intact water molecules')
    out = np.concatenate((oxygens[ids, None, :], oxygens[ids, None, :] + bonds), axis=1)
    return ids, out.reshape(-1, 3), xyz[center, 0]


def common_input(positions, spacing, padding):
    half = np.max(np.abs(positions), axis=0) + padding
    lines = ['# Input values use atomic units unless explicitly multiplied by a unit.', 'UnitsOutput = eV_Angstrom',
             'PeriodicDimensions = 0', 'BoxShape = parallelepiped', '%Lsize',
             ' | '.join(f'{v:.12g}*angstrom' for v in half), '%',
             f'Spacing = {spacing:.12g}*angstrom', 'PseudopotentialSet = standard',
             '# XC follows the bundled LDA pseudopotentials; benchmark before adoption.',
             'ExcessCharge = 0', 'SpinComponents = unpolarized', '%Coordinates']
    for element, point in zip(np.tile(['O','H','H'], len(positions)//3), positions):
        lines.append(f'"{element}" | ' + ' | '.join(f'{v:.12g}*angstrom' for v in point) + ' | no')
    return '\n'.join(lines + ['%', ''])


def prepare(args):
    for key in ('spacing', 'padding', 'dt_au', 'energy_ev', 'separation'):
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if args.steps < 1 or not np.isfinite(args.impact):
        raise ValueError('Invalid steps or impact parameter')
    structure, registry = registered_structure(args.phase, args.replica)
    ids, positions, origin = extract_cluster(structure, args.center, args.molecules)
    if args.separation <= np.max(np.abs(positions[:, 2])):
        raise ValueError('Projectile separation must lie beyond the cluster along z')
    common = common_input(positions, args.spacing, args.padding)
    gs = common + 'CalculationMode = gs\nFromScratch = yes\nMoveIons = no\n%Output\n density\n%\nOutputFormat = cube\n'
    td = common + f'''CalculationMode = td
FromScratch = no
MoveIons = no
TDPropagator = aetrs
TDTimeStep = {args.dt_au:.12g}
TDMaxSteps = {args.steps}
%TDOutput
 energy
 multipoles
%
'''
    velocity = np.sqrt(2 * args.energy_ev / EH / PROTON_MASS_AU)
    bohr_a = physical_constants['Bohr radius'][0] * 1e10
    flight_au = 2 * args.separation / bohr_a / velocity
    manifest = {
        'schema_version': 1, 'status': 'prepared_not_solver_tested',
        'source': {**structure.manifest_record(),
                   'path': str(structure.source_path.relative_to(PROJECT_ROOT))},
        'registry': str(registry.relative_to(PROJECT_ROOT)),
        'registry_sha256': file_sha256(registry), 'phase_id': args.phase,
        'molecule_ids_zero_based': ids.tolist(), 'center_molecule_zero_based': args.center,
        'wrapped_origin_angstrom': origin.tolist(),
        'extraction': 'oxygen minimum images about center; hydrogens unwrapped about their own oxygen; no rotation or relaxation',
        'embedding': 'none; finite neutral cluster, not bulk ice',
        'parameters': {k: v for k, v in vars(args).items() if k != 'output'},
        'collision': {'projectile': 'H+', 'energy_ev_total': args.energy_ev,
                      'energy_ev_per_nucleon': args.energy_ev, 'mass_electron_units': PROTON_MASS_AU,
                      'velocity_au': [0, 0, float(velocity)],
                      'initial_position_angstrom': [args.impact, 0, -args.separation],
                      'final_position_angstrom': [args.impact, 0, args.separation],
                      'flight_time_au': float(flight_au),
                      'minimum_steps_for_flight': int(np.ceil(flight_au / args.dt_au)),
                      'runnable': False,
                      'unresolved': ['neutral-target plus bare-projectile initial state',
                                     'collision box and absorber convergence',
                                     'capture/excitation/continuum analysis',
                                     'heavy-ion electronic states and potentials']}}
    xyz = f'{len(positions)}\nFinite cluster; coordinates in angstrom; see manifest.json\n'
    xyz += ''.join(f'{e} {x:.12f} {y:.12f} {z:.12f}\n' for e,(x,y,z) in zip(np.tile(['O','H','H'], len(ids)), positions))
    # Fragment only: cannot be mistaken for a complete runnable collision input.
    draft = f'''# Octopus 16.4 collision fragment -- NOT a complete input.
# Requires a separately verified neutral-target + H+ initial state/restart.
# Do not obtain it by blindly relaxing the combined +1 system.
ExperimentalFeatures = yes
MoveIons = yes
IonsConstantVelocity = yes
# Append a movable H projectile to Coordinates at (in angstrom):
# {args.impact:.12g} | 0 | {-args.separation:.12g} | yes
# Total ExcessCharge would be +1; this does not enforce projectile charge.
# Velocities use atomic units, the Octopus input convention.
%Velocities
'''
    draft += ''.join(f' "{e}" | 0 | 0 | 0\n' for e in np.tile(['O','H','H'],len(ids)))
    draft += f' "H" | 0 | 0 | {velocity:.12g}\n%\n'
    files = {'cluster.xyz': xyz, 'gs.inp': gs, 'control_td.inp': td, 'collision.fragment': draft}
    import hashlib
    manifest['input_sha256'] = {name: hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (output / name).write_text(content)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('amorphous_lda_80k','hexagonal_ih_100k'))
    parser.add_argument('--replica', type=int, default=0)
    parser.add_argument('--center', type=int, default=0, help='Zero-based OHH molecule index')
    parser.add_argument('--molecules', type=int, required=True)
    parser.add_argument('--spacing', type=float, required=True, help='Grid spacing in angstrom; convergence input')
    parser.add_argument('--padding', type=float, required=True, help='Target box padding in angstrom')
    parser.add_argument('--dt-au', type=float, required=True)
    parser.add_argument('--steps', type=int, required=True, help='Unperturbed control steps, not collision duration')
    parser.add_argument('--energy-ev', type=float, required=True, help='H+ total kinetic energy in eV')
    parser.add_argument('--impact', type=float, required=True, help='Signed x offset from central oxygen in angstrom')
    parser.add_argument('--separation', type=float, required=True, help='Initial/final z distance in angstrom')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths are never overwritten')
    args = parser.parse_args()
    try:
        print(prepare(args))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
