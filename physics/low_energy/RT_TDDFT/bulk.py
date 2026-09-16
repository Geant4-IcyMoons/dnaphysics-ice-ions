"""Prepare phase-independent periodic H stopping inputs; no physical validation implied."""
from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

import numpy as np
from scipy.constants import physical_constants

from physics.constants import EH, PROTON_MASS_AU
from physics.low_energy.RT_TDDFT.prepare import file_sha256, load_ice_structure

BOHR_ANGSTROM = physical_constants['Bohr radius'][0] * 1e10


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def trajectory(config, cell):
    """A Cartesian straight path contained in one periodic cell; no wake guarantee."""
    required = {'energy_ev', 'start_fractional', 'direction_cartesian', 'length_angstrom',
                'spacing_angstrom', 'dt_au', 'output_every', 'initialization'}
    if set(config) != required:
        raise ValueError(f'Expected settings keys {sorted(required)}')
    if config['initialization'] != 'relaxed_charged_cell':
        raise ValueError('Only relaxed_charged_cell is implemented; not a bare incoming H+ state')
    for key in ('energy_ev', 'length_angstrom', 'spacing_angstrom', 'dt_au'):
        positive(config[key], key)
    every = config['output_every']
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError('output_every must be a positive integer')
    start = np.asarray(config['start_fractional'], float)
    direction = np.asarray(config['direction_cartesian'], float)
    if start.shape != (3,) or direction.shape != (3,) or not np.isfinite([start, direction]).all():
        raise ValueError('Start and direction must be finite three-vectors')
    if np.any(start <= 0) or np.any(start >= 1) or np.linalg.norm(direction) == 0:
        raise ValueError('Start must be inside the cell and direction must be nonzero')
    direction = direction / np.linalg.norm(direction)
    speed = math.sqrt(2 * config['energy_ev'] / EH / PROTON_MASS_AU)
    # Round down so the propagated path never exceeds the requested length.
    steps = math.floor(config['length_angstrom'] / (speed * config['dt_au'] * BOHR_ANGSTROM))
    steps -= steps % every
    if steps < every:
        raise ValueError('Path is shorter than one output interval')
    length = steps * config['dt_au'] * speed * BOHR_ANGSTROM
    end = start + (direction * length) @ np.linalg.inv(cell)
    if np.any(end <= 0) or np.any(end >= 1):
        raise ValueError('Trajectory leaves the cell; shorten or reposition it')
    return {'steps': steps, 'velocity_au': (speed * direction).tolist(),
            'start_fractional': start.tolist(), 'end_fractional': end.tolist(),
            'start_angstrom': (start @ cell).tolist(), 'length_angstrom': length,
            'speed_au': speed}


def inputs(structure, config, path):
    cell = structure.lattice_angstrom
    lengths = np.linalg.norm(cell, axis=1)
    mass = PROTON_MASS_AU * physical_constants['electron mass in u'][0]
    text = f'''# Periodic frozen-host, constant-velocity H; relaxed charged-cell initialization.
UnitsOutput = atomic
PeriodicDimensions = 3
BoxShape = parallelepiped
%LatticeParameters
 {' | '.join(f'{v:.16g}*angstrom' for v in lengths)}
%
%LatticeVectors
'''
    text += '\n'.join(' | '.join(f'{v:.16g}' for v in row) for row in cell / lengths[:, None])
    text += f'''
%
Spacing = {config['spacing_angstrom']:.16g}*angstrom
%KPointsGrid
 1 | 1 | 1
%
KPointsUseSymmetries = no
SpinComponents = unpolarized
ExcessCharge = 1
XCFunctional = gga_x_pbe + gga_c_pbe
%Species
 "H" | species_pseudo | file | "pseudos/H.upf" | mass | {mass:.16g}
 "O" | species_pseudo | file | "pseudos/O.upf"
%
%Coordinates
'''
    # Center the cell explicitly: Octopus's parallelepiped is centered at the origin.
    origin = np.sum(cell, axis=0) / 2
    for element, xyz in zip(structure.species, structure.positions_angstrom - origin):
        text += f' "{element}" | ' + ' | '.join(f'{x:.16g}*angstrom' for x in xyz) + ' | no\n'
    xyz = np.asarray(path['start_angstrom']) - origin
    text += ' "H" | ' + ' | '.join(f'{x:.16g}*angstrom' for x in xyz) + ' | yes\n%\n'
    gs = text + '''CalculationMode = gs
FromScratch = yes
MoveIons = no
MaximumIter = 300
ConvRelDens = 1e-8
'''
    td = text + f'''CalculationMode = td
FromScratch = yes
ExperimentalFeatures = yes
MoveIons = yes
IonsConstantVelocity = yes
RecalculateGSDuringEvolution = no
%Velocities
'''
    td += ''.join(f' "{element}" | 0 | 0 | 0\n' for element in structure.species)
    td += ' "H" | ' + ' | '.join(f'{v:.16g}' for v in path['velocity_au']) + '\n%\n'
    td += f'''TDPropagator = etrs
TDTimeStep = {config['dt_au']:.16g}
TDMaxSteps = {path['steps']}
TDEnergyUpdateIter = {config['output_every']}
TDOutputComputeInterval = {config['output_every']}
RestartWriteInterval = {path['steps']}
%TDOutput
 energy
 geometry
%
'''
    return gs, td


def prepare(source, settings, pseudo_directory, output, *, metadata=None, frame=-1, diagnostic=False):
    config = json.loads(Path(settings).read_text())
    structure = load_ice_structure(Path(source), frame_index=frame, metadata_path=metadata,
                                   allow_unvalidated=diagnostic)
    if np.linalg.det(structure.lattice_angstrom) <= 0:
        raise ValueError('A right-handed cell is required')
    path = trajectory(config, structure.lattice_angstrom)
    pseudo_directory, output = Path(pseudo_directory), Path(output)
    for element in ('H', 'O'):
        pseudo = pseudo_directory / f'{element}.upf'
        if not pseudo.is_file():
            raise ValueError(f'Missing pseudopotential: {pseudo}')
        header = re.search(r'<PP_HEADER\b[^>]*>', pseudo.read_text(), re.DOTALL)
        if header is None:
            raise ValueError(f'Missing UPF header: {pseudo}')
        attributes = ET.fromstring(header.group().rstrip('> /') + '/>').attrib
        if (attributes.get('element', '').strip() != element
                or attributes.get('functional', '').strip() != 'PBE'
                or attributes.get('pseudo_type') != 'NC'
                or attributes.get('has_so') != 'F'
                or float(attributes.get('z_valence', 'nan')) != {'H': 1, 'O': 6}[element]):
            raise ValueError(f'Expected scalar norm-conserving PBE {element} with standard valence: {pseudo}')
    gs, td = inputs(structure, config, path)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'pseudos').mkdir()
    for element in ('H', 'O'):
        shutil.copyfile(pseudo_directory / f'{element}.upf', output / 'pseudos' / f'{element}.upf')
    (output / 'gs.inp').write_text(gs)
    (output / 'td.inp').write_text(td)
    (output / 'settings.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest = {'schema_version': 1, 'status': 'prepared_not_run',
                'source': structure.manifest_record(), 'settings': config, 'trajectory': path,
                'host_nuclei': 'frozen', 'projectile': 'H', 'cell_charge': 1,
                'valence_electrons': 8 * structure.water_molecule_count,
                'initialization': 'relaxed charged cell; projectile charge is not constrained to +1',
                'electrostatics': 'periodic charged cell; Octopus periodic Poisson convention',
                'code_sha256': file_sha256(Path(__file__)),
                'files_sha256': {str(p.relative_to(output)): file_sha256(p)
                                 for p in sorted(output.rglob('*')) if p.is_file()},
                'outstanding': ['grid, timestep, k points, cell size and trajectory convergence',
                                'initial transient and periodic wake effects',
                                'projectile charge partition; no exclusive capture/ionization probabilities',
                                'ensemble stopping and uncertainty',
                                'C/O/S projectile preparation and validation']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def analyze(case, fit_start, fit_end):
    """Report a finite-path energy slope, without asserting converged stopping."""
    case = Path(case)
    manifest = json.loads((case / 'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if file_sha256(case / name) != digest:
            raise ValueError(f'Input changed since preparation: {name}')
    energy_path = case / 'td.general/energy'
    data = np.atleast_2d(np.loadtxt(energy_path))
    if data.shape[1] < 3 or not np.isfinite(data).all():
        raise ValueError('Invalid atomic-unit Octopus energy table')
    steps, times, energy = data[:, 0], data[:, 1], data[:, 2]
    dt = manifest['settings']['dt_au']
    if len(steps) < 3 or steps[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError('Expected a complete, strictly increasing trajectory starting at step zero')
    if steps[-1] != manifest['trajectory']['steps'] or not np.allclose(times, steps * dt, atol=1e-6):
        raise ValueError('Incomplete run or inconsistent time units')
    distance = times * manifest['trajectory']['speed_au'] * BOHR_ANGSTROM
    if not 0 <= fit_start < fit_end <= distance[-1] + 1e-8:
        raise ValueError('Fit window must lie inside the completed path')
    mask = (distance >= fit_start) & (distance <= fit_end)
    if mask.sum() < 3:
        raise ValueError('Fit window needs at least three energy samples')
    change = (energy - energy[0]) * EH
    slope, intercept = np.polyfit(distance[mask], change[mask], 1)
    result = {
        'status': 'finite_path_diagnostic_not_validated_stopping',
        'source': manifest['source'], 'fit_window_angstrom': [fit_start, fit_end],
        'samples_in_fit': int(mask.sum()),
        'finite_path_energy_slope_ev_per_angstrom': float(slope),
        'fit_residual_rms_ev': float(np.sqrt(np.mean((change[mask] - slope * distance[mask] - intercept)**2))),
        'total_energy_change_ev': float(change[-1]),
        'energy_sha256': file_sha256(energy_path),
        'manifest_sha256': file_sha256(case / 'manifest.json'),
        'interpretation': 'Driven total-energy gain with frozen host; includes finite-path conservative variations. '
                          'Not an exclusive capture probability or a converged stopping coefficient.',
        'uncertainty': 'Ensemble and convergence uncertainty not evaluated; fit residual is not that uncertainty.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--structure', type=Path, required=True)
    preparation.add_argument('--metadata', type=Path)
    preparation.add_argument('--frame', type=int, default=-1)
    preparation.add_argument('--settings', type=Path, required=True)
    preparation.add_argument('--pseudo-directory', type=Path, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    preparation.add_argument('--diagnostic', action='store_true', help='Permit unaccepted structures; record diagnostic status')
    analysis = commands.add_parser('analyze')
    analysis.add_argument('--case', type=Path, required=True)
    analysis.add_argument('--fit-start', type=float, required=True, help='Distance in angstrom')
    analysis.add_argument('--fit-end', type=float, required=True, help='Distance in angstrom')
    args = parser.parse_args()
    if args.command == 'analyze':
        result = analyze(args.case, args.fit_start, args.fit_end)
    else:
        result = prepare(args.structure, args.settings, args.pseudo_directory, args.output,
                         metadata=args.metadata, frame=args.frame, diagnostic=args.diagnostic)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
