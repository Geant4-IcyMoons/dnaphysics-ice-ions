"""Independent 1 keV H+--H2O capture calculation with an Octopus orbital handoff.

This executable workflow is not an exact reconstruction of Hong et al. (2016).
All numerical settings and adopted differences are recorded in each case.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import time

import numpy as np
from scipy.constants import physical_constants

from physics.constants import EH, PROTON_MASS_AU
from .capture import read_mesh, read_obf, orbital_overlap, number_distribution
from .frame import translate_exports
from .prepare import file_sha256

BOHR_ANGSTROM = physical_constants['Bohr radius'][0] * 1e10


@dataclass(frozen=True)
class Settings:
    spacing: float = 0.33
    dt: float = 0.025
    impact: float = 2.0
    initial: float = 15.0
    flight_end: float = 45.0
    minimum_separation: float = 40.0
    radius: float = 25.0
    cap_width: float = 5.0
    cap_height: float = 2.0
    box_margin: float = 5.0
    maximum_projectile_displacement: float = 1.0
    clearing_time: float = 200.0
    samples: int = 5
    stationarity_tolerance: float = 1e-3
    threads: int = 2

    def validate(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0 or (key != 'impact' and value == 0):
                raise ValueError(f'{key} must be finite and positive (impact may be zero)')
        if self.samples < 3 or self.threads < 1 or self.cap_width >= self.radius:
            raise ValueError('Require >=3 clearing samples and 0 < CAP width < radius')
        if self.maximum_projectile_displacement >= self.radius-self.cap_width:
            raise ValueError('Projectile displacement limit must lie inside the absorber onset')
        if self.minimum_separation <= self.radius or self.flight_end <= self.minimum_separation:
            raise ValueError('Require flight_end > minimum_separation > radius')


def water_geometry():
    """NIST CCCBDB Cartesian geometry, oxygen shifted to origin, rotated to xy.

    Source: https://cccbdb.nist.gov/expgeom2x.asp?casno=7732185
    This orientation is explicitly named here, not assigned a Hong figure label.
    """
    return np.array([[0., 0., 0.], [0.5865, 0.7572, 0.], [0.5865, -0.7572, 0.]]) / BOHR_ANGSTROM


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def common(settings, pseudo, *, sphere=False):
    mass_u = PROTON_MASS_AU * physical_constants['electron mass in u'][0]
    text = f'''# Independent H+--H2O calculation, atomic input and output units.
UnitsOutput = atomic
PeriodicDimensions = 0
Spacing = {settings.spacing:.16g}
DerivativesOrder = 4
# Avoid the global METIS graph for the extended collision box.
MeshPartitionPackage = part_hilbert
SpinComponents = polarized
XCFunctional = lda_x + lda_c_pz
%Species
 "O" | species_pseudo | file | "{pseudo / 'O.psf'}" | lmax | 1 | lloc | 0
 "H" | species_pseudo | file | "{pseudo / 'H.psf'}" | lmax | 0 | lloc | 0 | mass | {mass_u:.16g}
%
'''
    if sphere:
        text += f'BoxShape = sphere\nRadius = {settings.radius:.16g}\n'
    else:
        padding = settings.radius + settings.cap_width + settings.box_margin
        half = [(settings.initial + settings.flight_end)/2 + padding, settings.impact/2 + padding, padding]
        center = [(settings.flight_end-settings.initial)/2, settings.impact/2, 0.]
        text += 'BoxShape = parallelepiped\n%Lsize\n ' + ' | '.join(map(str, half)) + '\n%\n'
        text += '%BoxCenter\n ' + ' | '.join(map(str, center)) + '\n%\n'
    return text


def coordinates(positions):
    return '%Coordinates\n' + ''.join(
        f' "{element}" | ' + ' | '.join(f'{x:.16g}' for x in point) + ' | yes\n'
        for element, point in zip(['O', 'H', 'H', 'H'], positions)) + '%\n'


def output_block(interval):
    return f'''%Output
 wfs | "output_format" | binary
 density | "output_format" | mesh_index
%
OutputWfsNumber = "1-4"
OutputInterval = {interval}
'''


def ground_input(settings, pseudo):
    return common(settings, pseudo) + coordinates(water_geometry()) + '''CalculationMode = gs
FromScratch = yes
ExcessCharge = 0
MoveIons = no
ConvRelDens = 1e-8
ConvEigenError = yes
EigensolverTolerance = 1e-9
EigensolverMaxIter = 100
MaximumIter = 400
''' + output_block(1)


def td_input(settings, pseudo, positions, velocities, steps, interval, *, sphere=False, files=None, probe=False):
    text = common(settings, pseudo, sphere=sphere) + coordinates(positions)
    text += f'''CalculationMode = td
FromScratch = yes
ExcessCharge = 1
MoveIons = yes
RecalculateGSDuringEvolution = no
TDPropagator = etrs
TDTimeStep = {settings.dt:.16g}
TDMaxSteps = {steps}
AbsorbingBoundaries = cap
ABWidth = {settings.cap_width:.16g}
ABCapHeight = {-settings.cap_height:.16g}
RestartWriteInterval = {max(1, steps)}
TDOutputComputeInterval = {interval}
TDEnergyUpdateIter = {interval}
%TDOutput
 geometry
 energy
%
%Velocities
'''
    text += ''.join(f' "{element}" | ' + ' | '.join(f'{x:.16g}' for x in v) + '\n'
                    for element, v in zip(['O', 'H', 'H', 'H'], velocities)) + '%\n'
    if files is not None or probe:
        text += 'OnlyUserDefinedInitialStates = yes\n%UserDefinedStates\n'
        for k, spin in enumerate(('up', 'down'), 1):
            for state in range(1, 5):
                source = 'formula | "0.01*exp(-r^2)"' if probe else f'file | "{files[spin][state-1]}"'
                text += f' 1 | {state} | {k} | {source} | normalize_no\n'
        text += '%\n'
    return text + output_block(interval)


def exports(directory):
    files = {spin: [Path(directory) / f'wf-k{k:06d}-st{state:05d}.obf' for state in range(1, 5)]
             for k, spin in enumerate(('up', 'down'), 1)}
    if any(not p.is_file() for group in files.values() for p in group):
        raise RuntimeError(f'Missing occupied spin orbitals: {directory}')
    return files


def snapshot(directory, step):
    return Path(directory) / 'output_iter' / f'td.{step:07d}'


def nuclear_frame(directory, step):
    rows = np.loadtxt(Path(directory) / 'td.general/coordinates', ndmin=2)
    match = rows[rows[:, 0] == step]
    if len(match) != 1 or not np.isfinite(match).all():
        raise RuntimeError('Missing or invalid measured nuclear frame')
    row = match[0]
    return float(row[1]), row[2:14].reshape(4, 3), row[14:26].reshape(4, 3)


def overlaps(files, points, spacing):
    return [orbital_overlap(files[s], points, spacing**3) for s in ('up', 'down')]


def matrix_error(a, b):
    return max(float(np.max(np.abs(x-y))) for x, y in zip(a, b))


def launch(executable, directory, name, text, settings, receipt, root):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}.inp').write_text(text)
    (directory / 'inp').write_text(text)
    log = directory / f'{name}.log'
    entry = {'stage': str(directory.relative_to(root) / name), 'input_sha256': file_sha256(directory / 'inp')}
    receipt['stages'].append(entry)
    write_json(root / 'result.json', receipt)
    started = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.Popen(receipt['launcher'] + [executable], cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, OMP_NUM_THREADS=str(settings.threads),
                                            OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='FALSE'))
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    entry.update(returncode=code, seconds=time.monotonic()-started, log_sha256=file_sha256(log))
    write_json(root / 'result.json', receipt)
    if code:
        raise RuntimeError(f'Octopus failed: {log}')
    if name == 'gs' and ('SCF converged' not in log.read_text() or 'not fully converged' in log.read_text()):
        raise RuntimeError(f'Ground state did not converge: {log}')


def run(settings, output, executable, pseudo_source, launcher=()):
    settings.validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pseudo = output / 'pseudopotentials'
    pseudo.mkdir()
    for element in ('H', 'O'):
        shutil.copyfile(Path(pseudo_source) / f'{element}.psf', pseudo / f'{element}.psf')
    input_pseudo = Path('../pseudopotentials')
    receipt = {'launcher': list(launcher), 'schema_version': 1, 'status': 'running', 'settings': asdict(settings),
               'reference_doi': '10.1103/PhysRevA.93.062706',
               'geometry_source': 'https://cccbdb.nist.gov/expgeom2x.asp?casno=7732185',
               'orientation': 'xy_bisector_positive_x', 'water_positions_bohr': water_geometry().tolist(),
               'energy_ev_total': 1000., 'xc': 'lda_x + lda_c_pz',
               'executable_sha256': file_sha256(Path(executable)),
               'version': subprocess.check_output([executable, '--version'], text=True),
               'source_sha256': {p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name('frame.py'), Path(__file__).with_name('capture.py'))},
               'pseudopotential_sha256': {p.name: file_sha256(p) for p in pseudo.iterdir()},
               'stages': [], 'limitations': [
                   'Independent setup; historical pseudopotential identities and geometry are not established.',
                   'Larger rectangular laboratory box precedes the spherical projectile-frame calculation.',
                   'Finite separation, interpolation, grid, timestep, CAP and clearing duration require convergence.',
                   'Cropping removes target electron density; residual electrostatic effects require separation convergence.',
                   'Post-handoff nonlocal pseudopotential Galilean covariance is not established.',
                   'P(1) is a Kohn-Sham determinant regional count inclusive over residual-target states.',
                   'No reference comparison, excitation or exclusive ionization result is claimed.']}
    lab, probe, post = (output / name for name in ('collision', 'mesh', 'projectile'))
    try:
        launch(executable, lab, 'gs', ground_input(settings, input_pseudo), settings, receipt, output)
        initial_points, _ = read_mesh(lab / 'static/density-sp1.mesh_index', [settings.spacing]*3)
        initial_files = exports(lab / 'static')
        gram = overlaps(initial_files, initial_points, settings.spacing)
        error = matrix_error(gram, [np.eye(4)]*2)
        receipt['initial_orthogonality_error'] = error
        if error > 1e-6:
            raise RuntimeError('Initial occupied orbitals are not orthonormal')
        positions = np.vstack((water_geometry(), [-settings.initial, settings.impact, 0.]))
        velocities = np.zeros((4, 3))
        velocity = math.sqrt(2*1000/EH/PROTON_MASS_AU)
        velocities[-1, 0] = velocity
        steps = math.ceil((settings.initial+settings.flight_end)/velocity/settings.dt)
        launch(executable, lab, 'collision', td_input(settings, input_pseudo, positions, velocities, steps, steps), settings, receipt, output)
        start_files = exports(snapshot(lab, 0))
        initial_error = max(float(np.max(np.abs(read_obf(a)-read_obf(b))))
                            for spin in initial_files for a, b in zip(initial_files[spin], start_files[spin]))
        receipt['neutral_target_restart_max_amplitude_error'] = initial_error
        if initial_error > 1e-10:
            raise RuntimeError('Collision did not start from unchanged neutral-water orbitals')
        time_au, positions, velocities = nuclear_frame(lab, steps)
        origin, boost = positions[-1].copy(), velocities[-1].copy()
        separation = float(np.min(np.linalg.norm(positions[:-1]-origin, axis=1)))
        receipt['handoff'] = {'time_au': time_au, 'origin_bohr': origin.tolist(), 'velocity_au': boost.tolist(),
                              'minimum_target_distance_bohr': separation}
        if separation < settings.minimum_separation:
            raise RuntimeError('Insufficient measured separation; increase flight_end')
        # The destination sphere must lie inside the laboratory box, before its CAP.
        padding = settings.radius + settings.cap_width + settings.box_margin
        center = np.array([(settings.flight_end-settings.initial)/2, settings.impact/2, 0.])
        half = np.array([(settings.initial+settings.flight_end)/2+padding, settings.impact/2+padding, padding])
        clearance = float(np.min(half-settings.cap_width-np.abs(origin-center)))
        receipt['handoff']['laboratory_absorber_clearance_bohr'] = clearance
        if clearance < settings.radius:
            raise RuntimeError('Destination sphere intersects the laboratory absorber; enlarge the box')
        positions, velocities = positions-origin, velocities-boost
        # A one-step mesh-probe run exports the exact native destination mesh.
        launch(executable, probe, 'mesh', td_input(settings, input_pseudo, positions, velocities, 1, 1, sphere=True, probe=True), settings, receipt, output)
        mesh = snapshot(probe, 0) / 'density-sp1.mesh_index'
        post.mkdir()
        shutil.copyfile(mesh, post / 'mesh.mesh_index')
        points, _ = read_mesh(post / 'mesh.mesh_index', [settings.spacing]*3)
        final_files = exports(snapshot(lab, steps))
        translated = translate_exports(final_files, initial_points, points, settings.spacing, origin, boost, post / 'orbitals')
        cropped_gram = overlaps(translated, points, settings.spacing)
        number_distribution(cropped_gram)  # reject noncontractive interpolation, never renormalize
        receipt['handoff']['retained_electron_norm'] = float(sum(np.trace(g).real for g in cropped_gram))
        receipt['handoff']['source_electron_norm'] = float(sum(np.trace(g).real for g in overlaps(final_files, initial_points, settings.spacing)))
        # Probe output has no physical content and is not a retained result.
        shutil.rmtree(probe / 'output_iter')
        for name in ('restart', 'td.general', 'static'):
            if (probe / name).exists(): shutil.rmtree(probe / name)
        interval = math.ceil(settings.clearing_time/settings.dt/(settings.samples-1))
        post_steps = interval*(settings.samples-1)
        launch(executable, post, 'clearing', td_input(settings, input_pseudo, positions, velocities, post_steps, interval, sphere=True, files={spin: [Path('orbitals')/p.name for p in paths] for spin, paths in translated.items()}), settings, receipt, output)
        imported = overlaps(exports(snapshot(post, 0)), points, settings.spacing)
        import_error = matrix_error(cropped_gram, imported)
        receipt['handoff']['import_overlap_error'] = import_error
        density_norm = sum(np.loadtxt(snapshot(post, 0)/f'density-sp{k}.mesh_index', usecols=4).sum() * settings.spacing**3 for k in (1, 2))
        receipt['handoff']['import_density_electron_count'] = float(density_norm)
        if abs(density_norm-receipt['handoff']['retained_electron_norm']) > 1e-8:
            raise RuntimeError('Imported density disagrees with singly occupied spin orbitals')
        if import_error > 1e-10:
            raise RuntimeError('Orbital import changed overlaps or normalized depleted states')
        rows = []
        for step in range(0, post_steps+1, interval):
            elapsed, measured, _ = nuclear_frame(post, step)
            center = measured[-1]
            distance = float(np.min(np.linalg.norm(measured[:-1], axis=1)))
            if distance <= settings.radius or np.linalg.norm(center) > settings.maximum_projectile_displacement:
                raise RuntimeError('Nuclei no longer satisfy the separated-region geometry')
            probabilities = number_distribution(overlaps(exports(snapshot(post, step)), points, settings.spacing))
            rows.append({'time_au': time_au+elapsed, 'clearing_time_au': elapsed,
                         'projectile_position_bohr': center.tolist(), 'target_positions_bohr': measured[:-1].tolist(),
                         'probabilities_by_electron_count': probabilities.tolist(),
                         'p_one_electron': float(probabilities[1]), 'mean_electrons': float(np.dot(np.arange(9), probabilities)),
                         'p_more_than_two': float(probabilities[3:].sum())})
        late = rows[-3:]
        variation = float(np.max(np.ptp([r['probabilities_by_electron_count'] for r in late], axis=0)))
        analysis = {'schema_version': 1, 'status': 'analyzed_not_literature_validated',
                    'collision': {'energy_ev_total': 1000., 'orientation': receipt['orientation'], 'impact_parameter_bohr': settings.impact},
                    'frames': rows, 'stationarity_frames': 3, 'maximum_probability_change': variation,
                    'stationarity_passed': variation <= settings.stationarity_tolerance,
                    'stationarity_tolerance': settings.stationarity_tolerance,
                    'method': 'determinant electron-number counting in the entire post-handoff sphere',
                    'limitations': receipt['limitations']}
        used = [post/'mesh.mesh_index', lab/'td.general/coordinates', post/'td.general/coordinates']
        used += [p for group in initial_files.values() for p in group]
        used += [p for group in final_files.values() for p in group]
        used += [p for step in range(0, post_steps+1, interval) for group in exports(snapshot(post, step)).values() for p in group]
        analysis['source_sha256'] = {str(p.relative_to(output)): file_sha256(p) for p in used}
        write_json(output / 'capture.json', analysis)
        receipt['status'] = 'executed_not_numerically_validated'
        receipt['stationarity_passed'] = analysis['stationarity_passed']
        receipt['capture_sha256'] = file_sha256(output / 'capture.json')
        return receipt
    except BaseException as exc:
        receipt.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(output / 'result.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', default='octopus')
    parser.add_argument('--pseudopotentials', type=Path, help='PSF directory; otherwise inferred from the executable prefix')
    parser.add_argument('--launcher', default='', help='Optional MPI launcher, e.g. srun --ntasks=32; no shell evaluation')
    parser.add_argument('--prepare-only', action='store_true', help='Write complete initial inputs and settings without running Octopus')
    defaults = Settings()
    for name, value in asdict(defaults).items():
        parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args()
    settings = Settings(**{name: getattr(args, name) for name in asdict(defaults)})
    settings.validate()
    executable = shutil.which(args.executable)
    if executable is None: parser.error('Octopus executable not found')
    executable = str(Path(executable).resolve())
    pseudo = args.pseudopotentials or Path(executable).parents[1] / 'share/octopus/pseudopotentials/PSF'
    if not all((pseudo / f'{element}.psf').is_file() for element in ('H', 'O')):
        parser.error('H.psf and O.psf are required')
    if args.prepare_only:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'gs.inp').write_text(ground_input(settings, pseudo.resolve()))
        positions = np.vstack((water_geometry(), [-settings.initial, settings.impact, 0.]))
        velocities = np.zeros((4, 3)); velocities[-1, 0] = math.sqrt(2*1000/EH/PROTON_MASS_AU)
        steps = math.ceil((settings.initial+settings.flight_end)/velocities[-1, 0]/settings.dt)
        (args.output / 'collision.inp').write_text(td_input(settings, pseudo.resolve(), positions, velocities, steps, steps))
        write_json(args.output / 'settings.json', asdict(settings))
    else:
        run(settings, args.output, executable, pseudo, shlex.split(args.launcher))
    print(args.output)


if __name__ == '__main__':
    main()
