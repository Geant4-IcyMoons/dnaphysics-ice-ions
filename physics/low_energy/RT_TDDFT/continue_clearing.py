"""Extend a completed projectile-frame calculation using native TD restarts.

Each bounded stage is copied from an intact predecessor. An interrupted stage
can be retried from that predecessor without repeating the collision.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from tqdm import tqdm

from .benchmark import Settings, exports, nuclear_frame, overlaps, snapshot
from .capture import number_distribution, read_mesh
from .prepare import file_sha256


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def restart_input(text, end_step):
    for key, value in [('FromScratch', 'no'), ('TDMaxSteps', str(end_step))]:
        text, count = re.subn(rf'(?m)^{key}\s*=.*$', f'{key} = {value}', text)
        if count != 1:
            raise ValueError(f'Expected one {key}')
    return text


def extend(case, executable, launcher, end_time):
    case = case.resolve()
    receipt = json.loads((case/'result.json').read_text())
    settings = Settings(**receipt['settings'])
    if file_sha256(executable) != receipt['executable_sha256']:
        raise ValueError('Solver differs from original calculation')
    analysis = json.loads((case/'capture.json').read_text())
    original = case/'projectile'
    start = round(analysis['frames'][-1]['clearing_time_au']/settings.dt)
    interval = round((analysis['frames'][-1]['clearing_time_au']-analysis['frames'][-2]['clearing_time_au'])/settings.dt)
    end = round(end_time/settings.dt)
    chunk = start
    if end <= start or (end-start) % chunk:
        raise ValueError('End time must extend by whole original clearing stages')
    points, _ = read_mesh(original/'mesh.mesh_index', [settings.spacing]*3)
    previous = original
    input_text = (original/'clearing.inp').read_text()
    for stop in tqdm(range(start+chunk, end+1, chunk), desc='Clearing stages', unit='stage'):
        directory = case/f'clearing_{stop:07d}'
        signature = {'parent': str(previous), 'parent_restart_sha256': file_sha256(previous/'restart/td/wfns'),
                     'input_sha256': file_sha256(original/'clearing.inp'), 'end_step': stop,
                     'executable_sha256': receipt['executable_sha256'],
                     'implementation_sha256': file_sha256(Path(__file__))}
        done = directory/'capture.json'
        if directory.exists():
            if json.loads((directory/'provenance.json').read_text()) != signature:
                raise ValueError('Incompatible continuation directory')
            if done.exists():
                analysis = json.loads(done.read_text())
                previous = directory
                continue
            # Only this owned, incomplete stage is discarded; its parent is intact.
            shutil.rmtree(directory)
        directory.mkdir()
        atomic_json(directory/'provenance.json', signature)
        shutil.copytree(previous/'restart', directory/'restart')
        shutil.copytree(previous/'td.general', directory/'td.general')
        (directory/'orbitals').symlink_to(original/'orbitals', target_is_directory=True)
        saved_step = int(re.search(r'Iter\s*=\s*(\d+)', (directory/'restart/td/wfns').read_text())[1])
        if saved_step != stop-chunk:
            raise ValueError('Unexpected checkpoint iteration')
        nuclear_frame(directory, saved_step)  # Octopus requires the complete trajectory to restore nuclei.
        checkpoint = {spin: [directory/'restart/td'/f'{k:010d}.obf' for k in indices]
                      for spin, indices in [('up', range(1,5)), ('down', range(5,9))]}
        distribution = number_distribution(overlaps(checkpoint, points, settings.spacing))
        error = float(np.max(np.abs(distribution-analysis['frames'][-1]['probabilities_by_electron_count'])))
        if error > 1e-10:
            raise ValueError(f'Checkpoint probability mismatch: {error}')
        (directory/'inp').write_text(restart_input(input_text, stop))
        log = directory/'octopus.log'
        with log.open('w') as stream:
            process = subprocess.Popen(launcher+[str(executable)], cwd=directory, stdout=stream, stderr=subprocess.STDOUT)
            with tqdm(total=chunk, desc=f'TD steps to {stop}', unit='step') as progress:
                while True:
                    try:
                        code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        code = None
                    coordinates = directory/'td.general/coordinates'
                    if coordinates.exists():
                        with coordinates.open('rb') as handle:
                            handle.seek(max(0, coordinates.stat().st_size-4096))
                            lines = handle.read().splitlines()
                        for line in reversed(lines[:-1]):
                            fields = line.split()
                            if fields and fields[0].isdigit():
                                progress.update(max(0, min(chunk, int(fields[0])-saved_step)-progress.n))
                                break
                    if code is not None:
                        break
        text = log.read_text()
        if code or 'Starting from scratch' in text or 'Starting simulation from initial geometry' in text:
            raise RuntimeError(f'Continuation failed or restart not loaded: {log}')
        for step in tqdm(range(saved_step+interval, stop+1, interval), desc='Capture samples', unit='frame'):
            elapsed, positions, _ = nuclear_frame(directory, step)
            if np.linalg.norm(positions[-1]) > settings.maximum_projectile_displacement or np.min(np.linalg.norm(positions[:-1], axis=1)) <= settings.radius:
                raise RuntimeError('Separated-region geometry failed')
            probabilities = number_distribution(overlaps(exports(snapshot(directory, step)), points, settings.spacing))
            analysis['frames'].append({'time_au': receipt['handoff']['time_au']+elapsed, 'clearing_time_au': elapsed,
                'projectile_position_bohr': positions[-1].tolist(), 'target_positions_bohr': positions[:-1].tolist(),
                'probabilities_by_electron_count': probabilities.tolist(), 'p_one_electron': float(probabilities[1]),
                'mean_electrons': float(np.dot(np.arange(9), probabilities)), 'p_more_than_two': float(probabilities[3:].sum())})
            for group in exports(snapshot(directory, step)).values():
                for path in group:
                    analysis['source_sha256'][str(path.relative_to(case))] = file_sha256(path)
        variation = float(np.max(np.ptp([r['probabilities_by_electron_count'] for r in analysis['frames'][-3:]], axis=0)))
        analysis.update(maximum_probability_change=variation, stationarity_passed=variation <= settings.stationarity_tolerance)
        analysis['source_sha256'][str((directory/'td.general/coordinates').relative_to(case))] = file_sha256(directory/'td.general/coordinates')
        analysis['continuation'] = dict(signature, checkpoint_probability_error=error, returncode=code, log_sha256=file_sha256(log))
        atomic_json(done, analysis)
        print(f'Clearing {stop*settings.dt:g} au: variation={variation:.8g}, passed={analysis["stationarity_passed"]}', flush=True)
        previous = directory
    return previous/'capture.json'


if __name__ == '__main__':
    import shlex
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--launcher', default='')
    parser.add_argument('--end-time', type=float, default=600.)
    args = parser.parse_args()
    print(extend(args.case, args.executable.resolve(), shlex.split(args.launcher), args.end_time))
