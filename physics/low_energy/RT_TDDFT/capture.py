"""Electron-number counting for an initially normalized Kohn--Sham determinant.

Hong et al., PRA 93, 062706 (2016), Eqs. 14--16. The coefficients of
 det(I - S_A + z S_A) give the electron-number distribution in region A.
This is a determinant approximation, not an exact interacting probability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


from .prepare import file_sha256 as sha256


def number_distribution(overlaps, *, tolerance=1e-8):
    """Convolve counting polynomials of independent, integer-occupied spin blocks.

    Do not normalize CAP-depleted orbitals: I-S_A includes electrons outside A,
    including norm absorbed at the boundary. Off-diagonal overlaps are essential.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be positive and finite')
    distribution = np.array([1.])
    spectra = []
    for block in overlaps:
        block = np.asarray(block, dtype=complex)
        if block.ndim != 2 or block.shape[0] != block.shape[1] or not block.size:
            raise ValueError('Each spin overlap must be a nonempty square matrix')
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite overlap')
        if np.max(np.abs(block - block.conj().T)) > tolerance:
            raise ValueError('Overlap is not Hermitian')
        eigenvalues = np.linalg.eigvalsh((block + block.conj().T) / 2)
        if eigenvalues.min() < -tolerance or eigenvalues.max() > 1 + tolerance:
            raise ValueError('Overlap is not a contraction: require 0 <= S_A <= I')
        eigenvalues = np.clip(eigenvalues, 0, 1)  # roundoff only, within tolerance
        spectra.extend(eigenvalues.tolist())
        for value in eigenvalues:
            distribution = np.convolve(distribution, [1 - value, value])
    if not spectra:
        raise ValueError('No occupied spin orbitals supplied')
    return distribution


def read_obf(path):
    """Memory-map Octopus v0 .obf, preserving complex amplitudes and endianness.

    Format: Octopus 16.4 src/basic/io_binary.c, 64-byte header. No Fortran
    record markers. Binary output stores atomic-unit amplitudes.
    """
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(64)
    if len(header) != 64 or header[:6] != b'pulpo\0' or header[7] != 0:
        raise ValueError(f'Unsupported Octopus binary header: {path}')
    endian = next((e for e in ('<', '>') if struct.unpack_from(e+'I', header, 8)[0] == 1), None)
    if endian is None:
        raise ValueError('Invalid OBF integer sentinel')
    if (struct.unpack_from(endian+'f', header, 12)[0] != 1 or
        struct.unpack_from(endian+'Q', header, 16)[0] != 1 or
        struct.unpack_from(endian+'d', header, 24)[0] != 1):
        raise ValueError('Unsupported mixed-endian OBF header')
    count, kind = struct.unpack_from(endian+'QI', header, 32)
    types = {0: 'f4', 1: 'f8', 2: 'c8', 3: 'c16'}
    if kind not in types or count == 0:
        raise ValueError('Expected a real or complex orbital array')
    dtype = np.dtype(endian + types[kind])
    if path.stat().st_size != 64 + count * dtype.itemsize:
        raise ValueError('Truncated or overlong OBF payload')
    return np.memmap(path, dtype=dtype, mode='r', offset=64, shape=(count,))


def read_mesh(path, spacing):
    """Read native mesh_index coordinates exported with UnitsOutput=atomic."""
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Three positive Cartesian grid spacings in bohr are required')
    table = np.loadtxt(path, comments='#', ndmin=2)
    if table.shape[1] < 4 or not len(table) or not np.isfinite(table[:, :4]).all():
        raise ValueError('Invalid mesh_index file')
    if not np.array_equal(table[:, 0], np.arange(1, len(table) + 1)):
        raise ValueError('Mesh indices must be contiguous and in native order')
    points = table[:, 1:4]
    indices = (points - points[0]) / spacing
    if np.max(np.abs(indices - np.rint(indices))) > 1e-5:
        raise ValueError('Mesh coordinates disagree with stated Cartesian spacing')
    if len(np.unique(np.rint(indices).astype(np.int64), axis=0)) != len(points):
        raise ValueError('Duplicate mesh points')
    return points, float(np.prod(spacing))


def orbital_overlap(files, points, volume, region=None, *, chunk_size=65536):
    """Accumulate all overlaps in bounded chunks; never discard orbital phases."""
    arrays = [read_obf(path) for path in files]
    if not arrays or any(len(array) != len(points) for array in arrays):
        raise ValueError('Orbital length differs from the shared native mesh')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if region is not None and (np.asarray(region).shape != (len(points),) or np.asarray(region).dtype != bool):
        raise ValueError('Region must be a Boolean mask on the mesh')
    result = np.zeros((len(arrays), len(arrays)), dtype=complex)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        values = np.column_stack([a[start:stop] for a in arrays])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite orbital amplitudes')
        if region is not None:
            values = values[region[start:stop]]
        result += volume * (values.conj().T @ values)
    return result


def analyze_manifest(path):
    """Analyze explicit native orbital exports; no automatic claim of H0 yield.

    The supplied late frames must follow the same eight initially orthonormal
    spin orbitals. Position and boundary metadata are required, never inferred
    from a nominal straight trajectory when nuclei move under Ehrenfest forces.
    """
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config['schema_version'] != 1 or config['units'] != 'atomic':
        raise ValueError('Expected schema 1 with atomic units')
    if config['projectile'] != 'H+' or config['active_electrons'] != 8:
        raise ValueError('This benchmark requires H+ and eight active target electrons')
    collision = config['collision']
    if (collision['energy_ev_total'] != 1000 or collision['orientation'] not in 'abcdef' or
        len(collision['orientation']) != 1 or not np.isfinite(collision['impact_parameter_bohr']) or
        collision['impact_parameter_bohr'] < 0):
        raise ValueError('Expected a 1 keV case, nonnegative b, and a Figure 1 orientation a--f')
    def resolve(name):
        return (path.parent / name).resolve()
    mesh_path = resolve(config['mesh'])
    points, volume = read_mesh(mesh_path, config['spacing_bohr'])
    files_used = {path, mesh_path}
    def orbitals(frame):
        blocks = frame['orbitals']
        if set(blocks) != {'up', 'down'} or any(len(blocks[s]) != 4 for s in blocks):
            raise ValueError('Require four singly occupied orbitals per spin')
        names = [resolve(name) for spin in ('up', 'down') for name in blocks[spin]]
        if len(set(names)) != 8:
            raise ValueError('Eight distinct spin-orbital files are required')
        files_used.update(names)
        return [names[:4], names[4:]]
    initial_tolerance = float(config['initial_orthogonality_tolerance'])
    if not 0 < initial_tolerance < 1e-2:
        raise ValueError('Invalid initial orthogonality tolerance')
    errors = [float(np.max(np.abs(orbital_overlap(block, points, volume) - np.eye(4))))
              for block in orbitals(config['initial'])]
    if max(errors) > initial_tolerance:
        raise ValueError('Initial occupied orbitals are not orthonormal on the supplied mesh')
    radius = float(config['region_radius_bohr'])
    minimum_separation = float(config['minimum_separation_bohr'])
    convergence = float(config['stationarity_tolerance'])
    if config['region'] != 'projectile_centered_spherical_box':
        raise ValueError('Use the post-translation projectile-centered box of Hong et al.')
    if not all(np.isfinite(x) and x > 0 for x in (radius, minimum_separation, convergence)):
        raise ValueError('Region, separation and stationarity settings must be positive')
    frames = config['frames']
    if len(frames) < 2:
        raise ValueError('At least two distinct late frames are required')
    rows = []
    previous_exports = set()
    previous_time = -np.inf
    for frame in frames:
        time = float(frame['time_au'])
        center = np.asarray(frame['projectile_position_bohr'], float)
        targets = np.asarray(frame['target_positions_bohr'], float)
        if (not np.isfinite(time) or time <= previous_time or center.shape != (3,) or
            targets.shape != (3,3) or not np.isfinite(center).all() or not np.isfinite(targets).all()):
            raise ValueError('Frames need increasing times and finite measured nuclear coordinates')
        previous_time = time
        separation = float(np.min(np.linalg.norm(targets-center, axis=1)))
        if separation <= radius or separation < minimum_separation:
            raise ValueError('Target nuclei are not separated from the projectile analysis region')
        if np.max(np.abs(center)) > 1e-8:
            raise ValueError('Late frames must be translated to the projectile-centered frame')
        mesh_tolerance = float(np.max(config['spacing_bohr'])) * 1.01
        if (np.linalg.norm(points, axis=1).max() > radius + 1e-8 or
            np.any(points.min(axis=0) > -radius + mesh_tolerance) or
            np.any(points.max(axis=0) < radius - mesh_tolerance)):
            raise ValueError('Mesh does not match the declared spherical analysis box')
        blocks = orbitals(frame)
        exports = {name for block in blocks for name in block}
        if exports & previous_exports:
            raise ValueError('Late frames must reference distinct orbital exports')
        previous_exports.update(exports)
        matrices = [orbital_overlap(block, points, volume) for block in blocks]
        probabilities = number_distribution(matrices)
        rows.append({'time_au':time, 'minimum_target_distance_bohr':separation,
                     'probabilities_by_electron_count':probabilities.tolist(),
                     'p_one_electron':float(probabilities[1]),
                     'mean_electrons':float(np.dot(np.arange(9), probabilities)),
                     'p_more_than_two':float(probabilities[3:].sum())})
    change = float(np.max(np.ptp([row['probabilities_by_electron_count'] for row in rows], axis=0)))
    return {'schema_version':1, 'status':'analyzed_not_literature_validated',
            'collision':collision,
            'method':'determinant electron-number counting in a separated projectile region',
            'initial_orthogonality_errors':errors, 'frames':rows,
            'maximum_probability_change':change, 'stationarity_passed':change <= convergence,
            'reference_doi':'10.1103/PhysRevA.93.062706',
            'source_sha256':{str(p):sha256(p) for p in sorted(files_used)},
            'analysis_sha256':sha256(Path(__file__)),
            'limitations':['Spatial occupancy needs post-collision continuum clearing and region convergence.',
                           'P(1) is inclusive over residual-target states; it does not prove intact H2O+.',
                           'Kohn-Sham determinant counting is an approximation to interacting probabilities.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    result = analyze_manifest(args.manifest)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
