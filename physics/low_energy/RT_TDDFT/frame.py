"""Translate native orbitals to an inertial projectile frame, without normalization."""
from pathlib import Path
import struct

import numpy as np
from scipy.ndimage import map_coordinates

from .capture import read_obf


def write_obf(path, values):
    """Write the Octopus 16.4 native complex128 format in little endian."""
    values = np.asarray(values, dtype='<c16')
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected a finite, nonempty orbital vector')
    header = bytearray(64)
    header[:6] = b'pulpo\0'
    struct.pack_into('<IfQdQI', header, 8, 1, 1., 1, 1., len(values), 3)
    with Path(path).open('xb') as stream:
        stream.write(header)
        values.tofile(stream)


def translate_orbital(values, source_points, destination_points, spacing, position, velocity):
    r"""Evaluate exp(-i v.r') psi(r'+R) in atomic units at the handoff time.

    The omitted global phase does not affect any overlap. Coordinates of every
    nucleus and their velocities must also change by -R and -v. This is one
    inertial frame change, not a continuously accelerating coordinate system.
    Cubic interpolation uses zero extension outside the source mesh. Neither
    interpolation nor cropping is unitary; callers must record and converge
    their effects. Integer grid translations are sampled exactly.
    """
    source_points = np.asarray(source_points, float)
    destination_points = np.asarray(destination_points, float)
    position, velocity = np.asarray(position, float), np.asarray(velocity, float)
    values = np.asarray(values)
    if (source_points.shape != (values.size, 3) or destination_points.ndim != 2 or
        destination_points.shape[1] != 3 or position.shape != (3,) or velocity.shape != (3,) or
        not np.isfinite(spacing) or spacing <= 0 or
        not all(np.isfinite(x).all() for x in (values, source_points, destination_points, position, velocity))):
        raise ValueError('Invalid Cartesian orbital translation')
    origin = source_points.min(axis=0)
    indices = (source_points - origin) / spacing
    if np.max(np.abs(indices - np.rint(indices))) > 1e-5:
        raise ValueError('Source mesh is inconsistent with spacing')
    indices = np.rint(indices).astype(int)
    shape = tuple(indices.max(axis=0) + 1)
    grid = np.zeros(shape, complex)
    grid[tuple(indices.T)] = values
    samples = (destination_points + position - origin) / spacing
    if np.max(np.abs(samples - np.rint(samples))) < 1e-8:
        samples = np.rint(samples).astype(int)
        inside = np.all((samples >= 0) & (samples < np.asarray(shape)), axis=1)
        result = np.zeros(len(samples), complex)
        result[inside] = grid[tuple(samples[inside].T)]
    else:
        result = map_coordinates(grid, samples.T, order=3, mode='constant', cval=0., prefilter=True)
    return result * np.exp(-1j * (destination_points @ velocity))


def translate_exports(files, source_points, destination_points, spacing, position, velocity, output):
    """Write all occupied orbitals, preserving spin labels and CAP norm loss."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    result = {}
    for spin, paths in files.items():
        result[spin] = []
        for state, path in enumerate(paths, 1):
            destination = output / f'{spin}-{state}.obf'
            write_obf(destination, translate_orbital(read_obf(path), source_points,
                      destination_points, spacing, position, velocity))
            result[spin].append(destination)
    return result
