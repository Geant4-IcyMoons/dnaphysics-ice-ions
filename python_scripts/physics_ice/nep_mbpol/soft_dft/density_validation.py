"""Common-grid density diagnostics for reciprocal CDFT branch solutions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class CubeDensity:
    atom_count: int
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    axes: tuple[tuple[float, float, float], ...]
    values: np.ndarray


def _tokens(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        yield from line.split()


def read_cube_density(path: Path) -> CubeDensity:
    """Read one scalar Gaussian-cube field without changing its units."""

    with path.open("r", encoding="utf-8") as handle:
        try:
            next(handle)
            next(handle)
            header = next(handle).split()
        except StopIteration as error:
            raise ValueError(f"Truncated cube header: {path}") from error
        if len(header) < 4:
            raise ValueError(f"Invalid cube origin record: {path}")
        signed_atom_count = int(header[0])
        atom_count = abs(signed_atom_count)
        origin = tuple(float(value) for value in header[1:4])
        counts: list[int] = []
        axes: list[tuple[float, float, float]] = []
        for _ in range(3):
            try:
                row = next(handle).split()
            except StopIteration as error:
                raise ValueError(f"Truncated cube grid: {path}") from error
            if len(row) < 4:
                raise ValueError(f"Invalid cube grid record: {path}")
            counts.append(abs(int(row[0])))
            axes.append(tuple(float(value) for value in row[1:4]))
        if any(count < 1 for count in counts):
            raise ValueError(f"Cube grid dimensions must be positive: {path}")
        for _ in range(atom_count):
            try:
                next(handle)
            except StopIteration as error:
                raise ValueError(f"Truncated cube atom list: {path}") from error
        # A negative atom count introduces an orbital-ID record in the cube
        # convention. Electronic-density cubes are scalar and must not use it.
        if signed_atom_count < 0:
            raise ValueError("Orbital cube files are not scalar density diagnostics.")
        values = np.fromiter(_tokens(iter(handle)), dtype=float)
    expected = math.prod(counts)
    if values.size != expected or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Cube data size/finite-value mismatch: expected {expected}, "
            f"found {values.size}."
        )
    return CubeDensity(
        atom_count=atom_count,
        origin=origin,
        shape=tuple(counts),
        axes=tuple(axes),
        values=values.reshape(tuple(counts)),
    )


def density_rms_comparison(
    first: CubeDensity,
    second: CubeDensity,
) -> dict[str, float | int | list[int]]:
    """Compare two same-geometry density fields on exactly the same grid."""

    if first.shape != second.shape:
        raise ValueError("Density cubes use different grid dimensions.")
    if first.atom_count != second.atom_count:
        raise ValueError("Density cubes use different atom counts.")
    if not np.allclose(first.origin, second.origin, rtol=0.0, atol=1.0e-12):
        raise ValueError("Density cubes use different origins.")
    if not np.allclose(first.axes, second.axes, rtol=0.0, atol=1.0e-12):
        raise ValueError("Density cubes use different grid vectors.")
    difference = first.values - second.values
    rms = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(
        np.sqrt(0.5 * np.mean(np.square(first.values) + np.square(second.values)))
    )
    relative = rms / reference_rms if reference_rms > 0.0 else (0.0 if rms == 0.0 else math.inf)
    return {
        "voxel_count": int(first.values.size),
        "shape": list(first.shape),
        "absolute_rms": rms,
        "reference_rms": reference_rms,
        "relative_rms": relative,
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
    }


def compare_density_cube_files(first: Path, second: Path) -> dict[str, object]:
    """Read and compare two cube files, retaining their exact provenance."""

    result: dict[str, object] = density_rms_comparison(
        read_cube_density(first), read_cube_density(second)
    )
    result["first"] = str(first.resolve())
    result["second"] = str(second.resolve())
    return result
