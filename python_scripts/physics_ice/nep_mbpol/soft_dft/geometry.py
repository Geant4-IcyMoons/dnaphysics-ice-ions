"""Deterministic ion--H2O scan geometries from a validated ion definition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np

from bca.scattering import turning_threshold_radius_angstrom

from .config import ProjectileDefinition


# The fixed molecular geometry is the NIST-published TIP4P/2005 geometry. It is
# a reproducible pilot geometry, not a substitute for phase-resolved ice
# environments in the later production dataset.
WATER_OH_ANGSTROM = 0.9572
WATER_HOH_DEGREES = 104.52
WATER_GEOMETRY_PROVENANCE = {
    "source": "NIST TIP4P/2005 benchmark",
    "url": (
        "https://www.nist.gov/mml/csd/chemical-informatics-group/"
        "benchmark-results-tip4p2005-water"
    ),
    "oh_angstrom": WATER_OH_ANGSTROM,
    "hoh_degrees": WATER_HOH_DEGREES,
    "scope": "fixed isolated-molecule pilot geometry",
}


@dataclass(frozen=True)
class Orientation:
    name: str
    anchor_index: int
    anchor_element: str
    direction: tuple[float, float, float]
    definition: str


@dataclass(frozen=True)
class ScanGeometry:
    orientation: str
    anchor_index: int
    anchor_element: str
    separation_angstrom: float
    coordinates_angstrom: tuple[tuple[str, float, float, float], ...]
    minimum_pair_distance_angstrom: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _unit(vector: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(tuple(vector), dtype=float)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Orientation vector must have positive finite length.")
    return tuple(float(value) for value in array / norm)


def water_coordinates() -> np.ndarray:
    """Return O, H1, H2 with the molecular bisector along +z."""

    half_angle = math.radians(WATER_HOH_DEGREES / 2.0)
    x = WATER_OH_ANGSTROM * math.sin(half_angle)
    z = WATER_OH_ANGSTROM * math.cos(half_angle)
    return np.asarray(((0.0, 0.0, 0.0), (x, 0.0, z), (-x, 0.0, z)))


_WATER = water_coordinates()
ORIENTATIONS: tuple[Orientation, ...] = (
    Orientation(
        "oxygen_back",
        0,
        "O",
        (0.0, 0.0, -1.0),
        "from oxygen opposite the H-O-H bisector",
    ),
    Orientation(
        "hydrogen_out",
        1,
        "H",
        _unit(_WATER[1] - _WATER[0]),
        "from H1 along the outward O-H bond direction",
    ),
    Orientation(
        "hydrogen_bisector",
        0,
        "O",
        (0.0, 0.0, 1.0),
        "from oxygen between the two O-H bonds",
    ),
    Orientation(
        "in_plane_side",
        0,
        "O",
        (1.0, 0.0, 0.0),
        "from oxygen in the molecular plane perpendicular to the bisector",
    ),
    Orientation(
        "plane_normal",
        0,
        "O",
        (0.0, 1.0, 0.0),
        "from oxygen normal to the molecular plane",
    ),
)


def scan_distances(
    projectile: ProjectileDefinition, anchor_element: str
) -> tuple[float, ...]:
    """Return only the radial coordinates declared by the ion definition."""

    projectile.validate()
    anchor = str(anchor_element)
    if anchor not in {"H", "O"}:
        raise ValueError("The water anchor element must be H or O.")
    grid = projectile.scan_grid
    if grid["kind"] == "nlh_overlap":
        overlap = [
            turning_threshold_radius_angstrom(
                projectile.symbol,
                anchor,
                minimum_turning_potential_ev=float(threshold),
            )
            for threshold in grid["thresholds_ev"]
        ]
        values = (*overlap, *(float(value) for value in grid["soft_probe_angstrom"]))
    else:
        values = tuple(float(value) for value in grid["distances_angstrom"][anchor])
    return tuple(sorted({round(value, 12) for value in values}))


def build_scan_geometry(
    orientation: Orientation,
    separation_angstrom: float,
    projectile_symbol: str,
) -> ScanGeometry:
    """Build one fixed-nuclei molecular scan point."""

    separation = float(separation_angstrom)
    if not math.isfinite(separation) or separation <= 0.0:
        raise ValueError("Scan separation must be positive and finite.")
    anchor = _WATER[orientation.anchor_index]
    direction = np.asarray(orientation.direction, dtype=float)
    projectile_position = anchor + separation * direction
    distances = np.linalg.norm(_WATER - projectile_position, axis=1)
    elements = ("O", "H", "H")
    coordinates = [(projectile_symbol, *projectile_position)]
    coordinates.extend(
        (element, *position)
        for element, position in zip(elements, _WATER, strict=True)
    )
    return ScanGeometry(
        orientation=orientation.name,
        anchor_index=orientation.anchor_index,
        anchor_element=orientation.anchor_element,
        separation_angstrom=separation,
        coordinates_angstrom=tuple(
            (element, float(x), float(y), float(z))
            for element, x, y, z in coordinates
        ),
        minimum_pair_distance_angstrom=float(np.min(distances)),
    )


def build_scan_geometries(
    projectile: ProjectileDefinition,
    orientations: tuple[Orientation, ...] = ORIENTATIONS,
) -> tuple[ScanGeometry, ...]:
    """Build every fixed-nuclei molecular orientation/separation point."""

    geometries = [
        build_scan_geometry(orientation, separation, projectile.symbol)
        for orientation in orientations
        for separation in scan_distances(projectile, orientation.anchor_element)
    ]
    return tuple(geometries)
