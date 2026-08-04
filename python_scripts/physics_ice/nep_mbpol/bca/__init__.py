"""Binary-collision building blocks for phase-resolved ion--ice scattering."""

from .config import (
    DEFAULT_ENERGY_MAX_EV,
    DEFAULT_ENERGY_MIN_EV,
    DEFAULT_ENERGY_POINTS,
    DEFAULT_IMPACT_POINTS,
    DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    DEFAULT_PROJECTILES,
    DEFAULT_QUADRATURE_ORDER,
    DEFAULT_WORKERS,
)
from .scattering import (
    CollisionResult,
    NLHCollisionKernel,
    PairKinematics,
    hard_cross_section_angstrom2,
    maximum_impact_parameter_angstrom,
    pair_kinematics,
    solve_nlh_collision,
)
from .structure import IceStructure, StructureValidationError, load_ice_structure

__all__ = [
    "CollisionResult",
    "DEFAULT_ENERGY_MAX_EV",
    "DEFAULT_ENERGY_MIN_EV",
    "DEFAULT_ENERGY_POINTS",
    "DEFAULT_IMPACT_POINTS",
    "DEFAULT_MINIMUM_TURNING_POTENTIAL_EV",
    "DEFAULT_PROJECTILES",
    "DEFAULT_QUADRATURE_ORDER",
    "DEFAULT_WORKERS",
    "IceStructure",
    "NLHCollisionKernel",
    "PairKinematics",
    "StructureValidationError",
    "hard_cross_section_angstrom2",
    "load_ice_structure",
    "maximum_impact_parameter_angstrom",
    "pair_kinematics",
    "solve_nlh_collision",
]
