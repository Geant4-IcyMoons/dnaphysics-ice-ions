"""Binary-collision building blocks for phase-resolved ion--ice scattering."""

from .config import (
    DEFAULT_AXIS_RELATIVE_TOLERANCE,
    DEFAULT_BASE_ENERGY_POINTS,
    DEFAULT_ENERGY_MAX_EV,
    DEFAULT_ENERGY_MIN_EV,
    DEFAULT_MAX_ENERGY_POINTS,
    DEFAULT_MAX_IMPACT_POINTS,
    DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    DEFAULT_PROJECTILES,
    DEFAULT_QUADRATURE_ORDER,
    DEFAULT_WORKERS,
)
from .scattering import (
    CollisionResult,
    NLHCollisionKernel,
    PairKinematics,
    TwoBodyOutcome,
    hard_cross_section_angstrom2,
    maximum_impact_parameter_angstrom,
    pair_kinematics,
    solve_nlh_collision,
    two_body_outcome_from_cm_angle,
    turning_threshold_radius_angstrom,
)
from .structure import IceStructure, StructureValidationError, load_ice_structure

__all__ = [
    "CollisionResult",
    "DEFAULT_AXIS_RELATIVE_TOLERANCE",
    "DEFAULT_BASE_ENERGY_POINTS",
    "DEFAULT_ENERGY_MAX_EV",
    "DEFAULT_ENERGY_MIN_EV",
    "DEFAULT_MAX_ENERGY_POINTS",
    "DEFAULT_MAX_IMPACT_POINTS",
    "DEFAULT_MINIMUM_TURNING_POTENTIAL_EV",
    "DEFAULT_PROJECTILES",
    "DEFAULT_QUADRATURE_ORDER",
    "DEFAULT_WORKERS",
    "IceStructure",
    "NLHCollisionKernel",
    "PairKinematics",
    "TwoBodyOutcome",
    "StructureValidationError",
    "hard_cross_section_angstrom2",
    "load_ice_structure",
    "maximum_impact_parameter_angstrom",
    "pair_kinematics",
    "solve_nlh_collision",
    "two_body_outcome_from_cm_angle",
    "turning_threshold_radius_angstrom",
]
