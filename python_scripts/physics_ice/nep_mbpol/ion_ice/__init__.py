"""Unified registry and workflow control for the ion--ice collision model."""

from .registry import (
    NEP_MBPOL_ROOT,
    REPOSITORY_ROOT,
    canonical_projectile,
    get_phase,
    get_projectile,
    phase_registry,
    projectile_registry,
)
from .schema import (
    IonChargeState,
    PhaseDefinition,
    ProjectileDefinition,
    load_phase_definition,
    load_projectile_definition,
)

__all__ = [
    "IonChargeState",
    "NEP_MBPOL_ROOT",
    "PhaseDefinition",
    "ProjectileDefinition",
    "REPOSITORY_ROOT",
    "canonical_projectile",
    "get_phase",
    "get_projectile",
    "load_phase_definition",
    "load_projectile_definition",
    "phase_registry",
    "projectile_registry",
]
