"""Load the canonical projectile and ice-phase registries."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .schema import (
    PhaseDefinition,
    ProjectileDefinition,
    load_phase_definition,
    load_projectile_definition,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
NEP_MBPOL_ROOT = PACKAGE_ROOT.parent
REPOSITORY_ROOT = NEP_MBPOL_ROOT.parents[2]
SPECIES_DIRECTORY = PACKAGE_ROOT / "species"
PHASE_DIRECTORY = PACKAGE_ROOT / "phases"


@lru_cache(maxsize=1)
def projectile_registry() -> dict[str, ProjectileDefinition]:
    loaded = [
        load_projectile_definition(path)
        for path in sorted(SPECIES_DIRECTORY.glob("*.json"))
    ]
    definitions = {
        definition.symbol: definition
        for definition in sorted(loaded, key=lambda item: item.atomic_number)
    }
    if not definitions:
        raise RuntimeError("The ion--ice projectile registry is empty.")
    for filename_symbol, definition in definitions.items():
        if filename_symbol != definition.symbol:
            raise ValueError(
                f"Species filename {filename_symbol}.json does not match "
                f"symbol {definition.symbol}."
            )
    aliases: dict[str, str] = {}
    for symbol, definition in definitions.items():
        for alias in definition.aliases:
            normalized = alias.strip().lower()
            previous = aliases.setdefault(normalized, symbol)
            if previous != symbol:
                raise ValueError(
                    f"Projectile alias {alias!r} is shared by {previous} and {symbol}."
                )
    return definitions


@lru_cache(maxsize=1)
def phase_registry() -> dict[str, PhaseDefinition]:
    definitions = {
        path.stem: load_phase_definition(path)
        for path in sorted(PHASE_DIRECTORY.glob("*.json"))
    }
    if not definitions:
        raise RuntimeError("The ion--ice phase registry is empty.")
    for filename_id, definition in definitions.items():
        if filename_id != definition.phase_id:
            raise ValueError(
                f"Phase filename {filename_id}.json does not match ID "
                f"{definition.phase_id}."
            )
    return definitions


def canonical_projectile(value: str) -> str:
    normalized = value.strip().lower()
    for symbol, definition in projectile_registry().items():
        if normalized in {alias.lower() for alias in definition.aliases}:
            return symbol
    available = ", ".join(projectile_registry())
    raise ValueError(
        f"Projectile {value!r} is not registered. Available: {available}."
    )


def get_projectile(value: str) -> ProjectileDefinition:
    return projectile_registry()[canonical_projectile(value)]


def get_phase(phase_id: str) -> PhaseDefinition:
    try:
        return phase_registry()[phase_id]
    except KeyError as exc:
        available = ", ".join(phase_registry())
        raise ValueError(
            f"Ice phase {phase_id!r} is not registered. Available: {available}."
        ) from exc
