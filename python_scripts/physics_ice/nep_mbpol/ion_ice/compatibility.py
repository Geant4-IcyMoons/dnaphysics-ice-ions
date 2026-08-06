"""Consistency checks between the unified registry and active backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

from bca import config as bca_config
from nlh import get_coefficients, supported_projectile_target_pairs

from .registry import (
    NEP_MBPOL_ROOT,
    REPOSITORY_ROOT,
    canonical_projectile,
    get_phase,
    projectile_registry,
)
from .resources import load_resource_profiles


@dataclass(frozen=True)
class CompatibilityCheck:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def backend_compatibility_checks() -> tuple[CompatibilityCheck, ...]:
    """Verify that the live legacy backends still match the shared registry."""

    registry = projectile_registry()
    nlh_symbols = tuple(
        definition.symbol
        for definition in sorted(registry.values(), key=lambda item: item.atomic_number)
        if definition.component_status("nlh_pair_potential") == "implemented"
    )
    checks: list[CompatibilityCheck] = [
        CompatibilityCheck(
            "nlh_projectile_set",
            nlh_symbols == bca_config.DEFAULT_PROJECTILES,
            f"registry={nlh_symbols}; backend={bca_config.DEFAULT_PROJECTILES}",
        )
    ]
    for symbol in nlh_symbols:
        definition = registry[symbol]
        backend_mass = bca_config.ISOTOPE_MASS_U.get(symbol)
        checks.append(
            CompatibilityCheck(
                f"isotope_mass_{symbol}",
                backend_mass is not None
                and math.isclose(
                    backend_mass,
                    definition.default_isotope.neutral_atomic_mass_u,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                (
                    f"registry={definition.default_isotope.neutral_atomic_mass_u}; "
                    f"backend={backend_mass}"
                ),
            )
        )
        for alias, backend_symbol in bca_config.ELEMENT_ALIASES.items():
            if backend_symbol != symbol:
                continue
            registry_symbol = canonical_projectile(alias)
            checks.append(
                CompatibilityCheck(
                    f"backend_alias_{symbol}_{alias}",
                    registry_symbol == backend_symbol,
                    f"registry={registry_symbol}; backend={backend_symbol}",
                )
            )
        for target in ("H", "O"):
            try:
                get_coefficients(symbol, target)
                available = True
            except (ValueError, RuntimeError):
                available = False
            checks.append(
                CompatibilityCheck(
                    f"nlh_pair_{symbol}_{target}",
                    available,
                    "published coefficient row present"
                    if available
                    else "coefficient row missing",
                )
            )
    expected_pairs = tuple(
        (symbol, target) for symbol in nlh_symbols for target in ("H", "O")
    )
    checks.append(
        CompatibilityCheck(
            "nlh_pair_inventory",
            supported_projectile_target_pairs() == expected_pairs,
            f"expected={expected_pairs}; backend={supported_projectile_target_pairs()}",
        )
    )

    framework = json.loads(
        (NEP_MBPOL_ROOT / "ion_ice" / "framework.json").read_text(
            encoding="utf-8"
        )
    )
    contracts = framework["numerical_contracts"]
    checks.extend(
        (
            CompatibilityCheck(
                "kernel_axis_tolerance",
                contracts["kernel_axis_relative_tolerance"]
                == bca_config.DEFAULT_AXIS_RELATIVE_TOLERANCE,
                "registry and BCA kernel-axis tolerances agree",
            ),
            CompatibilityCheck(
                "hard_boundary",
                framework["physical_boundaries"][
                    "nlh_preferred_hard_boundary_ev"
                ]
                == bca_config.DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
                "registry and BCA hard boundaries agree",
            ),
            CompatibilityCheck(
                "resource_profile_pointer",
                framework.get("pbs_resource_profiles") == "resources.json",
                str(framework.get("pbs_resource_profiles")),
            ),
        )
    )
    phase = get_phase("hexagonal_ih_100k")
    registry_path = phase.resolve(phase.structure_registry)
    checks.append(
        CompatibilityCheck(
            "hexagonal_structure_registry",
            registry_path is not None and registry_path.is_file(),
            str(registry_path),
        )
    )

    resource_profiles, _, _ = load_resource_profiles()
    resource_scripts = {
        "ice_structure_generation": (
            "pbs/run_nep_mbpol_amorphous.pbs",
            "pbs/run_nep_mbpol_hexagonal.pbs",
        ),
        "nlh_kernel_and_benchmark": (
            "pbs/generate_nlh_collision_kernels.pbs",
            "pbs/run_ion_ice_nlh_kernel.pbs",
        ),
        "hard_transport": (
            "pbs/run_nlh_hard_collision_trajectories.pbs",
            "pbs/run_adaptive_nlh_particle.pbs",
            "pbs/run_ion_ice_hard_transport.pbs",
        ),
        "soft_molecular_dft": ("pbs/run_adaptive_charge_resolved_dft.pbs",),
        "soft_dft_shard": ("pbs/run_charge_resolved_dft.pbs",),
    }
    for stage, paths in resource_scripts.items():
        expected = resource_profiles[stage]
        for relative in paths:
            path = REPOSITORY_ROOT / relative
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            match = re.search(r"^#PBS -l (select=.*)$", text, flags=re.MULTILINE)
            actual = match.group(1) if match else None
            tokens = {} if actual is None else {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in actual.split(":")[1:]
                if "=" in item
            }
            passed = (
                tokens.get("ncpus") == str(expected.ncpus)
                and tokens.get("mem") == f"{expected.memory_gb}gb"
                and int(tokens.get("ngpus", "0")) == expected.ngpus
                and all(tokens.get(key) == value for key, value in expected.select_extras)
            )
            checks.append(
                CompatibilityCheck(
                    f"pbs_resources_{path.name}",
                    passed,
                    f"profile={expected.select}; script={actual}",
                )
            )
    return tuple(checks)
