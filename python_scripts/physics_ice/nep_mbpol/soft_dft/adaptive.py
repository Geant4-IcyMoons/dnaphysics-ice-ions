"""Adaptive radial-mesh validation for charge-resolved ion--H2O potentials."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .config import ProjectileDefinition
from .geometry import ORIENTATIONS, build_scan_geometries


PROBE_FRACTIONS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class AdaptiveSettings:
    """Numerical interpolation contract for the molecular pilot."""

    relative_tolerance: float = 0.005
    maximum_refinement_depth: int = 8
    maximum_mesh_points_per_orientation: int = 257

    def validate(self) -> None:
        if not math.isfinite(self.relative_tolerance) or not (
            0.0 < self.relative_tolerance < 1.0
        ):
            raise ValueError("relative_tolerance must lie between zero and one.")
        if self.maximum_refinement_depth < 0:
            raise ValueError("maximum_refinement_depth cannot be negative.")
        if self.maximum_mesh_points_per_orientation < 2:
            raise ValueError(
                "maximum_mesh_points_per_orientation must be at least two."
            )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "probe_fractions": list(PROBE_FRACTIONS),
            "interpolation_coordinate": "linear separation_angstrom",
            "interpolation_rule": "piecewise linear interaction energy",
            "error_normalization": (
                "maximum absolute endpoint/direct-probe interaction energy "
                "within each charge-state interval"
            ),
            "sign_mismatch_policy": "fail if direct and interpolated signs differ",
            "scope": (
                "numerical potential-table interpolation only; not DFT or "
                "physical-model accuracy"
            ),
        }


DEFAULT_ADAPTIVE_SETTINGS = AdaptiveSettings()


def point_key(charge: int, orientation: str, separation_angstrom: float) -> str:
    """Return an exact, serialization-stable key for one DFT energy."""

    return f"q{charge}|{orientation}|{float(separation_angstrom).hex()}"


def initial_mesh_state(
    projectile: ProjectileDefinition, settings: AdaptiveSettings
) -> dict[str, Any]:
    """Construct common radial meshes and unresolved base intervals."""

    settings.validate()
    curves: dict[str, Any] = {}
    for orientation in ORIENTATIONS:
        separations = sorted(
            geometry.separation_angstrom
            for geometry in build_scan_geometries(
                projectile, orientations=(orientation,)
            )
        )
        if len(separations) > settings.maximum_mesh_points_per_orientation:
            raise ValueError("The base mesh exceeds the adaptive mesh-point limit.")
        curves[orientation.name] = {
            "mesh_separations_angstrom": separations,
            "active_intervals": [
                {"lower": lower, "upper": upper, "depth": 0}
                for lower, upper in zip(separations[:-1], separations[1:])
            ],
            "accepted_intervals": [],
            "refined_intervals": [],
        }
    return {
        "status": "awaiting_calculations",
        "settings": settings.as_dict(),
        "curves": curves,
        "maximum_scaled_interpolation_error": None,
        "sign_mismatch_count": 0,
        "historical_maximum_scaled_interpolation_error": 0.0,
        "historical_sign_mismatch_count": 0,
    }


def _probe_separations(lower: float, upper: float) -> tuple[float, ...]:
    # The base NLH boundary coordinates are stored to 12 decimal places.
    # Canonicalizing derived dyadic points at the same precision makes a point
    # reached through different bisection paths byte-identical on restart.
    return tuple(
        round(lower + fraction * (upper - lower), 12)
        for fraction in PROBE_FRACTIONS
    )


def required_probe_geometries(
    mesh_state: Mapping[str, Any],
    values_ev: Mapping[str, float],
    charges: tuple[int, ...],
) -> tuple[tuple[str, float], ...]:
    """Return missing validation geometries for all unresolved intervals."""

    missing: set[tuple[str, float]] = set()
    curves = mesh_state["curves"]
    for orientation, curve in curves.items():
        for interval in curve["active_intervals"]:
            for separation in _probe_separations(
                float(interval["lower"]), float(interval["upper"])
            ):
                if any(
                    point_key(charge, orientation, separation) not in values_ev
                    for charge in charges
                ):
                    missing.add((orientation, separation))
    return tuple(sorted(missing))


def interval_interpolation_report(
    *,
    orientation: str,
    lower: float,
    upper: float,
    depth: int,
    charges: tuple[int, ...],
    values_ev: Mapping[str, float],
    tolerance: float,
) -> dict[str, Any]:
    """Validate one interval at three direct, non-table probe positions."""

    worst_error = -1.0
    worst: dict[str, Any] = {}
    sign_mismatches: list[dict[str, Any]] = []
    probes = _probe_separations(lower, upper)
    for charge in charges:
        endpoint_values = (
            float(values_ev[point_key(charge, orientation, lower)]),
            float(values_ev[point_key(charge, orientation, upper)]),
        )
        actual_values = tuple(
            float(values_ev[point_key(charge, orientation, separation)])
            for separation in probes
        )
        scale = max(
            *(abs(value) for value in endpoint_values),
            *(abs(value) for value in actual_values),
        )
        if not (len(PROBE_FRACTIONS) == len(probes) == len(actual_values)):
            raise RuntimeError("Adaptive probe arrays have inconsistent lengths.")
        for fraction, separation, actual in zip(
            PROBE_FRACTIONS, probes, actual_values
        ):
            predicted = (
                (1.0 - fraction) * endpoint_values[0]
                + fraction * endpoint_values[1]
            )
            absolute_error = abs(actual - predicted)
            scaled_error = (
                absolute_error / scale
                if scale > 0.0
                else (0.0 if absolute_error == 0.0 else math.inf)
            )
            if scaled_error > worst_error:
                worst_error = scaled_error
                worst = {
                    "charge": charge,
                    "probe_fraction": fraction,
                    "probe_separation_angstrom": separation,
                    "actual_interaction_energy_ev": actual,
                    "interpolated_interaction_energy_ev": predicted,
                    "absolute_error_ev": absolute_error,
                    "interval_energy_scale_ev": scale,
                    "scaled_error": scaled_error,
                }
            if actual * predicted < 0.0:
                sign_mismatches.append(
                    {
                        "charge": charge,
                        "probe_fraction": fraction,
                        "actual_interaction_energy_ev": actual,
                        "interpolated_interaction_energy_ev": predicted,
                    }
                )
    return {
        "orientation": orientation,
        "lower_separation_angstrom": lower,
        "upper_separation_angstrom": upper,
        "depth": depth,
        "probe_fractions": list(PROBE_FRACTIONS),
        "worst_case": worst,
        "maximum_scaled_interpolation_error": worst_error,
        "sign_mismatch_count": len(sign_mismatches),
        "sign_mismatches": sign_mismatches,
        "relative_tolerance": tolerance,
        "passes": worst_error <= tolerance and not sign_mismatches,
    }


def refine_completed_intervals(
    mesh_state: dict[str, Any],
    values_ev: Mapping[str, float],
    charges: tuple[int, ...],
    settings: AdaptiveSettings,
) -> None:
    """Accept or bisect every active interval whose probes are available."""

    settings.validate()
    missing = required_probe_geometries(mesh_state, values_ev, charges)
    if missing:
        raise RuntimeError("Cannot refine while adaptive probe energies are missing.")

    global_error = 0.0
    global_sign_mismatches = 0
    any_active = False
    for orientation, curve in mesh_state["curves"].items():
        next_active: list[dict[str, Any]] = []
        for interval in curve["active_intervals"]:
            lower = float(interval["lower"])
            upper = float(interval["upper"])
            depth = int(interval["depth"])
            report = interval_interpolation_report(
                orientation=orientation,
                lower=lower,
                upper=upper,
                depth=depth,
                charges=charges,
                values_ev=values_ev,
                tolerance=settings.relative_tolerance,
            )
            global_error = max(
                global_error, float(report["maximum_scaled_interpolation_error"])
            )
            global_sign_mismatches += int(report["sign_mismatch_count"])
            if report["passes"]:
                report["decision"] = "accepted"
                curve["accepted_intervals"].append(report)
                continue
            if depth >= settings.maximum_refinement_depth:
                mesh_state["status"] = "refinement_limit_reached"
                mesh_state["failed_interval"] = report
                raise RuntimeError(
                    f"{orientation} interval {lower:g}--{upper:g} A failed "
                    f"the {settings.relative_tolerance:.3%} interpolation gate "
                    f"at maximum depth {depth}."
                )
            midpoint = round(0.5 * (lower + upper), 12)
            report["decision"] = "bisected"
            report["inserted_midpoint_angstrom"] = midpoint
            curve.setdefault("refined_intervals", []).append(report)
            mesh = set(float(value) for value in curve["mesh_separations_angstrom"])
            mesh.add(midpoint)
            if len(mesh) > settings.maximum_mesh_points_per_orientation:
                mesh_state["status"] = "refinement_limit_reached"
                mesh_state["failed_interval"] = report
                raise RuntimeError(
                    f"{orientation} exceeded the adaptive mesh-point limit."
                )
            curve["mesh_separations_angstrom"] = sorted(mesh)
            next_active.extend(
                (
                    {"lower": lower, "upper": midpoint, "depth": depth + 1},
                    {"lower": midpoint, "upper": upper, "depth": depth + 1},
                )
            )
        curve["active_intervals"] = next_active
        any_active = any_active or bool(next_active)

    mesh_state["historical_maximum_scaled_interpolation_error"] = max(
        global_error,
        float(mesh_state["historical_maximum_scaled_interpolation_error"]),
    )
    mesh_state["historical_sign_mismatch_count"] = int(
        mesh_state["historical_sign_mismatch_count"]
    ) + global_sign_mismatches
    mesh_state["status"] = "refining" if any_active else "complete"
    accepted = [
        report
        for curve in mesh_state["curves"].values()
        for report in curve["accepted_intervals"]
    ]
    mesh_state["maximum_scaled_interpolation_error"] = (
        max(
            float(report["maximum_scaled_interpolation_error"])
            for report in accepted
        )
        if accepted
        else None
    )
    mesh_state["sign_mismatch_count"] = sum(
        int(report["sign_mismatch_count"]) for report in accepted
    )
