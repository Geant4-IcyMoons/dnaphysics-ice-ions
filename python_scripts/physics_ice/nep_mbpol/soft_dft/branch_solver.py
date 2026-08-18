"""Restart-safe orchestration and acceptance for one-dimensional CDFT roots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from tqdm import tqdm

from .cdft_branch import (
    CalibrationSettings,
    CDFTBracket,
    PHYSICAL_STATE_STATUS,
    ProbeRecord,
    RECIPROCAL_VALIDATION_LIMITATION,
    RECIPROCAL_VALIDATION_SCOPE,
    bisect_injection_step_size,
    calibration_strengths,
    find_negative_response_bracket,
)


BRANCH_SOLVER_SCHEMA_VERSION = 2
ProbeCallback = Callable[[float, str], ProbeRecord]
RootCallback = Callable[
    [str, ProbeRecord, ProbeRecord, float], Mapping[str, Any]
]
DensityComparisonCallback = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]


@dataclass(frozen=True)
class BranchValidationSettings:
    """All numerical acceptance tolerances are explicit and signature-bound."""

    calibration: CalibrationSettings
    population_tolerance_electrons: float
    endpoint_strength_tolerance_hartree: float
    endpoint_residual_tolerance_electrons: float
    endpoint_population_tolerance_electrons: float
    endpoint_lagrangian_tolerance_hartree: float
    reciprocal_strength_tolerance_hartree: float
    reciprocal_energy_tolerance_hartree: float
    reciprocal_population_tolerance_electrons: float
    reciprocal_spin_squared_tolerance: float
    reciprocal_density_relative_rms_tolerance: float
    curvature_delta_hartree: float
    curvature_negative_margin_hartree: float

    def __post_init__(self) -> None:
        values = {
            "population tolerance": self.population_tolerance_electrons,
            "endpoint strength tolerance": self.endpoint_strength_tolerance_hartree,
            "endpoint residual reproduction tolerance": (
                self.endpoint_residual_tolerance_electrons
            ),
            "endpoint population reproduction tolerance": (
                self.endpoint_population_tolerance_electrons
            ),
            "endpoint Lagrangian reproduction tolerance": (
                self.endpoint_lagrangian_tolerance_hartree
            ),
            "reciprocal strength tolerance": (
                self.reciprocal_strength_tolerance_hartree
            ),
            "reciprocal energy tolerance": self.reciprocal_energy_tolerance_hartree,
            "reciprocal population tolerance": (
                self.reciprocal_population_tolerance_electrons
            ),
            "reciprocal spin tolerance": self.reciprocal_spin_squared_tolerance,
            "reciprocal density tolerance": (
                self.reciprocal_density_relative_rms_tolerance
            ),
            "curvature displacement": self.curvature_delta_hartree,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"CDFT {name} must be positive and finite.")
        if not math.isfinite(self.curvature_negative_margin_hartree) or (
            self.curvature_negative_margin_hartree < 0.0
        ):
            raise ValueError("CDFT curvature margin must be finite and nonnegative.")
        if self.curvature_delta_hartree <= (
            self.reciprocal_strength_tolerance_hartree
        ):
            raise ValueError(
                "CDFT curvature displacement must exceed the reciprocal "
                "strength tolerance."
            )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["calibration"] = self.calibration.as_dict()
        return payload


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _initial_checkpoint(
    solver_signature: str,
    settings: BranchValidationSettings,
    calibration_center_hartree: float,
    calibration_center_probe: bool,
) -> dict[str, Any]:
    return {
        "schema_version": BRANCH_SOLVER_SCHEMA_VERSION,
        "solver_signature": solver_signature,
        "settings": settings.as_dict(),
        "calibration_center_hartree": calibration_center_hartree,
        "calibration_center_probe": calibration_center_probe,
        "validation_scope": RECIPROCAL_VALIDATION_SCOPE,
        "physical_state_status": PHYSICAL_STATE_STATUS,
        "validation_limitation": RECIPROCAL_VALIDATION_LIMITATION,
        "status": "calibrating",
        "calibration_probes": {},
        "roots": {},
        "curvature_probes": {},
    }


def _load_checkpoint(
    path: Path,
    solver_signature: str,
    settings: BranchValidationSettings,
    calibration_center_hartree: float,
    calibration_center_probe: bool,
) -> dict[str, Any]:
    expected_settings = settings.as_dict()
    if not path.is_file():
        return _initial_checkpoint(
            solver_signature,
            settings,
            calibration_center_hartree,
            calibration_center_probe,
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != BRANCH_SOLVER_SCHEMA_VERSION:
        raise RuntimeError("Unsupported CDFT branch-solver checkpoint schema.")
    if payload.get("solver_signature") != solver_signature:
        raise RuntimeError("CDFT branch checkpoint belongs to another task.")
    if payload.get("settings") != expected_settings:
        raise RuntimeError("CDFT branch checkpoint settings are incompatible.")
    if payload.get("calibration_center_hartree") != calibration_center_hartree:
        raise RuntimeError("CDFT branch calibration center is incompatible.")
    if payload.get("calibration_center_probe") is not calibration_center_probe:
        raise RuntimeError("CDFT branch center-probe policy is incompatible.")
    for key in ("calibration_probes", "roots", "curvature_probes"):
        if not isinstance(payload.get(key), dict):
            raise RuntimeError(f"Malformed CDFT branch checkpoint field: {key}")
    return payload


def _probe_from_payload(payload: Mapping[str, Any]) -> ProbeRecord:
    return ProbeRecord(**dict(payload))


def _record_probe(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    category: str,
    label: str,
    strength: float,
    strength_tolerance: float,
    callback: ProbeCallback,
) -> ProbeRecord:
    records = checkpoint[category]
    existing = records.get(label)
    if existing is not None:
        record = _probe_from_payload(existing)
        if not math.isclose(
            record.strength_hartree,
            strength,
            rel_tol=0.0,
            abs_tol=strength_tolerance,
        ):
            raise RuntimeError("Checkpointed CDFT probe strength is incompatible.")
        return record
    record = callback(strength, label)
    if not isinstance(record, ProbeRecord):
        raise TypeError("CDFT probe callback must return ProbeRecord.")
    if not math.isclose(
        record.strength_hartree,
        strength,
        rel_tol=0.0,
        abs_tol=strength_tolerance,
    ):
        raise RuntimeError("CDFT probe callback returned the wrong multiplier.")
    records[label] = record.as_dict()
    _atomic_json(checkpoint_path, checkpoint)
    return record


def _trace_endpoint(
    point: Mapping[str, Any],
) -> tuple[float, float]:
    strength = point.get("strength")
    residual = point.get("residual")
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
    ):
        raise ValueError("CDFT root trace contains no finite strength.")
    if (
        isinstance(residual, bool)
        or not isinstance(residual, (int, float))
        or not math.isfinite(residual)
    ):
        raise ValueError("CDFT root trace contains no finite residual.")
    if point.get("inner_scf_converged") is not True:
        raise ValueError("CDFT root trace contains an unconverged endpoint.")
    return float(strength), float(residual)


def _trace_lagrangian(point: Mapping[str, Any]) -> float:
    value = point.get("energy")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("CDFT trace contains no finite constrained Lagrangian.")
    return float(value)


def _trace_value(point: Mapping[str, Any], key: str) -> float:
    value = point.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"CDFT trace contains no finite {key}.")
    return float(value)


def _validate_endpoint_reproduction(
    point: Mapping[str, Any],
    probe: ProbeRecord,
    settings: BranchValidationSettings,
) -> None:
    residual = _trace_value(point, "residual")
    target = _trace_value(point, "target")
    population = _trace_value(point, "current")
    lagrangian = _trace_lagrangian(point)
    if not math.isclose(
        residual,
        probe.residual_electrons,
        rel_tol=0.0,
        abs_tol=settings.endpoint_residual_tolerance_electrons,
    ):
        raise ValueError("CDFT BISECT endpoint residual was not reproduced.")
    if not math.isclose(
        population,
        probe.current_electrons,
        rel_tol=0.0,
        abs_tol=settings.endpoint_population_tolerance_electrons,
    ):
        raise ValueError("CDFT BISECT endpoint population was not reproduced.")
    if not math.isclose(
        target,
        probe.target_electrons,
        rel_tol=0.0,
        abs_tol=settings.endpoint_population_tolerance_electrons,
    ):
        raise ValueError("CDFT BISECT endpoint target was not reproduced.")
    if not math.isclose(
        population - target,
        residual,
        rel_tol=0.0,
        abs_tol=settings.endpoint_residual_tolerance_electrons,
    ):
        raise ValueError("CDFT BISECT endpoint population is internally inconsistent.")
    if not math.isclose(
        lagrangian,
        probe.energy_hartree,
        rel_tol=0.0,
        abs_tol=settings.endpoint_lagrangian_tolerance_hartree,
    ):
        raise ValueError(
            "CDFT BISECT endpoint constrained Lagrangian was not reproduced."
        )


def _final_trace_point(result: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = result.get("cdft_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("CDFT result contains no ordered branch trace.")
    point = trace[-1]
    if not isinstance(point, dict):
        raise ValueError("CDFT branch trace has a malformed final point.")
    return point


def _strict_same_sector_bracket(
    records: list[ProbeRecord],
) -> CDFTBracket:
    """Find a strict-sign bracket without treating an exact root as an endpoint."""

    grouped: dict[tuple[str, float], list[ProbeRecord]] = {}
    for record in records:
        fingerprint = record.branch_fingerprint
        if (
            not record.inner_scf_converged
            or record.residual_electrons == 0.0
            or not isinstance(fingerprint, str)
            or not fingerprint
        ):
            continue
        grouped.setdefault(
            (fingerprint, record.target_electrons), []
        ).append(record)
    candidates: list[CDFTBracket] = []
    for same_branch_records in grouped.values():
        try:
            bracket = find_negative_response_bracket(same_branch_records)
        except ValueError:
            continue
        if bracket.lower.residual_electrons * bracket.upper.residual_electrons >= 0.0:
            continue
        candidates.append(bracket)
    if not candidates:
        raise ValueError(
            "No strict, inner-converged same-sector negative-response bracket "
            "was found."
        )
    return min(
        candidates,
        key=lambda bracket: (
            bracket.upper.strength_hartree - bracket.lower.strength_hartree,
            abs(bracket.lower.residual_electrons)
            + abs(bracket.upper.residual_electrons),
            bracket.lower.strength_hartree,
        ),
    )


def _validate_injected_root(
    result: Mapping[str, Any],
    start: ProbeRecord,
    opposite: ProbeRecord,
    settings: BranchValidationSettings,
) -> None:
    required_gates = (
        result.get("status") == "complete",
        result.get("valid_completion") is True,
        result.get("cdft_outer_converged") is True,
        result.get("last_inner_scf_converged") is True,
        result.get("normal_end") is True,
    )
    if not all(required_gates):
        raise ValueError("Reciprocal CDFT root did not pass all completion gates.")
    trace = result.get("cdft_trace")
    if not isinstance(trace, list) or len(trace) < 3:
        raise ValueError(
            "CDFT BISECT trace did not retain both endpoints before bisection."
        )
    if not all(isinstance(point, dict) for point in (trace[0], trace[1])):
        raise ValueError("CDFT BISECT trace contains malformed endpoints.")
    first_strength, first_residual = _trace_endpoint(trace[0])
    second_strength, second_residual = _trace_endpoint(trace[1])
    tolerance = settings.endpoint_strength_tolerance_hartree
    if not math.isclose(
        first_strength, start.strength_hartree, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError("CDFT BISECT did not start at its calibrated endpoint.")
    if not math.isclose(
        second_strength,
        opposite.strength_hartree,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("CDFT BISECT first fallback did not visit the other side.")
    if first_residual * second_residual >= 0.0:
        raise ValueError("CDFT BISECT in-process history lacks opposite signs.")
    if (
        first_residual * start.residual_electrons <= 0.0
        or second_residual * opposite.residual_electrons <= 0.0
    ):
        raise ValueError("CDFT BISECT endpoint residual signs changed branch.")
    _validate_endpoint_reproduction(trace[0], start, settings)
    _validate_endpoint_reproduction(trace[1], opposite, settings)

    final_point = _final_trace_point(result)
    final_strength, trace_residual = _trace_endpoint(final_point)
    _trace_lagrangian(final_point)
    result_strength = _finite_result_value(result, "cdft_strength")
    final_residual = _finite_result_value(
        result, "cdft_deviation_electrons"
    )
    target = _finite_result_value(result, "cdft_target_electrons")
    population = _finite_result_value(result, "cdft_current_electrons")
    if not math.isclose(
        final_strength,
        result_strength,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("Final CDFT trace multiplier is inconsistent.")
    if not math.isclose(
        trace_residual,
        final_residual,
        rel_tol=0.0,
        abs_tol=settings.population_tolerance_electrons,
    ):
        raise ValueError("Final CDFT trace residual is inconsistent.")
    if not math.isclose(
        population - target,
        final_residual,
        rel_tol=0.0,
        abs_tol=settings.population_tolerance_electrons,
    ):
        raise ValueError("CDFT root population and residual are inconsistent.")
    if abs(final_residual) > settings.population_tolerance_electrons:
        raise ValueError("CDFT root does not satisfy the population tolerance.")


def _finite_result_value(result: Mapping[str, Any], key: str) -> float:
    value = result.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"CDFT result lacks finite {key}.")
    return float(value)


def _reciprocal_validation(
    lower_result: Mapping[str, Any],
    upper_result: Mapping[str, Any],
    density_comparison: Mapping[str, Any],
    settings: BranchValidationSettings,
) -> dict[str, Any]:
    lower_lambda = _finite_result_value(lower_result, "cdft_strength")
    upper_lambda = _finite_result_value(upper_result, "cdft_strength")
    lower_lagrangian = _trace_lagrangian(_final_trace_point(lower_result))
    upper_lagrangian = _trace_lagrangian(_final_trace_point(upper_result))
    lower_population = _finite_result_value(
        lower_result, "cdft_current_electrons"
    )
    upper_population = _finite_result_value(
        upper_result, "cdft_current_electrons"
    )
    differences = {
        "strength_hartree": abs(lower_lambda - upper_lambda),
        "constrained_lagrangian_hartree": abs(
            lower_lagrangian - upper_lagrangian
        ),
        "population_electrons": abs(lower_population - upper_population),
    }
    if differences["strength_hartree"] > (
        settings.reciprocal_strength_tolerance_hartree
    ):
        raise ValueError("Reciprocal CDFT roots disagree in multiplier.")
    if differences["constrained_lagrangian_hartree"] > (
        settings.reciprocal_energy_tolerance_hartree
    ):
        raise ValueError(
            "Reciprocal CDFT roots disagree in constrained Lagrangian."
        )
    if differences["population_electrons"] > (
        settings.reciprocal_population_tolerance_electrons
    ):
        raise ValueError("Reciprocal CDFT roots disagree in population.")
    lower_mode = lower_result.get("scf_spin_mode")
    upper_mode = upper_result.get("scf_spin_mode")
    if lower_mode not in {"RESTRICTED", "UNRESTRICTED"} or (
        upper_mode != lower_mode
    ):
        raise ValueError("Reciprocal CDFT roots have incompatible SCF spin modes.")
    lower_spin = lower_result.get("spin_squared_single_determinant")
    upper_spin = upper_result.get("spin_squared_single_determinant")
    if (lower_spin is None) != (upper_spin is None):
        raise ValueError("Reciprocal CDFT roots have asymmetric S squared output.")
    spin_difference: float | None = None
    if lower_spin is None:
        if lower_mode != "RESTRICTED":
            raise ValueError("Unrestricted reciprocal roots require S squared output.")
    else:
        lower_spin_value = _finite_result_value(
            lower_result, "spin_squared_single_determinant"
        )
        upper_spin_value = _finite_result_value(
            upper_result, "spin_squared_single_determinant"
        )
        spin_difference = abs(lower_spin_value - upper_spin_value)
        if spin_difference > settings.reciprocal_spin_squared_tolerance:
            raise ValueError("Reciprocal CDFT roots disagree in S squared.")
    relative_density_rms = density_comparison.get("relative_rms")
    if (
        isinstance(relative_density_rms, bool)
        or not isinstance(relative_density_rms, (int, float))
        or not math.isfinite(relative_density_rms)
        or relative_density_rms < 0.0
    ):
        raise ValueError("Reciprocal CDFT density comparison is absent.")
    if relative_density_rms > settings.reciprocal_density_relative_rms_tolerance:
        raise ValueError("Reciprocal CDFT roots have different electronic densities.")
    differences["spin_squared"] = spin_difference
    return {
        "differences": differences,
        "density": dict(density_comparison),
        "mean_strength_hartree": 0.5 * (lower_lambda + upper_lambda),
        "mean_constrained_lagrangian_hartree": 0.5
        * (lower_lagrangian + upper_lagrangian),
        "root_constrained_lagrangians_hartree": {
            "lower": lower_lagrangian,
            "upper": upper_lagrangian,
        },
    }


def solve_cdft_branch(
    checkpoint_path: Path,
    *,
    solver_signature: str,
    settings: BranchValidationSettings,
    calibration_center_hartree: float,
    calibration_center_probe: bool,
    probe_callback: ProbeCallback,
    root_callback: RootCallback,
    density_comparison_callback: DensityComparisonCallback,
    progress: bool = True,
) -> dict[str, Any]:
    """Calibrate, solve from both sides, and validate one CDFT branch."""

    if (
        isinstance(calibration_center_hartree, bool)
        or not isinstance(calibration_center_hartree, (int, float))
        or not math.isfinite(calibration_center_hartree)
    ):
        raise ValueError("CDFT calibration center must be finite.")
    if not isinstance(calibration_center_probe, bool):
        raise TypeError("CDFT center-probe policy must be Boolean.")
    if settings.calibration.include_zero_diagnostic:
        raise ValueError(
            "Branch calibration center probes are controlled by validated-parent "
            "continuation, not include_zero_diagnostic."
        )
    checkpoint = _load_checkpoint(
        checkpoint_path,
        solver_signature,
        settings,
        calibration_center_hartree,
        calibration_center_probe,
    )
    total = settings.calibration.max_probe_count + 4 + int(
        calibration_center_probe
    )
    with tqdm(
        total=total,
        desc="CDFT branch",
        unit="calculation",
        disable=not progress,
    ) as bar:
        bar.update(
            len(checkpoint["calibration_probes"])
            + len(checkpoint["roots"])
            + len(checkpoint["curvature_probes"])
        )
        bracket: CDFTBracket | None = None
        center_record: ProbeRecord | None = None
        if calibration_center_probe:
            label = "calibration_center"
            before = label in checkpoint["calibration_probes"]
            center_record = _record_probe(
                checkpoint,
                checkpoint_path,
                "calibration_probes",
                label,
                calibration_center_hartree,
                settings.endpoint_strength_tolerance_hartree,
                probe_callback,
            )
            if not before:
                bar.update()
            if not center_record.inner_scf_converged or not (
                isinstance(center_record.branch_fingerprint, str)
                and center_record.branch_fingerprint
            ):
                raise ValueError(
                    "Validated-parent center probe did not reproduce a converged "
                    "electronic sector."
                )
        offsets = calibration_strengths(settings.calibration)
        for index, offset in enumerate(offsets):
            strength = calibration_center_hartree + offset
            label = f"calibration_{index:03d}"
            before = label in checkpoint["calibration_probes"]
            _record_probe(
                checkpoint,
                checkpoint_path,
                "calibration_probes",
                label,
                strength,
                settings.endpoint_strength_tolerance_hartree,
                probe_callback,
            )
            if not before:
                bar.update()
            records = [
                _probe_from_payload(record)
                for record in checkpoint["calibration_probes"].values()
            ]
            try:
                bracket = _strict_same_sector_bracket(records)
            except ValueError:
                continue
            if center_record is not None and (
                bracket.lower.branch_fingerprint
                != center_record.branch_fingerprint
                or not math.isclose(
                    bracket.lower.target_electrons,
                    center_record.target_electrons,
                    rel_tol=0.0,
                    abs_tol=settings.population_tolerance_electrons,
                )
            ):
                bracket = None
                continue
            break
        if bracket is None:
            checkpoint["status"] = "no_bracket"
            _atomic_json(checkpoint_path, checkpoint)
            raise RuntimeError("CDFT calibration reached its bound without a bracket.")
        checkpoint["bracket"] = bracket.as_dict()
        checkpoint["status"] = "solving_reciprocal_roots"
        _atomic_json(checkpoint_path, checkpoint)

        root_definitions = (
            ("lower", bracket.lower, bracket.upper),
            ("upper", bracket.upper, bracket.lower),
        )
        for direction, start, opposite in root_definitions:
            existing = checkpoint["roots"].get(direction)
            if existing is None:
                step = bisect_injection_step_size(start, opposite)
                result = dict(root_callback(direction, start, opposite, step))
                _validate_injected_root(result, start, opposite, settings)
                callback_step = result.get("step_size")
                if callback_step is not None and (
                    isinstance(callback_step, bool)
                    or not isinstance(callback_step, (int, float))
                    or not math.isclose(
                        float(callback_step),
                        step,
                        rel_tol=0.0,
                        abs_tol=settings.endpoint_strength_tolerance_hartree,
                    )
                ):
                    raise ValueError("CDFT root used the wrong injected step.")
                result["injection_step_size"] = step
                checkpoint["roots"][direction] = result
                _atomic_json(checkpoint_path, checkpoint)
                bar.update()
            else:
                step = bisect_injection_step_size(start, opposite)
                _validate_injected_root(existing, start, opposite, settings)
                recorded_step = existing.get("injection_step_size")
                if (
                    isinstance(recorded_step, bool)
                    or not isinstance(recorded_step, (int, float))
                    or not math.isclose(
                        float(recorded_step),
                        step,
                        rel_tol=0.0,
                        abs_tol=settings.endpoint_strength_tolerance_hartree,
                    )
                ):
                    raise ValueError(
                        "Checkpointed CDFT root has the wrong injected step."
                    )
                existing["injection_step_size"] = step

        lower_result = checkpoint["roots"]["lower"]
        upper_result = checkpoint["roots"]["upper"]
        density_comparison = dict(
            density_comparison_callback(lower_result, upper_result)
        )
        reciprocal = _reciprocal_validation(
            lower_result, upper_result, density_comparison, settings
        )
        root_strength = float(reciprocal["mean_strength_hartree"])
        delta = settings.curvature_delta_hartree
        curvature_records: dict[str, ProbeRecord] = {}
        for label, strength in (
            ("lower", root_strength - delta),
            ("upper", root_strength + delta),
        ):
            before = label in checkpoint["curvature_probes"]
            curvature_records[label] = _record_probe(
                checkpoint,
                checkpoint_path,
                "curvature_probes",
                label,
                strength,
                settings.endpoint_strength_tolerance_hartree,
                probe_callback,
            )
            if not before:
                bar.update()
        lower_probe = curvature_records["lower"]
        upper_probe = curvature_records["upper"]
        if not (lower_probe.inner_scf_converged and upper_probe.inner_scf_converged):
            raise ValueError("CDFT curvature probes did not converge internally.")
        branch_fingerprint = bracket.lower.branch_fingerprint
        if not (
            branch_fingerprint
            and lower_probe.branch_fingerprint == branch_fingerprint
            and upper_probe.branch_fingerprint == branch_fingerprint
        ):
            raise ValueError(
                "CDFT curvature probes left the calibrated electronic sector."
            )
        if not (
            math.isclose(
                lower_probe.target_electrons,
                bracket.lower.target_electrons,
                rel_tol=0.0,
                abs_tol=settings.population_tolerance_electrons,
            )
            and math.isclose(
                upper_probe.target_electrons,
                bracket.lower.target_electrons,
                rel_tol=0.0,
                abs_tol=settings.population_tolerance_electrons,
            )
        ):
            raise ValueError("CDFT curvature probes changed the constraint target.")
        if not (
            lower_probe.residual_electrons > 0.0
            and upper_probe.residual_electrons < 0.0
        ):
            raise ValueError("CDFT root is not a negative-response maximum in lambda.")
        local_response = (
            upper_probe.current_electrons - lower_probe.current_electrons
        ) / (2.0 * delta)
        if local_response >= 0.0:
            raise ValueError("CDFT near-root population response is not negative.")
        root_lagrangians = reciprocal[
            "root_constrained_lagrangians_hartree"
        ]
        second_differences = {
            direction: (
                lower_probe.energy_hartree
                - 2.0 * float(root_lagrangians[direction])
                + upper_probe.energy_hartree
            )
            for direction in ("lower", "upper")
        }
        if max(second_differences.values()) >= (
            -settings.curvature_negative_margin_hartree
        ):
            raise ValueError(
                "CDFT constrained Lagrangian lacks negative curvature."
            )
        checkpoint["reciprocal_validation"] = reciprocal
        checkpoint["curvature_validation"] = {
            "delta_hartree": delta,
            "local_response_electrons_per_hartree": local_response,
            "second_differences_hartree": second_differences,
            "constrained_lagrangian_hartree": {
                "minus_delta": lower_probe.energy_hartree,
                "plus_delta": upper_probe.energy_hartree,
                **dict(root_lagrangians),
            },
        }
        checkpoint["status"] = "validated"
        checkpoint["directions"] = ["lower", "upper"]
        _atomic_json(checkpoint_path, checkpoint)
        return checkpoint
