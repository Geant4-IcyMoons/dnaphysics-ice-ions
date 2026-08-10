#!/usr/bin/env python3
"""Adapt trajectory count and energy density for one NLH projectile."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.config import DEFAULT_PROJECTILES  # noqa: E402
from bca.convergence import OBSERVABLES  # noqa: E402
from ion_ice import ICE_STRUCTURES_ROOT, PROCESS_EVIDENCE_ROOT  # noqa: E402


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 2
DEFAULT_BASE_ENERGIES_EV = (
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
    1.0e7,
    1.0e8,
)
DEFAULT_STRUCTURE_SEEDS = (1000, 2000, 3000)
ORIENTATIONS: dict[str, tuple[str, ...]] = {
    "c_axis": ("--direction", "0", "0", "1"),
    "basal_a_axis": ("--direction", "1", "0", "0"),
    "isotropic": ("--isotropic-directions",),
}


@dataclass(frozen=True)
class Case:
    structure_seed: int
    orientation: str
    energy_ev: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projectile", required=True, choices=DEFAULT_PROJECTILES)
    parser.add_argument(
        "--structure-directory",
        type=Path,
        default=ICE_STRUCTURES_ROOT / "hexagonal_ih_100K_experimental",
    )
    parser.add_argument(
        "--structure-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_STRUCTURE_SEEDS,
    )
    parser.add_argument(
        "--orientations",
        nargs="+",
        choices=tuple(ORIENTATIONS),
        default=tuple(ORIENTATIONS),
    )
    parser.add_argument(
        "--base-energies-ev",
        type=float,
        nargs="+",
        default=DEFAULT_BASE_ENERGIES_EV,
    )
    parser.add_argument("--kernels", type=Path, default=HERE / "collision_kernels")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROCESS_EVIDENCE_ROOT
            / "hard_nuclear_collisions"
            / "validation"
            / "runs"
            / "adaptive_particles"
        ),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--calibration-trajectories", type=int, default=10_000)
    parser.add_argument("--minimum-trajectories", type=int, default=200_000)
    parser.add_argument("--maximum-trajectories", type=int, default=64_000_000)
    parser.add_argument(
        "--extended-maximum-trajectories",
        type=int,
        default=None,
        help=(
            "Raise only the restart-time sampling ceiling without changing the "
            "controller signature or deterministic trajectory stream."
        ),
    )
    parser.add_argument(
        "--unlimited-trajectories",
        action="store_true",
        help=(
            "Continue checkpointed sampling until the statistical gate is met "
            "or the external scheduler stops the process."
        ),
    )
    parser.add_argument("--trajectory-batch-size", type=int, default=10_000)
    parser.add_argument("--path-length-angstrom", type=float, default=100.0)
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "collision_tube_mixture"),
        default="collision_tube_mixture",
    )
    parser.add_argument("--tube-mixture-fraction", type=float, default=0.5)
    parser.add_argument("--maximum-refinement-depth", type=int, default=8)
    parser.add_argument("--maximum-energy-points", type=int, default=257)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if len(set(args.structure_seeds)) != len(args.structure_seeds):
        raise ValueError("--structure-seeds contains duplicates.")
    energies = sorted(set(float(value) for value in args.base_energies_ev))
    if len(energies) < 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in energies
    ):
        raise ValueError("At least two distinct positive base energies are required.")
    if energies[0] < 1.0e3 or energies[-1] > 1.0e8:
        raise ValueError("The retained kernel range is 1 keV--100 MeV.")
    if not math.isfinite(args.tolerance) or not 0.0 < args.tolerance < 1.0:
        raise ValueError("--tolerance must lie strictly between zero and one.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one.")
    if args.calibration_trajectories < 2:
        raise ValueError("--calibration-trajectories must be at least two.")
    if not 2 <= args.minimum_trajectories <= args.maximum_trajectories:
        raise ValueError("Invalid minimum/maximum trajectory bounds.")
    if (
        args.extended_maximum_trajectories is not None
        and args.extended_maximum_trajectories < args.maximum_trajectories
    ):
        raise ValueError(
            "--extended-maximum-trajectories cannot lower the signed ceiling."
        )
    if args.unlimited_trajectories and args.extended_maximum_trajectories is not None:
        raise ValueError(
            "--unlimited-trajectories and --extended-maximum-trajectories are "
            "mutually exclusive."
        )
    if args.trajectory_batch_size < 1:
        raise ValueError("--trajectory-batch-size must be positive.")
    if args.sampling_mode == "collision_tube_mixture" and not (
        math.isfinite(args.tube_mixture_fraction)
        and 0.0 < args.tube_mixture_fraction < 1.0
    ):
        raise ValueError("--tube-mixture-fraction must lie strictly in (0, 1).")
    if args.maximum_refinement_depth < 0 or args.maximum_energy_points < len(
        energies
    ):
        raise ValueError("Invalid energy-refinement bounds.")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _energy_key(energy_ev: float) -> str:
    return f"{energy_ev:.12g}eV".replace("+", "")


def _interval_key(lower: float, upper: float) -> str:
    return f"{_energy_key(lower)}--{_energy_key(upper)}"


def _stable_seed(purpose: str, projectile: str, case: Case) -> int:
    encoded = (
        f"adaptive-nlh-v{IMPLEMENTATION_VERSION}|{purpose}|{projectile}|"
        f"{case.structure_seed}|{case.orientation}|{case.energy_ev:.17g}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _worker_count(requested: int) -> int:
    available = len(os.sched_getaffinity(0))
    workers = available if requested == 0 else requested
    if workers < 1 or workers > available:
        raise ValueError(
            f"--workers must be 0 or lie in 1-{available}; got {requested}."
        )
    return workers


def _runtime_trajectory_ceiling(args: argparse.Namespace) -> int:
    if getattr(args, "unlimited_trajectories", False):
        return sys.maxsize
    extension = args.extended_maximum_trajectories
    return args.maximum_trajectories if extension is None else int(extension)


def _case_directory(root: Path, stage: str, case: Case) -> Path:
    return (
        root
        / stage
        / f"seed{case.structure_seed}"
        / case.orientation
        / _energy_key(case.energy_ev)
    )


def _structure_paths(directory: Path, seed: int) -> tuple[Path, Path]:
    structure = directory / f"seed{seed}_final.xyz.gz"
    metadata = directory / f"seed{seed}_final.xyz.json"
    if not structure.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Missing attested structure inputs for seed {seed}.")
    return structure, metadata


def _manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Malformed manifest: {path}")
    return value


def _report_quality(report: dict[str, object]) -> float:
    observables = report.get("observables")
    if not isinstance(observables, dict) or set(observables) != set(OBSERVABLES):
        return math.inf
    widths: list[float] = []
    for name in OBSERVABLES:
        record = observables[name]
        if not isinstance(record, dict):
            return math.inf
        estimate = record.get("estimate")
        width = record.get("relative_confidence_half_width")
        if (
            estimate is None
            or width is None
            or not math.isfinite(float(estimate))
            or float(estimate) <= 0.0
            or not math.isfinite(float(width))
        ):
            return math.inf
        widths.append(float(width))
    return max(widths)


def _run_simulator(
    args: argparse.Namespace,
    workers: int,
    particle_root: Path,
    case: Case,
    *,
    stage: str,
    fixed_trajectories: int | None,
    control_variate: bool,
) -> tuple[int, Path, list[str]]:
    structure, metadata = _structure_paths(
        args.structure_directory, case.structure_seed
    )
    output = _case_directory(particle_root, stage, case)
    command = [
        sys.executable,
        "-u",
        str(HERE / "simulate_nlh_hard_collisions.py"),
        str(structure),
        "--metadata",
        str(metadata),
        "--kernels",
        str(args.kernels),
        "--projectile",
        args.projectile,
        "--energy-ev",
        f"{case.energy_ev:.17g}",
        "--path-length-angstrom",
        f"{args.path_length_angstrom:.17g}",
        "--seed",
        str(_stable_seed(stage, args.projectile, case)),
        "--workers",
        str(workers),
        "--trajectory-batch-size",
        str(args.trajectory_batch_size),
        "--statistical-relative-tolerance",
        f"{args.tolerance:.17g}",
        "--statistical-confidence",
        f"{args.confidence:.17g}",
        "--trajectory-cdf-tolerance",
        f"{args.tolerance:.17g}",
        "--output-detail",
        "summary",
        "--output-directory",
        str(output),
        "--sampling-mode",
        getattr(args, "sampling_mode", "collision_tube_mixture"),
        "--tube-mixture-fraction",
        f"{getattr(args, 'tube_mixture_fraction', 0.5):.17g}",
        *ORIENTATIONS[case.orientation],
    ]
    if fixed_trajectories is None:
        command.extend(
            (
                "--minimum-trajectories",
                str(args.minimum_trajectories),
                "--maximum-trajectories",
                str(_runtime_trajectory_ceiling(args)),
            )
        )
    else:
        command.extend(("--trajectories", str(fixed_trajectories)))
    if control_variate:
        command.append("--control-variate")
    if args.dry_run:
        return 0, output / "hard_collision_run.manifest.json", command
    # PBS captures inherited stdout/stderr through a pipe.  A long-running
    # nested tqdm stream can block in pipe_write if that capture path stops
    # draining, which in turn prevents the parent from consuming worker
    # results.  Keep the scientifically meaningful per-case progress, but
    # write it directly to the case directory rather than through PBS.
    output.mkdir(parents=True, exist_ok=True)
    progress_log = output / "hard_collision_progress.log"
    with progress_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode, output / "hard_collision_run.manifest.json", command


def _calibrate_case(
    args: argparse.Namespace,
    workers: int,
    particle_root: Path,
    case: Case,
) -> dict[str, object]:
    manifest_path = _case_directory(
        particle_root, "estimator_calibration", case
    ) / "hard_collision_run.manifest.json"
    if not manifest_path.is_file():
        return_code, manifest_path, command = _run_simulator(
            args,
            workers,
            particle_root,
            case,
            stage="estimator_calibration",
            fixed_trajectories=args.calibration_trajectories,
            control_variate=True,
        )
        if args.dry_run:
            return {
                "selected": "undetermined-dry-run",
                "command": command,
            }
        if return_code != 0:
            raise RuntimeError(
                f"Estimator calibration failed with status {return_code}: {case}"
            )
    manifest = _manifest(manifest_path)
    corrected = manifest["statistical_convergence"]
    raw = manifest["raw_statistical_convergence"]
    if not isinstance(corrected, dict) or not isinstance(raw, dict):
        raise RuntimeError(f"Calibration reports are malformed: {manifest_path}")
    corrected_quality = _report_quality(corrected)
    raw_quality = _report_quality(raw)
    use_control = corrected_quality < raw_quality
    return {
        "selected": "straight_line_control_variate" if use_control else "raw",
        "use_control_variate": use_control,
        "corrected_maximum_relative_half_width": corrected_quality,
        "raw_maximum_relative_half_width": raw_quality,
        "selection_data_reused_in_production": False,
        "manifest": str(manifest_path),
    }


def _run_production_case(
    args: argparse.Namespace,
    workers: int,
    particle_root: Path,
    case: Case,
    calibration: dict[str, object],
) -> Path:
    output = _case_directory(particle_root, "production", case)
    manifest_path = output / "hard_collision_run.manifest.json"
    if manifest_path.is_file():
        manifest = _manifest(manifest_path)
        report = manifest.get("statistical_convergence")
        if isinstance(report, dict) and bool(report.get("converged")):
            return manifest_path
    return_code, manifest_path, command = _run_simulator(
        args,
        workers,
        particle_root,
        case,
        stage="production",
        fixed_trajectories=None,
        control_variate=bool(calibration.get("use_control_variate", False)),
    )
    if args.dry_run:
        print("DRY RUN:", " ".join(command))
        return manifest_path
    if return_code == 2:
        raise RuntimeError(
            f"Case reached {_runtime_trajectory_ceiling(args):,} trajectories "
            "without "
            f"meeting every 0.5% gate: {case}. Checkpoints are resumable."
        )
    if return_code != 0:
        raise RuntimeError(f"Production case failed with status {return_code}: {case}")
    manifest = _manifest(manifest_path)
    report = manifest.get("statistical_convergence")
    if not isinstance(report, dict) or not bool(report.get("converged")):
        raise RuntimeError(f"Production manifest is not converged: {manifest_path}")
    return manifest_path


def _case_key(case: Case) -> str:
    return f"seed{case.structure_seed}/{case.orientation}/{_energy_key(case.energy_ev)}"


def _all_cases(args: argparse.Namespace, energy_ev: float) -> tuple[Case, ...]:
    return tuple(
        Case(seed, orientation, energy_ev)
        for seed in args.structure_seeds
        for orientation in args.orientations
    )


def _run_energy(
    args: argparse.Namespace,
    workers: int,
    particle_root: Path,
    energy_ev: float,
    state: dict[str, object],
    state_path: Path,
) -> None:
    cases_state = state.setdefault("cases", {})
    if not isinstance(cases_state, dict):
        raise RuntimeError("Malformed controller state.")
    cases = _all_cases(args, energy_ev)
    with tqdm(cases, desc=f"{args.projectile} {_energy_key(energy_ev)} cases") as bar:
        for case in bar:
            key = _case_key(case)
            existing = cases_state.get(key)
            if isinstance(existing, dict) and existing.get("converged"):
                continue
            calibration = _calibrate_case(args, workers, particle_root, case)
            if args.dry_run:
                cases_state[key] = {
                    "converged": False,
                    "calibration": calibration,
                }
                continue
            manifest_path = _run_production_case(
                args, workers, particle_root, case, calibration
            )
            manifest = _manifest(manifest_path)
            cases_state[key] = {
                "converged": True,
                "calibration": calibration,
                "production_manifest": str(manifest_path),
                "trajectories": manifest["configuration"]["completed_trajectories"],
            }
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(state_path, state)


def _case_manifest(
    state: dict[str, object], case: Case
) -> tuple[Path, dict[str, object]]:
    cases = state["cases"]
    if not isinstance(cases, dict):
        raise RuntimeError("Malformed controller state.")
    record = cases[_case_key(case)]
    if not isinstance(record, dict):
        raise RuntimeError(f"Missing case record: {case}")
    path = Path(str(record["production_manifest"]))
    return path, _manifest(path)


def _load_distribution(manifest_path: Path, field: str) -> np.ndarray:
    manifest = _manifest(manifest_path)
    output = manifest["outputs"]
    if not isinstance(output, dict) or output.get("detail") != "summary":
        raise RuntimeError(f"Compact distribution samples are unavailable: {manifest_path}")
    checkpoint = Path(str(output["checkpoint_directory"]))
    arrays: list[np.ndarray] = []
    for record_path in sorted(checkpoint.glob("batch_*.manifest.json")):
        record = _manifest(record_path)
        sample = checkpoint / str(record["distribution_sample"])
        with np.load(sample) as values:
            arrays.append(np.asarray(values[field], dtype=np.float64))
    if not arrays:
        raise RuntimeError(f"No distribution samples found below {checkpoint}.")
    result = np.concatenate(arrays)
    if np.any(~np.isfinite(result)):
        raise RuntimeError(f"Non-finite distribution sample below {checkpoint}.")
    result.sort()
    return result


def _cdf_interpolation_error(
    lower: np.ndarray,
    upper: np.ndarray,
    midpoint: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> float:
    """Supremum distance from the log-energy-linear endpoint CDF midpoint."""

    maximum = 0.0
    for candidates in (lower, upper, midpoint):
        for start in range(0, len(candidates), chunk_size):
            values = candidates[start : start + chunk_size]
            lower_cdf = np.searchsorted(lower, values, side="right") / len(lower)
            upper_cdf = np.searchsorted(upper, values, side="right") / len(upper)
            midpoint_cdf = (
                np.searchsorted(midpoint, values, side="right") / len(midpoint)
            )
            difference = np.abs(midpoint_cdf - 0.5 * (lower_cdf + upper_cdf))
            if difference.size:
                maximum = max(maximum, float(np.max(difference)))
    return maximum


def _observable_estimates(manifest: dict[str, object]) -> dict[str, float]:
    report = manifest["statistical_convergence"]
    if not isinstance(report, dict):
        raise RuntimeError("Malformed statistical report.")
    records = report["observables"]
    if not isinstance(records, dict):
        raise RuntimeError("Malformed observable report.")
    return {name: float(records[name]["estimate"]) for name in OBSERVABLES}


def _analyze_interval(
    args: argparse.Namespace,
    state: dict[str, object],
    lower_energy: float,
    upper_energy: float,
    midpoint_energy: float,
) -> dict[str, object]:
    worst_scalar = {"relative_error": -1.0}
    worst_cdf = {"absolute_error": -1.0}
    for seed in args.structure_seeds:
        for orientation in args.orientations:
            lower_path, lower = _case_manifest(
                state, Case(seed, orientation, lower_energy)
            )
            upper_path, upper = _case_manifest(
                state, Case(seed, orientation, upper_energy)
            )
            midpoint_path, midpoint = _case_manifest(
                state, Case(seed, orientation, midpoint_energy)
            )
            lower_values = _observable_estimates(lower)
            upper_values = _observable_estimates(upper)
            midpoint_values = _observable_estimates(midpoint)
            for observable in OBSERVABLES:
                values = (
                    lower_values[observable],
                    upper_values[observable],
                    midpoint_values[observable],
                )
                if any(not math.isfinite(value) or value <= 0.0 for value in values):
                    relative_error = math.inf
                else:
                    interpolated = math.sqrt(values[0] * values[1])
                    relative_error = abs(math.log(values[2] / interpolated))
                if relative_error > float(worst_scalar["relative_error"]):
                    worst_scalar = {
                        "relative_error": relative_error,
                        "structure_seed": seed,
                        "orientation": orientation,
                        "observable": observable,
                    }
            for field in (
                "total_recoil_energy_ev",
                "final_deflection_rad",
            ):
                cdf_error = _cdf_interpolation_error(
                    _load_distribution(lower_path, field),
                    _load_distribution(upper_path, field),
                    _load_distribution(midpoint_path, field),
                )
                if cdf_error > float(worst_cdf["absolute_error"]):
                    worst_cdf = {
                        "absolute_error": cdf_error,
                        "structure_seed": seed,
                        "orientation": orientation,
                        "distribution": field,
                    }
    scalar_passes = float(worst_scalar["relative_error"]) <= math.log1p(
        args.tolerance
    )
    cdf_passes = float(worst_cdf["absolute_error"]) <= args.tolerance
    return {
        "lower_energy_ev": lower_energy,
        "upper_energy_ev": upper_energy,
        "validation_midpoint_energy_ev": midpoint_energy,
        "scalar_interpolation": worst_scalar,
        "trajectory_cdf_interpolation": worst_cdf,
        "tolerance": args.tolerance,
        "passes": scalar_passes and cdf_passes,
    }


def _controller_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "projectile": args.projectile,
        "structure_directory": str(args.structure_directory.resolve()),
        "structure_seeds": list(args.structure_seeds),
        "orientations": list(args.orientations),
        "base_energies_ev": sorted(set(float(v) for v in args.base_energies_ev)),
        "kernel_path": str(args.kernels.resolve()),
        "tolerance": args.tolerance,
        "confidence": args.confidence,
        "calibration_trajectories": args.calibration_trajectories,
        "minimum_trajectories": args.minimum_trajectories,
        "maximum_trajectories": args.maximum_trajectories,
        "trajectory_batch_size": args.trajectory_batch_size,
        "path_length_angstrom": args.path_length_angstrom,
        "initial_condition_sampling": getattr(
            args, "sampling_mode", "collision_tube_mixture"
        ),
        "tube_mixture_fraction": getattr(args, "tube_mixture_fraction", 0.5),
        "maximum_refinement_depth": args.maximum_refinement_depth,
        "maximum_energy_points": args.maximum_energy_points,
        "numerical_contract": {
            "scalar_monte_carlo": (
                "95% simultaneous relative confidence half-width <= tolerance"
            ),
            "trajectory_cdf_monte_carlo": (
                "simultaneous DKW absolute half-width <= tolerance"
            ),
            "energy_interpolation": (
                "direct geometric-midpoint scalar and trajectory-CDF tests"
            ),
            "physical_accuracy": (
                "not claimed; NLH/DMol potential uncertainty is separate"
            ),
        },
    }


def _signature(configuration: dict[str, object]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    args.structure_directory = args.structure_directory.expanduser().resolve()
    args.kernels = args.kernels.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    workers = _worker_count(args.workers)
    configuration = _controller_configuration(args)
    signature = _signature(configuration)
    particle_root = args.output_root / args.projectile / signature
    state_path = particle_root / "adaptive_particle_state.json"
    if state_path.is_file():
        state = _manifest(state_path)
        if state.get("configuration_signature") != signature:
            raise RuntimeError(f"Controller state mismatch: {state_path}")
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "configuration_signature": signature,
            "configuration": configuration,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "runtime_workers": workers,
            "cases": {},
            "intervals": {},
        }
        _atomic_json(state_path, state)

    runtime_ceiling = _runtime_trajectory_ceiling(args)
    history = state.setdefault("runtime_trajectory_ceiling_history", [])
    if not isinstance(history, list):
        raise RuntimeError(f"Malformed trajectory-ceiling history: {state_path}")
    if not history or int(history[-1]["maximum_trajectories"]) != runtime_ceiling:
        history.append(
            {
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "maximum_trajectories": runtime_ceiling,
                "reason": (
                    "restart-time liveness ceiling; physical and numerical "
                    "acceptance gates unchanged"
                ),
            }
        )
    state["runtime_workers"] = workers
    _atomic_json(state_path, state)

    base = sorted(set(float(value) for value in args.base_energies_ev))
    for energy in base:
        _run_energy(args, workers, particle_root, energy, state, state_path)
    if args.dry_run:
        state["status"] = "dry_run_complete"
        _atomic_json(state_path, state)
        return 0

    intervals = [(base[i], base[i + 1], 0) for i in range(len(base) - 1)]
    cases_state = state.get("cases", {})
    evaluated_energies = set(base)
    if isinstance(cases_state, dict):
        for record in cases_state.values():
            if not isinstance(record, dict):
                continue
            manifest_path = record.get("production_manifest")
            if manifest_path:
                manifest = _manifest(Path(str(manifest_path)))
                evaluated_energies.add(
                    float(manifest["configuration"]["projectile_energy_ev"])
                )
    interval_records = state["intervals"]
    if not isinstance(interval_records, dict):
        raise RuntimeError("Malformed interval state.")
    while intervals:
        lower, upper, depth = intervals.pop(0)
        key = _interval_key(lower, upper)
        existing = interval_records.get(key)
        if isinstance(existing, dict):
            if existing.get("passes"):
                continue
            midpoint = float(existing["validation_midpoint_energy_ev"])
        else:
            midpoint = math.sqrt(lower * upper)
            evaluated_energies.add(midpoint)
            if len(evaluated_energies) > args.maximum_energy_points:
                raise RuntimeError(
                    "Adaptive energy grid exceeded --maximum-energy-points."
                )
            _run_energy(
                args, workers, particle_root, midpoint, state, state_path
            )
            existing = _analyze_interval(args, state, lower, upper, midpoint)
            existing["depth"] = depth
            interval_records[key] = existing
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(state_path, state)
        if not bool(existing["passes"]):
            if depth >= args.maximum_refinement_depth:
                state["status"] = "refinement_limit_reached"
                _atomic_json(state_path, state)
                raise RuntimeError(
                    f"Energy interval {lower:g}-{upper:g} eV did not meet "
                    f"the {args.tolerance:.3%} gate by depth {depth}."
                )
            intervals.extend(
                ((lower, midpoint, depth + 1), (midpoint, upper, depth + 1))
            )

    state["status"] = "complete"
    state["completed_utc"] = datetime.now(timezone.utc).isoformat()
    state["evaluated_energy_count"] = len(evaluated_energies)
    state["evaluated_energies_ev"] = sorted(evaluated_energies)
    _atomic_json(state_path, state)
    print(f"Completed adaptive {args.projectile} transport: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
