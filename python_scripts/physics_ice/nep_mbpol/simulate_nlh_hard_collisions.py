#!/usr/bin/env python3
"""Propagate hard H/He/C/O/S trajectories through periodic explicit ice."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.config import DEFAULT_PROJECTILES, DEFAULT_WORKERS  # noqa: E402
from bca.convergence import (  # noqa: E402
    DEFAULT_MAXIMUM_TRAJECTORIES,
    DEFAULT_MINIMUM_TRAJECTORIES,
    DEFAULT_STATISTICAL_CONFIDENCE,
    DEFAULT_STATISTICAL_RELATIVE_TOLERANCE,
    DEFAULT_TRAJECTORY_BATCH_SIZE,
    OBSERVABLES,
    RatioStatistics,
    convergence_report,
    doubling_schedule,
    merge_statistics,
)
from bca.runtime import AdaptiveKernelTable  # noqa: E402
from bca.structure import IceStructure, load_ice_structure  # noqa: E402
from bca.trajectory import (  # noqa: E402
    HardTrajectoryResult,
    PeriodicHardCollisionTransport,
)


_WORKER_TRANSPORT: PeriodicHardCollisionTransport | None = None
_WORKER_STRUCTURE: IceStructure | None = None
TRAJECTORY_IMPLEMENTATION_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--kernels",
        type=Path,
        default=HERE / "collision_kernels",
        help="Kernel directory or nlh_collision_kernels.manifest.json.",
    )
    parser.add_argument(
        "--projectile", choices=DEFAULT_PROJECTILES, required=True
    )
    parser.add_argument("--energy-ev", type=float, required=True)
    parser.add_argument(
        "--trajectories",
        type=int,
        help=(
            "Run an exact fixed trajectory count and report, but do not enforce, "
            "statistical convergence. If omitted, trajectory count is adaptive."
        ),
    )
    parser.add_argument(
        "--minimum-trajectories",
        type=int,
        default=DEFAULT_MINIMUM_TRAJECTORIES,
        help="First adaptive statistical check (default: 1000).",
    )
    parser.add_argument(
        "--maximum-trajectories",
        type=int,
        default=DEFAULT_MAXIMUM_TRAJECTORIES,
        help="Hard adaptive sampling limit (default: 1024000).",
    )
    parser.add_argument(
        "--trajectory-batch-size",
        type=int,
        default=DEFAULT_TRAJECTORY_BATCH_SIZE,
        help="Restart/checkpoint batch size (default: 1000).",
    )
    parser.add_argument(
        "--statistical-relative-tolerance",
        type=float,
        default=DEFAULT_STATISTICAL_RELATIVE_TOLERANCE,
        help="Required simultaneous relative confidence half-width (default: 0.005).",
    )
    parser.add_argument(
        "--statistical-confidence",
        type=float,
        default=DEFAULT_STATISTICAL_CONFIDENCE,
        help=(
            "Family-wise confidence across observables and looks "
            "(default: 0.95)."
        ),
    )
    parser.add_argument("--path-length-angstrom", type=float, default=100.0)
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument(
        "--direction",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Fixed crystal-frame direction; default is the Ih c-axis (0,0,1).",
    )
    orientation.add_argument(
        "--isotropic-directions",
        action="store_true",
        help="Sample an independent isotropic direction for every trajectory.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--search-window-angstrom", type=float, default=4.0)
    parser.add_argument("--max-collisions", type=int, default=10_000)
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Diagnostic only: permit a structure without collision attestation.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=HERE / "hard_collision_runs",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.energy_ev) or args.energy_ev <= 0.0:
        raise ValueError("--energy-ev must be finite and positive.")
    if args.trajectories is not None and args.trajectories < 2:
        raise ValueError("--trajectories must be at least two.")
    if args.minimum_trajectories < 2:
        raise ValueError("--minimum-trajectories must be at least two.")
    if args.maximum_trajectories < args.minimum_trajectories:
        raise ValueError(
            "--maximum-trajectories cannot be below --minimum-trajectories."
        )
    if args.trajectory_batch_size < 1:
        raise ValueError("--trajectory-batch-size must be positive.")
    if (
        not math.isfinite(args.statistical_relative_tolerance)
        or args.statistical_relative_tolerance <= 0.0
    ):
        raise ValueError(
            "--statistical-relative-tolerance must be finite and positive."
        )
    if not 0.0 < args.statistical_confidence < 1.0:
        raise ValueError(
            "--statistical-confidence must lie strictly between zero and one."
        )
    if (
        not math.isfinite(args.path_length_angstrom)
        or args.path_length_angstrom <= 0.0
    ):
        raise ValueError("--path-length-angstrom must be finite and positive.")
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.max_collisions < 1:
        raise ValueError("--max-collisions must be positive.")


def _initialize_worker(
    structure_path: str,
    metadata_path: str | None,
    kernel_path: str,
    allow_unvalidated: bool,
    search_window_angstrom: float,
) -> None:
    global _WORKER_STRUCTURE, _WORKER_TRANSPORT
    _WORKER_STRUCTURE = load_ice_structure(
        structure_path,
        metadata_path=metadata_path,
        allow_unvalidated=allow_unvalidated,
    )
    kernels = AdaptiveKernelTable(kernel_path)
    _WORKER_TRANSPORT = PeriodicHardCollisionTransport(
        _WORKER_STRUCTURE,
        kernels,
        search_window_angstrom=search_window_angstrom,
        allow_unvalidated_structure=allow_unvalidated,
    )


def _isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    cosine = 2.0 * float(rng.random()) - 1.0
    phi = 2.0 * math.pi * float(rng.random())
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return np.asarray(
        (sine * math.cos(phi), sine * math.sin(phi), cosine),
        dtype=np.float64,
    )


def _run_one(
    task: tuple[
        int,
        int,
        str,
        float,
        float,
        tuple[float, float, float] | None,
        int,
    ],
) -> tuple[int, HardTrajectoryResult]:
    if _WORKER_TRANSPORT is None or _WORKER_STRUCTURE is None:
        raise RuntimeError("Hard-collision worker was not initialized.")
    (
        trajectory_index,
        master_seed,
        projectile,
        energy_ev,
        path_length_angstrom,
        fixed_direction,
        max_collisions,
    ) = task
    seed = np.random.SeedSequence((master_seed, trajectory_index))
    rng = np.random.default_rng(seed)
    fractional_position = rng.random(3)
    initial_position = fractional_position @ _WORKER_STRUCTURE.lattice_angstrom
    direction = (
        _isotropic_direction(rng)
        if fixed_direction is None
        else np.asarray(fixed_direction, dtype=np.float64)
    )
    result = _WORKER_TRANSPORT.trace(
        projectile,
        energy_ev,
        initial_position,
        direction,
        path_length_angstrom,
        rng=rng,
        max_collisions=max_collisions,
    )
    return trajectory_index, result


def _temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _write_trajectories(
    path: Path, indexed_results: list[tuple[int, HardTrajectoryResult]]
) -> None:
    temporary = _temporary_path(path)
    columns = (
        "trajectory",
        "termination",
        "initial_energy_ev",
        "final_energy_ev",
        "recoil_energy_ev",
        "requested_path_length_angstrom",
        "traveled_path_length_angstrom",
        "collision_count",
        "ambiguous_event_count",
        "initial_x_angstrom",
        "initial_y_angstrom",
        "initial_z_angstrom",
        "final_x_angstrom",
        "final_y_angstrom",
        "final_z_angstrom",
        "initial_dx",
        "initial_dy",
        "initial_dz",
        "final_dx",
        "final_dy",
        "final_dz",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for index, result in indexed_results:
            writer.writerow(
                (
                    index,
                    result.termination,
                    f"{result.initial_energy_ev:.17g}",
                    f"{result.final_energy_ev:.17g}",
                    f"{result.recoil_energy_ev:.17g}",
                    f"{result.requested_path_length_angstrom:.17g}",
                    f"{result.traveled_path_length_angstrom:.17g}",
                    len(result.events),
                    result.ambiguous_event_count,
                    *result.initial_position_angstrom,
                    *result.final_position_angstrom,
                    *result.initial_direction,
                    *result.final_direction,
                )
            )
    temporary.replace(path)


def _write_events(
    path: Path, indexed_results: list[tuple[int, HardTrajectoryResult]]
) -> None:
    temporary = _temporary_path(path)
    columns = (
        "trajectory",
        "collision",
        "target_atom_index",
        "target",
        "target_image_i",
        "target_image_j",
        "target_image_k",
        "path_distance_angstrom",
        "collision_x_angstrom",
        "collision_y_angstrom",
        "collision_z_angstrom",
        "target_x_angstrom",
        "target_y_angstrom",
        "target_z_angstrom",
        "direction_in_x",
        "direction_in_y",
        "direction_in_z",
        "direction_out_x",
        "direction_out_y",
        "direction_out_z",
        "recoil_direction_x",
        "recoil_direction_y",
        "recoil_direction_z",
        "projectile_energy_in_ev",
        "projectile_energy_out_ev",
        "recoil_energy_ev",
        "impact_parameter_angstrom",
        "maximum_impact_parameter_angstrom",
        "theta_cm_rad",
        "theta_projectile_lab_rad",
        "competing_hard_candidates",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for trajectory_index, result in indexed_results:
            for event in result.events:
                writer.writerow(
                    (
                        trajectory_index,
                        event.collision_index,
                        event.target_atom_index,
                        event.target,
                        *event.target_image,
                        f"{event.path_distance_angstrom:.17g}",
                        *event.collision_position_angstrom,
                        *event.target_position_angstrom,
                        *event.direction_in,
                        *event.direction_out,
                        *event.recoil_direction,
                        f"{event.projectile_energy_in_ev:.17g}",
                        f"{event.projectile_energy_out_ev:.17g}",
                        f"{event.recoil_energy_ev:.17g}",
                        f"{event.impact_parameter_angstrom:.17g}",
                        f"{event.maximum_impact_parameter_angstrom:.17g}",
                        f"{event.theta_cm_rad:.17g}",
                        f"{event.theta_projectile_lab_rad:.17g}",
                        event.competing_hard_candidates,
                    )
                )
    temporary.replace(path)


def _batch_statistics(
    indexed_results: list[tuple[int, HardTrajectoryResult]],
) -> dict[str, RatioStatistics]:
    statistics = {name: RatioStatistics() for name in OBSERVABLES}
    for _, result in indexed_results:
        path = result.traveled_path_length_angstrom
        collisions = float(len(result.events))
        recoil = result.recoil_energy_ev
        transport = math.fsum(
            1.0 - math.cos(event.theta_projectile_lab_rad)
            for event in result.events
        )
        statistics["hard_collision_rate_per_angstrom"].add(collisions, path)
        statistics["hard_nuclear_stopping_ev_per_angstrom"].add(recoil, path)
        statistics["hard_transport_rate_per_angstrom"].add(transport, path)
        statistics["mean_recoil_energy_ev_per_collision"].add(
            recoil, collisions
        )
        statistics["mean_one_minus_cosine_per_collision"].add(
            transport, collisions
        )
    return statistics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_signature(
    args: argparse.Namespace,
    structure: IceStructure,
    kernels: AdaptiveKernelTable,
    fixed_direction: tuple[float, float, float] | None,
) -> tuple[str, dict[str, object]]:
    configuration: dict[str, object] = {
        "schema_version": 1,
        "trajectory_implementation_version": TRAJECTORY_IMPLEMENTATION_VERSION,
        "structure_sha256": structure.source_sha256,
        "structure_frame_index": structure.frame_index,
        "kernel_csv_sha256": kernels.csv_sha256,
        "projectile": args.projectile,
        "projectile_energy_ev": args.energy_ev,
        "path_length_angstrom": args.path_length_angstrom,
        "fixed_direction": fixed_direction,
        "isotropic_directions": args.isotropic_directions,
        "seed": args.seed,
        "search_window_angstrom": args.search_window_angstrom,
        "max_collisions": args.max_collisions,
    }
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), configuration


def _checkpoint_batch(
    directory: Path,
    signature: str,
    indexed_results: list[tuple[int, HardTrajectoryResult]],
) -> Path:
    start = indexed_results[0][0]
    stop = indexed_results[-1][0] + 1
    stem = f"batch_{start:09d}_{stop:09d}"
    trajectory_path = directory / f"{stem}.trajectories.csv"
    event_path = directory / f"{stem}.events.csv"
    record_path = directory / f"{stem}.manifest.json"
    _write_trajectories(trajectory_path, indexed_results)
    _write_events(event_path, indexed_results)
    statistics = _batch_statistics(indexed_results)
    target_counts = Counter(
        event.target for _, result in indexed_results for event in result.events
    )
    terminations = Counter(
        result.termination for _, result in indexed_results
    )
    record = {
        "schema_version": 1,
        "configuration_signature": signature,
        "trajectory_start": start,
        "trajectory_stop": stop,
        "trajectory_count": stop - start,
        "trajectory_csv": trajectory_path.name,
        "trajectory_csv_sha256": _sha256_file(trajectory_path),
        "event_csv": event_path.name,
        "event_csv_sha256": _sha256_file(event_path),
        "statistics": {
            name: value.to_dict() for name, value in statistics.items()
        },
        "summary": {
            "target_counts": dict(sorted(target_counts.items())),
            "termination_counts": dict(sorted(terminations.items())),
            "ambiguous_event_count": sum(
                result.ambiguous_event_count for _, result in indexed_results
            ),
        },
    }
    temporary = _temporary_path(record_path)
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(record_path)
    return record_path


def _load_checkpoint_batches(
    directory: Path, signature: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    expected_start = 0
    for path in sorted(directory.glob("batch_*.manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("configuration_signature") != signature:
            raise RuntimeError(f"Checkpoint signature mismatch in {path}.")
        start = int(record["trajectory_start"])
        stop = int(record["trajectory_stop"])
        if start != expected_start or stop <= start:
            raise RuntimeError(f"Non-contiguous checkpoint range in {path}.")
        trajectory_path = directory / str(record["trajectory_csv"])
        event_path = directory / str(record["event_csv"])
        for product, expected in (
            (trajectory_path, record["trajectory_csv_sha256"]),
            (event_path, record["event_csv_sha256"]),
        ):
            if not product.is_file() or _sha256_file(product) != expected:
                raise RuntimeError(f"Checkpoint checksum failure: {product}.")
        statistics = record.get("statistics")
        if not isinstance(statistics, dict) or set(statistics) != set(OBSERVABLES):
            raise RuntimeError(f"Incomplete checkpoint statistics in {path}.")
        records.append(record)
        expected_start = stop
    return records


def _statistics_from_records(
    records: list[dict[str, object]],
) -> dict[str, RatioStatistics]:
    batches = []
    for record in records:
        payload = record["statistics"]
        if not isinstance(payload, dict):
            raise RuntimeError("Malformed checkpoint statistics.")
        batches.append(
            {
                name: RatioStatistics.from_dict(payload[name])
                for name in OBSERVABLES
            }
        )
    return merge_statistics(batches)


def _combine_csv_fragments(
    output_path: Path,
    checkpoint_directory: Path,
    records: list[dict[str, object]],
    record_key: str,
) -> None:
    temporary = _temporary_path(output_path)
    with temporary.open("wb") as output_handle:
        for index, record in enumerate(records):
            fragment = checkpoint_directory / str(record[record_key])
            with fragment.open("rb") as input_handle:
                if index:
                    input_handle.readline()
                shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
    temporary.replace(output_path)


def _execute_tasks(
    tasks: list[
        tuple[
            int,
            int,
            str,
            float,
            float,
            tuple[float, float, float] | None,
            int,
        ]
    ],
    executor: ProcessPoolExecutor | None,
) -> list[tuple[int, HardTrajectoryResult]]:
    indexed_results: list[tuple[int, HardTrajectoryResult]] = []
    if executor is None:
        indexed_results = [_run_one(task) for task in tasks]
    else:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            indexed_results.append(future.result())
    indexed_results.sort(key=lambda item: item[0])
    return indexed_results


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    structure: IceStructure,
    kernels: AdaptiveKernelTable,
    records: list[dict[str, object]],
    statistics: dict[str, RatioStatistics],
    statistical_report: dict[str, object],
    signature: str,
    schedule: tuple[int, ...],
    fixed_direction: tuple[float, float, float] | None,
    checkpoint_directory: Path,
    trajectory_csv: Path,
    event_csv: Path,
) -> None:
    rate_statistics = statistics["hard_collision_rate_per_angstrom"]
    stopping_statistics = statistics[
        "hard_nuclear_stopping_ev_per_angstrom"
    ]
    transport_statistics = statistics["hard_transport_rate_per_angstrom"]
    total_path = rate_statistics.denominator_sum
    total_events = int(round(rate_statistics.numerator_sum))
    total_recoil = stopping_statistics.numerator_sum
    total_transport = transport_statistics.numerator_sum
    target_counts: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    ambiguous = 0
    for record in records:
        summary = record["summary"]
        if not isinstance(summary, dict):
            raise RuntimeError("Malformed checkpoint summary.")
        target_counts.update(summary["target_counts"])
        terminations.update(summary["termination_counts"])
        ambiguous += int(summary["ambiguous_event_count"])
    independent_atom_rate = sum(
        int(np.count_nonzero(structure.species == target))
        / structure.volume_angstrom3
        * kernels.hard_cross_section_angstrom2(
            args.projectile, target, args.energy_ev
        )
        for target in ("H", "O")
    )
    sampled_rate = total_events / total_path if total_path > 0.0 else 0.0
    manifest = {
        "schema_version": 2,
        "trajectory_implementation_version": TRAJECTORY_IMPLEMENTATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_signature": signature,
        "structure": structure.manifest_record(),
        "kernel_manifest": str(kernels.manifest_path),
        "kernel_csv_sha256": kernels.csv_sha256,
        "minimum_turning_potential_ev": kernels.minimum_turning_potential_ev,
        "configuration": {
            "projectile": args.projectile,
            "projectile_energy_ev": args.energy_ev,
            "sampling_mode": "fixed" if args.trajectories else "adaptive",
            "completed_trajectories": rate_statistics.count,
            "fixed_trajectories": args.trajectories,
            "adaptive_trajectory_schedule": list(schedule),
            "trajectory_batch_size": args.trajectory_batch_size,
            "statistical_relative_tolerance": (
                args.statistical_relative_tolerance
            ),
            "statistical_confidence": args.statistical_confidence,
            "path_length_angstrom": args.path_length_angstrom,
            "direction": fixed_direction,
            "isotropic_directions": args.isotropic_directions,
            "seed": args.seed,
            "workers": args.workers,
            "search_window_angstrom": args.search_window_angstrom,
            "max_collisions": args.max_collisions,
            "allow_unvalidated": args.allow_unvalidated,
        },
        "summary": {
            "total_traveled_path_angstrom": total_path,
            "hard_collision_count": total_events,
            "target_counts": dict(sorted(target_counts.items())),
            "total_recoil_energy_ev": total_recoil,
            "sampled_hard_collision_rate_per_angstrom": sampled_rate,
            "independent_atom_hard_rate_per_angstrom_at_initial_energy": (
                independent_atom_rate
            ),
            "sampled_to_independent_atom_rate_ratio": (
                sampled_rate / independent_atom_rate
                if independent_atom_rate > 0.0
                else None
            ),
            "sampled_hard_nuclear_stopping_ev_per_angstrom": (
                total_recoil / total_path if total_path > 0.0 else 0.0
            ),
            "sampled_hard_transport_rate_per_angstrom": (
                total_transport / total_path if total_path > 0.0 else 0.0
            ),
            "mean_recoil_energy_ev_per_collision": (
                total_recoil / total_events if total_events > 0 else None
            ),
            "mean_one_minus_cosine_per_collision": (
                total_transport / total_events if total_events > 0 else None
            ),
            "ambiguous_event_count": ambiguous,
            "termination_counts": dict(sorted(terminations.items())),
        },
        "statistical_convergence": {
            "required": args.trajectories is None,
            **statistical_report,
            "scope": (
                "normalization and first energy/angular moments only; this does "
                "not certify binned angular or recoil distribution tails"
            ),
        },
        "numerical_error_budgets": {
            "collision_kernel_interpolation": (
                "separate adaptive energy/impact grid with nominal 0.5% budget"
            ),
            "trajectory_monte_carlo": (
                f"{args.statistical_relative_tolerance:.3%} simultaneous "
                "relative confidence-half-width target"
            ),
            "combined_claim": (
                "not formed: interpolation and sampling errors are reported "
                "separately and are not the physical-model uncertainty"
            ),
        },
        "outputs": {
            "trajectories_csv": trajectory_csv.name,
            "events_csv": event_csv.name,
            "checkpoint_directory": str(checkpoint_directory),
        },
        "physics_scope": (
            "primary-projectile, static-lattice, retained-domain NLH hard collisions"
        ),
        "not_included": [
            "soft distant scattering below the turning-potential boundary",
            "simultaneous many-atom forces; overlaps are reported as ambiguous",
            "reinsertion and transport of emitted target recoils",
            "lattice relaxation, chemistry, and damage evolution",
            "electronic stopping and charge exchange",
        ],
    }
    temporary = _temporary_path(path)
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    _validate_args(args)
    structure = load_ice_structure(
        args.structure,
        metadata_path=args.metadata,
        allow_unvalidated=args.allow_unvalidated,
    )
    kernels = AdaptiveKernelTable(args.kernels)
    if args.projectile not in kernels.projectiles:
        raise ValueError(
            f"Kernel product contains {kernels.projectiles}, not {args.projectile}."
        )
    fixed_direction = (
        None
        if args.isotropic_directions
        else tuple(float(value) for value in (args.direction or (0.0, 0.0, 1.0)))
    )
    if fixed_direction is not None:
        norm = math.sqrt(sum(value * value for value in fixed_direction))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("--direction cannot be zero or non-finite.")
        fixed_direction = tuple(value / norm for value in fixed_direction)

    adaptive = args.trajectories is None
    schedule = (
        doubling_schedule(
            args.minimum_trajectories,
            args.maximum_trajectories,
            args.trajectory_batch_size,
        )
        if adaptive
        else (args.trajectories,)
    )
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature, signature_configuration = _run_signature(
        args, structure, kernels, fixed_direction
    )
    checkpoint_directory = output / ".trajectory_checkpoints" / signature
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    signature_path = checkpoint_directory / "configuration.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing.get("configuration_signature") != signature:
            raise RuntimeError(f"Checkpoint configuration mismatch: {signature_path}")
    else:
        temporary = _temporary_path(signature_path)
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "configuration_signature": signature,
                    "configuration": signature_configuration,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(signature_path)

    records = _load_checkpoint_batches(checkpoint_directory, signature)
    completed = int(records[-1]["trajectory_stop"]) if records else 0
    if completed > schedule[-1]:
        raise RuntimeError("Checkpoint exceeds the configured trajectory limit.")

    initializer_args = (
        str(args.structure.expanduser().resolve()),
        str(args.metadata.expanduser().resolve()) if args.metadata else None,
        str(args.kernels.expanduser().resolve()),
        args.allow_unvalidated,
        args.search_window_angstrom,
    )
    executor: ProcessPoolExecutor | None = None
    if args.workers == 1:
        _initialize_worker(*initializer_args)
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=initializer_args,
        )

    last_report: dict[str, object] | None = None
    try:
        for look_index, target in enumerate(schedule, start=1):
            if completed > target:
                continue
            if completed < target:
                with tqdm(
                    total=target - completed,
                    unit="trajectory",
                    desc=f"Sampling to statistical look {look_index}/{len(schedule)}",
                ) as progress:
                    while completed < target:
                        stop = min(
                            target, completed + args.trajectory_batch_size
                        )
                        tasks = [
                            (
                                index,
                                args.seed,
                                args.projectile,
                                args.energy_ev,
                                args.path_length_angstrom,
                                fixed_direction,
                                args.max_collisions,
                            )
                            for index in range(completed, stop)
                        ]
                        indexed_results = _execute_tasks(tasks, executor)
                        record_path = _checkpoint_batch(
                            checkpoint_directory, signature, indexed_results
                        )
                        records.append(
                            json.loads(record_path.read_text(encoding="utf-8"))
                        )
                        progress.update(stop - completed)
                        completed = stop
            statistics = _statistics_from_records(records)
            last_report = convergence_report(
                statistics,
                tolerance=args.statistical_relative_tolerance,
                confidence=args.statistical_confidence,
                scheduled_look_count=len(schedule),
                look_index=look_index,
            )
            maximum_width = last_report[
                "maximum_finite_relative_confidence_half_width"
            ]
            width_text = (
                f"{float(maximum_width):.3%}"
                if maximum_width is not None
                else "undefined"
            )
            print(
                f"Statistical look {look_index}/{len(schedule)}: "
                f"N={completed:,}, maximum finite relative half-width="
                f"{width_text}; converged={last_report['converged']}"
            )
            if not adaptive or bool(last_report["converged"]):
                break
    finally:
        if executor is not None:
            executor.shutdown()

    if last_report is None:
        raise RuntimeError("No statistical assessment was produced.")
    statistics = _statistics_from_records(records)
    trajectory_csv = output / "hard_collision_trajectories.csv"
    event_csv = output / "hard_collision_events.csv"
    manifest_path = output / "hard_collision_run.manifest.json"
    _combine_csv_fragments(
        trajectory_csv,
        checkpoint_directory,
        records,
        "trajectory_csv",
    )
    _combine_csv_fragments(
        event_csv,
        checkpoint_directory,
        records,
        "event_csv",
    )
    _write_manifest(
        manifest_path,
        args,
        structure,
        kernels,
        records,
        statistics,
        last_report,
        signature,
        schedule,
        fixed_direction,
        checkpoint_directory,
        trajectory_csv,
        event_csv,
    )
    print(f"Wrote {trajectory_csv}")
    print(f"Wrote {event_csv}")
    print(f"Wrote {manifest_path}")
    if adaptive and not bool(last_report["converged"]):
        print(
            "Statistical convergence was not reached before the configured "
            "trajectory limit.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
