#!/usr/bin/env python3
"""Propagate hard C/O/S trajectories through an explicit periodic ice cell."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.config import DEFAULT_WORKERS  # noqa: E402
from bca.runtime import AdaptiveKernelTable  # noqa: E402
from bca.structure import IceStructure, load_ice_structure  # noqa: E402
from bca.trajectory import (  # noqa: E402
    HardTrajectoryResult,
    PeriodicHardCollisionTransport,
)


_WORKER_TRANSPORT: PeriodicHardCollisionTransport | None = None
_WORKER_STRUCTURE: IceStructure | None = None


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
    parser.add_argument("--projectile", choices=("C", "O", "S"), required=True)
    parser.add_argument("--energy-ev", type=float, required=True)
    parser.add_argument("--trajectories", type=int, default=1000)
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
    if args.trajectories < 1:
        raise ValueError("--trajectories must be positive.")
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
    path: Path, results: list[HardTrajectoryResult]
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
        for index, result in enumerate(results):
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


def _write_events(path: Path, results: list[HardTrajectoryResult]) -> None:
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
        for trajectory_index, result in enumerate(results):
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


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    structure: IceStructure,
    kernels: AdaptiveKernelTable,
    results: list[HardTrajectoryResult],
    trajectory_csv: Path,
    event_csv: Path,
) -> None:
    total_path = sum(result.traveled_path_length_angstrom for result in results)
    total_recoil = sum(result.recoil_energy_ev for result in results)
    total_events = sum(len(result.events) for result in results)
    target_counts = Counter(
        event.target for result in results for event in result.events
    )
    terminations = Counter(result.termination for result in results)
    ambiguous = sum(result.ambiguous_event_count for result in results)
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
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "structure": structure.manifest_record(),
        "kernel_manifest": str(kernels.manifest_path),
        "kernel_csv_sha256": kernels.csv_sha256,
        "minimum_turning_potential_ev": kernels.minimum_turning_potential_ev,
        "configuration": {
            "projectile": args.projectile,
            "projectile_energy_ev": args.energy_ev,
            "trajectories": args.trajectories,
            "path_length_angstrom": args.path_length_angstrom,
            "direction": args.direction if args.direction else [0.0, 0.0, 1.0],
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
            "ambiguous_event_count": ambiguous,
            "termination_counts": dict(sorted(terminations.items())),
        },
        "outputs": {
            "trajectories_csv": trajectory_csv.name,
            "events_csv": event_csv.name,
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


def main() -> None:
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

    initializer_args = (
        str(args.structure.expanduser().resolve()),
        str(args.metadata.expanduser().resolve()) if args.metadata else None,
        str(args.kernels.expanduser().resolve()),
        args.allow_unvalidated,
        args.search_window_angstrom,
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
        for index in range(args.trajectories)
    ]
    indexed_results: list[tuple[int, HardTrajectoryResult]] = []
    if args.workers == 1:
        _initialize_worker(*initializer_args)
        indexed_results = [
            _run_one(task)
            for task in tqdm(tasks, unit="trajectory", desc="Hard trajectories")
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=initializer_args,
        ) as executor:
            futures = [executor.submit(_run_one, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                unit="trajectory",
                desc="Hard trajectories",
            ):
                indexed_results.append(future.result())
    indexed_results.sort(key=lambda item: item[0])
    results = [item[1] for item in indexed_results]

    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory_csv = output / "hard_collision_trajectories.csv"
    event_csv = output / "hard_collision_events.csv"
    manifest_path = output / "hard_collision_run.manifest.json"
    _write_trajectories(trajectory_csv, results)
    _write_events(event_csv, results)
    _write_manifest(
        manifest_path,
        args,
        structure,
        kernels,
        results,
        trajectory_csv,
        event_csv,
    )
    print(f"Wrote {trajectory_csv}")
    print(f"Wrote {event_csv}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
