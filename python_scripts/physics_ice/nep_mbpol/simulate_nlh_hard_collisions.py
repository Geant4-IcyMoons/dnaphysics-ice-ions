#!/usr/bin/env python3
"""Propagate hard H/He/C/O/S trajectories through periodic explicit ice."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from dataclasses import replace
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
    simultaneous_dkw_half_width,
)
from bca.runtime import AdaptiveKernelTable  # noqa: E402
from bca.structure import IceStructure, load_ice_structure  # noqa: E402
from bca.trajectory import (  # noqa: E402
    HardTrajectoryResult,
    PeriodicHardCollisionTransport,
)
from ion_ice import PROCESS_EVIDENCE_ROOT  # noqa: E402


_WORKER_TRANSPORT: PeriodicHardCollisionTransport | None = None
_WORKER_STRUCTURE: IceStructure | None = None
TRAJECTORY_IMPLEMENTATION_VERSION = 3


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
    parser.add_argument(
        "--trajectory-cdf-tolerance",
        type=float,
        help=(
            "Optional simultaneous absolute DKW half-width for the total-"
            "recoil and final-deflection trajectory CDFs."
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
        "--control-variate",
        action="store_true",
        help=(
            "Use the unbiased straight-line difference estimator for all five "
            "rate/moment observables."
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "collision_tube_mixture"),
        default="collision_tube_mixture",
        help=(
            "Initial-condition proposal. The default combines uniform cell "
            "translations with exactly weighted collision-tube strata."
        ),
    )
    parser.add_argument(
        "--tube-mixture-fraction",
        type=float,
        default=0.5,
        help=(
            "Probability of drawing from the collision-tube proposal; the "
            "remaining probability samples the target distribution directly "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--output-detail",
        choices=("full", "summary"),
        default="full",
        help=(
            "Write per-trajectory/event CSVs, or only restart-safe sufficient "
            "statistics (default: full)."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            PROCESS_EVIDENCE_ROOT
            / "hard_nuclear_collisions"
            / "validation"
            / "runs"
        ),
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
    if args.trajectory_cdf_tolerance is not None and (
        not math.isfinite(args.trajectory_cdf_tolerance)
        or args.trajectory_cdf_tolerance <= 0.0
    ):
        raise ValueError(
            "--trajectory-cdf-tolerance must be finite and positive."
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
    if args.sampling_mode == "collision_tube_mixture" and not (
        math.isfinite(args.tube_mixture_fraction)
        and 0.0 < args.tube_mixture_fraction < 1.0
    ):
        raise ValueError(
            "--tube-mixture-fraction must lie strictly in (0, 1)."
        )


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
        bool,
        str,
        float,
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
        use_control_variate,
        sampling_mode,
        tube_mixture_fraction,
    ) = task
    seed = np.random.SeedSequence((master_seed, trajectory_index))
    rng = np.random.default_rng(seed)
    direction = (
        _isotropic_direction(rng)
        if fixed_direction is None
        else np.asarray(fixed_direction, dtype=np.float64)
    )
    if sampling_mode == "collision_tube_mixture":
        initial_position, importance_sampling = (
            _WORKER_TRANSPORT.sample_collision_tube_mixture(
                projectile,
                energy_ev,
                direction,
                path_length_angstrom,
                tube_mixture_fraction,
                rng,
            )
        )
    elif sampling_mode == "uniform":
        fractional_position = rng.random(3)
        initial_position = fractional_position @ _WORKER_STRUCTURE.lattice_angstrom
        importance_sampling = None
    else:
        raise ValueError(f"Unknown sampling mode {sampling_mode!r}.")
    control_variate = (
        _WORKER_TRANSPORT.straight_line_control_variate(
            projectile,
            energy_ev,
            initial_position,
            direction,
            path_length_angstrom,
        )
        if use_control_variate
        else None
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
    if control_variate is not None or importance_sampling is not None:
        result = replace(
            result,
            control_variate=control_variate,
            importance_sampling=importance_sampling,
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
        "reference_collision_count",
        "reference_recoil_energy_ev",
        "reference_transport_moment",
        "expected_reference_collision_count",
        "expected_reference_recoil_energy_ev",
        "expected_reference_transport_moment",
        "control_variate_quadrature_relative_error",
        "sampling_component",
        "tube_mixture_fraction",
        "tube_density_over_uniform",
        "target_over_proposal_weight",
        "selected_target",
        "selected_atom_index",
        "selected_area_quantile",
        "selected_stratum_index",
        "selected_stratum_count",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for index, result in indexed_results:
            reference = result.control_variate
            importance = result.importance_sampling
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
                    *(
                        (
                            reference.collision_count,
                            reference.recoil_energy_ev,
                            reference.transport_moment,
                            reference.expected_collision_count,
                            reference.expected_recoil_energy_ev,
                            reference.expected_transport_moment,
                            reference.maximum_quadrature_relative_error,
                        )
                        if reference is not None
                        else ("", "", "", "", "", "", "")
                    ),
                    *(
                        (
                            importance.component,
                            importance.tube_mixture_fraction,
                            importance.tube_density_over_uniform,
                            importance.target_over_proposal_weight,
                            importance.selected_target or "",
                            (
                                importance.selected_atom_index
                                if importance.selected_atom_index is not None
                                else ""
                            ),
                            (
                                importance.selected_area_quantile
                                if importance.selected_area_quantile is not None
                                else ""
                            ),
                            (
                                importance.selected_stratum_index
                                if importance.selected_stratum_index is not None
                                else ""
                            ),
                            (
                                importance.selected_stratum_count
                                if importance.selected_stratum_count is not None
                                else ""
                            ),
                        )
                        if importance is not None
                        else ("uniform", "", "", 1.0, "", "", "", "", "")
                    ),
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


def _write_distribution_sample(
    path: Path, indexed_results: list[tuple[int, HardTrajectoryResult]]
) -> int:
    """Store only direct target draws, which retain ordinary unweighted CDFs."""

    target_results = [
        (index, result)
        for index, result in indexed_results
        if result.importance_sampling is None
        or result.importance_sampling.component == "uniform"
    ]

    trajectory = np.asarray(
        [index for index, _ in target_results], dtype=np.int64
    )
    total_recoil = np.asarray(
        [result.recoil_energy_ev for _, result in target_results],
        dtype=np.float64,
    )
    final_deflection = np.asarray(
        [
            math.acos(
                min(
                    1.0,
                    max(
                        -1.0,
                        float(
                            np.dot(
                                result.initial_direction,
                                result.final_direction,
                            )
                        ),
                    ),
                )
            )
            for _, result in target_results
        ],
        dtype=np.float64,
    )
    temporary = _temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            trajectory=trajectory,
            total_recoil_energy_ev=total_recoil,
            final_deflection_rad=final_deflection,
        )
    temporary.replace(path)
    return len(target_results)


def _batch_statistics(
    indexed_results: list[tuple[int, HardTrajectoryResult]],
    *,
    control_variate: bool = False,
) -> dict[str, RatioStatistics]:
    statistics = {name: RatioStatistics() for name in OBSERVABLES}
    for _, result in indexed_results:
        importance = result.importance_sampling
        weight = (
            importance.target_over_proposal_weight
            if importance is not None
            else 1.0
        )
        path = weight * result.traveled_path_length_angstrom
        sampled_collisions = float(len(result.events))
        sampled_recoil = result.recoil_energy_ev
        sampled_transport = math.fsum(
            1.0 - math.cos(event.theta_projectile_lab_rad)
            for event in result.events
        )
        if control_variate:
            reference = result.control_variate
            if reference is None:
                raise RuntimeError("Control-variate result is missing its reference.")
            collisions = reference.expected_collision_count + weight * (
                sampled_collisions - reference.collision_count
            )
            recoil = reference.expected_recoil_energy_ev + weight * (
                sampled_recoil - reference.recoil_energy_ev
            )
            transport = reference.expected_transport_moment + weight * (
                sampled_transport - reference.transport_moment
            )
        else:
            collisions = weight * sampled_collisions
            recoil = weight * sampled_recoil
            transport = weight * sampled_transport
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
        "control_variate": getattr(args, "control_variate", False),
        "initial_condition_sampling": getattr(args, "sampling_mode", "uniform"),
        "tube_mixture_fraction": getattr(args, "tube_mixture_fraction", 0.5),
        "output_detail": getattr(args, "output_detail", "full"),
    }
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), configuration


def _checkpoint_batch(
    directory: Path,
    signature: str,
    indexed_results: list[tuple[int, HardTrajectoryResult]],
    *,
    output_detail: str,
    control_variate: bool,
) -> Path:
    start = indexed_results[0][0]
    stop = indexed_results[-1][0] + 1
    stem = f"batch_{start:09d}_{stop:09d}"
    trajectory_path = directory / f"{stem}.trajectories.csv"
    event_path = directory / f"{stem}.events.csv"
    distribution_path = directory / f"{stem}.distributions.npz"
    record_path = directory / f"{stem}.manifest.json"
    target_distribution_sample_count = sum(
        result.importance_sampling is None
        or result.importance_sampling.component == "uniform"
        for _, result in indexed_results
    )
    if output_detail == "full":
        _write_trajectories(trajectory_path, indexed_results)
        _write_events(event_path, indexed_results)
    else:
        target_distribution_sample_count = _write_distribution_sample(
            distribution_path, indexed_results
        )
    statistics = _batch_statistics(
        indexed_results, control_variate=control_variate
    )
    raw_statistics = _batch_statistics(indexed_results)
    target_counts = Counter(
        event.target for _, result in indexed_results for event in result.events
    )
    terminations = Counter(
        result.termination for _, result in indexed_results
    )
    weighted_actual_collision_count = math.fsum(
        (
            result.importance_sampling.target_over_proposal_weight
            if result.importance_sampling is not None
            else 1.0
        )
        * len(result.events)
        for _, result in indexed_results
    )
    weighted_ambiguous_collision_count = math.fsum(
        (
            result.importance_sampling.target_over_proposal_weight
            if result.importance_sampling is not None
            else 1.0
        )
        for _, result in indexed_results
        for event in result.events
        if event.competing_hard_candidates > 0
    )
    weighted_actual_recoil_energy_ev = math.fsum(
        (
            result.importance_sampling.target_over_proposal_weight
            if result.importance_sampling is not None
            else 1.0
        )
        * event.recoil_energy_ev
        for _, result in indexed_results
        for event in result.events
    )
    weighted_ambiguous_recoil_energy_ev = math.fsum(
        (
            result.importance_sampling.target_over_proposal_weight
            if result.importance_sampling is not None
            else 1.0
        )
        * event.recoil_energy_ev
        for _, result in indexed_results
        for event in result.events
        if event.competing_hard_candidates > 0
    )
    weighted_actual_transport_moment = math.fsum(
        (
            result.importance_sampling.target_over_proposal_weight
            if result.importance_sampling is not None
            else 1.0
        )
        * (1.0 - math.cos(event.theta_projectile_lab_rad))
        for _, result in indexed_results
        for event in result.events
    )
    weighted_ambiguous_transport_moment = math.fsum(
        (
            result.importance_sampling.target_over_proposal_weight
            if result.importance_sampling is not None
            else 1.0
        )
        * (1.0 - math.cos(event.theta_projectile_lab_rad))
        for _, result in indexed_results
        for event in result.events
        if event.competing_hard_candidates > 0
    )
    record = {
        "schema_version": 4,
        "configuration_signature": signature,
        "trajectory_start": start,
        "trajectory_stop": stop,
        "trajectory_count": stop - start,
        "output_detail": output_detail,
        "statistics": {
            name: value.to_dict() for name, value in statistics.items()
        },
        "raw_statistics": {
            name: value.to_dict() for name, value in raw_statistics.items()
        },
        "summary": {
            "target_counts": dict(sorted(target_counts.items())),
            "termination_counts": dict(sorted(terminations.items())),
            "ambiguous_event_count": sum(
                result.ambiguous_event_count for _, result in indexed_results
            ),
            "event_row_count": sum(
                len(result.events) for _, result in indexed_results
            ),
            "importance_weighted_actual_collision_count": (
                weighted_actual_collision_count
            ),
            "importance_weighted_ambiguous_collision_count": (
                weighted_ambiguous_collision_count
            ),
            "importance_weighted_actual_recoil_energy_ev": (
                weighted_actual_recoil_energy_ev
            ),
            "importance_weighted_ambiguous_recoil_energy_ev": (
                weighted_ambiguous_recoil_energy_ev
            ),
            "importance_weighted_actual_transport_moment": (
                weighted_actual_transport_moment
            ),
            "importance_weighted_ambiguous_transport_moment": (
                weighted_ambiguous_transport_moment
            ),
            "target_distribution_sample_count": (
                target_distribution_sample_count
            ),
            "proposal_component_counts": dict(
                sorted(
                    Counter(
                        (
                            result.importance_sampling.component
                            if result.importance_sampling is not None
                            else "uniform"
                        )
                        for _, result in indexed_results
                    ).items()
                )
            ),
            "importance_weight_sum": math.fsum(
                (
                    result.importance_sampling.target_over_proposal_weight
                    if result.importance_sampling is not None
                    else 1.0
                )
                for _, result in indexed_results
            ),
            "importance_weight_square_sum": math.fsum(
                (
                    result.importance_sampling.target_over_proposal_weight
                    if result.importance_sampling is not None
                    else 1.0
                )
                ** 2
                for _, result in indexed_results
            ),
            "maximum_importance_weight": max(
                (
                    result.importance_sampling.target_over_proposal_weight
                    if result.importance_sampling is not None
                    else 1.0
                )
                for _, result in indexed_results
            ),
            "maximum_control_variate_quadrature_relative_error": max(
                (
                    result.control_variate.maximum_quadrature_relative_error
                    for _, result in indexed_results
                    if result.control_variate is not None
                ),
                default=0.0,
            ),
        },
    }
    if output_detail == "full":
        record.update(
            {
                "trajectory_csv": trajectory_path.name,
                "trajectory_csv_sha256": _sha256_file(trajectory_path),
                "event_csv": event_path.name,
                "event_csv_sha256": _sha256_file(event_path),
            }
        )
    else:
        record.update(
            {
                "distribution_sample": distribution_path.name,
                "distribution_sample_sha256": _sha256_file(
                    distribution_path
                ),
            }
        )
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
        if record.get("output_detail") == "full":
            trajectory_path = directory / str(record["trajectory_csv"])
            event_path = directory / str(record["event_csv"])
            for product, expected in (
                (trajectory_path, record["trajectory_csv_sha256"]),
                (event_path, record["event_csv_sha256"]),
            ):
                if not product.is_file() or _sha256_file(product) != expected:
                    raise RuntimeError(f"Checkpoint checksum failure: {product}.")
        elif record.get("output_detail") == "summary":
            distribution_path = directory / str(record["distribution_sample"])
            if (
                not distribution_path.is_file()
                or _sha256_file(distribution_path)
                != record["distribution_sample_sha256"]
            ):
                raise RuntimeError(
                    f"Checkpoint checksum failure: {distribution_path}."
                )
        else:
            raise RuntimeError(f"Unknown checkpoint output detail in {path}.")
        statistics = record.get("statistics")
        if not isinstance(statistics, dict) or set(statistics) != set(OBSERVABLES):
            raise RuntimeError(f"Incomplete checkpoint statistics in {path}.")
        raw_statistics = record.get("raw_statistics")
        if (
            not isinstance(raw_statistics, dict)
            or set(raw_statistics) != set(OBSERVABLES)
        ):
            raise RuntimeError(f"Incomplete raw checkpoint statistics in {path}.")
        records.append(record)
        expected_start = stop
    return records


def _statistics_from_records(
    records: list[dict[str, object]],
    record_key: str = "statistics",
) -> dict[str, RatioStatistics]:
    batches = []
    for record in records:
        payload = record[record_key]
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
            bool,
            str,
            float,
        ]
    ],
    executor: ProcessPoolExecutor | None,
    worker_count: int,
) -> list[tuple[int, HardTrajectoryResult]]:
    indexed_results: list[tuple[int, HardTrajectoryResult]] = []
    if executor is None:
        indexed_results = [_run_one(task) for task in tasks]
    else:
        # Keep the executor's wake-up pipe and pending-work dictionary bounded.
        # Submitting an entire 100,000-trajectory checkpoint batch at once can
        # eventually fill that pipe during long production runs, leaving the
        # parent blocked in write(2) while every worker waits for work.
        task_iterator = iter(tasks)
        maximum_pending = 4 * worker_count
        pending = {
            executor.submit(_run_one, task)
            for task in (
                next(task_iterator, None) for _ in range(maximum_pending)
            )
            if task is not None
        }
        while pending:
            completed_futures, pending = wait(
                pending, return_when=FIRST_COMPLETED
            )
            for future in completed_futures:
                indexed_results.append(future.result())
            for _ in range(maximum_pending - len(pending)):
                task = next(task_iterator, None)
                if task is None:
                    break
                pending.add(executor.submit(_run_one, task))
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
    raw_statistics = _statistics_from_records(records, "raw_statistics")
    raw_statistical_report = convergence_report(
        raw_statistics,
        tolerance=args.statistical_relative_tolerance,
        confidence=args.statistical_confidence,
        scheduled_look_count=len(schedule),
        look_index=int(statistical_report["look_index"]),
    )
    rate_statistics = statistics["hard_collision_rate_per_angstrom"]
    stopping_statistics = statistics[
        "hard_nuclear_stopping_ev_per_angstrom"
    ]
    transport_statistics = statistics["hard_transport_rate_per_angstrom"]
    total_path = rate_statistics.denominator_sum
    estimated_total_events = rate_statistics.numerator_sum
    estimated_total_recoil = stopping_statistics.numerator_sum
    estimated_total_transport = transport_statistics.numerator_sum
    raw_total_events = raw_statistics[
        "hard_collision_rate_per_angstrom"
    ].numerator_sum
    raw_total_recoil = raw_statistics[
        "hard_nuclear_stopping_ev_per_angstrom"
    ].numerator_sum
    raw_total_transport = raw_statistics[
        "hard_transport_rate_per_angstrom"
    ].numerator_sum
    target_counts: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    ambiguous = 0
    maximum_quadrature_error = 0.0
    target_distribution_sample_count = 0
    proposal_component_counts: Counter[str] = Counter()
    importance_weight_sum = 0.0
    importance_weight_square_sum = 0.0
    maximum_importance_weight = 0.0
    event_row_count = 0
    weighted_actual_collision_count = 0.0
    weighted_ambiguous_collision_count = 0.0
    weighted_actual_recoil_energy_ev = 0.0
    weighted_ambiguous_recoil_energy_ev = 0.0
    weighted_actual_transport_moment = 0.0
    weighted_ambiguous_transport_moment = 0.0
    for record in records:
        summary = record["summary"]
        if not isinstance(summary, dict):
            raise RuntimeError("Malformed checkpoint summary.")
        target_counts.update(summary["target_counts"])
        terminations.update(summary["termination_counts"])
        ambiguous += int(summary["ambiguous_event_count"])
        event_row_count += int(summary["event_row_count"])
        weighted_actual_collision_count += float(
            summary["importance_weighted_actual_collision_count"]
        )
        weighted_ambiguous_collision_count += float(
            summary["importance_weighted_ambiguous_collision_count"]
        )
        weighted_actual_recoil_energy_ev += float(
            summary["importance_weighted_actual_recoil_energy_ev"]
        )
        weighted_ambiguous_recoil_energy_ev += float(
            summary["importance_weighted_ambiguous_recoil_energy_ev"]
        )
        weighted_actual_transport_moment += float(
            summary["importance_weighted_actual_transport_moment"]
        )
        weighted_ambiguous_transport_moment += float(
            summary["importance_weighted_ambiguous_transport_moment"]
        )
        target_distribution_sample_count += int(
            summary["target_distribution_sample_count"]
        )
        proposal_component_counts.update(summary["proposal_component_counts"])
        importance_weight_sum += float(summary["importance_weight_sum"])
        importance_weight_square_sum += float(
            summary["importance_weight_square_sum"]
        )
        maximum_importance_weight = max(
            maximum_importance_weight,
            float(summary["maximum_importance_weight"]),
        )
        maximum_quadrature_error = max(
            maximum_quadrature_error,
            float(
                summary[
                    "maximum_control_variate_quadrature_relative_error"
                ]
            ),
        )
    independent_atom_rate = sum(
        int(np.count_nonzero(structure.species == target))
        / structure.volume_angstrom3
        * kernels.hard_cross_section_angstrom2(
            args.projectile, target, args.energy_ev
        )
        for target in ("H", "O")
    )
    sampled_rate = (
        estimated_total_events / total_path if total_path > 0.0 else 0.0
    )
    raw_sampled_rate = raw_total_events / total_path if total_path > 0.0 else 0.0
    manifest = {
        "schema_version": 4,
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
            "trajectory_cdf_tolerance": args.trajectory_cdf_tolerance,
            "path_length_angstrom": args.path_length_angstrom,
            "direction": fixed_direction,
            "isotropic_directions": args.isotropic_directions,
            "seed": args.seed,
            "workers": args.workers,
            "search_window_angstrom": args.search_window_angstrom,
            "max_collisions": args.max_collisions,
            "allow_unvalidated": args.allow_unvalidated,
            "control_variate": args.control_variate,
            "initial_condition_sampling": args.sampling_mode,
            "tube_mixture_fraction": args.tube_mixture_fraction,
            "output_detail": args.output_detail,
        },
        "summary": {
            "total_traveled_path_angstrom": total_path,
            "event_row_count": event_row_count,
            "importance_estimated_raw_hard_collision_count": raw_total_events,
            "estimated_hard_collision_count": estimated_total_events,
            "target_counts": dict(sorted(target_counts.items())),
            "raw_total_recoil_energy_ev": raw_total_recoil,
            "estimated_total_recoil_energy_ev": estimated_total_recoil,
            "sampled_hard_collision_rate_per_angstrom": sampled_rate,
            "raw_sampled_hard_collision_rate_per_angstrom": raw_sampled_rate,
            "independent_atom_hard_rate_per_angstrom_at_initial_energy": (
                independent_atom_rate
            ),
            "sampled_to_independent_atom_rate_ratio": (
                sampled_rate / independent_atom_rate
                if independent_atom_rate > 0.0
                else None
            ),
            "sampled_hard_nuclear_stopping_ev_per_angstrom": (
                estimated_total_recoil / total_path if total_path > 0.0 else 0.0
            ),
            "raw_sampled_hard_nuclear_stopping_ev_per_angstrom": (
                raw_total_recoil / total_path if total_path > 0.0 else 0.0
            ),
            "sampled_hard_transport_rate_per_angstrom": (
                estimated_total_transport / total_path
                if total_path > 0.0
                else 0.0
            ),
            "raw_sampled_hard_transport_rate_per_angstrom": (
                raw_total_transport / total_path if total_path > 0.0 else 0.0
            ),
            "mean_recoil_energy_ev_per_collision": (
                estimated_total_recoil / estimated_total_events
                if estimated_total_events > 0
                else None
            ),
            "raw_mean_recoil_energy_ev_per_collision": (
                raw_total_recoil / raw_total_events
                if raw_total_events > 0
                else None
            ),
            "mean_one_minus_cosine_per_collision": (
                estimated_total_transport / estimated_total_events
                if estimated_total_events > 0
                else None
            ),
            "raw_mean_one_minus_cosine_per_collision": (
                raw_total_transport / raw_total_events
                if raw_total_events > 0
                else None
            ),
            "ambiguous_event_count": ambiguous,
            "importance_weighted_ambiguous_fractions": {
                "collision_count": (
                    weighted_ambiguous_collision_count
                    / weighted_actual_collision_count
                    if weighted_actual_collision_count > 0.0
                    else None
                ),
                "recoil_energy": (
                    weighted_ambiguous_recoil_energy_ev
                    / weighted_actual_recoil_energy_ev
                    if weighted_actual_recoil_energy_ev > 0.0
                    else None
                ),
                "transport_moment": (
                    weighted_ambiguous_transport_moment
                    / weighted_actual_transport_moment
                    if weighted_actual_transport_moment > 0.0
                    else None
                ),
            },
            "termination_counts": dict(sorted(terminations.items())),
            "target_distribution_sample_count": (
                target_distribution_sample_count
            ),
            "proposal_component_counts": dict(
                sorted(proposal_component_counts.items())
            ),
            "importance_weight_mean": (
                importance_weight_sum / rate_statistics.count
                if rate_statistics.count
                else None
            ),
            "importance_weight_effective_sample_size": (
                importance_weight_sum * importance_weight_sum
                / importance_weight_square_sum
                if importance_weight_square_sum > 0.0
                else None
            ),
            "maximum_importance_weight": maximum_importance_weight,
        },
        "statistical_convergence": {
            "required": args.trajectories is None,
            **statistical_report,
            "scope": (
                "normalization and first energy/angular moments, plus the "
                "entire trajectory-level total-recoil/final-deflection CDFs "
                "when trajectory_cdf_tolerance is set"
            ),
        },
        "raw_statistical_convergence": {
            "required": False,
            **raw_statistical_report,
            "scope": "diagnostic estimator-selection evidence only",
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
            "control_variate_moment_quadrature_maximum_relative_error": (
                maximum_quadrature_error
            ),
        },
        "control_variate": {
            "enabled": args.control_variate,
            "estimator": (
                "atomistic trajectory minus its unperturbed straight-line "
                "retained collision sum plus the exact periodic-translation "
                "average"
            ),
            "coefficient": 1.0,
            "purpose": (
                "unbiased variance reduction; it does not replace or rescale "
                "the atomistic model"
            ),
        },
        "importance_sampling": {
            "mode": args.sampling_mode,
            "tube_mixture_fraction": args.tube_mixture_fraction,
            "estimator": (
                "exact target/proposal likelihood ratio for a mixture of "
                "uniform periodic translations and adaptive impact-area "
                "collision tubes"
            ),
            "purpose": (
                "unbiased rare-event variance reduction; it changes neither "
                "the NLH interaction nor the hard-collision definition"
            ),
        },
        "outputs": {
            "detail": args.output_detail,
            "trajectories_csv": (
                trajectory_csv.name if args.output_detail == "full" else None
            ),
            "events_csv": (
                event_csv.name if args.output_detail == "full" else None
            ),
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
                                args.control_variate,
                                args.sampling_mode,
                                args.tube_mixture_fraction,
                            )
                            for index in range(completed, stop)
                        ]
                        indexed_results = _execute_tasks(
                            tasks, executor, args.workers
                        )
                        record_path = _checkpoint_batch(
                            checkpoint_directory,
                            signature,
                            indexed_results,
                            output_detail=args.output_detail,
                            control_variate=args.control_variate,
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
            if args.trajectory_cdf_tolerance is not None:
                cdf_sample_count = sum(
                    int(record["summary"]["target_distribution_sample_count"])
                    for record in records
                )
                if cdf_sample_count < 1:
                    raise RuntimeError(
                        "No direct target-distribution draws are available "
                        "for the unweighted trajectory CDF gate."
                    )
                cdf_half_width, cdf_individual_confidence = (
                    simultaneous_dkw_half_width(
                        cdf_sample_count,
                        args.statistical_confidence,
                        distribution_count=2,
                        scheduled_look_count=len(schedule),
                    )
                )
                cdf_passes = cdf_half_width <= args.trajectory_cdf_tolerance
                last_report["trajectory_cdf_convergence"] = {
                    "passes": cdf_passes,
                    "absolute_confidence_half_width": cdf_half_width,
                    "absolute_tolerance": args.trajectory_cdf_tolerance,
                    "distribution_count": 2,
                    "target_distribution_sample_count": cdf_sample_count,
                    "individual_band_confidence": cdf_individual_confidence,
                    "distributions": [
                        "total_recoil_energy_ev_per_trajectory",
                        "final_projectile_deflection_rad_per_trajectory",
                    ],
                    "method": (
                        "Dvoretzky-Kiefer-Wolfowitz band with Bonferroni "
                        "correction across distributions and scheduled looks"
                    ),
                }
                last_report["converged"] = bool(
                    last_report["converged"] and cdf_passes
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
                f"{width_text}; "
                + (
                    "CDF half-width="
                    f"{float(last_report['trajectory_cdf_convergence']['absolute_confidence_half_width']):.3%}; "
                    if "trajectory_cdf_convergence" in last_report
                    else ""
                )
                + f"converged={last_report['converged']}"
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
    if args.output_detail == "full":
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
    if args.output_detail == "full":
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
