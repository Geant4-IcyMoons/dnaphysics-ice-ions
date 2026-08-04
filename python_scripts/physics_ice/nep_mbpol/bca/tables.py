"""Parallel, restartable generation of adaptive NLH collision kernels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from .adaptivity import adaptive_energy_mesh, adaptive_impact_mesh
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
    ICE_TARGETS,
    PUBLISHED_MINIMUM_TURNING_POTENTIAL_EV,
    canonical_element,
)
from .scattering import NLHCollisionKernel


SCHEMA_VERSION = 3
CSV_COLUMNS = (
    "projectile",
    "target",
    "projectile_energy_ev",
    "relative_kinetic_energy_ev",
    "minimum_turning_potential_ev",
    "threshold_radius_angstrom",
    "hard_cross_section_angstrom2",
    "maximum_impact_parameter_angstrom",
    "area_quantile",
    "impact_parameter_angstrom",
    "closest_approach_angstrom",
    "turning_potential_ev",
    "theta_cm_rad",
    "theta_projectile_lab_rad",
    "recoil_energy_ev",
    "projectile_out_energy_ev",
)


@dataclass(frozen=True)
class KernelTableConfig:
    projectiles: tuple[str, ...] = DEFAULT_PROJECTILES
    energy_min_ev: float = DEFAULT_ENERGY_MIN_EV
    energy_max_ev: float = DEFAULT_ENERGY_MAX_EV
    base_energy_points: int = DEFAULT_BASE_ENERGY_POINTS
    axis_relative_tolerance: float = DEFAULT_AXIS_RELATIVE_TOLERANCE
    max_energy_points: int = DEFAULT_MAX_ENERGY_POINTS
    max_impact_points: int = DEFAULT_MAX_IMPACT_POINTS
    minimum_turning_potential_ev: float = DEFAULT_MINIMUM_TURNING_POTENTIAL_EV
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER
    workers: int = DEFAULT_WORKERS

    def __post_init__(self) -> None:
        canonical = tuple(canonical_element(value) for value in self.projectiles)
        if len(set(canonical)) != len(canonical):
            raise ValueError("Projectiles must be unique.")
        object.__setattr__(self, "projectiles", canonical)
        if self.energy_min_ev <= 0.0 or self.energy_max_ev <= self.energy_min_ev:
            raise ValueError("Require 0 < energy_min_ev < energy_max_ev.")
        if self.base_energy_points < 2:
            raise ValueError("base_energy_points must be at least two.")
        if not 0.0 < self.axis_relative_tolerance < 0.1:
            raise ValueError("axis_relative_tolerance must lie between 0 and 0.1.")
        if self.max_energy_points < self.base_energy_points:
            raise ValueError("max_energy_points cannot be below base_energy_points.")
        if self.max_impact_points < 3:
            raise ValueError("max_impact_points must be at least three.")
        if self.minimum_turning_potential_ev < PUBLISHED_MINIMUM_TURNING_POTENTIAL_EV:
            raise ValueError(
                "The turning-potential threshold cannot be below the published "
                "NLH domain."
            )
        if self.quadrature_order < 16:
            raise ValueError("quadrature_order must be at least 16.")
        if self.workers < 1:
            raise ValueError("workers must be positive.")

    @property
    def config_hash(self) -> str:
        numerical_configuration = asdict(self)
        numerical_configuration.pop("workers")
        encoded = json.dumps(
            {"schema_version": SCHEMA_VERSION, **numerical_configuration},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class _EnergyTask:
    projectile: str
    target: str
    config: KernelTableConfig
    checkpoint_path: Path


@dataclass(frozen=True)
class _KernelTask:
    projectile: str
    target: str
    energy_index: int
    energy_ev: float
    config: KernelTableConfig
    checkpoint_path: Path


def _json_checkpoint_valid(path: Path, config_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value.get("schema_version") == SCHEMA_VERSION
            and value.get("config_hash") == config_hash
        )
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _npz_checkpoint_valid(path: Path, config_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            return (
                int(values["schema_version"].item()) == SCHEMA_VERSION
                and str(values["config_hash"].item()) == config_hash
            )
    except (OSError, ValueError, KeyError):
        return False


def _run_energy_task(task: _EnergyTask) -> str:
    config = task.config
    mesh = adaptive_energy_mesh(
        task.projectile,
        task.target,
        energy_min_ev=config.energy_min_ev,
        energy_max_ev=config.energy_max_ev,
        base_points=config.base_energy_points,
        minimum_turning_potential_ev=config.minimum_turning_potential_ev,
        quadrature_order=config.quadrature_order,
        relative_tolerance=config.axis_relative_tolerance,
        max_points=config.max_energy_points,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config.config_hash,
        "projectile": task.projectile,
        "target": task.target,
        "energies_ev": mesh.energies_ev.tolist(),
        "point_count": mesh.point_count,
        "maximum_theta_cm_relative_error": (
            mesh.maximum_theta_cm_relative_error
        ),
        "maximum_recoil_relative_error": mesh.maximum_recoil_relative_error,
        "validation_evaluations": mesh.validation_evaluations,
    }
    task.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.checkpoint_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(task.checkpoint_path)
    return str(task.checkpoint_path)


def _load_energy_record(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid energy-grid checkpoint {path}.")
    return value


def _run_kernel_task(task: _KernelTask) -> str:
    config = task.config
    kernel = NLHCollisionKernel(
        task.projectile,
        task.target,
        task.energy_ev,
        minimum_turning_potential_ev=config.minimum_turning_potential_ev,
        quadrature_order=config.quadrature_order,
    )
    mesh = adaptive_impact_mesh(
        kernel,
        relative_tolerance=config.axis_relative_tolerance,
        max_points=config.max_impact_points,
    )
    collisions = mesh.collisions
    task.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.checkpoint_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.int64(SCHEMA_VERSION),
            config_hash=np.asarray(config.config_hash),
            projectile=np.asarray(task.projectile),
            target=np.asarray(task.target),
            energy_index=np.int64(task.energy_index),
            projectile_energy_ev=np.float64(task.energy_ev),
            relative_kinetic_energy_ev=np.float64(
                kernel.kinematics.relative_kinetic_energy_ev
            ),
            minimum_turning_potential_ev=np.float64(
                config.minimum_turning_potential_ev
            ),
            threshold_radius_angstrom=np.float64(
                kernel.threshold_radius_angstrom
            ),
            hard_cross_section_angstrom2=np.float64(
                kernel.hard_cross_section_angstrom2
            ),
            maximum_impact_parameter_angstrom=np.float64(
                kernel.maximum_impact_parameter_angstrom
            ),
            impact_point_count=np.int64(mesh.point_count),
            recoil_l1_relative_error=np.float64(
                mesh.recoil_l1_relative_error
            ),
            transport_l1_relative_error=np.float64(
                mesh.transport_l1_relative_error
            ),
            theta_cm_l1_relative_error=np.float64(
                mesh.theta_cm_l1_relative_error
            ),
            theta_cm_max_error_over_pi=np.float64(
                mesh.theta_cm_max_error_over_pi
            ),
            impact_validation_evaluations=np.int64(
                mesh.validation_evaluations
            ),
            area_quantile=mesh.area_quantiles,
            impact_parameter_angstrom=np.asarray(
                [value.impact_parameter_angstrom for value in collisions]
            ),
            closest_approach_angstrom=np.asarray(
                [value.closest_approach_angstrom for value in collisions]
            ),
            turning_potential_ev=np.asarray(
                [value.turning_potential_ev for value in collisions]
            ),
            theta_cm_rad=np.asarray([value.theta_cm_rad for value in collisions]),
            theta_projectile_lab_rad=np.asarray(
                [value.theta_projectile_lab_rad for value in collisions]
            ),
            recoil_energy_ev=np.asarray(
                [value.recoil_energy_ev for value in collisions]
            ),
            projectile_out_energy_ev=np.asarray(
                [value.projectile_out_energy_ev for value in collisions]
            ),
        )
    temporary.replace(task.checkpoint_path)
    return str(task.checkpoint_path)


def _energy_tasks(
    config: KernelTableConfig, checkpoint_directory: Path
) -> list[_EnergyTask]:
    return [
        _EnergyTask(
            projectile=projectile,
            target=target,
            config=config,
            checkpoint_path=checkpoint_directory / f"{projectile}_{target}.json",
        )
        for projectile in config.projectiles
        for target in ICE_TARGETS
    ]


def _kernel_tasks(
    config: KernelTableConfig,
    energy_records: Iterable[dict[str, object]],
    checkpoint_directory: Path,
) -> list[_KernelTask]:
    tasks: list[_KernelTask] = []
    for record in energy_records:
        projectile = str(record["projectile"])
        target = str(record["target"])
        energies = record["energies_ev"]
        if not isinstance(energies, list):
            raise RuntimeError("Energy-grid checkpoint has no energy list.")
        for energy_index, energy in enumerate(energies):
            tasks.append(
                _KernelTask(
                    projectile=projectile,
                    target=target,
                    energy_index=energy_index,
                    energy_ev=float(energy),
                    config=config,
                    checkpoint_path=(
                        checkpoint_directory
                        / f"{projectile}_{target}_{energy_index:04d}.npz"
                    ),
                )
            )
    return tasks


def _run_parallel(
    function,
    tasks,
    *,
    workers: int,
    description: str,
    unit: str,
    show_progress: bool,
) -> None:
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, task) for task in tasks]
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(
                completed,
                total=len(futures),
                unit=unit,
                desc=description,
            )
        for future in completed:
            future.result()


def _write_csv(shards: Iterable[Path], destination: Path) -> int:
    temporary = destination.with_suffix(".tmp")
    row_count = 0
    scalar_columns = CSV_COLUMNS[:8]
    array_columns = CSV_COLUMNS[8:]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for shard in shards:
            with np.load(shard, allow_pickle=False) as values:
                scalars = {name: values[name].item() for name in scalar_columns}
                arrays = [values[name] for name in array_columns]
                for row in zip(*arrays, strict=True):
                    writer.writerow(
                        [
                            scalars["projectile"],
                            scalars["target"],
                            *(
                                f"{float(scalars[name]):.17g}"
                                for name in scalar_columns[2:]
                            ),
                            *(f"{float(value):.17g}" for value in row),
                        ]
                    )
                    row_count += 1
    temporary.replace(destination)
    return row_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _impact_summary(shards: Iterable[Path]) -> dict[str, object]:
    points = []
    errors: dict[str, list[float]] = {
        "recoil_l1_relative_error": [],
        "transport_l1_relative_error": [],
        "theta_cm_l1_relative_error": [],
        "theta_cm_max_error_over_pi": [],
    }
    for shard in shards:
        with np.load(shard, allow_pickle=False) as values:
            points.append(int(values["impact_point_count"].item()))
            for name in errors:
                errors[name].append(float(values[name].item()))
    return {
        "minimum_points": min(points),
        "median_points": float(np.median(points)),
        "maximum_points": max(points),
        "maximum_estimated_errors": {
            name: max(values) for name, values in errors.items()
        },
    }


def generate_kernel_tables(
    config: KernelTableConfig,
    output_directory: str | Path,
    *,
    resume: bool = True,
    show_progress: bool = True,
) -> tuple[Path, Path]:
    """Generate adaptive CSV kernels and an auditable convergence manifest."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / ".checkpoints" / config.config_hash

    energy_tasks = _energy_tasks(config, checkpoint_root / "energy")
    pending_energy = [
        task
        for task in energy_tasks
        if not (
            resume
            and _json_checkpoint_valid(task.checkpoint_path, config.config_hash)
        )
    ]
    _run_parallel(
        _run_energy_task,
        pending_energy,
        workers=config.workers,
        description="Adaptive energy grids",
        unit="pair",
        show_progress=show_progress,
    )
    energy_records = [
        _load_energy_record(task.checkpoint_path) for task in energy_tasks
    ]

    kernel_tasks = _kernel_tasks(
        config, energy_records, checkpoint_root / "kernels"
    )
    pending_kernels = [
        task
        for task in kernel_tasks
        if not (
            resume
            and _npz_checkpoint_valid(task.checkpoint_path, config.config_hash)
        )
    ]
    _run_parallel(
        _run_kernel_task,
        pending_kernels,
        workers=config.workers,
        description="Adaptive impact kernels",
        unit="kernel",
        show_progress=show_progress,
    )

    ordered_shards = [task.checkpoint_path for task in kernel_tasks]
    missing = [
        path
        for path in ordered_shards
        if not _npz_checkpoint_valid(path, config.config_hash)
    ]
    if missing:
        raise RuntimeError(f"Kernel generation left {len(missing)} invalid checkpoints.")

    csv_path = output / "nlh_collision_kernels.csv"
    row_count = _write_csv(ordered_shards, csv_path)
    energy_summary = [
        {
            key: record[key]
            for key in (
                "projectile",
                "target",
                "point_count",
                "maximum_theta_cm_relative_error",
                "maximum_recoil_relative_error",
                "validation_evaluations",
            )
        }
        for record in energy_records
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash,
        "configuration": asdict(config),
        "energy_variable": "total projectile kinetic energy, not energy per nucleon",
        "energy_mesh": {
            "method": (
                "base logarithmic grid with quarter/midpoint/three-quarter direct "
                "solver validation and pair-specific adaptive bisection"
            ),
            "pairs": energy_summary,
        },
        "impact_mesh": {
            "coordinate": "collision-area quantile q=(b/b_max)^2",
            "method": (
                "adaptive bisection with direct quarter/midpoint/three-quarter "
                "validation of a piecewise-linear CM angle followed by exact "
                "two-body recoil and lab-angle kinematics"
            ),
            **_impact_summary(ordered_shards),
        },
        "interpolation_contract": {
            "impact_axis": (
                "piecewise-linear theta_cm in q=(b/b_max)^2, then derive the "
                "lab angle and recoil from exact two-body kinematics"
            ),
            "energy_axis": (
                "piecewise log-log theta_cm in total projectile energy, then "
                "derive the lab angle and recoil at the requested energy"
            ),
            "cross_section": "evaluate the exact threshold expression",
        },
        "error_budget": {
            "per_axis_relative_tolerance": config.axis_relative_tolerance,
            "nominal_combined_two_axis_bound": 2.0
            * config.axis_relative_tolerance,
        },
        "hard_cross_section_evaluation": (
            "evaluate exactly at runtime: sigma=pi*r_threshold^2*"
            "(1-V_threshold/E_cm) for E_cm>V_threshold, otherwise zero; do not "
            "interpolate sigma across the threshold"
        ),
        "row_count": row_count,
        "csv": csv_path.name,
        "csv_sha256": _sha256_file(csv_path),
        "product_scope": (
            "independent-atom NLH hard-collision kernels for H and O target atoms"
        ),
        "not_yet_included": [
            "amorphous/hexagonal structure-aware collision sequencing",
            "long-range weak scattering below the turning-potential threshold",
            "NEP-MB-pol post-collision lattice relaxation and damage evolution",
            "Geant4 runtime table reader",
        ],
        "publication_status": (
            "adaptive infrastructure output; requires the independent dense-grid "
            "benchmark report and validated structure-aware transport stage"
        ),
    }
    manifest_path = output / "nlh_collision_kernels.manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return csv_path, manifest_path
