"""Parallel, restartable generation of independent-atom NLH collision kernels."""

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

from .config import (
    DEFAULT_ENERGY_MAX_EV,
    DEFAULT_ENERGY_MIN_EV,
    DEFAULT_ENERGY_POINTS,
    DEFAULT_IMPACT_POINTS,
    DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    DEFAULT_PROJECTILES,
    DEFAULT_QUADRATURE_ORDER,
    DEFAULT_WORKERS,
    ICE_TARGETS,
    PUBLISHED_MINIMUM_TURNING_POTENTIAL_EV,
    canonical_element,
)
from .scattering import NLHCollisionKernel


SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "projectile",
    "target",
    "projectile_energy_ev",
    "relative_kinetic_energy_ev",
    "minimum_turning_potential_ev",
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
    energy_points: int = DEFAULT_ENERGY_POINTS
    impact_points: int = DEFAULT_IMPACT_POINTS
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
        if self.energy_points < 2:
            raise ValueError("energy_points must be at least two.")
        if self.impact_points < 3:
            raise ValueError("impact_points must be at least three.")
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
    def energies_ev(self) -> np.ndarray:
        return np.geomspace(self.energy_min_ev, self.energy_max_ev, self.energy_points)

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
class _KernelTask:
    projectile: str
    target: str
    energy_index: int
    energy_ev: float
    impact_points: int
    minimum_turning_potential_ev: float
    quadrature_order: int
    config_hash: str
    checkpoint_path: Path


def _checkpoint_is_valid(path: Path, config_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            return str(values["config_hash"].item()) == config_hash
    except (OSError, ValueError, KeyError):
        return False


def _run_task(task: _KernelTask) -> str:
    kernel = NLHCollisionKernel(
        task.projectile,
        task.target,
        task.energy_ev,
        minimum_turning_potential_ev=task.minimum_turning_potential_ev,
        quadrature_order=task.quadrature_order,
    )
    if kernel.maximum_impact_parameter_angstrom <= 0.0:
        raise ValueError(
            f"{task.projectile}-{task.target} at {task.energy_ev:g} eV does not "
            "reach the requested turning-potential domain."
        )

    area_quantile = np.linspace(0.0, 1.0, task.impact_points)
    impacts = kernel.maximum_impact_parameter_angstrom * np.sqrt(area_quantile)
    closest = np.empty(task.impact_points)
    turning = np.empty(task.impact_points)
    theta_cm = np.empty(task.impact_points)
    theta_lab = np.empty(task.impact_points)
    recoil = np.empty(task.impact_points)
    projectile_out = np.empty(task.impact_points)
    for index, impact in enumerate(impacts):
        result = kernel.solve(float(impact))
        closest[index] = result.closest_approach_angstrom
        turning[index] = result.turning_potential_ev
        theta_cm[index] = result.theta_cm_rad
        theta_lab[index] = result.theta_projectile_lab_rad
        recoil[index] = result.recoil_energy_ev
        projectile_out[index] = result.projectile_out_energy_ev

    task.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.checkpoint_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.int64(SCHEMA_VERSION),
            config_hash=np.asarray(task.config_hash),
            projectile=np.asarray(task.projectile),
            target=np.asarray(task.target),
            energy_index=np.int64(task.energy_index),
            projectile_energy_ev=np.float64(task.energy_ev),
            relative_kinetic_energy_ev=np.float64(
                kernel.kinematics.relative_kinetic_energy_ev
            ),
            minimum_turning_potential_ev=np.float64(
                task.minimum_turning_potential_ev
            ),
            hard_cross_section_angstrom2=np.float64(
                kernel.hard_cross_section_angstrom2
            ),
            maximum_impact_parameter_angstrom=np.float64(
                kernel.maximum_impact_parameter_angstrom
            ),
            area_quantile=area_quantile,
            impact_parameter_angstrom=impacts,
            closest_approach_angstrom=closest,
            turning_potential_ev=turning,
            theta_cm_rad=theta_cm,
            theta_projectile_lab_rad=theta_lab,
            recoil_energy_ev=recoil,
            projectile_out_energy_ev=projectile_out,
        )
    temporary.replace(task.checkpoint_path)
    return str(task.checkpoint_path)


def _tasks(config: KernelTableConfig, checkpoint_directory: Path) -> list[_KernelTask]:
    tasks: list[_KernelTask] = []
    for projectile in config.projectiles:
        for target in ICE_TARGETS:
            for energy_index, energy in enumerate(config.energies_ev):
                filename = f"{projectile}_{target}_{energy_index:04d}.npz"
                tasks.append(
                    _KernelTask(
                        projectile=projectile,
                        target=target,
                        energy_index=energy_index,
                        energy_ev=float(energy),
                        impact_points=config.impact_points,
                        minimum_turning_potential_ev=(
                            config.minimum_turning_potential_ev
                        ),
                        quadrature_order=config.quadrature_order,
                        config_hash=config.config_hash,
                        checkpoint_path=checkpoint_directory / filename,
                    )
                )
    return tasks


def _write_csv(shards: Iterable[Path], destination: Path) -> int:
    temporary = destination.with_suffix(".tmp")
    row_count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for shard in shards:
            with np.load(shard, allow_pickle=False) as values:
                scalar = {
                    name: values[name].item()
                    for name in CSV_COLUMNS[:7]
                }
                arrays = [values[name] for name in CSV_COLUMNS[7:]]
                for row in zip(*arrays, strict=True):
                    writer.writerow(
                        [
                            scalar["projectile"],
                            scalar["target"],
                            *(f"{float(scalar[name]):.17g}" for name in CSV_COLUMNS[2:7]),
                            *(f"{float(value):.17g}" for value in row),
                        ]
                    )
                    row_count += 1
    temporary.replace(destination)
    return row_count


def generate_kernel_tables(
    config: KernelTableConfig,
    output_directory: str | Path,
    *,
    resume: bool = True,
    show_progress: bool = True,
) -> tuple[Path, Path]:
    """Generate CSV kernels and a scope/physics manifest.

    Tasks are independent projectile--target--energy blocks.  Checkpoints are
    keyed by the complete numerical configuration and are safe to reuse after
    interruption.  Final ordering is deterministic for any worker count.
    """

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output / ".checkpoints" / config.config_hash
    all_tasks = _tasks(config, checkpoint_directory)
    pending = [
        task
        for task in all_tasks
        if not (resume and _checkpoint_is_valid(task.checkpoint_path, config.config_hash))
    ]

    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {executor.submit(_run_task, task): task for task in pending}
            completed = as_completed(futures)
            if show_progress:
                completed = tqdm(
                    completed,
                    total=len(futures),
                    unit="kernel",
                    desc="NLH collision kernels",
                )
            for future in completed:
                future.result()

    ordered_shards = [task.checkpoint_path for task in all_tasks]
    missing = [path for path in ordered_shards if not _checkpoint_is_valid(path, config.config_hash)]
    if missing:
        raise RuntimeError(f"Kernel generation left {len(missing)} invalid checkpoints.")

    csv_path = output / "nlh_collision_kernels.csv"
    row_count = _write_csv(ordered_shards, csv_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash,
        "configuration": asdict(config),
        "energy_variable": "total projectile kinetic energy, not energy per nucleon",
        "impact_grid": "uniform in collision area: b=b_max*sqrt(area_quantile)",
        "row_count": row_count,
        "csv": csv_path.name,
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
            "infrastructure output; requires quadrature/grid convergence and the "
            "validated structure-aware transport stage"
        ),
    }
    manifest_path = output / "nlh_collision_kernels.manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return csv_path, manifest_path
