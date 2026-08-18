"""Fail-closed planning for low-energy ion-capture mixed-CDFT couplings."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from tqdm import tqdm

from ion_ice import ProjectileDefinition
from soft_dft.config import DEFAULT_CP2K_SETTINGS, CP2KSettings
from soft_dft.geometry import ScanGeometry, build_scan_geometries
from soft_dft.workflow import (
    VALIDATED_BRANCH_EXECUTION,
    VALIDATED_BRANCH_RUNNER,
)

from .channels import SingleElectronCaptureChannel, build_single_capture_channels


SCHEMA_VERSION = 2
IMPLEMENTATION_VERSION = 2
STATE_ROLES = ("entrance", "capture_product")
BRANCH_HANDOFF_PENDING = "branch_handoff_pending"
_BRANCH_HANDOFF_REQUIRED = (
    "Low-energy charge-exchange execution requires two reciprocal validated "
    "CDFT branch states for each exact geometry, total charge, multiplicity, "
    "and projectile population. Accepted-state handoff into MIXED_CDFT is not "
    "yet implemented; no CP2K calculation was started."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _configuration(
    projectile: ProjectileDefinition,
    settings: CP2KSettings,
    channels: tuple[SingleElectronCaptureChannel, ...],
    geometries: tuple[ScanGeometry, ...],
) -> dict[str, Any]:
    incident_charges = {
        channel.incident_state.charge for channel in channels
    }
    product_charges = {channel.product_state.charge for channel in channels}
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "projectile": projectile.soft_dft_dict(),
        "represented_projectile_charge_ladder": list(
            range(projectile.atomic_number + 1)
        ),
        "incident_capture_charges": [
            channel.incident_state.charge for channel in channels
        ],
        "charge_state_coverage": [
            {
                "charge": charge,
                "incident_in_selected_capture_channels": charge in incident_charges,
                "product_in_selected_capture_channels": charge in product_charges,
                "role": (
                    "incident_and_product"
                    if charge in incident_charges and charge in product_charges
                    else "incident_only"
                    if charge in incident_charges
                    else "product_endpoint"
                    if charge in product_charges
                    else "represented_but_not_selected"
                ),
            }
            for charge in range(projectile.atomic_number + 1)
        ],
        "channels": [channel.as_dict() for channel in channels],
        "cp2k": settings.as_dict(),
        "scan_geometries": [geometry.as_dict() for geometry in geometries],
        "method": {
            "electronic_state_requirement": (
                "two reciprocal validated charge-localized all-electron CDFT "
                "branches with exact state-identity matching"
            ),
            "electronic_state_handoff": BRANCH_HANDOFF_PENDING,
            "electronic_coupling_design": (
                "CP2K MIXED_CDFT with Lowdin orthogonalization"
            ),
            "transition_probability": "Landau-Zener using trajectory radial speed",
            "cross_section": "2*pi*integral b*P_capture(b) db",
        },
        "energy_scope": {
            "requested_low_energy_total_ev": [100.0, 1000.0],
            "overlap_requirement": (
                "Extend calculations to the CTMC lower boundary in keV/u; do "
                "not confuse total projectile energy with energy per nucleon."
            ),
        },
        "physics_status": "validation_pending",
        "excluded_processes": [
            "multiple-electron capture",
            "capture into excited H2O+ electronic channels",
            "projectile electron loss",
            "transfer ionisation",
            "spin-orbit-changing capture",
        ],
    }


def build_workflow(
    output_root: Path,
    *,
    projectile: ProjectileDefinition,
    settings: CP2KSettings | None = None,
    incident_charges: Iterable[int] | None = None,
    geometries: Iterable[ScanGeometry] | None = None,
) -> Path:
    """Plan the two required diabatic branches without rendering CDFT inputs."""

    projectile.validate()
    channels = build_single_capture_channels(projectile, incident_charges)
    selected_geometries = tuple(
        build_scan_geometries(projectile) if geometries is None else geometries
    )
    if not selected_geometries:
        raise ValueError("Select at least one projectile--H2O geometry.")
    if any(
        geometry.coordinates_angstrom[0][0] != projectile.symbol
        for geometry in selected_geometries
    ):
        raise ValueError("Scan geometry projectile does not match its definition.")
    geometry_keys = {
        (geometry.orientation, float(geometry.separation_angstrom).hex())
        for geometry in selected_geometries
    }
    if len(geometry_keys) != len(selected_geometries):
        raise ValueError("Scan geometries contain duplicate points.")
    if settings is None:
        settings = replace(
            DEFAULT_CP2K_SETTINGS,
            projectile_basis_set=projectile.projectile_basis_set,
            basis_file=projectile.basis_file,
        )

    configuration = _configuration(
        projectile, settings, channels, selected_geometries
    )
    signature = _sha256_bytes(_canonical_json(configuration))
    run_root = output_root.resolve() / signature
    manifest_path = run_root / "workflow.manifest.json"
    if manifest_path.exists():
        existing = load_workflow_manifest(manifest_path, verify_inputs=True)
        if existing["configuration_signature"] != signature:
            raise RuntimeError("Existing workflow has an incompatible signature.")
        return manifest_path

    geometry_indices: dict[str, int] = {}
    work_units: list[dict[str, Any]] = []
    total = len(channels) * len(selected_geometries)
    for channel in channels:
        geometry_indices.clear()
        for geometry in tqdm(
            selected_geometries,
            desc=f"Preparing {channel.projectile} q={channel.incident_state.charge}",
            unit="geometry",
            disable=total < 20,
        ):
            geometry_index = geometry_indices.get(geometry.orientation, 0)
            geometry_indices[geometry.orientation] = geometry_index + 1
            unit_id = (
                f"{channel.channel_id}_{geometry.orientation}_r{geometry_index:03d}"
            )
            unit_directory = run_root / "tasks" / unit_id
            state_tasks: dict[str, dict[str, Any]] = {}
            state_definitions = {
                "entrance": {
                    "projectile_fragment_charge": channel.incident_state.charge,
                    "target_fragment_charge": channel.target_state.initial_charge,
                    "electrons_on_projectile": (
                        channel.incident_state.electrons_on_projectile
                    ),
                },
                "capture_product": {
                    "projectile_fragment_charge": channel.product_state.charge,
                    "target_fragment_charge": channel.target_state.product_charge,
                    "electrons_on_projectile": (
                        channel.product_state.electrons_on_projectile
                    ),
                },
            }
            for role in STATE_ROLES:
                task_id = f"{unit_id}_{role}"
                directory = unit_directory / role
                definition = state_definitions[role]
                state_tasks[role] = {
                    "task_id": task_id,
                    "role": role,
                    **definition,
                    "total_charge": channel.total_charge,
                    "total_multiplicity": channel.total_multiplicity,
                    "coordinates_angstrom": [
                        list(row) for row in geometry.coordinates_angstrom
                    ],
                    "task_directory": str(directory.relative_to(run_root)),
                    "execution": VALIDATED_BRANCH_EXECUTION,
                    "validated_branch_runner": VALIDATED_BRANCH_RUNNER,
                    "integration_status": BRANCH_HANDOFF_PENDING,
                    "result_path": str(
                        (directory / "task_result.json").relative_to(run_root)
                    ),
                }

            mixed_directory = unit_directory / "mixed"
            mixed_task_id = f"{unit_id}_mixed"
            work_units.append(
                {
                    "work_unit_index": len(work_units),
                    "work_unit_id": unit_id,
                    "channel_id": channel.channel_id,
                    "reaction": channel.reaction,
                    "incident_charge": channel.incident_state.charge,
                    "product_charge": channel.product_state.charge,
                    "total_charge": channel.total_charge,
                    "total_multiplicity": channel.total_multiplicity,
                    "entrance_electrons_on_projectile": (
                        channel.incident_state.electrons_on_projectile
                    ),
                    "product_electrons_on_projectile": (
                        channel.product_state.electrons_on_projectile
                    ),
                    "orientation": geometry.orientation,
                    "geometry_index": geometry_index,
                    "anchor_element": geometry.anchor_element,
                    "separation_angstrom": geometry.separation_angstrom,
                    "minimum_pair_distance_angstrom": (
                        geometry.minimum_pair_distance_angstrom
                    ),
                    "coordinates_angstrom": [
                        list(row) for row in geometry.coordinates_angstrom
                    ],
                    "state_tasks": state_tasks,
                    "mixed_task": {
                        "task_id": mixed_task_id,
                        "task_directory": str(
                            mixed_directory.relative_to(run_root)
                        ),
                        "input_path": str(
                            (mixed_directory / "input.inp").relative_to(run_root)
                        ),
                        "result_path": str(
                            (mixed_directory / "task_result.json").relative_to(
                                run_root
                            )
                        ),
                        "integration_status": BRANCH_HANDOFF_PENDING,
                        "prepared_after_validated_state_handoff": True,
                    },
                }
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration_signature": signature,
        "configuration": configuration,
        "created_utc": _utc_now(),
        "channel_count": len(channels),
        "geometry_count": len(selected_geometries),
        "work_unit_count": len(work_units),
        "required_validated_branch_state_count": 2 * len(work_units),
        "integration_status": BRANCH_HANDOFF_PENDING,
        "work_units": work_units,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def load_workflow_manifest(
    manifest_path: Path, *, verify_inputs: bool = False
) -> dict[str, Any]:
    """Load a coupling plan and reject altered or executable legacy states."""

    path = manifest_path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported low-energy charge-exchange schema.")
    expected = _sha256_bytes(_canonical_json(payload["configuration"]))
    if payload.get("configuration_signature") != expected:
        raise RuntimeError("Charge-exchange workflow configuration checksum mismatch.")
    units = payload.get("work_units", [])
    if payload.get("work_unit_count") != len(units):
        raise RuntimeError("Charge-exchange work-unit count is inconsistent.")
    if payload.get("required_validated_branch_state_count") != 2 * len(units):
        raise RuntimeError("Required CDFT branch-state count is inconsistent.")
    if payload.get("integration_status") != BRANCH_HANDOFF_PENDING:
        raise RuntimeError("Unsupported charge-exchange branch-handoff status.")
    ids = {unit.get("work_unit_id") for unit in units}
    if len(ids) != len(units):
        raise RuntimeError("Charge-exchange work-unit IDs are not unique.")
    for index, unit in enumerate(units):
        if unit.get("work_unit_index") != index:
            raise RuntimeError("Work-unit indices are not stable and contiguous.")
        if set(unit.get("state_tasks", {})) != set(STATE_ROLES):
            raise RuntimeError(f"{unit.get('work_unit_id')} lacks a diabatic state.")
        for task in unit["state_tasks"].values():
            if task.get("execution") != VALIDATED_BRANCH_EXECUTION:
                raise RuntimeError(
                    f"{task.get('task_id')} is not a validated-branch dependency."
                )
            if task.get("validated_branch_runner") != VALIDATED_BRANCH_RUNNER:
                raise RuntimeError(
                    f"{task.get('task_id')} names an unsupported branch runner."
                )
            if task.get("integration_status") != BRANCH_HANDOFF_PENDING:
                raise RuntimeError(
                    f"{task.get('task_id')} has an unsupported handoff status."
                )
            if any(
                key in task
                for key in ("input_path", "input_sha256", "expected_wavefunction")
            ):
                raise RuntimeError(
                    f"{task.get('task_id')} contains a forbidden legacy CDFT input."
                )
        if unit.get("mixed_task", {}).get("integration_status") != (
            BRANCH_HANDOFF_PENDING
        ):
            raise RuntimeError(
                f"{unit.get('work_unit_id')} has an unsupported mixed-state status."
            )
    payload["_manifest_path"] = str(path)
    return payload


def run_workflow_tasks(
    manifest_path: Path,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
    cp2k_command: str | None = None,
    progress: bool = True,
) -> tuple[int, int]:
    """Refuse execution until accepted CDFT branch-state handoff is implemented."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    load_workflow_manifest(manifest_path, verify_inputs=True)
    raise RuntimeError(_BRANCH_HANDOFF_REQUIRED)


def collect_workflow(
    manifest_path: Path, output_directory: Path | None = None
) -> tuple[Path, Path]:
    """Refuse collection until accepted states can produce mixed results."""

    load_workflow_manifest(manifest_path, verify_inputs=True)
    raise RuntimeError(_BRANCH_HANDOFF_REQUIRED)
