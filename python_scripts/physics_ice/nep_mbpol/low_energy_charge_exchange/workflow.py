"""Restart-safe mixed-CDFT coupling workflow for low-energy ion capture."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Any, Iterable

from tqdm import tqdm

from ion_ice import ProjectileDefinition
from soft_dft.config import DEFAULT_CP2K_SETTINGS, CP2KSettings
from soft_dft.cp2k import COMPLEX_ROLE, parse_cp2k_output, render_cp2k_input
from soft_dft.geometry import ScanGeometry, build_scan_geometries

from .channels import SingleElectronCaptureChannel, build_single_capture_channels
from .cp2k import parse_mixed_cdft_output, render_mixed_cdft_input


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 1
STATE_ROLES = ("entrance", "capture_product")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _project_name(value: object) -> str:
    project = re.sub(r"[^A-Za-z0-9_-]", "_", str(value))[:100]
    if not project:
        raise ValueError("Task ID produced an empty CP2K project name.")
    return project


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
            "electronic_states": "two charge-localized all-electron CDFT states",
            "electronic_coupling": "CP2K MIXED_CDFT with Lowdin orthogonalization",
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
    """Prepare both diabatic states for every charge transition and geometry."""

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
                render_task = {
                    "task_id": task_id,
                    "role": COMPLEX_ROLE,
                    "charge": channel.total_charge,
                    "multiplicity": channel.total_multiplicity,
                    "electrons_on_projectile": definition[
                        "electrons_on_projectile"
                    ],
                    "coordinates_angstrom": [
                        list(row) for row in geometry.coordinates_angstrom
                    ],
                }
                input_text = render_cp2k_input(render_task, settings)
                input_path = directory / "input.inp"
                _atomic_text(input_path, input_text)
                state_tasks[role] = {
                    "task_id": task_id,
                    "role": role,
                    **definition,
                    "total_charge": channel.total_charge,
                    "total_multiplicity": channel.total_multiplicity,
                    "task_directory": str(directory.relative_to(run_root)),
                    "input_path": str(input_path.relative_to(run_root)),
                    "input_sha256": _sha256_bytes(input_text.encode("utf-8")),
                    "result_path": str(
                        (directory / "task_result.json").relative_to(run_root)
                    ),
                    "expected_wavefunction": (
                        f"{_project_name(task_id)}-RESTART.wfn"
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
                        "prepared_after_state_convergence": True,
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
        "prepared_cdft_state_count": 2 * len(work_units),
        "work_units": work_units,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def load_workflow_manifest(
    manifest_path: Path, *, verify_inputs: bool = False
) -> dict[str, Any]:
    """Load a coupling workflow and reject altered configuration or inputs."""

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
    ids = {unit.get("work_unit_id") for unit in units}
    if len(ids) != len(units):
        raise RuntimeError("Charge-exchange work-unit IDs are not unique.")
    for index, unit in enumerate(units):
        if unit.get("work_unit_index") != index:
            raise RuntimeError("Work-unit indices are not stable and contiguous.")
        if set(unit.get("state_tasks", {})) != set(STATE_ROLES):
            raise RuntimeError(f"{unit.get('work_unit_id')} lacks a diabatic state.")
        if verify_inputs:
            for task in unit["state_tasks"].values():
                input_path = path.parent / task["input_path"]
                if not input_path.is_file() or _sha256_file(input_path) != task[
                    "input_sha256"
                ]:
                    raise RuntimeError(
                        f"CDFT state input checksum mismatch: {input_path}"
                    )
    payload["_manifest_path"] = str(path)
    return payload


def _valid_result(
    path: Path,
    *,
    task_id: str,
    signature: str,
    input_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        result.get("schema_version") == SCHEMA_VERSION
        and result.get("status") == "complete"
        and result.get("task_id") == task_id
        and result.get("configuration_signature") == signature
        and result.get("input_sha256") == input_sha256
        and result.get("valid_completion")
    )


def _execute_cp2k(
    *,
    root: Path,
    task_id: str,
    task_directory: Path,
    input_path: Path,
    result_path: Path,
    input_sha256: str,
    signature: str,
    command: list[str],
    mixed: bool,
) -> bool:
    if _valid_result(
        result_path,
        task_id=task_id,
        signature=signature,
        input_sha256=input_sha256,
    ):
        return False
    task_directory.mkdir(parents=True, exist_ok=True)
    temporary_output = task_directory / f".cp2k.{os.getpid()}.out.tmp"
    started = _utc_now()
    with temporary_output.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            [*command, "-i", input_path.name],
            cwd=task_directory,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    text = temporary_output.read_text(encoding="utf-8", errors="replace")
    parsed = (
        parse_mixed_cdft_output(text)
        if mixed
        else parse_cp2k_output(text, require_cdft=True)
    )
    if not mixed:
        strength = parsed.get("cdft_strength")
        parsed["valid_completion"] = bool(
            parsed["valid_completion"]
            and strength is not None
            and math.isfinite(float(strength))
        )
    if completed.returncode != 0 or not parsed["valid_completion"]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failed = task_directory / f"cp2k.failed.{stamp}.out"
        os.replace(temporary_output, failed)
        _atomic_json(
            task_directory / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "task_id": task_id,
                "configuration_signature": signature,
                "input_sha256": input_sha256,
                "return_code": completed.returncode,
                "parsed": parsed,
                "output_path": str(failed.relative_to(root)),
                "started_utc": started,
                "failed_utc": _utc_now(),
            },
        )
        raise RuntimeError(f"CP2K task {task_id} failed; see {failed}")
    output_path = task_directory / "cp2k.out"
    os.replace(temporary_output, output_path)
    _atomic_json(
        result_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "task_id": task_id,
            "configuration_signature": signature,
            "input_sha256": input_sha256,
            "command": command,
            "return_code": completed.returncode,
            "started_utc": started,
            "completed_utc": _utc_now(),
            "output_path": str(output_path.relative_to(root)),
            "output_sha256": _sha256_file(output_path),
            **parsed,
        },
    )
    (task_directory / "failure.json").unlink(missing_ok=True)
    return True


def _restart_wavefunction(root: Path, task: dict[str, Any]) -> Path:
    directory = root / task["task_directory"]
    expected = directory / task["expected_wavefunction"]
    if expected.is_file():
        return expected
    candidates = sorted(directory.glob("*-RESTART.wfn"))
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"Expected one converged restart wavefunction for {task['task_id']}; "
        f"found {len(candidates)} in {directory}."
    )


def _record_or_verify_restart(
    root: Path,
    task: dict[str, Any],
    restart: Path,
) -> dict[str, Any]:
    """Checksum-link a state result to the wavefunction used by mixed CDFT."""

    result_path = root / task["result_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    relative = str(restart.relative_to(root))
    checksum = _sha256_file(restart)
    recorded_path = result.get("wavefunction_path")
    recorded_checksum = result.get("wavefunction_sha256")
    if recorded_path is not None or recorded_checksum is not None:
        if recorded_path != relative or recorded_checksum != checksum:
            raise RuntimeError(
                f"Restart wavefunction changed after {task['task_id']} completed."
            )
        return result
    result["wavefunction_path"] = relative
    result["wavefunction_sha256"] = checksum
    _atomic_json(result_path, result)
    return result


def run_workflow_tasks(
    manifest_path: Path,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
    cp2k_command: str | None = None,
    progress: bool = True,
) -> tuple[int, int]:
    """Run complete channel/geometry units so mixed-state dependencies stay local."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    root = Path(manifest["_manifest_path"]).parent
    signature = manifest["configuration_signature"]
    settings = CP2KSettings(**manifest["configuration"]["cp2k"])
    command = shlex.split(
        cp2k_command or os.environ.get("CP2K_COMMAND", "cp2k.psmp")
    )
    if not command:
        raise ValueError("CP2K command is empty.")
    selected = [
        unit
        for unit in manifest["work_units"]
        if int(unit["work_unit_index"]) % shard_count == shard_index
    ]
    completed_count = 0
    skipped_count = 0
    for unit in tqdm(
        selected,
        desc=f"Low-energy capture shard {shard_index}",
        unit="channel-geometry",
        disable=not progress,
    ):
        state_results: dict[str, dict[str, Any]] = {}
        restarts: dict[str, Path] = {}
        for role in STATE_ROLES:
            task = unit["state_tasks"][role]
            changed = _execute_cp2k(
                root=root,
                task_id=task["task_id"],
                task_directory=root / task["task_directory"],
                input_path=root / task["input_path"],
                result_path=root / task["result_path"],
                input_sha256=task["input_sha256"],
                signature=signature,
                command=command,
                mixed=False,
            )
            completed_count += int(changed)
            skipped_count += int(not changed)
            restarts[role] = _restart_wavefunction(root, task)
            state_results[role] = _record_or_verify_restart(
                root, task, restarts[role]
            )

        mixed = unit["mixed_task"]
        mixed_directory = root / mixed["task_directory"]
        entrance_restart = os.path.relpath(
            restarts["entrance"], start=mixed_directory
        ).replace(os.sep, "/")
        product_restart = os.path.relpath(
            restarts["capture_product"], start=mixed_directory
        ).replace(os.sep, "/")
        mixed_text = render_mixed_cdft_input(
            {**unit, "task_id": mixed["task_id"]},
            settings,
            entrance_wavefunction=entrance_restart,
            product_wavefunction=product_restart,
            entrance_strength=float(state_results["entrance"]["cdft_strength"]),
            product_strength=float(
                state_results["capture_product"]["cdft_strength"]
            ),
        )
        mixed_input = root / mixed["input_path"]
        _atomic_text(mixed_input, mixed_text)
        mixed_sha256 = _sha256_bytes(mixed_text.encode("utf-8"))
        changed = _execute_cp2k(
            root=root,
            task_id=mixed["task_id"],
            task_directory=mixed_directory,
            input_path=mixed_input,
            result_path=root / mixed["result_path"],
            input_sha256=mixed_sha256,
            signature=signature,
            command=command,
            mixed=True,
        )
        completed_count += int(changed)
        skipped_count += int(not changed)
    return completed_count, skipped_count


def collect_workflow(
    manifest_path: Path, output_directory: Path | None = None
) -> tuple[Path, Path]:
    """Collect diabatic gaps and couplings without claiming cross sections."""

    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    root = Path(manifest["_manifest_path"]).parent
    destination = (output_directory or root / "collected").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    symbol = manifest["configuration"]["projectile"]["symbol"]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for unit in manifest["work_units"]:
        mixed = unit["mixed_task"]
        path = root / mixed["result_path"]
        input_path = root / mixed["input_path"]
        if not input_path.is_file():
            missing.append(mixed["task_id"])
            continue
        input_sha256 = _sha256_file(input_path)
        if not _valid_result(
            path,
            task_id=mixed["task_id"],
            signature=manifest["configuration_signature"],
            input_sha256=input_sha256,
        ):
            missing.append(mixed["task_id"])
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "work_unit_id": unit["work_unit_id"],
                "projectile": symbol,
                "incident_charge": unit["incident_charge"],
                "product_charge": unit["product_charge"],
                "total_charge": unit["total_charge"],
                "total_multiplicity": unit["total_multiplicity"],
                "orientation": unit["orientation"],
                "anchor_element": unit["anchor_element"],
                "separation_angstrom": unit["separation_angstrom"],
                "minimum_pair_distance_angstrom": unit[
                    "minimum_pair_distance_angstrom"
                ],
                "charge_transfer_energy_ev": result[
                    "charge_transfer_energy_ev"
                ],
                "state_overlap": result["state_overlap"],
                "coupling_lowdin_ev": result["coupling_lowdin_ev"],
                "coupling_rotation_mhartree": result.get(
                    "coupling_rotation_mhartree"
                ),
            }
        )
    table_path = destination / f"{symbol.lower()}_mixed_cdft_capture_couplings.csv"
    fieldnames = [
        "work_unit_id",
        "projectile",
        "incident_charge",
        "product_charge",
        "total_charge",
        "total_multiplicity",
        "orientation",
        "anchor_element",
        "separation_angstrom",
        "minimum_pair_distance_angstrom",
        "charge_transfer_energy_ev",
        "state_overlap",
        "coupling_lowdin_ev",
        "coupling_rotation_mhartree",
    ]
    fd, temporary = tempfile.mkstemp(prefix=f".{table_path.name}.", dir=destination)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, table_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    collection = {
        "schema_version": SCHEMA_VERSION,
        "source_workflow": manifest["_manifest_path"],
        "source_configuration_signature": manifest["configuration_signature"],
        "created_utc": _utc_now(),
        "row_count": len(rows),
        "expected_work_unit_count": manifest["work_unit_count"],
        "missing_count": len(missing),
        "missing_tasks": missing,
        "numerical_status": "complete" if not missing else "incomplete",
        "physics_status": "validation_pending",
        "cross_section_status": "not_computed",
        "table_sha256": _sha256_file(table_path),
        "next_required_stage": (
            "locate and converge diabatic crossings, obtain trajectory radial "
            "speeds, apply Landau-Zener probabilities, integrate over impact "
            "parameter and orientation, then compare with CTMC SC"
        ),
    }
    collection_path = destination / f"{symbol.lower()}_mixed_cdft_capture.manifest.json"
    _atomic_json(collection_path, collection)
    return table_path, collection_path
