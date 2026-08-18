"""Restart-safe preparation, execution, and collection of ion--H2O CDFT scans."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
from typing import Any, Iterable

from tqdm import tqdm

from nlh import potential_ev

from .config import (
    CP2K_PROVENANCE,
    DEFAULT_CP2K_SETTINGS,
    DEFAULT_PROJECTILE,
    HARTREE_TO_EV,
    CP2KSettings,
    IonChargeState,
    ProjectileDefinition,
)
from .cp2k import (
    COMPLEX_ROLE,
    PROJECTILE_COUNTERPOISE_ROLE,
    ROLES,
    WATER_COUNTERPOISE_ROLE,
    parse_cp2k_output,
    render_cp2k_input,
)
from .geometry import (
    ORIENTATIONS,
    ScanGeometry,
    WATER_GEOMETRY_PROVENANCE,
    build_scan_geometries,
)


SCHEMA_VERSION = 3
IMPLEMENTATION_VERSION = 14
VALIDATED_BRANCH_EXECUTION = "validated_cdft_branch"
VALIDATED_BRANCH_RUNNER = "soft_dft.branch_execution.CP2KBranchExecutor"
_VALIDATED_BRANCH_REQUIRED = (
    "Legacy soft-DFT mesh execution cannot run or accept complex CDFT tasks. "
    f"Use {VALIDATED_BRANCH_RUNNER} to obtain a reciprocal validated branch "
    "state; handoff of those states into the full mesh is not yet implemented."
)


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


def _configuration(
    projectile: ProjectileDefinition,
    settings: CP2KSettings,
    charge_states: tuple[IonChargeState, ...],
    geometries: tuple[ScanGeometry, ...],
    workflow_context: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "projectile": projectile.soft_dft_dict(),
        "charge_states": [state.as_dict() for state in charge_states],
        "cp2k": settings.as_dict(),
        "molecule": "H2O",
        "water_geometry": WATER_GEOMETRY_PROVENANCE,
        "orientations": [
            {
                "name": item.name,
                "anchor_index": item.anchor_index,
                "anchor_element": item.anchor_element,
                "direction": list(item.direction),
                "definition": item.definition,
            }
            for item in ORIENTATIONS
        ],
        "scan_geometries": [geometry.as_dict() for geometry in geometries],
        "counterpoise_definition": (
            "Boys-Bernardi: E_int = E_projectile+H2O^(full basis) - "
            "E_projectile^(full basis) - E_H2O^(full basis), at fixed nuclei"
        ),
        "charge_state_interface": (
            "Each DFT table is conditional on fixed q. CTMC is not used in "
            "DFT; Geant4 selects V_q and changes the selected table only "
            "after a sampled CTMC q-to-q' event."
        ),
        "provenance": {
            "projectile_states_and_scan": projectile.soft_dft_dict(),
            "cp2k": CP2K_PROVENANCE,
        },
        "scientific_scope": "fixed-nuclei isolated-molecule ion--H2O CDFT pilot",
        "validation_status": "validation_pending",
        "workflow_context": workflow_context or {"kind": "fixed_base_scan"},
    }


def _point_id(
    symbol: str, charge: int, orientation: str, geometry_index: int
) -> str:
    return f"{symbol}_q{charge}_{orientation}_r{geometry_index:03d}"


def _task_id(point_id: str, role: str) -> str:
    suffix = {
        COMPLEX_ROLE: "complex",
        PROJECTILE_COUNTERPOISE_ROLE: "projectile_cp",
        WATER_COUNTERPOISE_ROLE: "water_cp",
    }[role]
    return f"{point_id}_{suffix}"


def build_workflow(
    output_root: Path,
    *,
    projectile: ProjectileDefinition = DEFAULT_PROJECTILE,
    settings: CP2KSettings | None = None,
    charges: Iterable[int] | None = None,
    geometries: Iterable[ScanGeometry] | None = None,
    workflow_context: dict[str, Any] | None = None,
) -> Path:
    """Create deterministic CP2K tasks and return the workflow manifest."""

    projectile.validate()
    if settings is None:
        settings = replace(
            DEFAULT_CP2K_SETTINGS,
            projectile_basis_set=projectile.projectile_basis_set,
            basis_file=projectile.basis_file,
        )
    requested_charges = (
        range(projectile.atomic_number + 1) if charges is None else charges
    )
    selected = tuple(projectile.state(int(charge)) for charge in requested_charges)
    if not selected or len({state.charge for state in selected}) != len(selected):
        raise ValueError("Select at least one unique projectile charge state.")
    selected_geometries = tuple(
        build_scan_geometries(projectile) if geometries is None else geometries
    )
    if not selected_geometries:
        raise ValueError("Select at least one ion--H2O scan geometry.")
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
        raise ValueError(
            "Scan geometries contain duplicate orientation/separation points."
        )
    configuration = _configuration(
        projectile, settings, selected, selected_geometries, workflow_context
    )
    signature = _sha256_bytes(_canonical_json(configuration))
    run_root = output_root.resolve() / signature
    manifest_path = run_root / "workflow.manifest.json"

    if manifest_path.exists():
        existing = load_workflow_manifest(manifest_path, verify_inputs=True)
        if existing["configuration_signature"] != signature:
            raise RuntimeError("Existing workflow has an incompatible signature.")
        return manifest_path

    geometry_records: list[tuple[ScanGeometry, int, str]] = []
    per_orientation_index: dict[str, int] = {}
    for geometry in selected_geometries:
        geometry_index = per_orientation_index.get(geometry.orientation, 0)
        per_orientation_index[geometry.orientation] = geometry_index + 1
        geometry_id = (
            f"{projectile.symbol}_{geometry.orientation}_r{geometry_index:03d}"
        )
        geometry_records.append((geometry, geometry_index, geometry_id))

    tasks: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    water_task_by_geometry: dict[str, str] = {}
    total = len(geometry_records) + 2 * len(selected) * len(geometry_records)

    def append_task(
        *,
        task_id: str,
        point_id: str | None,
        role: str,
        geometry: ScanGeometry,
        geometry_index: int,
        geometry_id: str,
        state: IonChargeState | None,
    ) -> dict[str, Any]:
        task_directory = run_root / "tasks" / task_id
        task: dict[str, Any] = {
            "task_index": len(tasks),
            "task_id": task_id,
            "point_id": point_id,
            "geometry_id": geometry_id,
            "role": role,
            "projectile": projectile.symbol,
            "projectile_atomic_number": projectile.atomic_number,
            "charge": state.charge if state is not None else None,
            "electrons_on_projectile": (
                state.electrons_on_projectile if state is not None else None
            ),
            "multiplicity": state.multiplicity if state is not None else 1,
            "scf_spin_mode": (
                state.scf_spin_mode if state is not None else "RESTRICTED"
            ),
            "configuration": (
                state.configuration if state is not None else "neutral H2O"
            ),
            "term": state.term if state is not None else "singlet",
            "cp2k_atomic_guess": (
                state.atomic_guess_dict() if state is not None else None
            ),
            "orientation": geometry.orientation,
            "geometry_index": geometry_index,
            "anchor_index": geometry.anchor_index,
            "anchor_element": geometry.anchor_element,
            "separation_angstrom": geometry.separation_angstrom,
            "minimum_pair_distance_angstrom": (
                geometry.minimum_pair_distance_angstrom
            ),
            "coordinates_angstrom": [
                list(row) for row in geometry.coordinates_angstrom
            ],
            "task_directory": str(task_directory.relative_to(run_root)),
        }
        if role == COMPLEX_ROLE:
            # The legacy mesh runner previously rendered STRENGTH 0 and could
            # restart from an unpaired WFN.  Neither defines a diabatic CDFT
            # branch.  Keep the task in the manifest, but do not create an
            # executable input until the validated branch handoff exists.
            task.update(
                {
                    "execution": VALIDATED_BRANCH_EXECUTION,
                    "validated_branch_runner": VALIDATED_BRANCH_RUNNER,
                    "integration_status": "mesh_handoff_pending",
                }
            )
            task["result_path"] = str(
                (task_directory / "task_result.json").relative_to(run_root)
            )
        elif (
            role == PROJECTILE_COUNTERPOISE_ROLE
            and state is not None
            and state.charge == projectile.atomic_number
        ):
            # A bare isolated nucleus has no electrons and no other real
            # nucleus, hence its all-electron electronic energy is exactly
            # zero. Ghost basis functions do not alter it.
            task.update(
                {
                    "execution": "analytic",
                    "analytic_energy_hartree": 0.0,
                    "definition": "isolated bare nucleus",
                }
            )
            result_path = task_directory / "task_result.json"
            _atomic_json(
                result_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                    "execution": "analytic",
                    "task_id": task_id,
                    "configuration_signature": signature,
                    "energy_hartree": 0.0,
                    "valid_completion": True,
                    "definition": "isolated bare nucleus",
                    "completed_utc": _utc_now(),
                },
            )
            task["result_path"] = str(result_path.relative_to(run_root))
        else:
            task["execution"] = "cp2k"
            input_text = render_cp2k_input(task, settings)
            input_path = task_directory / "input.inp"
            _atomic_text(input_path, input_text)
            task["input_path"] = str(input_path.relative_to(run_root))
            task["input_sha256"] = _sha256_bytes(input_text.encode("utf-8"))
            task["result_path"] = str(
                (task_directory / "task_result.json").relative_to(run_root)
            )
        tasks.append(task)
        return task

    with tqdm(
        total=total,
        desc=f"Preparing {projectile.symbol} CDFT tasks",
        unit="task",
    ) as progress:
        # E_H2O in the full geometry-dependent ghost basis is independent of
        # projectile charge. Store it once per geometry and reference it from all q.
        for geometry, geometry_index, geometry_id in geometry_records:
            task_id = f"{geometry_id}_water_cp"
            append_task(
                task_id=task_id,
                point_id=None,
                role=WATER_COUNTERPOISE_ROLE,
                geometry=geometry,
                geometry_index=geometry_index,
                geometry_id=geometry_id,
                state=None,
            )
            water_task_by_geometry[geometry_id] = task_id
            progress.update()

        for state in selected:
            for geometry, geometry_index, geometry_id in geometry_records:
                point_id = _point_id(
                    projectile.symbol,
                    state.charge,
                    geometry.orientation,
                    geometry_index,
                )
                component_task_ids = {
                    WATER_COUNTERPOISE_ROLE: water_task_by_geometry[geometry_id]
                }
                for role in (COMPLEX_ROLE, PROJECTILE_COUNTERPOISE_ROLE):
                    task_id = _task_id(point_id, role)
                    append_task(
                        task_id=task_id,
                        point_id=point_id,
                        role=role,
                        geometry=geometry,
                        geometry_index=geometry_index,
                        geometry_id=geometry_id,
                        state=state,
                    )
                    component_task_ids[role] = task_id
                    progress.update()
                points.append(
                    {
                        "point_index": len(points),
                        "point_id": point_id,
                        "charge": state.charge,
                        "geometry_id": geometry_id,
                        "component_task_ids": component_task_ids,
                    }
                )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration_signature": signature,
        "configuration": configuration,
        "created_utc": _utc_now(),
        "point_count": len(points),
        "task_count": len(tasks),
        "cp2k_task_count": sum(task["execution"] == "cp2k" for task in tasks),
        "validated_branch_task_count": sum(
            task["execution"] == VALIDATED_BRANCH_EXECUTION for task in tasks
        ),
        "analytic_task_count": sum(
            task["execution"] == "analytic" for task in tasks
        ),
        "points": points,
        "tasks": tasks,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def load_workflow_manifest(
    manifest_path: Path,
    *,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    """Load a workflow and reject altered configuration or task inputs."""

    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported CDFT workflow schema.")
    expected = _sha256_bytes(_canonical_json(payload["configuration"]))
    if payload.get("configuration_signature") != expected:
        raise RuntimeError("CDFT workflow configuration checksum mismatch.")
    if payload.get("task_count") != len(payload.get("tasks", [])):
        raise RuntimeError("CDFT workflow task count is inconsistent.")
    if payload.get("point_count") != len(payload.get("points", [])):
        raise RuntimeError("CDFT workflow point count is inconsistent.")
    task_ids = {task.get("task_id") for task in payload["tasks"]}
    if len(task_ids) != len(payload["tasks"]):
        raise RuntimeError("CDFT workflow task IDs are not unique.")
    for index, task in enumerate(payload["tasks"]):
        if task.get("task_index") != index:
            raise RuntimeError("CDFT task indices are not stable and contiguous.")
        if verify_inputs and task.get("execution") == "cp2k":
            path = manifest_path.parent / task["input_path"]
            if not path.is_file() or _sha256_file(path) != task["input_sha256"]:
                raise RuntimeError(f"CDFT task input checksum mismatch: {path}")
    for index, point in enumerate(payload["points"]):
        if point.get("point_index") != index:
            raise RuntimeError("CDFT point indices are not stable and contiguous.")
        component_ids = point.get("component_task_ids", {})
        if set(component_ids) != set(ROLES):
            raise RuntimeError(f"CDFT point {point.get('point_id')} lacks a CP role.")
        if not set(component_ids.values()).issubset(task_ids):
            raise RuntimeError(
                f"CDFT point {point.get('point_id')} references an unknown task."
            )
    payload["_manifest_path"] = str(manifest_path)
    return payload


def _valid_existing_result(
    result_path: Path,
    task: dict[str, Any],
    configuration_signature: str,
) -> bool:
    # Full-mesh consumption of reciprocal branch records is deliberately not
    # implemented yet.  In particular, an old generic `valid_completion`
    # flag must never promote a complex CDFT result.
    if task.get("role") == COMPLEX_ROLE:
        return False
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if task["execution"] == "analytic":
        return bool(
            result.get("schema_version") == SCHEMA_VERSION
            and result.get("status") == "complete"
            and result.get("configuration_signature") == configuration_signature
            and result.get("task_id") == task["task_id"]
            and result.get("valid_completion")
        )
    return bool(
        result.get("schema_version") == SCHEMA_VERSION
        and result.get("status") == "complete"
        and result.get("configuration_signature") == configuration_signature
        and result.get("task_id") == task["task_id"]
        and result.get("input_sha256") == task["input_sha256"]
        and result.get("valid_completion")
    )


def _recover_valid_failed_result(
    result_path: Path,
    task_directory: Path,
    task: dict[str, Any],
    configuration_signature: str,
) -> bool:
    """Promote a completed CP2K output rejected by an older parser."""

    if task.get("role") == COMPLEX_ROLE:
        return False

    failure_path = task_directory / "failure.json"
    if not failure_path.is_file():
        return False
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    output_name = failure.get("output_path")
    if not isinstance(output_name, str) or Path(output_name).name != output_name:
        return False
    failed_output = task_directory / output_name
    if not (
        failure.get("schema_version") == SCHEMA_VERSION
        and failure.get("status") == "failed"
        and failure.get("task_id") == task["task_id"]
        and failure.get("configuration_signature") == configuration_signature
        and failure.get("input_sha256") == task["input_sha256"]
        and failure.get("return_code") == 0
        and failed_output.is_file()
    ):
        return False

    text = failed_output.read_text(encoding="utf-8", errors="replace")
    parsed = parse_cp2k_output(text, require_cdft=task["role"] == COMPLEX_ROLE)
    if not parsed["valid_completion"]:
        return False

    output_path = task_directory / "cp2k.out"
    if output_path.exists():
        return False
    os.replace(failed_output, output_path)
    completed_utc = failure.get("failed_utc") or _utc_now()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "execution": "cp2k",
        "task_id": task["task_id"],
        "configuration_signature": configuration_signature,
        "input_sha256": task["input_sha256"],
        "command": failure.get("command", []),
        "cp2k_data_dir": None,
        "return_code": 0,
        "started_utc": failure.get("started_utc"),
        "completed_utc": completed_utc,
        "recovered_utc": _utc_now(),
        "recovered_from_parser_failure": True,
        "output_path": output_path.name,
        "output_sha256": _sha256_file(output_path),
        **parsed,
    }
    _atomic_json(result_path, result)
    failure_path.unlink()
    return True


def _restart_execution_input(
    input_path: Path,
    task_directory: Path,
    task: dict[str, Any],
    configuration_signature: str,
) -> tuple[Path, dict[str, Any]]:
    """Reuse a failed task's latest CP2K wavefunction without changing physics.

    The immutable manifest input remains the source of truth.  A derived input
    changes only CP2K's SCF initial guess and records both its checksum and the
    checksum of the wavefunction that supplied that guess.
    """

    if task.get("role") == COMPLEX_ROLE:
        raise RuntimeError(_VALIDATED_BRANCH_REQUIRED)

    failure_path = task_directory / "failure.json"
    if not failure_path.is_file():
        return input_path, {}
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return input_path, {}
    if not (
        failure.get("schema_version") == SCHEMA_VERSION
        and failure.get("status") == "failed"
        and failure.get("task_id") == task["task_id"]
        and failure.get("configuration_signature") == configuration_signature
        and failure.get("input_sha256") == task["input_sha256"]
    ):
        return input_path, {}

    prior_execution_name = failure.get("execution_input_path")
    prior_execution_sha256 = failure.get("execution_input_sha256")
    if (
        isinstance(prior_execution_name, str)
        and Path(prior_execution_name).name == prior_execution_name
        and isinstance(prior_execution_sha256, str)
    ):
        prior_execution = task_directory / prior_execution_name
        if (
            prior_execution.is_file()
            and _sha256_file(prior_execution) == prior_execution_sha256
        ):
            metadata = {
                key: value
                for key, value in failure.items()
                if key.startswith("wavefunction_restart_")
                or key.startswith("execution_input_")
            }
            return prior_execution, metadata

    wavefunction = task_directory / f"{task['task_id']}-RESTART.wfn"
    if not wavefunction.is_file() or wavefunction.stat().st_size == 0:
        return input_path, {}

    input_text = input_path.read_text(encoding="utf-8")
    atomic_guess = "SCF_GUESS ATOMIC"
    spin_lines = [line for line in ("UKS TRUE", "UKS FALSE") if line in input_text]
    if input_text.count(atomic_guess) != 1 or len(spin_lines) != 1:
        raise RuntimeError(
            f"Cannot derive an unambiguous restart input for {task['task_id']}."
        )
    uks_line = spin_lines[0]
    restart_text = input_text.replace(
        uks_line,
        f"{uks_line}\n    WFN_RESTART_FILE_NAME {wavefunction.name}",
        1,
    ).replace(atomic_guess, "SCF_GUESS RESTART", 1)
    restart_path = task_directory / "input.restart.inp"
    _atomic_text(restart_path, restart_text)
    return restart_path, {
        "restarted_from_wavefunction": wavefunction.name,
        "wavefunction_restart_sha256": _sha256_file(wavefunction),
        "execution_input_path": restart_path.name,
        "execution_input_sha256": _sha256_bytes(restart_text.encode("utf-8")),
    }


def run_workflow_tasks(
    manifest_path: Path,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
    cp2k_command: str | None = None,
    progress: bool = True,
) -> tuple[int, int]:
    """Run one deterministic task shard; completed tasks are checksum-skipped."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    root = Path(manifest["_manifest_path"]).parent
    signature = manifest["configuration_signature"]
    selected = [
        task
        for task in manifest["tasks"]
        if int(task["task_index"]) % shard_count == shard_index
    ]
    if any(task.get("role") == COMPLEX_ROLE for task in selected):
        raise RuntimeError(_VALIDATED_BRANCH_REQUIRED)
    command_text = cp2k_command or os.environ.get("CP2K_COMMAND", "cp2k.psmp")
    command = shlex.split(command_text)
    if not command:
        raise ValueError("CP2K command is empty.")

    completed = 0
    skipped = 0
    active_process: subprocess.Popen[str] | None = None
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: object) -> None:
        if active_process is not None and active_process.poll() is None:
            os.killpg(active_process.pid, signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, forward_signal)
    try:
        for task in tqdm(
            selected,
            desc=f"CDFT shard {shard_index}",
            unit="task",
            disable=not progress,
        ):
            result_path = root / task["result_path"]
            if _valid_existing_result(result_path, task, signature):
                skipped += 1
                continue
            task_directory = root / task["task_directory"]
            if _recover_valid_failed_result(
                result_path, task_directory, task, signature
            ):
                skipped += 1
                continue
            if task["execution"] == "analytic":
                raise RuntimeError(
                    f"Analytic task result is missing or incompatible: {result_path}"
                )

            input_path = root / task["input_path"]
            execution_input = input_path
            restart_metadata: dict[str, Any] = {}
            execution_input, restart_metadata = _restart_execution_input(
                input_path, task_directory, task, signature
            )
            temporary_output = task_directory / f".cp2k.{os.getpid()}.out.tmp"
            started = _utc_now()
            with temporary_output.open("w", encoding="utf-8") as output:
                active_process = subprocess.Popen(
                    [*command, "-i", execution_input.name],
                    cwd=task_directory,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                return_code = active_process.wait()
            active_process = None
            text = temporary_output.read_text(encoding="utf-8", errors="replace")
            parsed = parse_cp2k_output(text, require_cdft=task["role"] == COMPLEX_ROLE)
            if return_code != 0 or not parsed["valid_completion"]:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                failed_output = task_directory / f"cp2k.failed.{stamp}.out"
                os.replace(temporary_output, failed_output)
                _atomic_json(
                    task_directory / "failure.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "failed",
                        "task_id": task["task_id"],
                        "configuration_signature": signature,
                        "input_sha256": task["input_sha256"],
                        "command": command,
                        "return_code": return_code,
                        "parsed": parsed,
                        "started_utc": started,
                        "failed_utc": _utc_now(),
                        "output_path": failed_output.name,
                        **restart_metadata,
                    },
                )
                raise RuntimeError(
                    f"CP2K task {task['task_id']} failed; see {failed_output}"
                )

            output_path = task_directory / "cp2k.out"
            os.replace(temporary_output, output_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "execution": "cp2k",
                "task_id": task["task_id"],
                "configuration_signature": signature,
                "input_sha256": task["input_sha256"],
                "command": command,
                "cp2k_data_dir": os.environ.get("CP2K_DATA_DIR"),
                "return_code": return_code,
                "started_utc": started,
                "completed_utc": _utc_now(),
                "output_path": output_path.name,
                "output_sha256": _sha256_file(output_path),
                **restart_metadata,
                **parsed,
            }
            _atomic_json(result_path, result)
            (task_directory / "failure.json").unlink(missing_ok=True)
            completed += 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return completed, skipped


def pending_workflow_tasks(manifest_path: Path) -> list[dict[str, Any]]:
    """Return tasks without a checksum-compatible completed result."""

    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    root = Path(manifest["_manifest_path"]).parent
    signature = manifest["configuration_signature"]
    return [
        task
        for task in manifest["tasks"]
        if not _valid_existing_result(root / task["result_path"], task, signature)
    ]


def audit_workflow_electronic_states(manifest_path: Path) -> dict[str, Any]:
    """Audit exact alpha/beta populations for every completed CP2K task.

    Population acceptance uses no fitted tolerance: CP2K reports integer
    orbital occupations, which must equal those implied by the declared total
    electron count and multiplicity.  ``S**2`` is reported as a diagnostic but
    is not thresholded here because an acceptable spin-contamination bound is
    method- and state-dependent.
    """

    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    root = Path(manifest["_manifest_path"]).parent
    signature = manifest["configuration_signature"]
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid_outputs: list[str] = []
    inconsistent: list[str] = []
    for task in tqdm(
        manifest["tasks"], desc="Auditing CDFT electronic states", unit="task"
    ):
        if task["execution"] == "analytic":
            continue
        result_path = root / task["result_path"]
        if not _valid_existing_result(result_path, task, signature):
            missing.append(task["task_id"])
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        output_name = result.get("output_path")
        output = (
            root / task["task_directory"] / output_name
            if isinstance(output_name, str) and Path(output_name).name == output_name
            else None
        )
        if not (
            output is not None
            and output.is_file()
            and result.get("output_sha256") == _sha256_file(output)
        ):
            invalid_outputs.append(task["task_id"])
            continue
        parsed = parse_cp2k_output(
            output.read_text(encoding="utf-8", errors="replace"),
            require_cdft=task["role"] == COMPLEX_ROLE,
        )
        projectile_electrons = task.get("electrons_on_projectile")
        if task["role"] == WATER_COUNTERPOISE_ROLE:
            expected_total = 10
        elif task["role"] == PROJECTILE_COUNTERPOISE_ROLE:
            expected_total = int(projectile_electrons)
        else:
            expected_total = 10 + int(projectile_electrons)
        expected_spin_difference = int(task["multiplicity"]) - 1
        expected_alpha_numerator = expected_total + expected_spin_difference
        expected_beta_numerator = expected_total - expected_spin_difference
        if expected_alpha_numerator % 2 or expected_beta_numerator % 2:
            raise RuntimeError(
                f"Electron count and multiplicity have inconsistent parity: "
                f"{task['task_id']}"
            )
        expected_alpha = expected_alpha_numerator // 2
        expected_beta = expected_beta_numerator // 2
        observed_alpha = parsed["electron_count_alpha"]
        observed_beta = parsed["electron_count_beta"]
        population_consistent = bool(
            observed_alpha == expected_alpha and observed_beta == expected_beta
        )
        if not population_consistent:
            inconsistent.append(task["task_id"])
        ideal_s = 0.5 * expected_spin_difference
        ideal_s2_from_multiplicity = ideal_s * (ideal_s + 1.0)
        records.append(
            {
                "task_id": task["task_id"],
                "role": task["role"],
                "charge": task.get("charge"),
                "multiplicity": task["multiplicity"],
                "expected_electron_count_alpha": expected_alpha,
                "expected_electron_count_beta": expected_beta,
                "observed_electron_count_alpha": observed_alpha,
                "observed_electron_count_beta": observed_beta,
                "population_consistent": population_consistent,
                "spin_squared_from_multiplicity": ideal_s2_from_multiplicity,
                "spin_squared_ideal_cp2k": parsed["spin_squared_ideal"],
                "spin_squared_single_determinant": parsed[
                    "spin_squared_single_determinant"
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration_signature": signature,
        "completed_cp2k_task_count": len(records),
        "missing_task_ids": missing,
        "invalid_output_task_ids": invalid_outputs,
        "population_inconsistent_task_ids": inconsistent,
        "all_completed_populations_consistent": not invalid_outputs
        and not inconsistent,
        "complete_workflow_audited": not missing
        and not invalid_outputs
        and not inconsistent,
        "records": records,
    }


def reuse_compatible_workflow_results(
    source_manifest_path: Path,
    target_manifest_path: Path,
) -> tuple[int, int]:
    """Reuse completed CP2K tasks whose rendered inputs are byte-identical.

    Whole-workflow signatures intentionally change whenever any numerical
    setting changes.  This helper permits exact task-level reuse without
    weakening that contract: task ID, input SHA-256, source output SHA-256,
    and a fresh parse of the copied output must all agree.  Analytic tasks are
    generated by :func:`build_workflow` and are not imported.

    Returns ``(reused, incompatible_or_incomplete)`` for target CP2K tasks
    that do not already have a valid result.
    """

    source = load_workflow_manifest(source_manifest_path, verify_inputs=True)
    target = load_workflow_manifest(target_manifest_path, verify_inputs=True)
    source_root = Path(source["_manifest_path"]).parent
    target_root = Path(target["_manifest_path"]).parent
    source_signature = source["configuration_signature"]
    target_signature = target["configuration_signature"]
    source_tasks = {task["task_id"]: task for task in source["tasks"]}

    reused = 0
    incompatible = 0
    for target_task in tqdm(
        target["tasks"], desc="Checking reusable CDFT tasks", unit="task"
    ):
        if target_task.get("role") == COMPLEX_ROLE:
            # Legacy result JSON cannot establish reciprocal-branch identity.
            continue
        if target_task["execution"] != "cp2k":
            continue
        target_result_path = target_root / target_task["result_path"]
        if _valid_existing_result(
            target_result_path, target_task, target_signature
        ):
            continue
        source_task = source_tasks.get(target_task["task_id"])
        if not (
            source_task is not None
            and source_task.get("execution") == "cp2k"
            and source_task.get("input_sha256")
            == target_task.get("input_sha256")
        ):
            incompatible += 1
            continue
        source_result_path = source_root / source_task["result_path"]
        if not _valid_existing_result(
            source_result_path, source_task, source_signature
        ):
            incompatible += 1
            continue
        source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
        output_name = source_result.get("output_path")
        if not isinstance(output_name, str) or Path(output_name).name != output_name:
            incompatible += 1
            continue
        source_output = source_root / source_task["task_directory"] / output_name
        expected_output_sha256 = source_result.get("output_sha256")
        if not (
            source_output.is_file()
            and isinstance(expected_output_sha256, str)
            and _sha256_file(source_output) == expected_output_sha256
        ):
            incompatible += 1
            continue
        output_text = source_output.read_text(encoding="utf-8", errors="replace")
        parsed = parse_cp2k_output(
            output_text, require_cdft=target_task["role"] == COMPLEX_ROLE
        )
        if not parsed["valid_completion"]:
            incompatible += 1
            continue

        target_directory = target_root / target_task["task_directory"]
        target_output = target_directory / "cp2k.out"
        if target_output.exists():
            if _sha256_file(target_output) != expected_output_sha256:
                raise RuntimeError(
                    f"Refusing to replace incompatible CDFT output: {target_output}"
                )
        else:
            fd, temporary = tempfile.mkstemp(
                prefix=".cp2k.reuse.", dir=target_directory
            )
            os.close(fd)
            try:
                shutil.copyfile(source_output, temporary)
                if _sha256_file(Path(temporary)) != expected_output_sha256:
                    raise RuntimeError("Copied CDFT output checksum mismatch.")
                os.replace(temporary, target_output)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        _atomic_json(
            target_result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "execution": "cp2k",
                "task_id": target_task["task_id"],
                "configuration_signature": target_signature,
                "input_sha256": target_task["input_sha256"],
                "command": source_result.get("command", []),
                "cp2k_data_dir": source_result.get("cp2k_data_dir"),
                "return_code": 0,
                "started_utc": source_result.get("started_utc"),
                "completed_utc": source_result.get("completed_utc"),
                "reused_utc": _utc_now(),
                "reused_from_configuration_signature": source_signature,
                "reused_from_task_id": source_task["task_id"],
                "output_path": target_output.name,
                "output_sha256": expected_output_sha256,
                **parsed,
            },
        )
        reused += 1
    return reused, incompatible


def _read_result(
    root: Path,
    task: dict[str, Any],
    signature: str,
) -> dict[str, Any] | None:
    path = root / task["result_path"]
    if not _valid_existing_result(path, task, signature):
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_workflow(
    manifest_path: Path,
    output_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Collect raw, counterpoise-corrected fixed-q interaction energies."""

    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    if any(task.get("role") == COMPLEX_ROLE for task in manifest["tasks"]):
        raise RuntimeError(_VALIDATED_BRANCH_REQUIRED)
    root = Path(manifest["_manifest_path"]).parent
    signature = manifest["configuration_signature"]
    projectile = manifest["configuration"]["projectile"]
    symbol = str(projectile["symbol"])
    output_stem = f"{symbol.lower()}_charge_resolved_cdft"
    nlh_diagnostics = projectile["scan_grid"]["kind"] == "nlh_overlap"
    destination = (output_directory or root / "collected").resolve()
    destination.mkdir(parents=True, exist_ok=True)

    tasks_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    results_by_id: dict[str, dict[str, Any] | None] = {}
    for task in tqdm(manifest["tasks"], desc="Reading CDFT results", unit="task"):
        results_by_id[task["task_id"]] = _read_result(root, task, signature)

    rows: list[dict[str, Any]] = []
    missing_tasks: list[str] = []
    invalid_cdft: list[str] = []
    for point in tqdm(manifest["points"], desc="Collecting CDFT points", unit="point"):
        point_id = point["point_id"]
        component_ids = point["component_task_ids"]
        role_tasks = {role: tasks_by_id[component_ids[role]] for role in ROLES}
        task = role_tasks[COMPLEX_ROLE]
        results = {role: results_by_id[component_ids[role]] for role in ROLES}
        absent = [
            role_tasks[role]["task_id"] for role in ROLES if results[role] is None
        ]
        if absent:
            missing_tasks.extend(absent)
            continue
        complex_result = results[COMPLEX_ROLE]
        projectile_result = results[PROJECTILE_COUNTERPOISE_ROLE]
        water_result = results[WATER_COUNTERPOISE_ROLE]
        assert complex_result is not None
        assert projectile_result is not None
        assert water_result is not None
        interaction_hartree = (
            float(complex_result["energy_hartree"])
            - float(projectile_result["energy_hartree"])
            - float(water_result["energy_hartree"])
        )
        deviation = complex_result.get("cdft_deviation_electrons")
        if deviation is None or abs(float(deviation)) > float(
            manifest["configuration"]["cp2k"]["cdft_eps"]
        ):
            invalid_cdft.append(task["task_id"])
        anchor_nlh: float | str = ""
        pair_nlh: list[float] = []
        if nlh_diagnostics:
            anchor_nlh = float(
                potential_ev(
                    float(task["separation_angstrom"]),
                    symbol,
                    str(task["anchor_element"]),
                    enforce_fit_domain=False,
                )
            )
            projectile_xyz = tuple(
                float(value) for value in task["coordinates_angstrom"][0][1:]
            )
            for target in task["coordinates_angstrom"][1:]:
                target_xyz = tuple(float(value) for value in target[1:])
                pair_nlh.append(
                    float(
                        potential_ev(
                            math.dist(projectile_xyz, target_xyz),
                            symbol,
                            str(target[0]),
                            enforce_fit_domain=False,
                        )
                    )
                )
        retained_pair_nlh = [value for value in pair_nlh if value >= 10.0]
        rows.append(
            {
                "point_id": point_id,
                "projectile": symbol,
                "charge": task["charge"],
                "electrons_on_projectile": task["electrons_on_projectile"],
                "multiplicity": task["multiplicity"],
                "orientation": task["orientation"],
                "anchor_element": task["anchor_element"],
                "separation_angstrom": task["separation_angstrom"],
                "minimum_pair_distance_angstrom": task[
                    "minimum_pair_distance_angstrom"
                ],
                "interaction_energy_ev_counterpoise": (
                    interaction_hartree * HARTREE_TO_EV
                ),
                "complex_energy_hartree": complex_result["energy_hartree"],
                "projectile_counterpoise_energy_hartree": projectile_result[
                    "energy_hartree"
                ],
                "water_counterpoise_energy_hartree": water_result[
                    "energy_hartree"
                ],
                "cdft_target_electrons": complex_result.get(
                    "cdft_target_electrons"
                ),
                "cdft_current_electrons": complex_result.get(
                    "cdft_current_electrons"
                ),
                "cdft_deviation_electrons": deviation,
                "anchor_nlh_potential_ev_if_in_domain": (
                    anchor_nlh
                    if isinstance(anchor_nlh, float) and anchor_nlh >= 10.0
                    else ""
                ),
                "nlh_retained_pair_count": len(retained_pair_nlh),
                "nlh_retained_pair_sum_ev": sum(retained_pair_nlh),
                "nlh_max_pair_potential_ev": max(pair_nlh) if pair_nlh else "",
            }
        )

    csv_path = destination / f"{output_stem}_interactions.csv"
    fieldnames = [
        "point_id",
        "projectile",
        "charge",
        "electrons_on_projectile",
        "multiplicity",
        "orientation",
        "anchor_element",
        "separation_angstrom",
        "minimum_pair_distance_angstrom",
        "interaction_energy_ev_counterpoise",
        "complex_energy_hartree",
        "projectile_counterpoise_energy_hartree",
        "water_counterpoise_energy_hartree",
        "cdft_target_electrons",
        "cdft_current_electrons",
        "cdft_deviation_electrons",
        "anchor_nlh_potential_ev_if_in_domain",
        "nlh_retained_pair_count",
        "nlh_retained_pair_sum_ev",
        "nlh_max_pair_potential_ev",
    ]
    fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=destination)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    collection = {
        "schema_version": SCHEMA_VERSION,
        "source_workflow": str(Path(manifest["_manifest_path"])),
        "source_configuration_signature": signature,
        "created_utc": _utc_now(),
        "row_count": len(rows),
        "expected_point_count": len(manifest["points"]),
        "missing_task_count": len(set(missing_tasks)),
        "missing_tasks": sorted(set(missing_tasks)),
        "invalid_cdft_count": len(invalid_cdft),
        "invalid_cdft_tasks": invalid_cdft,
        "numerical_status": (
            "calculations_complete"
            if not missing_tasks and not invalid_cdft
            else "incomplete_or_failed"
        ),
        "physics_status": "validation_pending",
        "projectile": projectile,
        "table_sha256": _sha256_file(csv_path),
        "no_asymptotic_shift_applied": True,
        "no_nlh_blend_or_scaling_applied": True,
        "ctmc_dependency": (
            "none during DFT generation; Geant4 will select V_q before and "
            "V_q' after an independently sampled CTMC transition"
        ),
        "required_acceptance_gates": [
            "repeat basis, GAPW grid, nonperiodic cell, and SCF/CDFT convergence",
            "demonstrate projectile charge localization for every q and geometry",
            "test exchange-correlation-functional dependence",
            "show the unshifted long-range interaction converges to zero",
            "validate the 10--30 eV overlap against NLH without a gap or overlap",
            (
                "extend the molecular pilot to representative amorphous and "
                "ice-Ih environments"
            ),
            "derive and independently benchmark soft stopping and angular moments",
        ],
        "excluded_claims": [
            "physical accuracy of PBE",
            "phase-resolved ice potential",
            "soft stopping or transport cross section",
            "production Geant4 readiness",
        ],
    }
    collection_path = destination / f"{output_stem}.manifest.json"
    _atomic_json(collection_path, collection)
    return csv_path, collection_path
