#!/usr/bin/env python3
"""Recursively refine and run parallel fixed-charge ion--H2O DFT scans."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from tqdm import tqdm

from ion_ice import PROCESS_EVIDENCE_ROOT
from soft_dft import (
    DEFAULT_ADAPTIVE_SETTINGS,
    DEFAULT_CP2K_SETTINGS,
    OT_LINESEARCHES,
    OT_MINIMIZERS,
    AdaptiveSettings,
    CP2KSettings,
    ProjectileDefinition,
    ScanGeometry,
    build_scan_geometries,
    build_scan_geometry,
    build_workflow,
    collect_workflow,
    load_builtin_projectile,
    load_projectile_definition,
    pending_workflow_tasks,
)
from soft_dft.adaptive import (
    initial_mesh_state,
    point_key,
    refine_completed_intervals,
    required_probe_geometries,
)
from soft_dft.geometry import ORIENTATIONS


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 2
IMPLEMENTATION_VERSION = 2


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


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty adaptive DFT table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output parent; defaults to process_evidence/soft_nuclear_collisions/validation/runs/ELEMENT_adaptive_pilot.",
    )
    parser.add_argument(
        "--projectile",
        help="Reviewed built-in element symbol; defaults to C.",
    )
    parser.add_argument(
        "--projectile-definition",
        type=Path,
        help="External complete, provenance-bearing projectile JSON definition.",
    )
    parser.add_argument(
        "--charges",
        nargs="+",
        type=int,
        help="Charge-state subset; defaults to every q=0..Z from the definition.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_ADAPTIVE_SETTINGS.relative_tolerance,
    )
    parser.add_argument(
        "--maximum-refinement-depth",
        type=int,
        default=DEFAULT_ADAPTIVE_SETTINGS.maximum_refinement_depth,
    )
    parser.add_argument(
        "--maximum-mesh-points-per-orientation",
        type=int,
        default=DEFAULT_ADAPTIVE_SETTINGS.maximum_mesh_points_per_orientation,
    )
    parser.add_argument(
        "--projectile-basis-set",
        help="Override the definition's all-electron projectile basis for convergence tests.",
    )
    parser.add_argument(
        "--water-basis-set", default=DEFAULT_CP2K_SETTINGS.water_basis_set
    )
    parser.add_argument("--basis-file", default=DEFAULT_CP2K_SETTINGS.basis_file)
    parser.add_argument(
        "--cell-angstrom", type=float, default=DEFAULT_CP2K_SETTINGS.cell_angstrom
    )
    parser.add_argument(
        "--mgrid-cutoff-ry",
        type=float,
        default=DEFAULT_CP2K_SETTINGS.mgrid_cutoff_ry,
    )
    parser.add_argument(
        "--mgrid-rel-cutoff-ry",
        type=float,
        default=DEFAULT_CP2K_SETTINGS.mgrid_rel_cutoff_ry,
    )
    parser.add_argument("--scf-eps", type=float, default=DEFAULT_CP2K_SETTINGS.scf_eps)
    parser.add_argument(
        "--complex-ot-inner-scf-max",
        type=int,
        default=DEFAULT_CP2K_SETTINGS.complex_ot_inner_scf_max,
        help=(
            "Inner OT steps between constrained-complex preconditioner "
            "refreshes; recorded in the immutable workflow signature."
        ),
    )
    parser.add_argument(
        "--ot-minimizer",
        choices=OT_MINIMIZERS,
        default=DEFAULT_CP2K_SETTINGS.ot_minimizer,
    )
    parser.add_argument(
        "--ot-linesearch",
        choices=OT_LINESEARCHES,
        default=DEFAULT_CP2K_SETTINGS.ot_linesearch,
    )
    parser.add_argument(
        "--cdft-eps", type=float, default=DEFAULT_CP2K_SETTINGS.cdft_eps
    )
    parser.add_argument(
        "--cp2k-command",
        help=(
            "CP2K command for each calculation. The default is a directly "
            "invoked cp2k.psmp using OpenMP threads."
        ),
    )
    parser.add_argument("--cpus-per-calculation", type=int, default=16)
    parser.add_argument(
        "--parallel-calculations",
        type=int,
        default=0,
        help="Concurrent CP2K calculations; zero uses all allocated CPUs.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Prepare or advance one immutable adaptive batch without running "
            "CP2K; use its manifest with multi-node PBS arrays."
        ),
    )
    return parser.parse_args()


def _load_projectile(args: argparse.Namespace) -> ProjectileDefinition:
    if args.projectile_definition is not None:
        projectile = load_projectile_definition(args.projectile_definition)
        if args.projectile is not None and args.projectile != projectile.symbol:
            raise ValueError(
                "--projectile does not match --projectile-definition symbol."
            )
        return projectile
    return load_builtin_projectile(args.projectile or "C")


def _settings(
    args: argparse.Namespace, projectile: ProjectileDefinition
) -> tuple[CP2KSettings, AdaptiveSettings, tuple[int, ...]]:
    charges = tuple(
        range(projectile.atomic_number + 1)
        if args.charges is None
        else sorted(int(value) for value in args.charges)
    )
    if not charges or len(set(charges)) != len(charges) or any(
        value < 0 or value > projectile.atomic_number for value in charges
    ):
        raise ValueError(
            f"--charges must contain unique values from 0 through "
            f"{projectile.atomic_number}."
        )
    cp2k = replace(
        DEFAULT_CP2K_SETTINGS,
        projectile_basis_set=(
            args.projectile_basis_set or projectile.projectile_basis_set
        ),
        water_basis_set=args.water_basis_set,
        basis_file=args.basis_file,
        cell_angstrom=args.cell_angstrom,
        mgrid_cutoff_ry=args.mgrid_cutoff_ry,
        mgrid_rel_cutoff_ry=args.mgrid_rel_cutoff_ry,
        scf_eps=args.scf_eps,
        complex_ot_inner_scf_max=args.complex_ot_inner_scf_max,
        ot_minimizer=args.ot_minimizer,
        ot_linesearch=args.ot_linesearch,
        cdft_eps=args.cdft_eps,
    )
    adaptive = AdaptiveSettings(
        relative_tolerance=args.relative_tolerance,
        maximum_refinement_depth=args.maximum_refinement_depth,
        maximum_mesh_points_per_orientation=(
            args.maximum_mesh_points_per_orientation
        ),
    )
    adaptive.validate()
    return cp2k, adaptive, charges


def _configuration(
    projectile: ProjectileDefinition,
    charges: tuple[int, ...],
    cp2k: CP2KSettings,
    adaptive: AdaptiveSettings,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "projectile": projectile.soft_dft_dict(),
        "charges": list(charges),
        "cp2k": cp2k.as_dict(),
        "adaptive": adaptive.as_dict(),
        "base_geometries": [
            {
                "orientation": geometry.orientation,
                "separation_angstrom": geometry.separation_angstrom,
            }
            for geometry in build_scan_geometries(projectile)
        ],
        "parallelism": (
            "immutable calculation batches; deterministic task shards; "
            "runtime worker count does not affect the scientific result"
        ),
        "physical_status": "validation_pending molecular pilot",
    }


def _new_batch(
    state: dict[str, Any],
    run_root: Path,
    cp2k: CP2KSettings,
    projectile: ProjectileDefinition,
    charges: tuple[int, ...],
    geometries: tuple[ScanGeometry, ...],
    purpose: str,
) -> Path:
    index = len(state["batches"])
    manifest_path = build_workflow(
        run_root / "batches" / f"batch_{index:04d}",
        projectile=projectile,
        settings=cp2k,
        charges=charges,
        geometries=geometries,
        workflow_context={
            "kind": "adaptive_charge_resolved_dft_batch",
            "adaptive_configuration_signature": state["configuration_signature"],
            "batch_index": index,
            "purpose": purpose,
        },
    )
    state["batches"].append(
        {
            "batch_index": index,
            "purpose": purpose,
            "manifest": str(manifest_path),
            "geometry_count": len(geometries),
            "created_utc": _utc_now(),
        }
    )
    state["updated_utc"] = _utc_now()
    return manifest_path


def _initialize(
    run_root: Path,
    configuration: dict[str, Any],
    signature: str,
    cp2k: CP2KSettings,
    projectile: ProjectileDefinition,
    adaptive: AdaptiveSettings,
    charges: tuple[int, ...],
) -> tuple[dict[str, Any], Path]:
    state_path = run_root / "adaptive_charge_resolved_dft.state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_signature") != signature:
            raise RuntimeError(f"Adaptive controller state mismatch: {state_path}")
        return state, state_path
    state = {
        "schema_version": SCHEMA_VERSION,
        "configuration_signature": signature,
        "configuration": configuration,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "awaiting_calculations",
        "batches": [],
        "mesh": initial_mesh_state(projectile, adaptive),
    }
    _new_batch(
        state,
        run_root,
        cp2k,
        projectile,
        charges,
        build_scan_geometries(projectile),
        "base_mesh",
    )
    _atomic_json(state_path, state)
    return state, state_path


def _parallel_layout(
    cpus_per_calculation: int, requested_parallel: int
) -> tuple[list[int], int]:
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    else:
        affinity = list(range(os.cpu_count() or 1))
    if cpus_per_calculation < 1 or cpus_per_calculation > len(affinity):
        raise ValueError(
            f"--cpus-per-calculation must lie in 1-{len(affinity)}."
        )
    maximum_parallel = len(affinity) // cpus_per_calculation
    parallel = maximum_parallel if requested_parallel == 0 else requested_parallel
    if parallel < 1 or parallel > maximum_parallel:
        raise ValueError(
            f"--parallel-calculations must lie in 1-{maximum_parallel}, or be zero."
        )
    return affinity, parallel


def _run_manifest_parallel(
    manifest_path: Path,
    *,
    run_root: Path,
    batch_index: int,
    cp2k_command: str | None,
    cpus_per_calculation: int,
    requested_parallel: int,
) -> None:
    pending = pending_workflow_tasks(manifest_path)
    if not pending:
        return
    affinity, parallel = _parallel_layout(
        cpus_per_calculation, requested_parallel
    )
    parallel = min(parallel, len(pending))
    command_text = cp2k_command
    if command_text is None:
        executable = shutil.which("cp2k.psmp")
        if executable is None:
            raise RuntimeError(
                "CP2K is unavailable; set --cp2k-command or use --prepare-only."
            )
        command_text = shlex.quote(executable)
    if "{cpus}" in command_text:
        command_text = command_text.format(cpus=cpus_per_calculation)

    runner = HERE / "run_charge_resolved_dft.py"
    workflow_root = manifest_path.parent
    pending_result_paths = [
        workflow_root / task["result_path"] for task in pending
    ]
    log_directory = run_root / "controller_logs" / f"batch_{batch_index:04d}"
    log_directory.mkdir(parents=True, exist_ok=True)
    taskset = shutil.which("taskset")
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[Any] = []
    previous_handlers: dict[int, Any] = {}

    def stop_children(signum: int, _frame: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, stop_children)
    try:
        for shard_index in range(parallel):
            core_start = shard_index * cpus_per_calculation
            core_stop = (shard_index + 1) * cpus_per_calculation
            core_group = affinity[core_start:core_stop]
            command = [
                sys.executable,
                os.fspath(runner),
                os.fspath(manifest_path),
                "--shard-count",
                str(parallel),
                "--shard-index",
                str(shard_index),
                "--cp2k-command",
                command_text,
                "--no-progress",
            ]
            if taskset is not None:
                command = [
                    taskset,
                    "-c",
                    ",".join(str(core) for core in core_group),
                    *command,
                ]
            handle = (log_directory / f"shard_{shard_index:04d}.log").open(
                "a", encoding="utf-8"
            )
            handles.append(handle)
            environment = os.environ.copy()
            omp_threads = environment.get(
                "OMP_THREADS_PER_RANK", str(cpus_per_calculation)
            )
            environment.update(
                {
                    "OMP_NUM_THREADS": omp_threads,
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=HERE.parent.parent.parent,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            )

        initial_count = len(pending)
        with tqdm(
            total=initial_count,
            desc=f"CDFT adaptive batch {batch_index}",
            unit="task",
        ) as progress:
            while any(process.poll() is None for process in processes):
                remaining = sum(
                    not path.is_file() for path in pending_result_paths
                )
                progress.n = initial_count - remaining
                progress.refresh()
                failed = [
                    process.returncode
                    for process in processes
                    if process.poll() not in (None, 0)
                ]
                if failed:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                    raise RuntimeError(
                        f"Adaptive CP2K batch {batch_index} failed; see "
                        f"{log_directory}."
                    )
                time.sleep(1.0)
            remaining = sum(not path.is_file() for path in pending_result_paths)
            progress.n = initial_count - remaining
            progress.refresh()
        if any(process.wait() != 0 for process in processes):
            raise RuntimeError(
                f"Adaptive CP2K batch {batch_index} failed; see {log_directory}."
            )
        if pending_workflow_tasks(manifest_path):
            raise RuntimeError(
                f"Adaptive CP2K batch {batch_index} ended with incomplete tasks."
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _collected_values(
    state: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    values: dict[str, float] = {}
    rows_by_key: dict[str, dict[str, str]] = {}
    for batch in state["batches"]:
        manifest_path = Path(batch["manifest"])
        if pending_workflow_tasks(manifest_path):
            raise RuntimeError(f"Batch remains incomplete: {manifest_path}")
        csv_path, collection_path = collect_workflow(manifest_path)
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        if collection["numerical_status"] != "calculations_complete":
            raise RuntimeError(
                f"Batch failed its numerical/CDFT checks: {collection_path}"
            )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = point_key(
                    int(row["charge"]),
                    row["orientation"],
                    float(row["separation_angstrom"]),
                )
                value = float(row["interaction_energy_ev_counterpoise"])
                if key in values and value != values[key]:
                    raise RuntimeError(f"Conflicting duplicate DFT point: {key}")
                values[key] = value
                rows_by_key[key] = row
        batch["collection_manifest"] = str(collection_path)
        batch["status"] = "complete"
    return values, rows_by_key


def _add_probe_batch(
    state: dict[str, Any],
    state_path: Path,
    run_root: Path,
    cp2k: CP2KSettings,
    projectile: ProjectileDefinition,
    charges: tuple[int, ...],
    probes: tuple[tuple[str, float], ...],
) -> Path:
    orientation_map = {item.name: item for item in ORIENTATIONS}
    geometries = tuple(
        build_scan_geometry(orientation_map[name], separation, projectile.symbol)
        for name, separation in probes
    )
    path = _new_batch(
        state,
        run_root,
        cp2k,
        projectile,
        charges,
        geometries,
        "quarter_midpoint_three_quarter_validation",
    )
    state["status"] = "awaiting_calculations"
    _atomic_json(state_path, state)
    return path


def _finalize(
    state: dict[str, Any],
    state_path: Path,
    run_root: Path,
    projectile: ProjectileDefinition,
    charges: tuple[int, ...],
    rows_by_key: dict[str, dict[str, str]],
) -> None:
    output = run_root / "collected"
    output_stem = f"{projectile.symbol.lower()}_cdft_adaptive"
    evaluations = sorted(
        rows_by_key.values(),
        key=lambda row: (
            row["orientation"],
            float(row["separation_angstrom"]),
            int(row["charge"]),
        ),
    )
    mesh_rows: list[dict[str, str]] = []
    for orientation, curve in state["mesh"]["curves"].items():
        for separation in curve["mesh_separations_angstrom"]:
            for charge in charges:
                mesh_rows.append(
                    rows_by_key[point_key(charge, orientation, float(separation))]
                )
    mesh_rows.sort(
        key=lambda row: (
            row["orientation"],
            int(row["charge"]),
            float(row["separation_angstrom"]),
        )
    )
    evaluations_path = output / f"{output_stem}_all_evaluations.csv"
    mesh_path = output / f"{output_stem}_mesh.csv"
    _atomic_csv(evaluations_path, evaluations)
    _atomic_csv(mesh_path, mesh_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "configuration_signature": state["configuration_signature"],
        "created_utc": _utc_now(),
        "numerical_status": "adaptive_interpolation_converged",
        "physics_status": "validation_pending",
        "relative_tolerance": state["configuration"]["adaptive"][
            "relative_tolerance"
        ],
        "tolerance_definition": (
            "numerical radial potential interpolation only; not physical accuracy"
        ),
        "projectile": projectile.soft_dft_dict(),
        "charge_mesh_policy": (
            "one common mesh refined by the worst error among requested charges "
            + ",".join(str(charge) for charge in charges)
        ),
        "evaluation_row_count": len(evaluations),
        "mesh_row_count": len(mesh_rows),
        "mesh_points_per_orientation": {
            orientation: len(curve["mesh_separations_angstrom"])
            for orientation, curve in state["mesh"]["curves"].items()
        },
        "accepted_interval_count": sum(
            len(curve["accepted_intervals"])
            for curve in state["mesh"]["curves"].values()
        ),
        "bisected_interval_count": sum(
            len(curve.get("refined_intervals", []))
            for curve in state["mesh"]["curves"].values()
        ),
        "maximum_scaled_interpolation_error": state["mesh"][
            "maximum_scaled_interpolation_error"
        ],
        "all_evaluations": str(evaluations_path),
        "all_evaluations_sha256": _sha256_file(evaluations_path),
        "adaptive_mesh": str(mesh_path),
        "adaptive_mesh_sha256": _sha256_file(mesh_path),
        "controller_state": str(state_path),
        "excluded_claims": [
            "DFT functional or basis convergence",
            "phase-resolved ice validity",
            "soft stopping or angular-transport convergence",
            "production Geant4 readiness",
        ],
    }
    result_path = output / f"{output_stem}.manifest.json"
    _atomic_json(result_path, result)
    state["status"] = "complete"
    state["completed_utc"] = _utc_now()
    state["final_manifest"] = str(result_path)
    _atomic_json(state_path, state)


def main() -> int:
    args = parse_args()
    projectile = _load_projectile(args)
    cp2k, adaptive, charges = _settings(args, projectile)
    configuration = _configuration(projectile, charges, cp2k, adaptive)
    signature = _sha256_bytes(_canonical_json(configuration))
    output_root = args.output_root or (
        PROCESS_EVIDENCE_ROOT
        / "soft_nuclear_collisions"
        / "validation"
        / "runs"
        / f"{projectile.symbol.lower()}_adaptive_pilot"
    )
    run_root = output_root.expanduser().resolve() / signature
    state, state_path = _initialize(
        run_root,
        configuration,
        signature,
        cp2k,
        projectile,
        adaptive,
        charges,
    )
    if state.get("status") == "complete":
        print(f"Adaptive DFT workflow already complete: {state['final_manifest']}")
        return 0

    while True:
        incomplete = [
            batch
            for batch in state["batches"]
            if pending_workflow_tasks(Path(batch["manifest"]))
        ]
        if incomplete:
            if args.prepare_only:
                for batch in incomplete:
                    print(f"Prepared batch manifest: {batch['manifest']}")
                print("Status: awaiting_calculations")
                return 0
            for batch in incomplete:
                _run_manifest_parallel(
                    Path(batch["manifest"]),
                    run_root=run_root,
                    batch_index=int(batch["batch_index"]),
                    cp2k_command=args.cp2k_command,
                    cpus_per_calculation=args.cpus_per_calculation,
                    requested_parallel=args.parallel_calculations,
                )

        values, rows_by_key = _collected_values(state)
        probes = required_probe_geometries(state["mesh"], values, charges)
        if probes:
            manifest_path = _add_probe_batch(
                state,
                state_path,
                run_root,
                cp2k,
                projectile,
                charges,
                probes,
            )
            print(f"Prepared adaptive batch: {manifest_path}")
            if args.prepare_only:
                print("Status: awaiting_calculations")
                return 0
            continue

        try:
            refine_completed_intervals(state["mesh"], values, charges, adaptive)
        except RuntimeError:
            state["updated_utc"] = _utc_now()
            state["status"] = state["mesh"]["status"]
            _atomic_json(state_path, state)
            raise
        state["updated_utc"] = _utc_now()
        state["status"] = state["mesh"]["status"]
        _atomic_json(state_path, state)
        if state["mesh"]["status"] == "complete":
            _finalize(
                state,
                state_path,
                run_root,
                projectile,
                charges,
                rows_by_key,
            )
            print(f"Completed adaptive DFT workflow: {state['final_manifest']}")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
