#!/usr/bin/env python3
"""Prepare, run, and reduce deterministic adaptive NLH case shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adaptive_nlh_particle_transport as serial  # noqa: E402
from bca.config import DEFAULT_PROJECTILES  # noqa: E402
from ion_ice import ICE_STRUCTURES_ROOT, PROCESS_EVIDENCE_ROOT  # noqa: E402


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _common_prepare_arguments(parser: argparse.ArgumentParser) -> None:
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
        default=serial.DEFAULT_STRUCTURE_SEEDS,
    )
    parser.add_argument(
        "--orientations",
        nargs="+",
        choices=tuple(serial.ORIENTATIONS),
        default=tuple(serial.ORIENTATIONS),
    )
    parser.add_argument(
        "--base-energies-ev",
        type=float,
        nargs="+",
        default=serial.DEFAULT_BASE_ENERGIES_EV,
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
            / "sharded_adaptive_particles"
        ),
    )
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
            "Raise only the restart-time ceiling while retaining a finite, "
            "auditable stopping boundary."
        ),
    )
    parser.add_argument("--trajectory-batch-size", type=int, default=100_000)
    parser.add_argument("--path-length-angstrom", type=float, default=100.0)
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "collision_tube_mixture"),
        default="collision_tube_mixture",
    )
    parser.add_argument("--tube-mixture-fraction", type=float, default=0.5)
    parser.add_argument("--maximum-refinement-depth", type=int, default=8)
    parser.add_argument("--maximum-energy-points", type=int, default=257)
    parser.add_argument(
        "--unlimited-trajectories",
        action="store_true",
        help="Let each restartable case sample until its statistical gate passes.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Reduce results and prepare one wave.")
    _common_prepare_arguments(prepare)
    run = commands.add_parser("run", help="Run one deterministic subset of a wave.")
    run.add_argument("wave_manifest", type=Path)
    run.add_argument("--shard-count", type=int, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def _serial_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return the complete argument contract used by the proven serial case code."""

    return SimpleNamespace(
        implementation_version=serial.IMPLEMENTATION_VERSION,
        projectile=args.projectile,
        structure_directory=args.structure_directory.expanduser().resolve(),
        structure_seeds=list(args.structure_seeds),
        orientations=list(args.orientations),
        base_energies_ev=list(args.base_energies_ev),
        kernels=args.kernels.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        workers=0,
        tolerance=args.tolerance,
        confidence=args.confidence,
        calibration_trajectories=args.calibration_trajectories,
        minimum_trajectories=args.minimum_trajectories,
        maximum_trajectories=args.maximum_trajectories,
        extended_maximum_trajectories=args.extended_maximum_trajectories,
        unlimited_trajectories=args.unlimited_trajectories,
        trajectory_batch_size=args.trajectory_batch_size,
        path_length_angstrom=args.path_length_angstrom,
        sampling_mode=getattr(args, "sampling_mode", "collision_tube_mixture"),
        tube_mixture_fraction=getattr(args, "tube_mixture_fraction", 0.5),
        maximum_refinement_depth=args.maximum_refinement_depth,
        maximum_energy_points=args.maximum_energy_points,
        dry_run=False,
    )


def _case_dict(case: serial.Case) -> dict[str, Any]:
    return {
        "structure_seed": case.structure_seed,
        "orientation": case.orientation,
        "energy_ev": case.energy_ev,
        "case_key": serial._case_key(case),
    }


def _case_from_dict(record: dict[str, Any]) -> serial.Case:
    return serial.Case(
        int(record["structure_seed"]),
        str(record["orientation"]),
        float(record["energy_ev"]),
    )


def _case_sort_key(case: serial.Case) -> tuple[float, int, str]:
    return (case.energy_ev, case.structure_seed, case.orientation)


def _deduplicate_cases(cases: list[serial.Case]) -> list[serial.Case]:
    return sorted(set(cases), key=_case_sort_key)


def _cases_for_shard(
    cases: list[serial.Case], shard_count: int, shard_index: int
) -> list[serial.Case]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    return [case for index, case in enumerate(cases) if index % shard_count == shard_index]


def _state_root(args: argparse.Namespace) -> tuple[Path, dict[str, Any], str]:
    configuration = serial._controller_configuration(args)
    configuration["sharded_orchestration"] = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "ownership": "one immutable case owner per wave and global shard index",
    }
    signature = serial._signature(configuration)
    root = args.output_root / args.projectile / signature
    return root, configuration, signature


def _initialize_state(
    root: Path, configuration: dict[str, Any], signature: str
) -> tuple[dict[str, Any], Path]:
    state_path = root / "adaptive_particle_shards.state.json"
    if state_path.is_file():
        state = serial._manifest(state_path)
        if state.get("configuration_signature") != signature:
            raise RuntimeError(f"Sharded controller state mismatch: {state_path}")
        return state, state_path
    base = sorted(float(value) for value in configuration["base_energies_ev"])
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "configuration_signature": signature,
        "configuration": configuration,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "awaiting_base_wave",
        "cases": {},
        "intervals": {},
        "frontier": [
            {"lower_energy_ev": base[index], "upper_energy_ev": base[index + 1], "depth": 0}
            for index in range(len(base) - 1)
        ],
        "waves": [],
    }
    _atomic_json(state_path, state)
    return state, state_path


def _expected_production_manifest(
    root: Path, case: serial.Case
) -> Path:
    return serial._case_directory(root, "production", case) / "hard_collision_run.manifest.json"


def _reconcile_cases(
    state: dict[str, Any], state_path: Path, root: Path
) -> None:
    cases_state = state["cases"]
    changed = False
    for wave in state["waves"]:
        all_complete = True
        for item in wave["cases"]:
            case = _case_from_dict(item)
            key = serial._case_key(case)
            existing = cases_state.get(key)
            if isinstance(existing, dict) and existing.get("converged"):
                continue
            path = _expected_production_manifest(root, case)
            if not path.is_file():
                all_complete = False
                continue
            manifest = serial._manifest(path)
            report = manifest.get("statistical_convergence")
            if not isinstance(report, dict) or not bool(report.get("converged")):
                all_complete = False
                continue
            cases_state[key] = {
                "converged": True,
                "production_manifest": str(path),
                "trajectories": int(manifest["configuration"]["completed_trajectories"]),
            }
            changed = True
        new_status = "complete" if all_complete else "awaiting_calculations"
        if wave.get("status") != new_status:
            wave["status"] = new_status
            changed = True
    if changed:
        state["updated_utc"] = _utc_now()
        _atomic_json(state_path, state)


def _missing_cases(
    state: dict[str, Any], cases: list[serial.Case]
) -> list[serial.Case]:
    records = state["cases"]
    return [
        case
        for case in cases
        if not (
            isinstance(records.get(serial._case_key(case)), dict)
            and records[serial._case_key(case)].get("converged")
        )
    ]


def _write_wave(
    state: dict[str, Any],
    state_path: Path,
    root: Path,
    args: argparse.Namespace,
    cases: list[serial.Case],
    purpose: str,
) -> Path:
    cases = _deduplicate_cases(cases)
    if not cases:
        raise ValueError("Cannot write an empty hard-collision wave.")
    index = len(state["waves"])
    path = root / "waves" / f"wave_{index:04d}.manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "configuration_signature": state["configuration_signature"],
        "particle_root": str(root),
        "purpose": purpose,
        "wave_index": index,
        "created_utc": _utc_now(),
        "configuration": state["configuration"],
        "runtime": {
            "unlimited_trajectories": bool(args.unlimited_trajectories),
            "maximum_trajectories": serial._runtime_trajectory_ceiling(args),
        },
        "cases": [_case_dict(case) for case in cases],
    }
    _atomic_json(path, payload)
    state["waves"].append(
        {
            "wave_index": index,
            "purpose": purpose,
            "manifest": str(path),
            "case_count": len(cases),
            "cases": payload["cases"],
            "status": "awaiting_calculations",
        }
    )
    state["status"] = "awaiting_calculations"
    state["updated_utc"] = _utc_now()
    _atomic_json(state_path, state)
    return path


def _refresh_wave_runtime(path: Path, args: argparse.Namespace) -> None:
    """Atomically update only the restart-time sampling limit of a pending wave."""

    payload = serial._manifest(path)
    requested = {
        "unlimited_trajectories": bool(args.unlimited_trajectories),
        "maximum_trajectories": serial._runtime_trajectory_ceiling(args),
    }
    if payload.get("runtime") == requested:
        return
    history = payload.setdefault("runtime_history", [])
    if not isinstance(history, list):
        raise RuntimeError(f"Malformed wave runtime history: {path}")
    history.append(
        {
            "recorded_utc": _utc_now(),
            "previous": payload.get("runtime"),
            "replacement": requested,
            "reason": (
                "restart-time liveness ceiling only; configuration signature "
                "and numerical acceptance gates unchanged"
            ),
        }
    )
    payload["runtime"] = requested
    _atomic_json(path, payload)


def _frontier_key(record: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(record["lower_energy_ev"]),
        float(record["upper_energy_ev"]),
        int(record["depth"]),
    )


def _finalize(state: dict[str, Any], state_path: Path, root: Path) -> Path:
    energies = sorted(
        {float(key.rsplit("/", 1)[1][:-2]) for key in state["cases"]}
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "configuration_signature": state["configuration_signature"],
        "created_utc": _utc_now(),
        "numerical_status": "adaptive_sampling_and_energy_grid_converged",
        "physics_status": "validation_pending",
        "projectile": state["configuration"]["projectile"],
        "evaluated_energy_count": len(energies),
        "evaluated_energies_ev": energies,
        "case_count": len(state["cases"]),
        "controller_state": str(state_path),
        "excluded_claims": [
            "soft nuclear scattering",
            "electronic stopping",
            "charge exchange",
            "recoil-cascade and lattice-damage evolution",
            "physical accuracy beyond the retained NLH domain",
        ],
    }
    path = root / "hard_collision_sharded_adaptive.manifest.json"
    _atomic_json(path, result)
    state["status"] = "complete"
    state["completed_utc"] = _utc_now()
    state["final_manifest"] = str(path)
    _atomic_json(state_path, state)
    return path


def prepare_wave(raw_args: argparse.Namespace) -> tuple[str, Path, int]:
    args = _serial_args(raw_args)
    serial._validate_args(args)
    root, configuration, signature = _state_root(args)
    state, state_path = _initialize_state(root, configuration, signature)
    if state.get("status") == "complete":
        return "complete", Path(state["final_manifest"]), 0
    _reconcile_cases(state, state_path, root)
    incomplete_waves = [wave for wave in state["waves"] if wave["status"] != "complete"]
    if incomplete_waves:
        wave = incomplete_waves[-1]
        wave_path = Path(wave["manifest"])
        _refresh_wave_runtime(wave_path, args)
        missing = _missing_cases(
            state, [_case_from_dict(item) for item in wave["cases"]]
        )
        return "awaiting_calculations", wave_path, len(missing)

    base_energies = sorted(set(float(value) for value in args.base_energies_ev))
    base_cases = [
        case for energy in base_energies for case in serial._all_cases(args, energy)
    ]
    missing_base = _missing_cases(state, base_cases)
    if missing_base:
        path = _write_wave(state, state_path, root, args, missing_base, "base_energy_grid")
        return "prepared", path, len(missing_base)

    while True:
        pending: list[serial.Case] = []
        next_frontier: list[dict[str, Any]] = []
        for item in sorted(state["frontier"], key=_frontier_key):
            lower, upper, depth = _frontier_key(item)
            key = serial._interval_key(lower, upper)
            record = state["intervals"].get(key)
            midpoint = math.sqrt(lower * upper)
            if record is None:
                midpoint_cases = list(serial._all_cases(args, midpoint))
                missing = _missing_cases(state, midpoint_cases)
                if missing:
                    pending.extend(missing)
                    next_frontier.append(item)
                    continue
                record = serial._analyze_interval(args, state, lower, upper, midpoint)
                record["depth"] = depth
                state["intervals"][key] = record
            if bool(record["passes"]):
                continue
            if depth >= args.maximum_refinement_depth:
                state["status"] = "refinement_limit_reached"
                _atomic_json(state_path, state)
                raise RuntimeError(
                    f"Energy interval {lower:g}-{upper:g} eV failed the "
                    f"{args.tolerance:.3%} gate at depth {depth}."
                )
            next_frontier.extend(
                (
                    {"lower_energy_ev": lower, "upper_energy_ev": midpoint, "depth": depth + 1},
                    {"lower_energy_ev": midpoint, "upper_energy_ev": upper, "depth": depth + 1},
                )
            )

        deduplicated = {
            _frontier_key(item): item for item in next_frontier
        }
        state["frontier"] = [deduplicated[key] for key in sorted(deduplicated)]
        state["updated_utc"] = _utc_now()
        _atomic_json(state_path, state)
        if pending:
            evaluated = {
                float(key.rsplit("/", 1)[1][:-2]) for key in state["cases"]
            }
            candidate = {case.energy_ev for case in pending}
            if len(evaluated | candidate) > args.maximum_energy_points:
                raise RuntimeError("Adaptive energy grid exceeded --maximum-energy-points.")
            path = _write_wave(
                state,
                state_path,
                root,
                args,
                pending,
                "geometric_midpoint_validation",
            )
            return "prepared", path, len(_deduplicate_cases(pending))
        if not state["frontier"]:
            return "complete", _finalize(state, state_path, root), 0


def _args_from_wave(payload: dict[str, Any]) -> argparse.Namespace:
    configuration = payload["configuration"]
    runtime = payload["runtime"]
    return SimpleNamespace(
        implementation_version=serial.IMPLEMENTATION_VERSION,
        projectile=configuration["projectile"],
        structure_directory=Path(configuration["structure_directory"]),
        structure_seeds=list(configuration["structure_seeds"]),
        orientations=list(configuration["orientations"]),
        base_energies_ev=list(configuration["base_energies_ev"]),
        kernels=Path(configuration["kernel_path"]),
        output_root=Path(payload["particle_root"]).parent.parent,
        workers=0,
        tolerance=float(configuration["tolerance"]),
        confidence=float(configuration["confidence"]),
        calibration_trajectories=int(configuration["calibration_trajectories"]),
        minimum_trajectories=int(configuration["minimum_trajectories"]),
        maximum_trajectories=int(configuration["maximum_trajectories"]),
        extended_maximum_trajectories=(
            None
            if runtime["unlimited_trajectories"]
            else int(runtime["maximum_trajectories"])
        ),
        unlimited_trajectories=bool(runtime["unlimited_trajectories"]),
        trajectory_batch_size=int(configuration["trajectory_batch_size"]),
        path_length_angstrom=float(configuration["path_length_angstrom"]),
        sampling_mode=str(configuration["initial_condition_sampling"]),
        tube_mixture_fraction=float(configuration["tube_mixture_fraction"]),
        maximum_refinement_depth=int(configuration["maximum_refinement_depth"]),
        maximum_energy_points=int(configuration["maximum_energy_points"]),
        dry_run=False,
    )


def run_shard(
    wave_manifest: Path, shard_count: int, shard_index: int, requested_workers: int
) -> Path:
    payload = serial._manifest(wave_manifest.expanduser().resolve())
    args = _args_from_wave(payload)
    configuration = dict(payload["configuration"])
    if configuration.get("sharded_orchestration", {}).get("implementation_version") != IMPLEMENTATION_VERSION:
        raise RuntimeError("Unsupported hard-shard workflow implementation.")
    if serial._signature(configuration) != payload["configuration_signature"]:
        raise RuntimeError("Hard-shard wave configuration signature mismatch.")
    workers = serial._worker_count(requested_workers)
    cases = [_case_from_dict(item) for item in payload["cases"]]
    selected = _cases_for_shard(cases, shard_count, shard_index)
    root = Path(payload["particle_root"])
    completed: list[dict[str, Any]] = []
    for case in tqdm(selected, desc=f"hard wave {payload['wave_index']} shard {shard_index}", unit="case"):
        calibration = serial._calibrate_case(args, workers, root, case)
        path = serial._run_production_case(args, workers, root, case, calibration)
        manifest = serial._manifest(path)
        completed.append(
            {
                **_case_dict(case),
                "production_manifest": str(path),
                "completed_trajectories": int(manifest["configuration"]["completed_trajectories"]),
            }
        )
    receipt = wave_manifest.parent / f"wave_{int(payload['wave_index']):04d}.receipts" / f"shard_{shard_index:06d}.json"
    _atomic_json(
        receipt,
        {
            "schema_version": SCHEMA_VERSION,
            "configuration_signature": payload["configuration_signature"],
            "wave_index": payload["wave_index"],
            "shard_count": shard_count,
            "shard_index": shard_index,
            "workers": workers,
            "completed_utc": _utc_now(),
            "cases": completed,
        },
    )
    return receipt


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        status, path, count = prepare_wave(args)
        if status == "complete":
            print(f"Completed hard-shard campaign: {path}")
        else:
            print(f"Prepared wave manifest: {path}")
            print(f"Pending case count: {count}")
            print(f"Status: {status}")
        return 0
    receipt = run_shard(
        args.wave_manifest, args.shard_count, args.shard_index, args.workers
    )
    print(f"Completed shard receipt: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
