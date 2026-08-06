#!/usr/bin/env python3
"""Inspect, plan, submit, and assemble the unified ion--ice model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ion_ice.compatibility import backend_compatibility_checks  # noqa: E402
from ion_ice.registry import (  # noqa: E402
    SPECIES_DIRECTORY,
    phase_registry,
    projectile_registry,
)
from ion_ice.schema import ELEMENT_SYMBOLS, load_projectile_definition  # noqa: E402
from ion_ice.resources import PARALLEL_STAGES, calibrate_resources  # noqa: E402
from ion_ice.workflow import (  # noqa: E402
    assemble_geant4_bundle,
    build_plan,
    inspect_model,
    run_local_tasks,
    save_plan,
    submit_plan,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _selection(values: list[str] | None, available: tuple[str, ...]) -> tuple[str, ...]:
    if values is None or values == ["all"]:
        return available
    return tuple(values)


def _reports(args: argparse.Namespace) -> list[dict[str, Any]]:
    projectiles = _selection(args.projectiles, tuple(projectile_registry()))
    phases = _selection(args.phases, tuple(phase_registry()))
    return [
        inspect_model(projectile, phase)
        for projectile in projectiles
        for phase in phases
    ]


def _print_reports(reports: list[dict[str, Any]]) -> None:
    for report in reports:
        readiness = "READY" if report["ready_for_geant4"] else "INCOMPLETE"
        print(f"{report['projectile']} / {report['phase_id']}: {readiness}")
        for stage in report["stages"]:
            print(
                f"  {stage['stage']:<27} {stage['state']:<18} "
                f"{stage['summary']}"
            )
            for blocker in stage["blockers"]:
                print(f"    missing: {blocker}")


def _validate(_: argparse.Namespace) -> int:
    for definition in projectile_registry().values():
        definition.validate()
    for definition in phase_registry().values():
        definition.validate()
    checks = backend_compatibility_checks()
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    return 0 if all(check.passed for check in checks) else 1


def _status(args: argparse.Namespace) -> int:
    reports = _reports(args)
    _print_reports(reports)
    if args.json_output:
        _json(args.json_output.expanduser().resolve(), {"reports": reports})
        print(f"Wrote {args.json_output.expanduser().resolve()}")
    return 0


def _plan(args: argparse.Namespace) -> int:
    projectiles = _selection(args.projectiles, tuple(projectile_registry()))
    phases = _selection(args.phases, tuple(phase_registry()))
    plan = build_plan(
        projectiles,
        phases,
        cp2k_exe=args.cp2k_exe,
        cp2k_module=args.cp2k_module,
        resource_profile_path=args.resource_profiles,
    )
    path = save_plan(plan, args.output)
    for task in plan["tasks"]:
        print(
            f"{task['state']:<28} {task['kind']:<5} {task['task_id']}"
        )
        for blocker in task.get("blockers", []):
            print(f"  missing: {blocker}")
    print(f"Wrote signed plan {path}")
    print("No job or local reducer was started.")
    return 0


def _submit(args: argparse.Namespace) -> int:
    receipt = submit_plan(args.plan, confirm=args.confirm)
    print(f"Wrote PBS submission receipt {receipt}")
    return 0


def _run_local(args: argparse.Namespace) -> int:
    receipt = run_local_tasks(args.plan, confirm=args.confirm)
    print(f"Wrote local-task receipt {receipt}")
    return 0


def _assemble(args: argparse.Namespace) -> int:
    path = assemble_geant4_bundle(args.projectile, args.phase, args.output_directory)
    print(path)
    return 0


def _calibrate_resources(args: argparse.Namespace) -> int:
    report = calibrate_resources(args.stage, args.job_ids)
    _json(args.output.expanduser().resolve(), report)
    current = report["current_profile"]
    proposed = report["proposed_profile"]
    print(
        f"Current: {current['ncpus']} CPUs, {current['memory_gb']} GB "
        f"({current['requested_gb_per_cpu']:.4g} GB/CPU)"
    )
    print(
        f"Proposed: {proposed['ncpus']} CPUs, {proposed['memory_gb']} GB "
        f"({proposed['requested_gb_per_cpu']:.4g} GB/CPU)"
    )
    if not report["all_jobs_completed_successfully"]:
        print("Provisional only: at least one accounting record is incomplete.")
    print(f"Wrote {args.output.expanduser().resolve()}")
    return 0


def _init_species(args: argparse.Namespace) -> int:
    if not 1 <= args.atomic_number < len(ELEMENT_SYMBOLS):
        raise ValueError("--atomic-number must lie in 1..118.")
    if ELEMENT_SYMBOLS[args.atomic_number] != args.symbol:
        raise ValueError(
            f"Atomic number {args.atomic_number} belongs to "
            f"{ELEMENT_SYMBOLS[args.atomic_number]}, not {args.symbol}."
        )
    output = (args.output or (SPECIES_DIRECTORY / f"{args.symbol}.json")).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing species file {output}.")
    reason = "Not yet supplied and independently validated for this projectile."
    payload = {
        "schema_version": 1,
        "symbol": args.symbol,
        "name": args.name,
        "atomic_number": args.atomic_number,
        "aliases": list(
            dict.fromkeys((args.symbol.lower(), args.name.lower()))
        ),
        "default_isotope": {
            "mass_number": args.mass_number,
            "neutral_atomic_mass_u": args.neutral_atomic_mass_u,
            "provenance": {
                "source": args.mass_source,
                "url": args.mass_url,
            },
        },
        "components": {
            name: {"status": "missing", "reason": reason}
            for name in (
                "nlh_pair_potential",
                "hard_transport",
                "soft_dft",
                "geant4_hard_runtime",
                "geant4_soft_runtime",
            )
        },
    }
    _json(output, payload)
    load_projectile_definition(output)
    print(output)
    print(
        "Registered a scientifically empty projectile skeleton; status/plan will "
        "identify every component that still needs evidence."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate registries/backends.")
    validate.set_defaults(function=_validate)

    status = subparsers.add_parser("status", help="Report model readiness and gaps.")
    status.add_argument("--projectiles", nargs="+", default=["all"])
    status.add_argument("--phases", nargs="+", default=["all"])
    status.add_argument("--json-output", type=Path)
    status.set_defaults(function=_status)

    plan = subparsers.add_parser("plan", help="Create a signed, non-executing plan.")
    plan.add_argument("--projectiles", nargs="+", default=["all"])
    plan.add_argument("--phases", nargs="+", default=["all"])
    plan.add_argument("--cp2k-exe")
    plan.add_argument("--cp2k-module")
    plan.add_argument(
        "--resource-profiles",
        type=Path,
        default=HERE / "ion_ice" / "resources.json",
    )
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(function=_plan)

    submit = subparsers.add_parser("submit", help="Submit PBS tasks in a signed plan.")
    submit.add_argument("plan", type=Path)
    submit.add_argument("--confirm", action="store_true", required=True)
    submit.set_defaults(function=_submit)

    local = subparsers.add_parser(
        "run-local", help="Run deterministic local reducers in a signed plan."
    )
    local.add_argument("plan", type=Path)
    local.add_argument("--confirm", action="store_true", required=True)
    local.set_defaults(function=_run_local)

    assemble = subparsers.add_parser(
        "assemble", help="Assemble only a fully accepted Geant4 bundle."
    )
    assemble.add_argument("--projectile", required=True)
    assemble.add_argument("--phase", required=True)
    assemble.add_argument("--output-directory", type=Path, required=True)
    assemble.set_defaults(function=_assemble)

    calibrate = subparsers.add_parser(
        "calibrate-resources",
        help="Evaluate CPUs and peak GB/CPU from PBS accounting.",
    )
    calibrate.add_argument(
        "--stage", choices=tuple(sorted(PARALLEL_STAGES)), required=True
    )
    calibrate.add_argument("--job-ids", nargs="+", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.set_defaults(function=_calibrate_resources)

    initialise = subparsers.add_parser(
        "init-species", help="Create a strict missing-component species skeleton."
    )
    initialise.add_argument("--symbol", required=True)
    initialise.add_argument("--name", required=True)
    initialise.add_argument("--atomic-number", type=int, required=True)
    initialise.add_argument("--mass-number", type=int, required=True)
    initialise.add_argument("--neutral-atomic-mass-u", type=float, required=True)
    initialise.add_argument("--mass-source", required=True)
    initialise.add_argument("--mass-url", required=True)
    initialise.add_argument("--output", type=Path)
    initialise.set_defaults(function=_init_species)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return int(args.function(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
