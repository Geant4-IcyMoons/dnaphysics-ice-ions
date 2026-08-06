#!/usr/bin/env python3
"""Prepare deterministic all-electron ion(q+)--H2O constrained-DFT tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from soft_dft import (
    DEFAULT_CP2K_SETTINGS,
    build_workflow,
    build_scan_geometries,
    load_builtin_projectile,
    load_projectile_definition,
    load_workflow_manifest,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output parent; defaults to soft_collision_dft_runs/ELEMENT_molecular_pilot.",
    )
    parser.add_argument("--projectile", help="Built-in element symbol; defaults to C.")
    parser.add_argument("--projectile-definition", type=Path)
    parser.add_argument(
        "--charges",
        nargs="+",
        type=int,
        help="Charge-state subset; defaults to every q=0..Z.",
    )
    parser.add_argument(
        "--runtime-smoke",
        action="store_true",
        help=(
            "Prepare one 6-A oxygen-back geometry for every selected charge. "
            "This checks the executable, basis, and charge ladder only; it is "
            "not a physical or phase-specific potential table."
        ),
    )
    parser.add_argument("--projectile-basis-set")
    parser.add_argument(
        "--water-basis-set", default=DEFAULT_CP2K_SETTINGS.water_basis_set
    )
    parser.add_argument("--basis-file")
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
        "--cdft-eps", type=float, default=DEFAULT_CP2K_SETTINGS.cdft_eps
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.projectile_definition is not None:
        projectile = load_projectile_definition(args.projectile_definition)
        if args.projectile is not None and args.projectile != projectile.symbol:
            raise ValueError(
                "--projectile does not match --projectile-definition symbol."
            )
    else:
        projectile = load_builtin_projectile(args.projectile or "C")
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        projectile_basis_set=(
            args.projectile_basis_set or projectile.projectile_basis_set
        ),
        water_basis_set=args.water_basis_set,
        basis_file=args.basis_file or projectile.basis_file,
        cell_angstrom=args.cell_angstrom,
        mgrid_cutoff_ry=args.mgrid_cutoff_ry,
        mgrid_rel_cutoff_ry=args.mgrid_rel_cutoff_ry,
        scf_eps=args.scf_eps,
        cdft_eps=args.cdft_eps,
    )
    output_root = args.output_root or (
        HERE
        / "soft_collision_dft_runs"
        / (
            f"{projectile.symbol.lower()}_runtime_smoke"
            if args.runtime_smoke
            else f"{projectile.symbol.lower()}_molecular_pilot"
        )
    )
    charges = None if args.charges is None else tuple(args.charges)
    geometries = None
    workflow_context = None
    if args.runtime_smoke:
        candidates = [
            geometry
            for geometry in build_scan_geometries(projectile)
            if geometry.orientation == "oxygen_back"
            and geometry.separation_angstrom == 6.0
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "The registered scan must contain exactly one oxygen-back 6-A "
                "runtime-smoke geometry."
            )
        geometries = tuple(candidates)
        workflow_context = {
            "kind": "all_charge_runtime_smoke",
            "scientific_use": "none",
            "definition": (
                "Executable, basis, SCF/CDFT, and complete charge-ladder "
                "plumbing check at one asymptotic geometry."
            ),
        }
    manifest_path = build_workflow(
        output_root,
        projectile=projectile,
        settings=settings,
        charges=charges,
        geometries=geometries,
        workflow_context=workflow_context,
    )
    manifest = load_workflow_manifest(manifest_path)
    print(f"Workflow: {manifest_path}")
    print(f"Configuration: {manifest['configuration_signature']}")
    print(
        f"Tasks: {manifest['task_count']} total, "
        f"{manifest['cp2k_task_count']} CP2K, "
        f"{manifest['analytic_task_count']} analytic"
    )
    print("Physics status: validation_pending (molecular pilot only)")


if __name__ == "__main__":
    main()
