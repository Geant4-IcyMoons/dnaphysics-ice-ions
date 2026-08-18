#!/usr/bin/env python3
"""Prepare deterministic all-electron ion(q+)--H2O constrained-DFT tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ion_ice import PROCESS_EVIDENCE_ROOT
from soft_dft import (
    CDFT_CONSTRAINT_TYPES,
    DEFAULT_CP2K_SETTINGS,
    MIXING_METHODS,
    CDFT_OPTIMIZERS,
    OT_ALGORITHMS,
    OT_LINESEARCHES,
    OT_MINIMIZERS,
    SCF_SOLVERS,
    build_workflow,
    build_scan_geometries,
    load_builtin_projectile,
    load_projectile_definition,
    load_workflow_manifest,
    reuse_compatible_workflow_results,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output parent; defaults to process_evidence/soft_nuclear_collisions/validation/runs/ELEMENT_molecular_pilot.",
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
        "--complex-ot-inner-scf-max",
        type=int,
        default=DEFAULT_CP2K_SETTINGS.complex_ot_inner_scf_max,
        help=(
            "Inner OT steps between preconditioner refreshes for constrained "
            "complexes; this is signature-bound and does not change EPS_SCF."
        ),
    )
    parser.add_argument(
        "--ot-algorithm",
        choices=OT_ALGORITHMS,
        default=DEFAULT_CP2K_SETTINGS.ot_algorithm,
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
        "--cdft-optimizer",
        choices=CDFT_OPTIMIZERS,
        default=DEFAULT_CP2K_SETTINGS.cdft_optimizer,
        help=(
            "CP2K optimizer for the one-dimensional CDFT constraint. "
            "BISECT is the documented difficult-case validation option."
        ),
    )
    parser.add_argument(
        "--cdft-constraint-type",
        choices=CDFT_CONSTRAINT_TYPES,
        default=DEFAULT_CP2K_SETTINGS.cdft_constraint_type,
        help=(
            "Population weight used to define the projectile electron count. "
            "HIRSHFELD uses CP2K's density-based, parameter-free weights."
        ),
    )
    parser.add_argument(
        "--complex-scf-solver",
        choices=SCF_SOLVERS,
        default=DEFAULT_CP2K_SETTINGS.complex_scf_solver,
        help=(
            "SCF solver for constrained ion--water complexes. Changes are "
            "numerical convergence tests and create a new workflow signature."
        ),
    )
    parser.add_argument(
        "--water-counterpoise-scf-solver",
        choices=SCF_SOLVERS,
        default=DEFAULT_CP2K_SETTINGS.water_counterpoise_scf_solver,
        help=(
            "SCF solver for neutral-water counterpoise terms."
        ),
    )
    parser.add_argument(
        "--projectile-counterpoise-scf-solver",
        choices=SCF_SOLVERS,
        default=DEFAULT_CP2K_SETTINGS.projectile_counterpoise_scf_solver,
        help=(
            "SCF solver for isolated projectile counterpoise terms. "
            "DIAGONALIZATION preserves the explicitly audited ionic "
            "electron populations."
        ),
    )
    parser.add_argument(
        "--complex-mixing-method",
        choices=MIXING_METHODS,
        default=DEFAULT_CP2K_SETTINGS.complex_mixing_method,
        help="Density-matrix mixing method for diagonalization complex SCF.",
    )
    parser.add_argument(
        "--complex-mixing-alpha",
        type=float,
        default=DEFAULT_CP2K_SETTINGS.complex_mixing_alpha,
        help="New-density fraction for diagonalization/Pulay complex SCF.",
    )
    parser.add_argument(
        "--complex-mixing-npulay",
        type=int,
        default=DEFAULT_CP2K_SETTINGS.complex_mixing_npulay,
        help="Prior steps retained by constrained-complex Pulay mixing.",
    )
    parser.add_argument(
        "--counterpoise-mixing-method",
        choices=MIXING_METHODS,
        default=DEFAULT_CP2K_SETTINGS.counterpoise_mixing_method,
        help="Density-matrix mixing method for diagonalization counterpoise SCF.",
    )
    parser.add_argument(
        "--counterpoise-mixing-alpha",
        type=float,
        default=DEFAULT_CP2K_SETTINGS.counterpoise_mixing_alpha,
        help="New-density fraction for diagonalization/Pulay counterpoise SCF.",
    )
    parser.add_argument(
        "--counterpoise-mixing-npulay",
        type=int,
        default=DEFAULT_CP2K_SETTINGS.counterpoise_mixing_npulay,
        help="Prior steps retained by counterpoise Pulay mixing.",
    )
    parser.add_argument(
        "--counterpoise-ot-algorithm",
        choices=OT_ALGORITHMS,
        default=DEFAULT_CP2K_SETTINGS.counterpoise_ot_algorithm,
    )
    parser.add_argument(
        "--counterpoise-ot-minimizer",
        choices=OT_MINIMIZERS,
        default=DEFAULT_CP2K_SETTINGS.counterpoise_ot_minimizer,
    )
    parser.add_argument(
        "--counterpoise-ot-linesearch",
        choices=OT_LINESEARCHES,
        default=DEFAULT_CP2K_SETTINGS.counterpoise_ot_linesearch,
    )
    parser.add_argument(
        "--reuse-compatible-from",
        action="append",
        type=Path,
        default=[],
        metavar="MANIFEST",
        help=(
            "Reuse completed tasks from another workflow only when task IDs "
            "and rendered CP2K input SHA-256 values are identical. May be "
            "specified more than once."
        ),
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
        complex_ot_inner_scf_max=args.complex_ot_inner_scf_max,
        ot_algorithm=args.ot_algorithm,
        ot_minimizer=args.ot_minimizer,
        ot_linesearch=args.ot_linesearch,
        complex_scf_solver=args.complex_scf_solver,
        complex_mixing_method=args.complex_mixing_method,
        complex_mixing_alpha=args.complex_mixing_alpha,
        complex_mixing_npulay=args.complex_mixing_npulay,
        counterpoise_ot_algorithm=args.counterpoise_ot_algorithm,
        counterpoise_ot_minimizer=args.counterpoise_ot_minimizer,
        counterpoise_ot_linesearch=args.counterpoise_ot_linesearch,
        water_counterpoise_scf_solver=(
            args.water_counterpoise_scf_solver
        ),
        projectile_counterpoise_scf_solver=(
            args.projectile_counterpoise_scf_solver
        ),
        counterpoise_mixing_alpha=args.counterpoise_mixing_alpha,
        counterpoise_mixing_npulay=args.counterpoise_mixing_npulay,
        counterpoise_mixing_method=args.counterpoise_mixing_method,
        cdft_eps=args.cdft_eps,
        cdft_optimizer=args.cdft_optimizer,
        cdft_constraint_type=args.cdft_constraint_type,
    )
    output_root = args.output_root or (
        PROCESS_EVIDENCE_ROOT
        / "soft_nuclear_collisions"
        / "validation"
        / "runs"
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
    for source_manifest in args.reuse_compatible_from:
        reused, incompatible = reuse_compatible_workflow_results(
            source_manifest, manifest_path
        )
        print(
            f"Compatible reuse from {source_manifest}: {reused} reused, "
            f"{incompatible} incompatible or incomplete"
        )
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
