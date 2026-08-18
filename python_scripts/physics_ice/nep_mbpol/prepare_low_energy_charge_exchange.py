#!/usr/bin/env python3
"""Plan requested low-energy capture channels pending CDFT branch handoff."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ion_ice import PROCESS_EVIDENCE_ROOT
from low_energy_charge_exchange import build_workflow, load_workflow_manifest
from soft_dft import (
    DEFAULT_CP2K_SETTINGS,
    build_scan_geometries,
    load_builtin_projectile,
    load_projectile_definition,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projectile", help="Built-in element symbol; defaults to C.")
    parser.add_argument("--projectile-definition", type=Path)
    parser.add_argument(
        "--incident-charges",
        nargs="+",
        type=int,
        help="Subset of incident q values; defaults to every q=1..Z.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="Prepare one asymptotic geometry per capture channel.",
    )
    parser.add_argument("--projectile-basis-set")
    parser.add_argument("--water-basis-set")
    parser.add_argument("--basis-file")
    parser.add_argument("--cell-angstrom", type=float)
    parser.add_argument("--mgrid-cutoff-ry", type=float)
    parser.add_argument("--mgrid-rel-cutoff-ry", type=float)
    parser.add_argument("--scf-eps", type=float)
    parser.add_argument("--cdft-eps", type=float)
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
    replacements = {
        "projectile_basis_set": (
            args.projectile_basis_set or projectile.projectile_basis_set
        ),
        "basis_file": args.basis_file or projectile.basis_file,
    }
    optional = {
        "water_basis_set": args.water_basis_set,
        "cell_angstrom": args.cell_angstrom,
        "mgrid_cutoff_ry": args.mgrid_cutoff_ry,
        "mgrid_rel_cutoff_ry": args.mgrid_rel_cutoff_ry,
        "scf_eps": args.scf_eps,
        "cdft_eps": args.cdft_eps,
    }
    replacements.update(
        {key: value for key, value in optional.items() if value is not None}
    )
    settings = replace(DEFAULT_CP2K_SETTINGS, **replacements)
    geometries = None
    if args.runtime_smoke:
        candidates = [
            geometry
            for geometry in build_scan_geometries(projectile)
            if geometry.orientation == "oxygen_back"
            and geometry.separation_angstrom == 6.0
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "The registered scan needs one oxygen-back 6-A smoke geometry."
            )
        geometries = tuple(candidates)
    output_root = args.output_root or (
        PROCESS_EVIDENCE_ROOT
        / "low_energy_charge_exchange"
        / "validation"
        / "runs"
        / f"{projectile.symbol.lower()}_single_capture"
    )
    manifest_path = build_workflow(
        output_root,
        projectile=projectile,
        settings=settings,
        incident_charges=args.incident_charges,
        geometries=geometries,
    )
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    print(f"Workflow: {manifest_path}")
    print(
        f"Projectile ladder: q=0..{projectile.atomic_number}; "
        f"capture channels: {manifest['channel_count']}"
    )
    for channel in manifest["configuration"]["channels"]:
        print(f"  {channel['reaction']} [{channel['status']}]")
    print(f"Channel/geometry work units: {manifest['work_unit_count']}")
    print(
        "Integration status: branch_handoff_pending; no CDFT inputs, "
        "couplings, or cross sections generated"
    )


if __name__ == "__main__":
    main()
